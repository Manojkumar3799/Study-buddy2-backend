"""Service layer for LLM completion with automatic provider fallback."""

import time
from dataclasses import dataclass

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

from app.core.config import get_settings
from app.core.exceptions import (
    AllProvidersFailedError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    NetworkError,
)
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

litellm.suppress_debug_info = True

# Exceptions worth retrying within the same provider (transient failures)
RETRIABLE_EXCEPTIONS = (APIConnectionError, RateLimitError, Timeout, APIError)

# Exceptions that should never be retried (config/auth/input problems)
NON_RETRIABLE_EXCEPTIONS = (AuthenticationError, BadRequestError)


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    model: str
    api_key: str


def _get_provider_chain() -> list[ProviderConfig]:
    """
    Build the ordered fallback chain of providers: Gemini -> Groq -> Grok.

    Returns:
        list[ProviderConfig]: Providers in fallback priority order.
    """
    return [
        ProviderConfig(name="gemini", model=settings.gemini_model, api_key=settings.gemini_api_key),
        ProviderConfig(name="groq", model=settings.groq_model, api_key=settings.groq_api_key),
        ProviderConfig(name="grok", model=settings.grok_model, api_key=settings.grok_api_key),
    ]


def _classify_final_failure(exc: Exception) -> Exception:
    """
    Map a low-level LiteLLM/provider exception to a specific StudyForge
    exception type, used only when ALL providers have been exhausted and
    we need to report a meaningful reason to the client.

    Args:
        exc: The last exception raised by the final provider attempted.

    Returns:
        Exception: A StudyForgeException subtype describing the failure.
    """
    if isinstance(exc, RateLimitError):
        message = str(exc).lower()
        if "quota" in message or "billing" in message:
            return LLMQuotaExceededError()
        return LLMRateLimitError()
    if isinstance(exc, Timeout):
        return LLMTimeoutError()
    if isinstance(exc, APIConnectionError):
        return NetworkError()
    return AllProvidersFailedError()


def _call_provider_with_retry(provider: ProviderConfig, messages: list[dict]) -> str:
    """
    Call a single LLM provider with exponential-backoff retries.

    Args:
        provider: The provider configuration to call.
        messages: Chat messages to send.

    Returns:
        str: The generated answer text.

    Raises:
        Exception: The last exception encountered if all retries for this
            provider are exhausted (caller decides whether to try next provider).
    """
    if not provider.api_key:
        logger.warning(f"Skipping provider '{provider.name}': no API key configured")
        raise ValueError(f"No API key configured for provider '{provider.name}'")

    last_exception: Exception | None = None
    max_attempts = settings.llm_max_retries_per_provider

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Calling provider '{provider.name}' (attempt {attempt}/{max_attempts})")
            start = time.perf_counter()

            response = litellm.completion(
                model=provider.model,
                messages=messages,
                api_key=provider.api_key,
                timeout=settings.llm_request_timeout_seconds,
                stream=False,
            )

            elapsed = time.perf_counter() - start
            answer = response.choices[0].message.content

            logger.info(f"Provider '{provider.name}' succeeded in {elapsed:.2f}s")
            return answer

        except NON_RETRIABLE_EXCEPTIONS as exc:
            logger.error(f"Provider '{provider.name}' non-retriable error: {exc.__class__.__name__}: {exc}")
            raise
        except RETRIABLE_EXCEPTIONS as exc:
            last_exception = exc
            delay = settings.llm_retry_base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                f"Provider '{provider.name}' attempt {attempt}/{max_attempts} failed: "
                f"{exc.__class__.__name__}: {exc}"
            )
            if attempt < max_attempts:
                logger.info(f"Retrying '{provider.name}' in {delay:.1f}s")
                time.sleep(delay)
        except Exception as exc:
            logger.error(f"Provider '{provider.name}' unexpected error: {exc.__class__.__name__}: {exc}")
            raise

    raise last_exception  # type: ignore[misc]


def generate_answer(messages: list[dict]) -> tuple[str, str]:
    """
    Generate an answer using the provider fallback chain: Gemini -> Groq -> Grok.

    Each provider is retried up to the configured number of attempts with
    exponential backoff before falling through to the next provider.

    Args:
        messages: Chat messages (system + user prompt) to send to the LLM.

    Returns:
        tuple[str, str]: The generated answer text and the provider name that succeeded.

    Raises:
        StudyForgeException: A specific subtype (quota/rate-limit/timeout/network/
            generic) describing why every provider failed.
    """
    providers = _get_provider_chain()
    last_exception: Exception | None = None

    for provider in providers:
        try:
            answer = _call_provider_with_retry(provider, messages)
            return answer, provider.name
        except Exception as exc:
            last_exception = exc
            logger.error(f"Provider '{provider.name}' exhausted all attempts: {exc}")
            continue

    logger.error("All LLM providers failed")
    if last_exception is not None:
        raise _classify_final_failure(last_exception)
    raise AllProvidersFailedError()


def _stream_provider_with_retry(provider: ProviderConfig, messages: list[dict]):
    """
    Stream a completion from a single provider with exponential-backoff retries.

    Retries only apply to establishing the stream (e.g. connection/auth/rate-limit
    errors on the initial request). Once tokens start flowing, a mid-stream
    interruption is raised to the caller rather than silently retried, since
    partial output may have already been sent to the client.

    Args:
        provider: The provider configuration to call.
        messages: Chat messages to send.

    Yields:
        str: Incremental text tokens as they arrive.

    Raises:
        Exception: The last exception encountered if all retries are exhausted.
    """
    if not provider.api_key:
        logger.warning(f"Skipping provider '{provider.name}': no API key configured")
        raise ValueError(f"No API key configured for provider '{provider.name}'")

    last_exception: Exception | None = None
    max_attempts = settings.llm_max_retries_per_provider

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                f"Streaming from provider '{provider.name}' (attempt {attempt}/{max_attempts})"
            )
            response_stream = litellm.completion(
                model=provider.model,
                messages=messages,
                api_key=provider.api_key,
                timeout=settings.llm_request_timeout_seconds,
                stream=True,
            )

            token_yielded = False
            try:
                for chunk in response_stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        token_yielded = True
                        yield delta
            except Exception as mid_stream_exc:
                if token_yielded:
                    # Partial output already sent - do not retry, surface as interruption
                    logger.error(
                        f"Provider '{provider.name}' interrupted mid-stream: {mid_stream_exc}"
                    )
                    raise
                raise mid_stream_exc

            if not token_yielded:
                raise APIError(
                    message="Stream produced no content",
                    llm_provider=provider.name,
                    model=provider.model,
                    status_code=502,
                )

            logger.info(f"Provider '{provider.name}' stream completed successfully")
            return

        except NON_RETRIABLE_EXCEPTIONS as exc:
            logger.error(
                f"Provider '{provider.name}' non-retriable streaming error: "
                f"{exc.__class__.__name__}: {exc}"
            )
            raise
        except RETRIABLE_EXCEPTIONS as exc:
            last_exception = exc
            delay = settings.llm_retry_base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                f"Provider '{provider.name}' stream attempt {attempt}/{max_attempts} failed: "
                f"{exc.__class__.__name__}: {exc}"
            )
            if attempt < max_attempts:
                logger.info(f"Retrying '{provider.name}' stream in {delay:.1f}s")
                time.sleep(delay)
        except Exception as exc:
            logger.error(
                f"Provider '{provider.name}' unexpected streaming error: "
                f"{exc.__class__.__name__}: {exc}"
            )
            raise

    raise last_exception  # type: ignore[misc]


def stream_answer(messages: list[dict]):
    """
    Stream an answer using the provider fallback chain: Gemini -> Groq -> Grok.

    If a provider fails before yielding any tokens, the next provider in the
    chain is tried. If a provider fails mid-stream (after some tokens were
    already sent), a StreamingInterruptedError is raised rather than switching
    providers, to avoid sending a duplicate/mixed answer to the client.

    Args:
        messages: Chat messages (system + user prompt) to send to the LLM.

    Yields:
        str: Incremental text tokens, in order, from whichever provider succeeds.

    Raises:
        StudyForgeException: A specific subtype describing why streaming failed.
    """
    from app.core.exceptions import StreamingInterruptedError

    providers = _get_provider_chain()
    last_exception: Exception | None = None

    for provider in providers:
        any_token_sent = False
        try:
            for token in _stream_provider_with_retry(provider, messages):
                any_token_sent = True
                yield token
            return
        except Exception as exc:
            last_exception = exc
            if any_token_sent:
                logger.error(
                    f"Provider '{provider.name}' failed mid-stream after partial output: {exc}"
                )
                raise StreamingInterruptedError() from exc
            logger.error(f"Provider '{provider.name}' failed before any output: {exc}")
            continue

    logger.error("All LLM providers failed to produce a streaming response")
    if last_exception is not None:
        raise _classify_final_failure(last_exception)
    raise AllProvidersFailedError()