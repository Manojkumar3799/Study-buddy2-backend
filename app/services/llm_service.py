"""Service layer for LLM completion with automatic provider fallback.

LLM stack: LangChain chat model integrations
  Primary:   ChatGoogleGenerativeAI  (langchain-google-genai >= 4.x)
  Secondary: ChatGroq                (langchain-groq)
  Tertiary:  ChatXAI                 (langchain-xai)

Fallback / retry strategy mirrors the original LiteLLM implementation:
  - Each provider is attempted up to `llm_max_retries_per_provider` times
    with exponential back-off before falling through to the next provider.
  - Non-retriable errors (auth, bad-request) abort the current provider
    immediately without retry and without trying subsequent providers.
  - Once all providers are exhausted the last low-level exception is
    mapped to a typed StudyForge exception via `_classify_final_failure`.

Provider SDK exception taxonomy (verified against installed versions):
  Gemini (google-genai >= 2.x, via langchain-google-genai >= 4.x):
    google.genai.errors.APIError   – base class (.code: int, .status: str)
    google.genai.errors.ClientError  – 4xx errors (auth=401/403, bad-req=400,
                                        rate-limit=429)
    google.genai.errors.ServerError  – 5xx errors (transient)
    NOTE: 429 rate-limit surfaces as ClientError with .code == 429 /
          .status == "RESOURCE_EXHAUSTED", NOT as its own class.

  Groq (groq SDK):
    groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError,
    groq.APITimeoutError  → retriable
    groq.AuthenticationError, groq.BadRequestError  → non-retriable

  xAI / Grok (openai SDK via langchain-openai, used by langchain-xai):
    openai.RateLimitError, openai.InternalServerError,
    openai.APIConnectionError, openai.APITimeoutError  → retriable
    openai.AuthenticationError, openai.BadRequestError  → non-retriable

NOTE: We intentionally do NOT use LangChain's `.with_fallbacks()` here
because it cannot distinguish between retriable and non-retriable errors
in the way we need (non-retriable errors must halt the entire chain, not
trigger the next fallback). The manual loop below reproduces the exact
same semantics as the original LiteLLM code.
"""

import time
from dataclasses import dataclass

# ── LangChain chat model imports ─────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI

# ── Provider SDK exceptions ──────────────────────────────────────────────────
# Gemini — google-genai SDK v2.x (used by langchain-google-genai >= 4.0)
import google.genai.errors as _google_exc

# Groq — groq SDK
import groq as _groq_sdk

# xAI — openai SDK (ChatXAI inherits from BaseChatOpenAI / langchain-openai)
import openai as _openai_sdk

# ── Application internals ────────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# Exception helpers for the google-genai SDK
# ---------------------------------------------------------------------------

def _is_gemini_retriable(exc: Exception) -> bool:
    """
    Decide whether a google.genai exception is worth retrying.

    Rules:
      - ServerError (5xx): always retriable (transient).
      - ClientError (4xx):
          • 429 RESOURCE_EXHAUSTED (rate-limit) → retriable
          • everything else (400/401/403/404 …) → non-retriable
      - APIError (other): retriable (treat as transient / unknown).
      - Anything else: not a google-genai exception → return False.
    """
    if isinstance(exc, _google_exc.ServerError):
        return True
    if isinstance(exc, _google_exc.ClientError):
        # 429 is rate-limiting — retry, then fall through
        return getattr(exc, "code", None) == 429
    if isinstance(exc, _google_exc.APIError):
        return True
    return False


def _is_gemini_non_retriable(exc: Exception) -> bool:
    """
    Decide whether a google.genai exception should abort immediately
    (no retry, no next-provider fallback).

    Only ClientError with a non-429 4xx code qualifies.
    """
    if isinstance(exc, _google_exc.ClientError):
        code = getattr(exc, "code", None)
        return code != 429  # 400/401/403/404 etc.
    return False


# ---------------------------------------------------------------------------
# Groq / xAI retriable / non-retriable tuples (unchanged SDK types)
# ---------------------------------------------------------------------------

_GROQ_RETRIABLE = (
    _groq_sdk.RateLimitError,
    _groq_sdk.InternalServerError,
    _groq_sdk.APIConnectionError,
    _groq_sdk.APITimeoutError,
)

_GROQ_NON_RETRIABLE = (
    _groq_sdk.AuthenticationError,
    _groq_sdk.BadRequestError,
)

_OPENAI_RETRIABLE = (
    _openai_sdk.RateLimitError,
    _openai_sdk.InternalServerError,
    _openai_sdk.APIConnectionError,
    _openai_sdk.APITimeoutError,
)

_OPENAI_NON_RETRIABLE = (
    _openai_sdk.AuthenticationError,
    _openai_sdk.BadRequestError,
)


def _is_retriable(exc: Exception) -> bool:
    """Return True if this exception should trigger a retry on the same provider."""
    return (
        _is_gemini_retriable(exc)
        or isinstance(exc, _GROQ_RETRIABLE)
        or isinstance(exc, _OPENAI_RETRIABLE)
    )


def _is_non_retriable(exc: Exception) -> bool:
    """Return True if this exception should abort immediately (no retry, no fallback)."""
    return (
        _is_gemini_non_retriable(exc)
        or isinstance(exc, _GROQ_NON_RETRIABLE)
        or isinstance(exc, _OPENAI_NON_RETRIABLE)
    )


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    model: str          # bare model name (no "provider/" prefix)
    api_key: str


def _get_provider_chain() -> list[ProviderConfig]:
    """
    Build the ordered fallback chain of providers: Gemini → Groq → Grok.

    LangChain integrations use bare model names — the "gemini/", "groq/", and
    "xai/" prefixes used by the LiteLLM routing convention are stripped here.

    Returns:
        list[ProviderConfig]: Providers in fallback priority order.
    """
    def _strip_prefix(model: str) -> str:
        """Remove 'provider/' prefix (e.g. 'gemini/gemini-2.5-flash' → 'gemini-2.5-flash')."""
        return model.split("/", 1)[1] if "/" in model else model

    return [
        ProviderConfig(
            name="gemini",
            model=_strip_prefix(settings.gemini_model),
            api_key=settings.gemini_api_key,
        ),
        ProviderConfig(
            name="groq",
            model=_strip_prefix(settings.groq_model),
            api_key=settings.groq_api_key,
        ),
        ProviderConfig(
            name="grok",
            model=_strip_prefix(settings.grok_model),
            api_key=settings.grok_api_key,
        ),
    ]


def _build_langchain_model(provider: ProviderConfig):
    """
    Instantiate the correct LangChain chat model class for a provider.

    Args:
        provider: The provider configuration.

    Returns:
        A LangChain BaseChatModel instance.
    """
    timeout = settings.llm_request_timeout_seconds
    if provider.name == "gemini":
        return ChatGoogleGenerativeAI(
            model=provider.model,
            google_api_key=provider.api_key,
            request_timeout=timeout,
            # Disable internal SDK retries — we manage retries ourselves
            max_retries=1,
        )
    if provider.name == "groq":
        return ChatGroq(
            model=provider.model,
            groq_api_key=provider.api_key,
            request_timeout=timeout,
            max_retries=0,  # we manage retries
        )
    if provider.name == "grok":
        return ChatXAI(
            model=provider.model,
            xai_api_key=provider.api_key,
            request_timeout=timeout,
            max_retries=0,  # we manage retries
        )
    raise ValueError(f"Unknown provider: {provider.name!r}")


def _classify_final_failure(exc: Exception) -> Exception:
    """
    Map a low-level provider exception to a typed StudyForge exception.
    Called only when ALL providers / retries have been exhausted.

    The mapping preserves the original LiteLLM-based classification:
      - Rate-limit / quota  → LLMQuotaExceededError or LLMRateLimitError
      - Timeout             → LLMTimeoutError
      - Connection error    → NetworkError
      - Anything else       → AllProvidersFailedError

    Args:
        exc: The last exception raised by the final provider attempted.

    Returns:
        Exception: A StudyForgeException subtype describing the failure.
    """
    # ── Gemini ─────────────────────────────────────────────────────────────
    if isinstance(exc, _google_exc.ClientError) and getattr(exc, "code", None) == 429:
        status = (getattr(exc, "status", "") or "").lower()
        if "quota" in status or "billing" in str(exc).lower():
            return LLMQuotaExceededError()
        return LLMRateLimitError()

    if isinstance(exc, _google_exc.ServerError):
        return NetworkError()   # 5xx → treat as transient network issue

    # ── Groq ───────────────────────────────────────────────────────────────
    if isinstance(exc, _groq_sdk.RateLimitError):
        message = str(exc).lower()
        if "quota" in message or "billing" in message:
            return LLMQuotaExceededError()
        return LLMRateLimitError()

    if isinstance(exc, _groq_sdk.APITimeoutError):
        return LLMTimeoutError()

    if isinstance(exc, _groq_sdk.APIConnectionError):
        return NetworkError()

    # ── xAI / Grok (openai SDK) ────────────────────────────────────────────
    if isinstance(exc, _openai_sdk.RateLimitError):
        message = str(exc).lower()
        if "quota" in message or "billing" in message:
            return LLMQuotaExceededError()
        return LLMRateLimitError()

    if isinstance(exc, _openai_sdk.APITimeoutError):
        return LLMTimeoutError()

    if isinstance(exc, _openai_sdk.APIConnectionError):
        return NetworkError()

    return AllProvidersFailedError()


def _call_provider_with_retry(provider: ProviderConfig, messages: list[dict]) -> str:
    """
    Call a single LLM provider with exponential-backoff retries.

    Args:
        provider: The provider configuration to call.
        messages: Chat messages in OpenAI dict format.

    Returns:
        str: The generated answer text.

    Raises:
        Exception: The last exception if all retries for this provider are
            exhausted (caller decides whether to try the next provider).
        Exception (non-retriable): Raised immediately; caller must NOT
            fall through to the next provider.
    """
    if not provider.api_key:
        logger.warning(f"Skipping provider '{provider.name}': no API key configured")
        raise ValueError(f"No API key configured for provider '{provider.name}'")

    model = _build_langchain_model(provider)
    last_exception: Exception | None = None
    max_attempts = settings.llm_max_retries_per_provider

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Calling provider '{provider.name}' (attempt {attempt}/{max_attempts})")
            start = time.perf_counter()

            response = model.invoke(messages)

            elapsed = time.perf_counter() - start
            # LangChain returns AIMessage; .content is the string answer
            answer: str = response.content  # type: ignore[assignment]

            logger.info(f"Provider '{provider.name}' succeeded in {elapsed:.2f}s")
            return answer

        except Exception as exc:
            if _is_non_retriable(exc):
                logger.error(
                    f"Provider '{provider.name}' non-retriable error: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise

            if _is_retriable(exc):
                last_exception = exc
                delay = settings.llm_retry_base_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"Provider '{provider.name}' attempt {attempt}/{max_attempts} failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                if attempt < max_attempts:
                    logger.info(f"Retrying '{provider.name}' in {delay:.1f}s")
                    time.sleep(delay)
            else:
                # Unknown / unexpected exception — log and re-raise immediately
                logger.error(
                    f"Provider '{provider.name}' unexpected error: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise

    raise last_exception  # type: ignore[misc]


def generate_answer(messages: list[dict]) -> tuple[str, str]:
    """
    Generate an answer using the provider fallback chain: Gemini → Groq → Grok.

    Each provider is retried up to the configured number of attempts with
    exponential backoff before falling through to the next provider.

    Args:
        messages: Chat messages (system + user prompt) in OpenAI dict format.

    Returns:
        tuple[str, str]: The generated answer text and the provider name
            that succeeded.

    Raises:
        StudyForgeException: A specific subtype (quota/rate-limit/timeout/
            network/generic) describing why every provider failed.
    """
    providers = _get_provider_chain()
    last_exception: Exception | None = None

    for provider in providers:
        try:
            answer = _call_provider_with_retry(provider, messages)
            return answer, provider.name
        except Exception as exc:
            last_exception = exc
            if _is_non_retriable(exc):
                # Auth/bad-request: fail immediately, do not try next provider
                logger.error(
                    f"Provider '{provider.name}' non-retriable — "
                    f"aborting fallback chain: {exc}"
                )
                break
            logger.error(f"Provider '{provider.name}' exhausted all attempts: {exc}")
            continue

    logger.error("All LLM providers failed")
    if last_exception is not None:
        raise _classify_final_failure(last_exception)
    raise AllProvidersFailedError()


def _stream_provider_with_retry(provider: ProviderConfig, messages: list[dict]):
    """
    Stream a completion from a single provider with exponential-backoff retries.

    Retries apply only to establishing the stream (connection/auth/rate-limit
    errors on the initial request). Once tokens start flowing, a mid-stream
    interruption is raised to the caller rather than silently retried, since
    partial output may have already been sent to the client.

    Args:
        provider: The provider configuration to call.
        messages: Chat messages in OpenAI dict format.

    Yields:
        str: Incremental text tokens as they arrive.

    Raises:
        Exception: The last exception if all retries are exhausted before
            streaming starts.
        Exception (non-retriable): Raised immediately on first occurrence.
    """
    if not provider.api_key:
        logger.warning(f"Skipping provider '{provider.name}': no API key configured")
        raise ValueError(f"No API key configured for provider '{provider.name}'")

    model = _build_langchain_model(provider)
    last_exception: Exception | None = None
    max_attempts = settings.llm_max_retries_per_provider

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                f"Streaming from provider '{provider.name}' (attempt {attempt}/{max_attempts})"
            )

            token_yielded = False
            try:
                for chunk in model.stream(messages):
                    # LangChain stream yields AIMessageChunk objects;
                    # .content is the token string (may be "" for metadata chunks)
                    delta: str = chunk.content  # type: ignore[assignment]
                    if delta:
                        token_yielded = True
                        yield delta
            except Exception as mid_stream_exc:
                if token_yielded:
                    # Partial output already sent — do not retry, surface as interruption
                    logger.error(
                        f"Provider '{provider.name}' interrupted mid-stream: {mid_stream_exc}"
                    )
                    raise
                raise mid_stream_exc

            if not token_yielded:
                # Stream connected but produced zero content — treat as transient
                raise RuntimeError(
                    f"Provider '{provider.name}' stream produced no content"
                )

            logger.info(f"Provider '{provider.name}' stream completed successfully")
            return

        except Exception as exc:
            if _is_non_retriable(exc):
                logger.error(
                    f"Provider '{provider.name}' non-retriable streaming error: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise

            if _is_retriable(exc):
                last_exception = exc
                delay = settings.llm_retry_base_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"Provider '{provider.name}' stream attempt {attempt}/{max_attempts} failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                if attempt < max_attempts:
                    logger.info(f"Retrying '{provider.name}' stream in {delay:.1f}s")
                    time.sleep(delay)
            else:
                logger.error(
                    f"Provider '{provider.name}' unexpected streaming error: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise

    raise last_exception  # type: ignore[misc]


def stream_answer(messages: list[dict]):
    """
    Stream an answer using the provider fallback chain: Gemini → Groq → Grok.

    If a provider fails before yielding any tokens, the next provider in the
    chain is tried. If a provider fails mid-stream (after some tokens were
    already sent), a StreamingInterruptedError is raised rather than switching
    providers, to avoid sending a duplicate/mixed answer to the client.

    Args:
        messages: Chat messages (system + user prompt) in OpenAI dict format.

    Yields:
        str | tuple[str, str]: Incremental text tokens as plain ``str``, followed
            by a single sentinel ``("__provider__", provider_name)`` tuple after
            all tokens have been yielded.  Callers should check
            ``isinstance(item, tuple)`` to detect the sentinel; all other items
            are token strings.

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
            # Sentinel: lets the caller know which provider succeeded.
            yield ("__provider__", provider.name)
            return
        except Exception as exc:
            last_exception = exc
            if any_token_sent:
                logger.error(
                    f"Provider '{provider.name}' failed mid-stream after partial output: {exc}"
                )
                raise StreamingInterruptedError() from exc
            if _is_non_retriable(exc):
                # Auth/bad-request: fail immediately, do not try next provider
                logger.error(
                    f"Provider '{provider.name}' non-retriable streaming error — "
                    f"aborting fallback chain: {exc}"
                )
                break
            logger.error(f"Provider '{provider.name}' failed before any output: {exc}")
            continue

    logger.error("All LLM providers failed to produce a streaming response")
    if last_exception is not None:
        raise _classify_final_failure(last_exception)
    raise AllProvidersFailedError()