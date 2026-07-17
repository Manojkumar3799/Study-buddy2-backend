"""API routes for question-answering over a stored document."""

import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.exceptions import StudyForgeException
from app.core.logging_config import get_logger
from app.models.schemas import AskRequest, AskResponse, ErrorResponse, RetrievedChunkResponse
from app.prompts.rag_prompt import build_rag_prompt
from app.services.llm_service import generate_answer, stream_answer
from app.services.retrieval_service import retrieve_relevant_chunks

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ask", tags=["Question Answering"])

NO_CONTEXT_MESSAGE = (
    "The uploaded document does not contain relevant information for your question."
)


def _resolve_retrieval_params(request: AskRequest) -> tuple[int, float]:
    """
    Resolve effective top_k and similarity_threshold from request overrides
    and configured defaults.

    Args:
        request: The incoming ask request.

    Returns:
        tuple[int, float]: Effective top_k and similarity_threshold.
    """
    top_k = request.top_k or settings.retrieval_top_k
    threshold = (
        request.similarity_threshold
        if request.similarity_threshold is not None
        else settings.similarity_threshold
    )
    return top_k, threshold


@router.post(
    "/{document_id}",
    response_model=AskResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Vector store not found"},
        503: {"model": ErrorResponse, "description": "All LLM providers failed"},
    },
    summary="Ask a question (non-streaming, full JSON response)",
)
async def ask_question(document_id: str, request: AskRequest) -> AskResponse:
    """
    Answer a question using only the content of a previously stored document.

    Non-streaming variant: waits for the full answer before responding.
    Useful for testing/debugging; the frontend should use /ask/{document_id}/stream.

    Args:
        document_id: Unique identifier of a previously stored document.
        request: The question and optional retrieval overrides.

    Returns:
        AskResponse: The generated answer, sources, and provider used.
    """
    ask_start = time.perf_counter()
    logger.info(f"Ask requested: document_id={document_id}, question='{request.question[:80]}'")

    top_k, threshold = _resolve_retrieval_params(request)

    retrieved = retrieve_relevant_chunks(
        document_id=document_id,
        question=request.question,
        top_k=top_k,
        similarity_threshold=threshold,
    )

    if not retrieved:
        logger.info(f"No sufficient context found for document_id={document_id}; skipping LLM call")
        return AskResponse(
            document_id=document_id,
            question=request.question,
            answer=NO_CONTEXT_MESSAGE,
            provider_used=None,
            sources=[],
            has_sufficient_context=False,
        )

    context_chunks = [
        {"text": c.text, "start_page": c.start_page, "end_page": c.end_page} for c in retrieved
    ]
    messages = build_rag_prompt(request.question, context_chunks)

    answer, provider_used = generate_answer(messages)

    sources = [
        RetrievedChunkResponse(
            chunk_id=c.chunk_id,
            text=c.text,
            start_page=c.start_page,
            end_page=c.end_page,
            score=c.score,
        )
        for c in retrieved
    ]

    ask_elapsed = time.perf_counter() - ask_start
    logger.info(
        f"Ask complete: document_id={document_id}, provider={provider_used}, "
        f"time={ask_elapsed:.3f}s"
    )

    return AskResponse(
        document_id=document_id,
        question=request.question,
        answer=answer,
        provider_used=provider_used,
        sources=sources,
        has_sufficient_context=True,
    )


@router.post(
    "/{document_id}/stream",
    responses={
        404: {"model": ErrorResponse, "description": "Vector store not found"},
    },
    summary="Ask a question (streaming, Server-Sent-Events style)",
)
async def ask_question_stream(document_id: str, request: AskRequest) -> StreamingResponse:
    """
    Answer a question with the response streamed token-by-token as it is
    generated, ChatGPT-style.

    The stream emits newline-delimited JSON events:
    - {"type": "sources", "sources": [...], "has_sufficient_context": bool}
      sent once, immediately, before any answer tokens.
    - {"type": "token", "content": "..."} sent repeatedly as tokens arrive.
    - {"type": "done", "provider_used": "gemini"} sent once at the end.
    - {"type": "error", "detail": "..."} sent if all providers fail.

    Args:
        document_id: Unique identifier of a previously stored document.
        request: The question and optional retrieval overrides.

    Returns:
        StreamingResponse: A newline-delimited JSON event stream.
    """
    logger.info(
        f"Streaming ask requested: document_id={document_id}, question='{request.question[:80]}'"
    )

    top_k, threshold = _resolve_retrieval_params(request)

    # Retrieval happens before streaming starts, so a 404 (bad document_id)
    # still surfaces as a normal HTTP error rather than being swallowed
    # inside the stream.
    retrieved = retrieve_relevant_chunks(
        document_id=document_id,
        question=request.question,
        top_k=top_k,
        similarity_threshold=threshold,
    )

    sources_payload = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "score": c.score,
        }
        for c in retrieved
    ]

    async def event_generator():
        """Yield newline-delimited JSON events for the streaming response."""
        stream_start = time.perf_counter()
        logger.info(f"Stream started: document_id={document_id}")

        yield json.dumps(
            {
                "type": "sources",
                "sources": sources_payload,
                "has_sufficient_context": bool(retrieved),
            }
        ) + "\n"

        if not retrieved:
            logger.info(
                f"No sufficient context found for document_id={document_id}; skipping LLM call"
            )
            yield json.dumps({"type": "token", "content": NO_CONTEXT_MESSAGE}) + "\n"
            yield json.dumps({"type": "done", "provider_used": None}) + "\n"
            stream_elapsed = time.perf_counter() - stream_start
            logger.info(f"Stream ended: document_id={document_id}, time={stream_elapsed:.3f}s")
            return

        context_chunks = [
            {"text": c.text, "start_page": c.start_page, "end_page": c.end_page} for c in retrieved
        ]
        messages = build_rag_prompt(request.question, context_chunks)

        provider_used = None
        try:
            for token in stream_answer(messages):
                yield json.dumps({"type": "token", "content": token}) + "\n"
        except StudyForgeException as exc:
            logger.error(
                f"Streaming failed for document_id={document_id}: "
                f"{exc.__class__.__name__}: {exc.message}"
            )
            yield json.dumps(
                {"type": "error", "error": exc.__class__.__name__, "detail": exc.message}
            ) + "\n"
            stream_elapsed = time.perf_counter() - stream_start
            logger.info(
                f"Stream ended with error: document_id={document_id}, time={stream_elapsed:.3f}s"
            )
            return

        yield json.dumps({"type": "done", "provider_used": provider_used}) + "\n"
        stream_elapsed = time.perf_counter() - stream_start
        logger.info(f"Stream ended: document_id={document_id}, time={stream_elapsed:.3f}s")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")