"""API routes for question-answering over a stored document.

Routing logic (mode field on AskRequest):
  'pdf'  → always use PDF RAG (existing path, unchanged)
  'web'  → always use Firecrawl MCP web research
  'auto' → LLM / keyword router decides (default)

Streaming event protocol (newline-delimited JSON):
  PDF path:
    {"type": "sources", "source_type": "pdf",
     "sources": [{chunk_id, text, start_page, end_page, score}, ...],
     "has_sufficient_context": bool}
    {"type": "token", "content": "..."}  × N
    {"type": "done", "provider_used": "gemini"}

  Web path:
    {"type": "sources", "source_type": "web",
     "sources": [{"title": "...", "url": "...", "domain": "..."}, ...],
     "has_sufficient_context": bool}
    {"type": "token", "content": "..."}  × N
    {"type": "done", "provider_used": "gemini"}

  Both paths:
    {"type": "error", "error": "...", "detail": "..."} on failure
"""

import asyncio
import json
import time
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.exceptions import StudyForgeException
from app.core.logging_config import get_logger
from app.models.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    RetrievedChunkResponse,
    WebSourceResponse,
)
from app.prompts.rag_prompt import build_rag_prompt
from app.prompts.web_prompt import build_web_prompt
from app.services.llm_service import generate_answer, stream_answer
from app.services.mcp_client import firecrawl_search
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.routing_service import classify_route

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ask", tags=["Question Answering"])

NO_PDF_CONTEXT_MESSAGE = (
    "The uploaded document does not contain relevant information for your question."
)
NO_WEB_RESULTS_MESSAGE = (
    "🔬 Web research did not return results for your question. "
    "Please try a more specific query, or check that FIRECRAWL_API_KEY is configured."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


async def _resolve_route(
    question: str,
    mode: Literal["auto", "pdf", "web"],
) -> Literal["pdf_rag", "web_search"]:
    """
    Determine the answering route based on the request mode.

    'pdf' → pdf_rag directly (no LLM cost).
    'web' → web_search directly.
    'auto' → run classify_route in a thread executor (it may invoke the LLM).

    Args:
        question: The user's question.
        mode: The mode from the AskRequest.

    Returns:
        Literal['pdf_rag', 'web_search']
    """
    if mode == "pdf":
        logger.info("Route forced to pdf_rag (mode='pdf')")
        return "pdf_rag"
    if mode == "web":
        logger.info("Route forced to web_search (mode='web')")
        return "web_search"
    # mode == "auto": consult routing service (may block on LLM — run in executor)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, classify_route, question)


# ---------------------------------------------------------------------------
# Non-streaming endpoint
# ---------------------------------------------------------------------------

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
    Answer a question using either the stored PDF (RAG) or Firecrawl web research,
    routed according to the ``mode`` field.

    Non-streaming variant: waits for the full answer before responding.
    For real-time token streaming use ``POST /ask/{document_id}/stream``.

    Args:
        document_id: Unique identifier of a previously stored document.
        request: The question, optional retrieval overrides, and mode.

    Returns:
        AskResponse: Generated answer, sources, provider, and source_type.
    """
    ask_start = time.perf_counter()
    logger.info(
        f"Ask requested: document_id={document_id}, mode={request.mode}, "
        f"question='{request.question[:80]}'"
    )

    route = await _resolve_route(request.question, request.mode)

    # ------------------------------------------------------------------
    # Web research path
    # ------------------------------------------------------------------
    if route == "web_search":
        web_results = await firecrawl_search(request.question)
        has_context = bool(web_results)

        if not has_context:
            logger.info("No web results — returning no-context message")
            return AskResponse(
                document_id=document_id,
                question=request.question,
                answer=NO_WEB_RESULTS_MESSAGE,
                provider_used=None,
                source_type="web",
                sources=[],
                web_sources=[],
                has_sufficient_context=False,
            )

        web_context = [
            {"title": r.title, "url": r.url, "domain": r.domain, "content": r.content}
            for r in web_results
        ]
        messages = build_web_prompt(request.question, web_context)
        answer, provider_used = generate_answer(messages)

        web_sources = [
            WebSourceResponse(title=r.title, url=r.url, domain=r.domain)
            for r in web_results
        ]
        elapsed = time.perf_counter() - ask_start
        logger.info(
            f"Web ask complete: document_id={document_id}, provider={provider_used}, "
            f"sources={len(web_sources)}, time={elapsed:.3f}s"
        )
        return AskResponse(
            document_id=document_id,
            question=request.question,
            answer=answer,
            provider_used=provider_used,
            source_type="web",
            sources=[],
            web_sources=web_sources,
            has_sufficient_context=True,
        )

    # ------------------------------------------------------------------
    # PDF RAG path (existing logic, unchanged)
    # ------------------------------------------------------------------
    top_k, threshold = _resolve_retrieval_params(request)
    retrieved = await retrieve_relevant_chunks(
        document_id=document_id,
        question=request.question,
        top_k=top_k,
        similarity_threshold=threshold,
    )

    if not retrieved:
        logger.info(
            f"No sufficient context found for document_id={document_id}; skipping LLM call"
        )
        return AskResponse(
            document_id=document_id,
            question=request.question,
            answer=NO_PDF_CONTEXT_MESSAGE,
            provider_used=None,
            source_type="pdf",
            sources=[],
            web_sources=[],
            has_sufficient_context=False,
        )

    context_chunks = [
        {"text": c.text, "start_page": c.start_page, "end_page": c.end_page}
        for c in retrieved
    ]
    messages = build_rag_prompt(request.question, context_chunks)
    answer, provider_used = generate_answer(messages)

    pdf_sources = [
        RetrievedChunkResponse(
            chunk_id=c.chunk_id,
            text=c.text,
            start_page=c.start_page,
            end_page=c.end_page,
            score=c.score,
        )
        for c in retrieved
    ]
    elapsed = time.perf_counter() - ask_start
    logger.info(
        f"PDF ask complete: document_id={document_id}, provider={provider_used}, "
        f"chunks={len(pdf_sources)}, time={elapsed:.3f}s"
    )
    return AskResponse(
        document_id=document_id,
        question=request.question,
        answer=answer,
        provider_used=provider_used,
        source_type="pdf",
        sources=pdf_sources,
        web_sources=[],
        has_sufficient_context=True,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/{document_id}/stream",
    responses={
        404: {"model": ErrorResponse, "description": "Vector store not found"},
    },
    summary="Ask a question (streaming, Server-Sent-Events style)",
)
async def ask_question_stream(document_id: str, request: AskRequest) -> StreamingResponse:
    """
    Answer a question with the response streamed token-by-token.

    The stream emits newline-delimited JSON events:
    - ``{"type": "sources", "source_type": "pdf"|"web", "sources": [...],
         "has_sufficient_context": bool}``
      sent once, immediately, before any answer tokens.
    - ``{"type": "token", "content": "..."}`` sent repeatedly as tokens arrive.
    - ``{"type": "done", "provider_used": "gemini"}`` sent once at the end.
    - ``{"type": "error", "detail": "..."}`` sent if all providers fail.

    Args:
        document_id: Unique identifier of a previously stored document.
        request: The question, optional retrieval overrides, and mode.

    Returns:
        StreamingResponse: A newline-delimited JSON event stream.
    """
    logger.info(
        f"Streaming ask requested: document_id={document_id}, mode={request.mode}, "
        f"question='{request.question[:80]}'"
    )

    # ------------------------------------------------------------------
    # Pre-stream: resolve route and fetch source data.
    # Both steps happen BEFORE the StreamingResponse starts so that
    # HTTP-level errors (404 from bad document_id, etc.) still surface
    # as proper HTTP status codes rather than being swallowed in the stream.
    # ------------------------------------------------------------------
    route = await _resolve_route(request.question, request.mode)

    if route == "web_search":
        # Fetch web results before streaming starts
        web_results = await firecrawl_search(request.question)
        retrieved = None  # not used on web path
    else:
        # Fetch PDF chunks before streaming starts (may raise VectorStoreNotFoundError → 404)
        top_k, threshold = _resolve_retrieval_params(request)
        retrieved = await retrieve_relevant_chunks(
            document_id=document_id,
            question=request.question,
            top_k=top_k,
            similarity_threshold=threshold,
        )
        web_results = []

    # ------------------------------------------------------------------
    # Build sources payloads for the first stream event
    # ------------------------------------------------------------------
    if route == "web_search":
        sources_payload = [
            {"title": r.title, "url": r.url, "domain": r.domain}
            for r in web_results
        ]
        has_context = bool(web_results)
    else:
        sources_payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "start_page": c.start_page,
                "end_page": c.end_page,
                "score": c.score,
            }
            for c in (retrieved or [])
        ]
        has_context = bool(retrieved)

    # ------------------------------------------------------------------
    # Async event generator (captures route + data via closure)
    # ------------------------------------------------------------------
    async def event_generator():
        """Yield newline-delimited JSON events for the streaming response."""
        stream_start = time.perf_counter()
        logger.info(f"Stream started: document_id={document_id}, route={route}")

        # ---- sources event (always first) --------------------------------
        yield json.dumps(
            {
                "type": "sources",
                "source_type": "web" if route == "web_search" else "pdf",
                "sources": sources_payload,
                "has_sufficient_context": has_context,
            }
        ) + "\n"

        # ---- no-context short-circuit ------------------------------------
        if not has_context:
            no_ctx_msg = (
                NO_WEB_RESULTS_MESSAGE
                if route == "web_search"
                else NO_PDF_CONTEXT_MESSAGE
            )
            logger.info(
                f"No context found for document_id={document_id} (route={route})"
            )
            yield json.dumps({"type": "token", "content": no_ctx_msg}) + "\n"
            yield json.dumps({"type": "done", "provider_used": None}) + "\n"
            logger.info(
                f"Stream ended (no context): document_id={document_id}, "
                f"time={time.perf_counter() - stream_start:.3f}s"
            )
            return

        # ---- build LLM prompt --------------------------------------------
        if route == "web_search":
            web_context = [
                {
                    "title": r.title,
                    "url": r.url,
                    "domain": r.domain,
                    "content": r.content,
                }
                for r in web_results
            ]
            messages = build_web_prompt(request.question, web_context)
        else:
            context_chunks = [
                {"text": c.text, "start_page": c.start_page, "end_page": c.end_page}
                for c in (retrieved or [])
            ]
            messages = build_rag_prompt(request.question, context_chunks)

        # ---- stream tokens from LLM fallback chain -----------------------
        provider_used = None
        try:
            for item in stream_answer(messages):
                if isinstance(item, tuple) and item[0] == "__provider__":
                    # Sentinel from stream_answer carrying the winning provider name
                    provider_used = item[1]
                else:
                    yield json.dumps({"type": "token", "content": item}) + "\n"
        except StudyForgeException as exc:
            logger.error(
                f"Streaming failed for document_id={document_id}: "
                f"{exc.__class__.__name__}: {exc.message}"
            )
            yield json.dumps(
                {"type": "error", "error": exc.__class__.__name__, "detail": exc.message}
            ) + "\n"
            logger.info(
                f"Stream ended with error: document_id={document_id}, "
                f"time={time.perf_counter() - stream_start:.3f}s"
            )
            return

        yield json.dumps({"type": "done", "provider_used": provider_used}) + "\n"
        logger.info(
            f"Stream ended: document_id={document_id}, route={route}, "
            f"provider={provider_used}, time={time.perf_counter() - stream_start:.3f}s"
        )

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")