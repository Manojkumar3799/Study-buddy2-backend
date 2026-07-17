"""API routes for question-based chunk retrieval."""
import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import ErrorResponse, RetrievalRequest, RetrievalResponse, RetrievedChunkResponse
from app.services.retrieval_service import retrieve_relevant_chunks

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])


@router.post(
    "/{document_id}",
    response_model=RetrievalResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Vector store not found"},
        500: {"model": ErrorResponse, "description": "Retrieval failed"},
    },
)
async def retrieve_chunks(document_id: str, request: RetrievalRequest) -> RetrievalResponse:
    """
    Retrieve the most relevant chunks for a question from a stored document.

    This endpoint is for inspecting retrieval quality before the LLM answer
    generation step is wired in.

    Args:
        document_id: Unique identifier of a previously stored document.
        request: The question and optional retrieval overrides.

    Returns:
        RetrievalResponse: Matching chunks and whether sufficient context was found.
    """
    top_k = request.top_k or settings.retrieval_top_k
    threshold = (
        request.similarity_threshold
        if request.similarity_threshold is not None
        else settings.similarity_threshold
    )

    start = time.perf_counter()
    logger.info(f"Retrieval endpoint called: document_id={document_id}")

    retrieved = retrieve_relevant_chunks(
        document_id=document_id,
        question=request.question,
        top_k=top_k,
        similarity_threshold=threshold,
    )

    elapsed = time.perf_counter() - start
    logger.info(
        f"Retrieval endpoint complete: document_id={document_id}, "
        f"matches={len(retrieved)}, time={elapsed:.3f}s"
    )

    chunk_responses = [
        RetrievedChunkResponse(
            chunk_id=c.chunk_id,
            text=c.text,
            start_page=c.start_page,
            end_page=c.end_page,
            score=c.score,
        )
        for c in retrieved
    ]

    return RetrievalResponse(
        document_id=document_id,
        question=request.question,
        top_k=top_k,
        similarity_threshold=threshold,
        total_matches=len(chunk_responses),
        chunks=chunk_responses,
        has_sufficient_context=len(chunk_responses) > 0,
    )