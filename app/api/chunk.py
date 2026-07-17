"""API routes for text chunking."""
import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import ChunkingResponse, ChunkResponse, ErrorResponse
from app.services.chunking_service import chunk_pages
from app.services.text_extraction_service import extract_text_from_pdf

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/chunk", tags=["Chunking"])


@router.post(
    "/{document_id}",
    response_model=ChunkingResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "No meaningful text extracted"},
    },
)
async def chunk_document(document_id: str) -> ChunkingResponse:
    """
    Extract text from a document and split it into overlapping chunks.

    Args:
        document_id: Unique identifier returned by the /upload endpoint.

    Returns:
        ChunkingResponse: Ordered chunks with page-range metadata.
    """
    start = time.perf_counter()
    logger.info(f"Chunking requested: document_id={document_id}")

    pages = extract_text_from_pdf(document_id)
    chunks = chunk_pages(pages)

    chunk_responses = [
        ChunkResponse(
            chunk_id=c.chunk_id,
            text=c.text,
            word_count=c.word_count,
            start_page=c.start_page,
            end_page=c.end_page,
        )
        for c in chunks
    ]

    elapsed = time.perf_counter() - start
    logger.info(
        f"Chunking endpoint complete: document_id={document_id}, "
        f"chunks={len(chunk_responses)}, time={elapsed:.3f}s"
    )

    return ChunkingResponse(
        document_id=document_id,
        total_chunks=len(chunk_responses),
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
        chunks=chunk_responses,
    )