"""API routes for embedding generation."""

import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import EmbeddingPreview, EmbeddingResponse, ErrorResponse
from app.services.chunking_service import chunk_pages
from app.services.embedding_service import generate_embeddings, get_embedding_dimension
from app.services.text_extraction_service import extract_text_from_pdf

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/embed", tags=["Embeddings"])

PREVIEW_CHUNK_LIMIT = 5
PREVIEW_VECTOR_LENGTH = 8


@router.post(
    "/{document_id}",
    response_model=EmbeddingResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "No meaningful text extracted"},
        500: {"model": ErrorResponse, "description": "Embedding generation failed"},
    },
)
async def embed_document(document_id: str) -> EmbeddingResponse:
    """
    Extract, chunk, and generate embeddings for a document's text.

    This endpoint is for inspection/testing of the embedding step only.
    Embeddings are not yet persisted (added in the FAISS storage feature).

    Args:
        document_id: Unique identifier returned by the /upload endpoint.

    Returns:
        EmbeddingResponse: Embedding metadata and truncated vector previews.
    """
    logger.info(f"Embedding requested: document_id={document_id}")

    pages = extract_text_from_pdf(document_id)
    chunks = chunk_pages(pages)

    start = time.perf_counter()
    embeddings = await generate_embeddings(chunks)
    elapsed = time.perf_counter() - start

    logger.info(
        f"Embedding endpoint complete: document_id={document_id}, "
        f"chunks_embedded={len(embeddings)}, time={elapsed:.3f}s"
    )

    previews = [
        EmbeddingPreview(
            chunk_id=chunk.chunk_id,
            word_count=chunk.word_count,
            embedding_preview=embedding[:PREVIEW_VECTOR_LENGTH],
        )
        for chunk, embedding in zip(chunks[:PREVIEW_CHUNK_LIMIT], embeddings[:PREVIEW_CHUNK_LIMIT])
    ]

    return EmbeddingResponse(
        document_id=document_id,
        model_name=settings.embedding_model_name,
        embedding_dimension=get_embedding_dimension(),
        total_chunks_embedded=len(embeddings),
        processing_time_seconds=round(elapsed, 3),
        previews=previews,
    )