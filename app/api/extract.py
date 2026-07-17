"""API routes for PDF text extraction."""
import time

from fastapi import APIRouter

from app.core.logging_config import get_logger
from app.models.schemas import ErrorResponse, ExtractionResponse, PageTextResponse
from app.services.text_extraction_service import extract_text_from_pdf

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["Extraction"])


@router.post(
    "/{document_id}",
    response_model=ExtractionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "No meaningful text extracted"},
    },
)
async def extract_document_text(document_id: str) -> ExtractionResponse:
    """
    Extract and clean text from a previously uploaded PDF.

    Args:
        document_id: Unique identifier returned by the /upload endpoint.

    Returns:
        ExtractionResponse: Per-page cleaned text and word counts.
    """
    start = time.perf_counter()
    logger.info(f"Extraction requested: document_id={document_id}")

    pages = extract_text_from_pdf(document_id)

    page_responses = [
        PageTextResponse(
            page_number=p.page_number,
            text=p.text,
            word_count=len(p.text.split()),
        )
        for p in pages
    ]

    total_words = sum(p.word_count for p in page_responses)

    elapsed = time.perf_counter() - start
    logger.info(
        f"Extraction endpoint complete: document_id={document_id}, "
        f"pages={len(page_responses)}, words={total_words}, time={elapsed:.3f}s"
    )

    return ExtractionResponse(
        document_id=document_id,
        total_pages=len(page_responses),
        total_words=total_words,
        pages=page_responses,
    )