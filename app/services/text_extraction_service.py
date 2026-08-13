"""Service layer for extracting and cleaning text from PDFs.

Text extraction now operates on raw PDF bytes rather than a local file path,
making it compatible with Supabase Storage-backed deployments where no local
PDF file exists on disk.
"""

import fitz  # PyMuPDF

from app.core.exceptions import TextExtractionError
from app.core.logging_config import get_logger
from app.utils.text_cleaning import clean_text, is_text_meaningful

logger = get_logger(__name__)


class PageText:
    """Container for a single page's extracted text."""

    def __init__(self, page_number: int, text: str) -> None:
        self.page_number = page_number
        self.text = text


def extract_text_from_pdf_bytes(pdf_bytes: bytes, document_id: str = "<unknown>") -> list[PageText]:
    """
    Extract and clean text from every page of a PDF supplied as raw bytes.

    Args:
        pdf_bytes: Raw bytes of the PDF document.
        document_id: Optional identifier used in log messages.

    Returns:
        list[PageText]: Cleaned text for each page.

    Raises:
        TextExtractionError: If no meaningful text is found in the PDF.
    """
    logger.info(f"Starting text extraction: document_id={document_id}")

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[PageText] = []

    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            pages.append(PageText(page_number=page_index + 1, text=cleaned))
    finally:
        document.close()

    total_words = sum(len(p.text.split()) for p in pages)
    combined_text = " ".join(p.text for p in pages)

    if not is_text_meaningful(combined_text, min_words=10):
        logger.warning(f"Extraction produced insufficient text: document_id={document_id}")
        raise TextExtractionError()

    logger.info(
        f"Extraction complete: document_id={document_id}, "
        f"pages={len(pages)}, total_words={total_words}"
    )

    return pages


async def extract_text_from_pdf(document_id: str) -> list[PageText]:
    """
    Backward-compatible entry point that fetches PDF from Supabase Storage and extracts text.
    """
    from app.services.pdf_service import download_pdf_from_storage
    pdf_bytes = await download_pdf_from_storage(document_id)
    return extract_text_from_pdf_bytes(pdf_bytes, document_id=document_id)


def get_combined_text(pages: list[PageText]) -> str:
    """
    Combine per-page text into a single document string with page markers.

    Args:
        pages: List of extracted page texts.

    Returns:
        str: Combined text with page boundary markers for traceability.
    """
    return "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages if p.text)
