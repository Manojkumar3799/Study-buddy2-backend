"""Service layer for extracting and cleaning text from PDFs."""

from pathlib import Path

import fitz  # PyMuPDF

from app.core.exceptions import DocumentNotFoundError, TextExtractionError
from app.core.logging_config import get_logger
from app.utils.text_cleaning import clean_text, is_text_meaningful

logger = get_logger(__name__)

UPLOAD_DIR = Path("storage/uploads")


class PageText:
    """Container for a single page's extracted text."""

    def __init__(self, page_number: int, text: str) -> None:
        self.page_number = page_number
        self.text = text


def get_pdf_path(document_id: str) -> Path:
    """
    Resolve the file path for a given document ID.

    Args:
        document_id: Unique identifier of the uploaded document.

    Returns:
        Path: Path to the stored PDF file.

    Raises:
        DocumentNotFoundError: If no file exists for the given ID.
    """
    file_path = UPLOAD_DIR / f"{document_id}.pdf"
    if not file_path.exists():
        logger.warning(f"Document not found: {document_id}")
        raise DocumentNotFoundError(document_id)
    return file_path


def extract_text_from_pdf(document_id: str) -> list[PageText]:
    """
    Extract and clean text from every page of a stored PDF.

    Args:
        document_id: Unique identifier of the uploaded document.

    Returns:
        list[PageText]: Cleaned text for each page.

    Raises:
        DocumentNotFoundError: If the document does not exist.
        TextExtractionError: If no meaningful text is found in the PDF.
    """
    file_path = get_pdf_path(document_id)

    logger.info(f"Starting text extraction: document_id={document_id}")

    document = fitz.open(file_path)
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


def get_combined_text(pages: list[PageText]) -> str:
    """
    Combine per-page text into a single document string with page markers.

    Args:
        pages: List of extracted page texts.

    Returns:
        str: Combined text with page boundary markers for traceability.
    """
    return "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages if p.text)