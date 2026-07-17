"""Service layer for PDF validation and storage."""

import uuid
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.exceptions import (
    CorruptedPDFError,
    EmptyPDFError,
    InvalidPDFError,
    PDFTooLargeError,
    StudyForgeException,
)
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = settings.max_pdf_size_mb * 1024 * 1024


def validate_content_type(content_type: str | None) -> None:
    """
    Validate that the uploaded file has a PDF content type.

    Args:
        content_type: The MIME type reported by the upload.

    Raises:
        InvalidPDFError: If the content type is not application/pdf.
    """
    if content_type != "application/pdf":
        logger.warning(f"Rejected upload with invalid content type: {content_type}")
        raise InvalidPDFError(
            f"Invalid file type '{content_type}'. Only PDF files are accepted."
        )


def validate_file_size(file_bytes: bytes) -> None:
    """
    Validate that the uploaded file does not exceed the maximum allowed size.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Raises:
        PDFTooLargeError: If the file exceeds the configured size limit.
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        logger.warning(f"Rejected upload exceeding size limit: {len(file_bytes)} bytes")
        raise PDFTooLargeError(
            f"File size {len(file_bytes) / (1024 * 1024):.2f}MB exceeds "
            f"the {settings.max_pdf_size_mb}MB limit."
        )


def validate_and_open_pdf(file_bytes: bytes) -> fitz.Document:
    """
    Open the PDF using PyMuPDF and validate it is readable and non-empty.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.

    Returns:
        fitz.Document: The opened PDF document.

    Raises:
        CorruptedPDFError: If the file cannot be opened as a PDF.
        EmptyPDFError: If the PDF has zero pages.
    """
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.error(f"Failed to open PDF: {exc}")
        raise CorruptedPDFError() from exc

    if document.page_count == 0:
        document.close()
        logger.warning("Rejected PDF with zero pages")
        raise EmptyPDFError()

    return document


def save_pdf_to_disk(file_bytes: bytes, document_id: str) -> Path:
    """
    Persist the uploaded PDF bytes to local storage.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        document_id: Unique identifier used as the filename.

    Returns:
        Path: The path where the file was saved.

    Raises:
        StudyForgeException: If the file cannot be written to disk
            (e.g. permissions issue, disk full).
    """
    file_path = UPLOAD_DIR / f"{document_id}.pdf"
    try:
        file_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.error(f"Failed to write PDF to disk: {exc}")
        raise StudyForgeException(
            "Failed to save the uploaded file. Please try again.", status_code=500
        ) from exc
    logger.info(f"Saved PDF to disk: {file_path}")
    return file_path

def generate_document_id() -> str:
    """Generate a unique document identifier."""
    return str(uuid.uuid4())