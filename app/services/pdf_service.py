"""Service layer for PDF validation and Supabase Storage upload."""

import uuid

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.exceptions import (
    CorruptedPDFError,
    EmptyPDFError,
    InvalidPDFError,
    PDFTooLargeError,
    StorageError,
    StudyForgeException,
)
from app.core.logging_config import get_logger
from app.services.supabase_client import get_storage_client

logger = get_logger(__name__)
settings = get_settings()

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


async def upload_pdf_to_storage(file_bytes: bytes, document_id: str, user_id: str) -> str:
    """
    Upload the PDF bytes to Supabase Storage.

    The file is stored at ``{user_id}/{document_id}.pdf`` inside the configured
    bucket, providing per-user file isolation at the storage layer.  The upload
    uses upsert semantics so re-processing the same document_id simply overwrites
    the previous file rather than raising a conflict error.

    Args:
        file_bytes: Raw bytes of the validated PDF.
        document_id: Unique identifier used as the storage key.
        user_id: The authenticated user's UUID (from Supabase Auth JWT).

    Returns:
        str: The storage path (``{user_id}/{document_id}.pdf``) for reference.

    Raises:
        StorageError: If the upload to Supabase Storage fails.
    """
    import asyncio
    from functools import partial

    storage_path = f"{user_id}/{document_id}.pdf"
    bucket = settings.supabase_storage_bucket

    try:
        client = get_storage_client()
        # supabase-py storage is synchronous; run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                client.storage.from_(bucket).upload,
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": "application/pdf", "upsert": "true"},
            ),
        )
    except Exception as exc:
        logger.error(f"Failed to upload PDF to Supabase Storage: {exc}")
        raise StorageError(
            f"Failed to upload PDF to cloud storage: {exc}"
        ) from exc

    logger.info(f"PDF uploaded to Supabase Storage: bucket={bucket}, path={storage_path}")
    return storage_path


async def download_pdf_from_storage(document_id: str, user_id: str) -> bytes:
    """
    Download PDF bytes from Supabase Storage.

    Args:
        document_id: Unique identifier of the previously uploaded document.
        user_id: The authenticated user's UUID. Used to construct the
            per-user storage path ``{user_id}/{document_id}.pdf``.
            If the path does not exist, the user either does not own this
            document or it was never uploaded.

    Returns:
        bytes: The raw PDF bytes.

    Raises:
        DocumentNotFoundError: If no object exists for this user/document_id.
        StorageError: If the download fails for any other reason.
    """
    import asyncio
    from functools import partial
    from app.core.exceptions import DocumentNotFoundError

    storage_path = f"{user_id}/{document_id}.pdf"
    bucket = settings.supabase_storage_bucket

    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()
        pdf_bytes: bytes = await loop.run_in_executor(
            None,
            partial(client.storage.from_(bucket).download, storage_path),
        )
    except Exception as exc:
        err_str = str(exc).lower()
        if "not found" in err_str or "404" in err_str:
            raise DocumentNotFoundError(document_id) from exc
        logger.error(f"Failed to download PDF from Supabase Storage: {exc}")
        raise StorageError(f"Failed to download PDF from cloud storage: {exc}") from exc

    logger.info(f"PDF downloaded from Supabase Storage: path={storage_path}, size={len(pdf_bytes)}")
    return pdf_bytes


def generate_document_id() -> str:
    """Generate a unique document identifier."""
    return str(uuid.uuid4())
