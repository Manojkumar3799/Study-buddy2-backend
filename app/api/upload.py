"""API routes for PDF upload."""

import time

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.auth import get_current_user
from app.core.exceptions import StudyForgeException
from app.core.logging_config import get_logger
from app.models.schemas import ErrorResponse, UploadResponse
from app.services.pdf_service import (
    generate_document_id,
    upload_pdf_to_storage,
    validate_and_open_pdf,
    validate_content_type,
    validate_file_size,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid, corrupted, or empty PDF"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authentication token"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    user_id: str = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload and validate a PDF document.

    Requires a valid Supabase Auth JWT in the ``Authorization: Bearer <token>``
    header.  The uploaded file is stored under the authenticated user's folder
    in Supabase Storage (``{user_id}/{document_id}.pdf``) so users cannot
    access each other's files.

    Args:
        file: The uploaded PDF file.
        user_id: Injected by the ``get_current_user`` dependency.

    Returns:
        UploadResponse: Metadata about the stored document.
    """
    start = time.perf_counter()
    logger.info(
        f"Upload received: filename={file.filename}, content_type={file.content_type}, "
        f"user_id={user_id}"
    )

    validate_content_type(file.content_type)

    file_bytes = await file.read()
    validate_file_size(file_bytes)

    document = validate_and_open_pdf(file_bytes)
    page_count = document.page_count
    document.close()

    document_id = generate_document_id()
    await upload_pdf_to_storage(file_bytes, document_id, user_id)

    elapsed = time.perf_counter() - start
    logger.info(
        f"Upload successful: document_id={document_id}, user_id={user_id}, "
        f"pages={page_count}, time={elapsed:.3f}s"
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown.pdf",
        page_count=page_count,
        size_bytes=len(file_bytes),
        message="PDF uploaded and validated successfully.",
    )
