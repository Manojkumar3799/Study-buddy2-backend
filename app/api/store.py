"""API routes for building and inspecting per-document vector stores."""

from fastapi import APIRouter

from app.core.logging_config import get_logger
from app.models.schemas import ErrorResponse, StoreResponse, VectorStoreInfoResponse
from app.services.vector_store_service import build_and_save_vector_store, get_vector_store_info

logger = get_logger(__name__)

router = APIRouter(prefix="/store", tags=["Vector Store"])


@router.post(
    "/{document_id}",
    response_model=StoreResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "No meaningful text extracted"},
        500: {"model": ErrorResponse, "description": "Vector store build failed"},
    },
)
async def store_document(document_id: str) -> StoreResponse:
    """
    Run the full pipeline (extract -> chunk -> embed -> store) and persist
    a FAISS index plus chunk metadata for the document.

    Args:
        document_id: Unique identifier returned by the /upload endpoint.

    Returns:
        StoreResponse: Summary of the storage operation.
    """
    logger.info(f"Store requested: document_id={document_id}")

    result = build_and_save_vector_store(document_id)

    return StoreResponse(
        document_id=result["document_id"],
        total_chunks_stored=result["total_chunks_stored"],
        embedding_dimension=result["embedding_dimension"],
        processing_time_seconds=result["processing_time_seconds"],
        message="Document successfully embedded and stored in FAISS.",
    )


@router.get(
    "/{document_id}",
    response_model=VectorStoreInfoResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Vector store not found"},
        500: {"model": ErrorResponse, "description": "Vector store load failed"},
    },
)
async def get_document_store_info(document_id: str) -> VectorStoreInfoResponse:
    """
    Retrieve summary information about a document's existing vector store.

    Args:
        document_id: Unique identifier of a previously stored document.

    Returns:
        VectorStoreInfoResponse: Chunk count and embedding dimension info.
    """
    logger.info(f"Vector store info requested: document_id={document_id}")

    info = get_vector_store_info(document_id)

    return VectorStoreInfoResponse(
        document_id=info["document_id"],
        total_chunks_stored=info["total_chunks_stored"],
        embedding_dimension=info["embedding_dimension"],
    )