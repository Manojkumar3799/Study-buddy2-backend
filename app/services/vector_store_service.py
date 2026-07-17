"""Service layer for persisting and querying per-document FAISS vector stores."""

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.core.exceptions import VectorStoreError, VectorStoreNotFoundError
from app.core.logging_config import get_logger
from app.services.chunking_service import Chunk, chunk_pages
from app.services.embedding_service import generate_embeddings, get_embedding_dimension
from app.services.text_extraction_service import extract_text_from_pdf

logger = get_logger(__name__)

VECTOR_STORE_DIR = Path("storage/vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _index_path(document_id: str) -> Path:
    """Return the FAISS index file path for a document."""
    return VECTOR_STORE_DIR / f"{document_id}.index"


def _metadata_path(document_id: str) -> Path:
    """Return the chunk metadata file path for a document."""
    return VECTOR_STORE_DIR / f"{document_id}.metadata.json"


def vector_store_exists(document_id: str) -> bool:
    """
    Check whether a FAISS index and metadata file exist for a document.

    Args:
        document_id: Unique identifier of the document.

    Returns:
        bool: True if both index and metadata files exist.
    """
    return _index_path(document_id).exists() and _metadata_path(document_id).exists()


def build_and_save_vector_store(document_id: str) -> dict[str, Any]:
    """
    Run the full pipeline (extract -> chunk -> embed) and persist a FAISS
    index plus chunk metadata for the given document.

    Args:
        document_id: Unique identifier of the previously uploaded document.

    Returns:
        dict[str, Any]: Summary of the storage operation.

    Raises:
        DocumentNotFoundError: If the document does not exist.
        TextExtractionError: If no meaningful text is found.
        EmbeddingGenerationError: If embedding generation fails.
        VectorStoreError: If FAISS index construction or saving fails.
    """
    logger.info(f"Building vector store: document_id={document_id}")
    start = time.perf_counter()

    pages = extract_text_from_pdf(document_id)
    chunks: list[Chunk] = chunk_pages(pages)

    if not chunks:
        raise VectorStoreError("No chunks available to build a vector store.")

    embeddings = generate_embeddings(chunks)
    dimension = get_embedding_dimension()

    try:
        vectors = np.array(embeddings, dtype="float32")
        # Embeddings are already normalized -> inner product == cosine similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        faiss.write_index(index, str(_index_path(document_id)))
    except Exception as exc:
        logger.error(f"FAISS index build/save failed: {exc}")
        raise VectorStoreError() from exc

    metadata = {
        "document_id": document_id,
        "embedding_dimension": dimension,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
            }
            for chunk in chunks
        ],
    }

    try:
        _metadata_path(document_id).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.error(f"Metadata save failed: {exc}")
        raise VectorStoreError() from exc

    elapsed = time.perf_counter() - start
    logger.info(
        f"Vector store built: document_id={document_id}, chunks={len(chunks)}, "
        f"dimension={dimension}, time={elapsed:.2f}s"
    )

    return {
        "document_id": document_id,
        "total_chunks_stored": len(chunks),
        "embedding_dimension": dimension,
        "processing_time_seconds": round(elapsed, 3),
    }


def load_vector_store(document_id: str) -> tuple[faiss.Index, dict[str, Any]]:
    """
    Load a document's FAISS index and metadata from disk.

    Args:
        document_id: Unique identifier of the document.

    Returns:
        tuple[faiss.Index, dict[str, Any]]: The FAISS index and its metadata.

    Raises:
        VectorStoreNotFoundError: If no store exists for the document.
        VectorStoreError: If loading fails unexpectedly.
    """
    if not vector_store_exists(document_id):
        raise VectorStoreNotFoundError(document_id)

    try:
        index = faiss.read_index(str(_index_path(document_id)))
        metadata = json.loads(_metadata_path(document_id).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Failed to load vector store for {document_id}: {exc}")
        raise VectorStoreError() from exc

    return index, metadata


def get_vector_store_info(document_id: str) -> dict[str, Any]:
    """
    Return summary info about a document's stored vector index.

    Args:
        document_id: Unique identifier of the document.

    Returns:
        dict[str, Any]: Summary metadata (chunk count, dimension, etc).

    Raises:
        VectorStoreNotFoundError: If no store exists for the document.
    """
    _, metadata = load_vector_store(document_id)
    return {
        "document_id": metadata["document_id"],
        "total_chunks_stored": metadata["total_chunks"],
        "embedding_dimension": metadata["embedding_dimension"],
    }