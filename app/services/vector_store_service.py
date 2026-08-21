"""Service layer for persisting and querying per-document chunk embeddings in pgvector.

Replaces the former FAISS-based implementation. All data lives in the
``document_chunks`` Supabase Postgres table; no local index files are written.
"""

import time
from typing import Any
import asyncio
from functools import partial

from app.core.exceptions import VectorStoreError, VectorStoreNotFoundError
from app.core.logging_config import get_logger
from app.services.chunking_service import Chunk, chunk_pages
from app.services.embedding_service import generate_embeddings, get_embedding_dimension
from app.services.pdf_service import download_pdf_from_storage
from app.services.supabase_client import get_db_pool
from app.services.text_extraction_service import extract_text_from_pdf_bytes

logger = get_logger(__name__)


def _serialize_embedding(embedding: list[float]) -> str:
    """Serialize a float list to the pgvector text literal ``'[0.1, ...]'``."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def _db_exists_sync(document_id: str, user_id: str) -> bool:
    pool = get_db_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s AND user_id = %s",
                (document_id, user_id),
            )
            count = cur.fetchone()[0]
    return count > 0


async def vector_store_exists(document_id: str, user_id: str) -> bool:
    """
    Check whether any chunk rows exist for a document in pgvector.

    Args:
        document_id: Unique identifier of the document.
        user_id: The authenticated user's UUID.

    Returns:
        bool: True if at least one chunk row exists for this user + document.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _db_exists_sync, document_id, user_id)


def _db_save_sync(document_id: str, user_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    pool = get_db_pool()
    with pool.connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # Delete any pre-existing rows for this document OWNED BY this user
                cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s AND user_id = %s",
                    (document_id, user_id),
                )

                # Bulk insert all chunks with their embeddings and user_id
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO document_chunks
                            (document_id, user_id, chunk_id, text, start_page, end_page, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            document_id,
                            user_id,
                            chunk.chunk_id,
                            chunk.text,
                            chunk.start_page,
                            chunk.end_page,
                            _serialize_embedding(emb),
                        )
                    )


async def build_and_save_vector_store(document_id: str, user_id: str) -> dict[str, Any]:
    """
    Run the full pipeline (download -> extract -> chunk -> embed -> upsert) and
    persist all chunk embeddings to the ``document_chunks`` pgvector table.

    Existing rows for this document_id + user_id are deleted first so
    re-processing is always idempotent.

    Args:
        document_id: Unique identifier of the previously uploaded document.
        user_id: The authenticated user's UUID. Used for storage path lookup
            and to tag chunk rows so only this user can query them.

    Returns:
        dict[str, Any]: Summary of the storage operation.

    Raises:
        DocumentNotFoundError: If the PDF is not in Supabase Storage.
        TextExtractionError: If no meaningful text is found.
        EmbeddingGenerationError: If embedding generation fails.
        VectorStoreError: If the Postgres upsert fails.
    """
    logger.info(f"Building pgvector store: document_id={document_id}, user_id={user_id}")
    start = time.perf_counter()

    # 1. Fetch PDF bytes from Supabase Storage (user-scoped path)
    pdf_bytes = await download_pdf_from_storage(document_id, user_id)

    # 2. Extract text (works on raw bytes, no local file needed)
    pages = extract_text_from_pdf_bytes(pdf_bytes, document_id=document_id)

    # 3. Chunk
    chunks: list[Chunk] = chunk_pages(pages)
    if not chunks:
        raise VectorStoreError("No chunks available to build a vector store.")

    # 4. Embed
    embeddings = await generate_embeddings(chunks)
    dimension = get_embedding_dimension()

    # 5. Upsert into pgvector (delete-then-insert for idempotency, scoped to user_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _db_save_sync, document_id, user_id, chunks, embeddings)
    except Exception as exc:
        logger.error(f"pgvector upsert failed for document_id={document_id}: {exc}")
        raise VectorStoreError(f"Failed to store embeddings in pgvector: {exc}") from exc

    elapsed = time.perf_counter() - start
    logger.info(
        f"pgvector store built: document_id={document_id}, chunks={len(chunks)}, "
        f"dimension={dimension}, time={elapsed:.2f}s"
    )

    return {
        "document_id": document_id,
        "total_chunks_stored": len(chunks),
        "embedding_dimension": dimension,
        "processing_time_seconds": round(elapsed, 3),
    }


def _db_info_sync(document_id: str, user_id: str) -> int:
    pool = get_db_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s AND user_id = %s",
                (document_id, user_id),
            )
            return cur.fetchone()[0]


async def get_vector_store_info(document_id: str, user_id: str) -> dict[str, Any]:
    """
    Return summary info about a document's stored chunk embeddings.

    Args:
        document_id: Unique identifier of the document.
        user_id: The authenticated user's UUID. Restricts the query to only
            the requesting user's chunks.

    Returns:
        dict[str, Any]: Summary metadata (chunk count, embedding dimension).

    Raises:
        VectorStoreNotFoundError: If no chunks exist for this user + document.
    """
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, _db_info_sync, document_id, user_id)

    if count == 0:
        raise VectorStoreNotFoundError(document_id)

    return {
        "document_id": document_id,
        "total_chunks_stored": int(count),
        "embedding_dimension": get_embedding_dimension(),
    }
