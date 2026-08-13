"""Service layer for embedding questions and retrieving relevant chunks via Supabase pgvector."""

import time
import asyncio
from functools import partial

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.embedding_service import get_embedding_model
from app.services.supabase_client import get_db_pool
from app.core.exceptions import VectorStoreNotFoundError

logger = get_logger(__name__)
settings = get_settings()


class RetrievedChunk:
    """A single retrieved chunk with its similarity score."""

    def __init__(
        self,
        chunk_id: int,
        text: str,
        start_page: int,
        end_page: int,
        score: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.start_page = start_page
        self.end_page = end_page
        self.score = score


async def embed_question(question: str) -> list[float]:
    """
    Generate a normalized embedding vector for a user question.

    Args:
        question: The user's natural language question.

    Returns:
        list[float]: The embedding vector as a list of floats.
    """
    model = get_embedding_model()
    embedding = await model.embed_query(question)
    return embedding


def _db_retrieve_sync(document_id: str, emb_str: str, similarity_threshold: float, top_k: int) -> tuple[int, list[dict]]:
    pool = get_db_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks WHERE document_id = %s", (document_id,))
            count = cur.fetchone()[0]
            if count == 0:
                return 0, []

            # Execute pgvector similarity search query using cosine distance (<=>)
            cur.execute(
                """
                SELECT chunk_id, text, start_page, end_page,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE document_id = %s
                  AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (emb_str, document_id, emb_str, similarity_threshold, emb_str, top_k),
            )
            rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "chunk_id": r[0],
                    "text": r[1],
                    "start_page": r[2],
                    "end_page": r[3],
                    "similarity": r[4],
                })
            return count, results


async def retrieve_relevant_chunks(
    document_id: str,
    question: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a question from Supabase pgvector,
    filtered by a minimum similarity threshold.

    Args:
        document_id: Unique identifier of a previously stored document.
        question: The user's natural language question.
        top_k: Number of top candidates to retrieve before thresholding.
        similarity_threshold: Minimum cosine similarity score to keep a chunk.

    Returns:
        list[RetrievedChunk]: Chunks that passed the similarity threshold,
            ordered by descending relevance. May be empty if none pass.

    Raises:
        VectorStoreNotFoundError: If no store exists for the document.
        VectorStoreError: If query execution fails.
    """
    top_k = top_k or settings.retrieval_top_k
    similarity_threshold = (
        similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
    )

    logger.info(
        f"Retrieval requested: document_id={document_id}, top_k={top_k}, "
        f"threshold={similarity_threshold}, question='{question[:80]}'"
    )

    question_emb = await embed_question(question)
    emb_str = "[" + ",".join(str(v) for v in question_emb) + "]"

    loop = asyncio.get_event_loop()
    count, rows = await loop.run_in_executor(
        None,
        _db_retrieve_sync,
        document_id,
        emb_str,
        similarity_threshold,
        top_k,
    )

    if count == 0:
        raise VectorStoreNotFoundError(document_id)

    retrieved: list[RetrievedChunk] = []
    for r in rows:
        retrieved.append(
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                start_page=r["start_page"],
                end_page=r["end_page"],
                score=round(r["similarity"], 4),
            )
        )

    logger.info(
        f"Retrieval complete: document_id={document_id}, "
        f"candidates_found={len(rows)}, passed_threshold={len(retrieved)}"
    )

    return retrieved
