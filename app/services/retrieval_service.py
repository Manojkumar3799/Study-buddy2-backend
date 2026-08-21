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


def _db_retrieve_sync(
    document_id: str,
    user_id: str,
    emb_str: str,
    question: str,
    similarity_threshold: float,
    top_k: int,
    search_mode: str = "hybrid",
    apply_threshold: bool = True,
) -> tuple[int, list[dict], int, int]:
    pool = get_db_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Count chunks for this document AND this user (0 means no access)
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s AND user_id = %s",
                (document_id, user_id),
            )
            count = cur.fetchone()[0]
            if count == 0:
                return 0, [], 0, 0

            if search_mode == "vector":
                # Execute pgvector similarity search query using cosine distance (<=>)
                if apply_threshold:
                    cur.execute(
                        """
                        SELECT chunk_id, text, start_page, end_page,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM document_chunks
                        WHERE document_id = %s
                          AND user_id = %s
                          AND 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s
                        """,
                        (emb_str, document_id, user_id, emb_str, similarity_threshold, emb_str, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chunk_id, text, start_page, end_page,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM document_chunks
                        WHERE document_id = %s
                          AND user_id = %s
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s
                        """,
                        (emb_str, document_id, user_id, emb_str, top_k),
                    )
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "chunk_id": r[0],
                        "text": r[1],
                        "start_page": r[2],
                        "end_page": r[3],
                        "score": r[4],
                    })
                return count, results, len(rows), 0

            else:
                # Hybrid search using Reciprocal Rank Fusion (RRF)
                candidate_limit = max(50, top_k * 3)
                # Raw RRF max possible score is (1/61 + 1/61) = 2/61.
                # To normalise to [0.0, 1.0], we multiply raw score by (61/2) = 30.5
                query = """
                    WITH vector_search AS (
                      SELECT chunk_id, text, start_page, end_page,
                             1 - (embedding <=> %(embedding)s::vector) AS similarity,
                             row_number() OVER (ORDER BY embedding <=> %(embedding)s::vector ASC) AS rank
                      FROM document_chunks
                      WHERE document_id = %(document_id)s
                        AND user_id = %(user_id)s
                      ORDER BY embedding <=> %(embedding)s::vector ASC
                      LIMIT %(candidate_limit)s
                    ),
                    fts_search AS (
                      SELECT chunk_id, text, start_page, end_page,
                             row_number() OVER (ORDER BY ts_rank_cd(fts_vector, plainto_tsquery('english', %(query_str)s)) DESC) AS rank
                      FROM document_chunks
                      WHERE document_id = %(document_id)s
                        AND user_id = %(user_id)s
                        AND fts_vector @@ plainto_tsquery('english', %(query_str)s)
                      ORDER BY ts_rank_cd(fts_vector, plainto_tsquery('english', %(query_str)s)) DESC
                      LIMIT %(candidate_limit)s
                    )
                    SELECT 
                      COALESCE(vs.chunk_id, fs.chunk_id) AS chunk_id,
                      COALESCE(vs.text, fs.text) AS text,
                      COALESCE(vs.start_page, fs.start_page) AS start_page,
                      COALESCE(vs.end_page, fs.end_page) AS end_page,
                      (COALESCE(1.0 / (60 + vs.rank), 0.0) + COALESCE(1.0 / (60 + fs.rank), 0.0)) * 30.5 AS fused_score,
                      vs.similarity AS raw_similarity,
                      CASE WHEN vs.chunk_id IS NOT NULL THEN 1 ELSE 0 END AS matched_vector,
                      CASE WHEN fs.chunk_id IS NOT NULL THEN 1 ELSE 0 END AS matched_fts
                    FROM vector_search vs
                    FULL OUTER JOIN fts_search fs ON vs.chunk_id = fs.chunk_id
                    ORDER BY fused_score DESC
                """
                params = {
                    "embedding": emb_str,
                    "document_id": document_id,
                    "user_id": user_id,
                    "query_str": question,
                    "candidate_limit": candidate_limit,
                }
                cur.execute(query, params)
                rows = cur.fetchall()
                
                vector_count = 0
                fts_count = 0
                results = []
                
                for r in rows:
                    score = r[4]
                    matched_vector = r[6]
                    matched_fts = r[7]
                    
                    if matched_vector:
                        vector_count += 1
                    if matched_fts:
                        fts_count += 1
                        
                    if apply_threshold and score < similarity_threshold:
                        continue
                        
                    results.append({
                        "chunk_id": r[0],
                        "text": r[1],
                        "start_page": r[2],
                        "end_page": r[3],
                        "score": score,
                    })
                
                # Limit the final combined results to top_k
                results = results[:top_k]
                return count, results, vector_count, fts_count


async def retrieve_relevant_chunks(
    document_id: str,
    user_id: str,
    question: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    search_mode: str = "hybrid",
    apply_threshold: bool = True,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a question from Supabase pgvector,
    filtered by a minimum similarity threshold, using hybrid search by default.

    All queries are scoped to the authenticated user's rows so that users
    cannot access each other's document data.

    Args:
        document_id: Unique identifier of a previously stored document.
        user_id: The authenticated user's UUID. Restricts results to chunks
            owned by this user.
        question: The user's natural language question.
        top_k: Number of top candidates to retrieve before thresholding.
        similarity_threshold: Minimum combined score to keep a chunk.
        search_mode: Mode to search — 'hybrid' (default) or 'vector' (pure pgvector).
        apply_threshold: If False, does not filter results by threshold.

    Returns:
        list[RetrievedChunk]: Chunks that passed the similarity threshold,
            ordered by descending relevance. May be empty if none pass.

    Raises:
        VectorStoreNotFoundError: If no store exists for this user + document.
        VectorStoreError: If query execution fails.
    """
    top_k = top_k or settings.retrieval_top_k
    similarity_threshold = (
        similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
    )

    logger.info(
        f"Retrieval requested: document_id={document_id}, user_id={user_id}, top_k={top_k}, "
        f"threshold={similarity_threshold}, mode={search_mode}, "
        f"question='{question[:80]}'"
    )

    question_emb = await embed_question(question)
    emb_str = "[" + ",".join(str(v) for v in question_emb) + "]"

    loop = asyncio.get_event_loop()
    count, rows, vector_count, fts_count = await loop.run_in_executor(
        None,
        _db_retrieve_sync,
        document_id,
        user_id,
        emb_str,
        question,
        similarity_threshold,
        top_k,
        search_mode,
        apply_threshold,
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
                score=round(r["score"], 4),
            )
        )

    logger.info(
        f"Retrieval complete: document_id={document_id}, "
        f"vector_candidates={vector_count}, fts_candidates={fts_count}, "
        f"passed_threshold={len(retrieved)}"
    )

    return retrieved
