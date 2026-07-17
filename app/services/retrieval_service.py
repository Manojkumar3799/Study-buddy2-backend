"""Service layer for embedding questions and retrieving relevant chunks via FAISS."""

import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.embedding_service import load_embedding_model
from app.services.vector_store_service import load_vector_store

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


def embed_question(question: str) -> np.ndarray:
    """
    Generate a normalized embedding vector for a user question.

    Args:
        question: The user's natural language question.

    Returns:
        np.ndarray: A (1, dimension) float32 embedding vector.
    """
    model = load_embedding_model()
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector.astype("float32")


def retrieve_relevant_chunks(
    document_id: str,
    question: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a question from a document's
    FAISS index, filtered by a minimum similarity threshold.

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
        VectorStoreError: If loading the index fails.
    """
    top_k = top_k or settings.retrieval_top_k
    similarity_threshold = (
        similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
    )

    logger.info(
        f"Retrieval requested: document_id={document_id}, top_k={top_k}, "
        f"threshold={similarity_threshold}, question='{question[:80]}'"
    )

    index, metadata = load_vector_store(document_id)
    chunks_metadata = metadata["chunks"]

    effective_k = min(top_k, index.ntotal)
    if effective_k == 0:
        logger.warning(f"Vector store for {document_id} has no vectors")
        return []

    question_vector = embed_question(question)
    scores, indices = index.search(question_vector, effective_k)

    retrieved: list[RetrievedChunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        score_float = float(score)
        if score_float < similarity_threshold:
            continue

        chunk_meta = chunks_metadata[idx]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_meta["chunk_id"],
                text=chunk_meta["text"],
                start_page=chunk_meta["start_page"],
                end_page=chunk_meta["end_page"],
                score=round(score_float, 4),
            )
        )

    logger.info(
        f"Retrieval complete: document_id={document_id}, "
        f"candidates={effective_k}, passed_threshold={len(retrieved)}"
    )

    return retrieved