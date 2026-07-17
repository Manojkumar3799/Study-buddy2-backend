"""Service layer for generating sentence embeddings from text chunks."""

import time

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import StudyForgeException
from app.core.logging_config import get_logger
from app.services.chunking_service import Chunk

logger = get_logger(__name__)
settings = get_settings()

_model: SentenceTransformer | None = None


class EmbeddingGenerationError(StudyForgeException):
    """Raised when embedding generation fails."""

    def __init__(self, message: str = "Failed to generate embeddings for the document.") -> None:
        super().__init__(message, status_code=500)


def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformers model as a singleton.

    Loading is expensive (~1-3s) so this must only happen once per
    application lifetime, not per request.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        start = time.perf_counter()
        _model = SentenceTransformer(settings.embedding_model_name)
        elapsed = time.perf_counter() - start
        logger.info(f"Embedding model loaded in {elapsed:.2f}s")
    return _model


def get_embedding_dimension() -> int:
    """
    Return the output vector dimension of the loaded embedding model.

    Returns:
        int: Embedding vector dimension.
    """
    model = load_embedding_model()
    return model.get_sentence_embedding_dimension()


def generate_embeddings(chunks: list[Chunk]) -> list[list[float]]:
    """
    Generate embedding vectors for a list of text chunks.

    Args:
        chunks: Ordered list of text chunks to embed.

    Returns:
        list[list[float]]: One embedding vector per chunk, same order as input.

    Raises:
        EmbeddingGenerationError: If the model fails to produce embeddings.
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation")
        return []

    model = load_embedding_model()
    texts = [chunk.text for chunk in chunks]

    logger.info(f"Generating embeddings for {len(texts)} chunks")
    start = time.perf_counter()

    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        logger.error(f"Embedding generation failed: {exc}")
        raise EmbeddingGenerationError() from exc

    elapsed = time.perf_counter() - start
    logger.info(
        f"Embedding generation complete: {len(texts)} chunks in {elapsed:.2f}s "
        f"({len(texts) / elapsed:.1f} chunks/sec)"
    )

    return embeddings.tolist()