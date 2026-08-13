import litellm
import numpy as np
from app.services.chunking_service import Chunk


def _normalize_vector(v: list[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm = 1.0)."""
    arr = np.array(v, dtype="float32")
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = "gemini/gemini-embedding-001", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = await litellm.aembedding(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.extend(_normalize_vector(item["embedding"]) for item in response.data)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        response = await litellm.aembedding(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
            task_type="RETRIEVAL_QUERY",
        )
        return _normalize_vector(response.data[0]["embedding"])


_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Get the singleton instance of the EmbeddingModel."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def get_embedding_dimension() -> int:
    """Return the output vector dimension."""
    return get_embedding_model().dimensions


async def generate_embeddings(chunks: list[Chunk]) -> list[list[float]]:
    """Generate embedding vectors for a list of text chunks asynchronously."""
    if not chunks:
        return []
    model = get_embedding_model()
    texts = [chunk.text for chunk in chunks]
    return await model.embed_texts(texts)


def load_embedding_model() -> None:
    """No-op loader for backward compatibility with startup routines."""
    pass