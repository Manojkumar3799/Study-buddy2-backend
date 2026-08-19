"""Pydantic request/response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class WebSourceResponse(BaseModel):
    """A single web search result returned when answering via Firecrawl web research."""

    title: str = Field(..., description="Page title from the web search result.")
    url: str = Field(..., description="Full URL of the web page.")
    domain: str = Field(..., description="Bare hostname / domain of the web page.")


class UploadResponse(BaseModel):
    """Response returned after a successful PDF upload."""

    document_id: str = Field(..., description="Unique identifier for the uploaded document.")
    filename: str = Field(..., description="Original filename of the uploaded PDF.")
    page_count: int = Field(..., description="Number of pages detected in the PDF.")
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes.")
    message: str = Field(..., description="Human-readable status message.")


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Short error code/category.")
    detail: str = Field(..., description="Human-readable error detail.")


class PageTextResponse(BaseModel):
    """Extracted text for a single page."""

    page_number: int = Field(..., description="1-indexed page number.")
    text: str = Field(..., description="Cleaned extracted text for this page.")
    word_count: int = Field(..., description="Number of words on this page.")


class ExtractionResponse(BaseModel):
    """Response returned after successful text extraction."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    total_pages: int = Field(..., description="Total number of pages processed.")
    total_words: int = Field(..., description="Total word count across all pages.")
    pages: list[PageTextResponse] = Field(..., description="Per-page extracted text.")


class ChunkResponse(BaseModel):
    """A single text chunk with source metadata."""

    chunk_id: int = Field(..., description="0-indexed chunk sequence number.")
    text: str = Field(..., description="Chunk text content.")
    word_count: int = Field(..., description="Number of words in this chunk.")
    start_page: int = Field(..., description="First page this chunk's text originates from.")
    end_page: int = Field(..., description="Last page this chunk's text originates from.")


class ChunkingResponse(BaseModel):
    """Response returned after successful chunking."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    total_chunks: int = Field(..., description="Total number of chunks produced.")
    chunk_size_words: int = Field(..., description="Configured chunk size in words.")
    chunk_overlap_words: int = Field(..., description="Configured overlap in words.")
    chunks: list[ChunkResponse] = Field(..., description="Ordered list of chunks.")


class EmbeddingPreview(BaseModel):
    """Preview of a single chunk's embedding (truncated for readability)."""

    chunk_id: int = Field(..., description="0-indexed chunk sequence number.")
    word_count: int = Field(..., description="Number of words in this chunk.")
    embedding_preview: list[float] = Field(
        ..., description="First 8 values of the embedding vector (for sanity checking)."
    )


class EmbeddingResponse(BaseModel):
    """Response returned after successful embedding generation."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    model_name: str = Field(..., description="Name of the embedding model used.")
    embedding_dimension: int = Field(..., description="Dimensionality of each embedding vector.")
    total_chunks_embedded: int = Field(..., description="Number of chunks that were embedded.")
    processing_time_seconds: float = Field(..., description="Time taken to generate all embeddings.")
    previews: list[EmbeddingPreview] = Field(
        ..., description="Truncated embedding previews for the first few chunks."
    )


class StoreResponse(BaseModel):
    """Response returned after successfully building and saving a vector store."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    total_chunks_stored: int = Field(..., description="Number of chunks stored in the FAISS index.")
    embedding_dimension: int = Field(..., description="Dimensionality of stored embedding vectors.")
    processing_time_seconds: float = Field(..., description="Total pipeline processing time.")
    message: str = Field(..., description="Human-readable status message.")


class VectorStoreInfoResponse(BaseModel):
    """Response returned when inspecting an existing vector store."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    total_chunks_stored: int = Field(..., description="Number of chunks stored in the FAISS index.")
    embedding_dimension: int = Field(..., description="Dimensionality of stored embedding vectors.")


class RetrievalRequest(BaseModel):
    """Request body for a retrieval query."""

    question: str = Field(..., min_length=1, description="The user's natural language question.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Override the number of candidates to retrieve."
    )
    similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override the minimum similarity score."
    )


class RetrievedChunkResponse(BaseModel):
    """A single retrieved chunk with its similarity score."""

    chunk_id: int = Field(..., description="0-indexed chunk sequence number.")
    text: str = Field(..., description="Chunk text content.")
    start_page: int = Field(..., description="First page this chunk's text originates from.")
    end_page: int = Field(..., description="Last page this chunk's text originates from.")
    score: float = Field(..., description="Cosine similarity score (0-1, higher is more relevant).")


class RetrievalResponse(BaseModel):
    """Response returned after a retrieval query."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    question: str = Field(..., description="The question that was searched.")
    top_k: int = Field(..., description="Number of candidates that were considered.")
    similarity_threshold: float = Field(..., description="Minimum similarity score applied.")
    total_matches: int = Field(..., description="Number of chunks that passed the threshold.")
    chunks: list[RetrievedChunkResponse] = Field(
        ..., description="Retrieved chunks ordered by descending relevance."
    )
    has_sufficient_context: bool = Field(
        ..., description="Whether any chunks passed the threshold (usable for downstream LLM step)."
    )


class AskRequest(BaseModel):
    """Request body for a question-answering query."""

    question: str = Field(..., min_length=1, description="The user's natural language question.")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override retrieval top_k.")
    similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override similarity threshold."
    )
    mode: Literal["auto", "pdf", "web"] = Field(
        default="auto",
        description=(
            "Answering mode. "
            "'auto' lets the LLM decide between PDF RAG and web research. "
            "'pdf' always uses the uploaded document. "
            "'web' always uses Firecrawl web research."
        ),
    )


class AskResponse(BaseModel):
    """Response returned after a question-answering query."""

    document_id: str = Field(..., description="Unique identifier for the document.")
    question: str = Field(..., description="The question that was asked.")
    answer: str = Field(..., description="The generated answer.")
    provider_used: str | None = Field(
        default=None, description="Which LLM provider generated the answer (null if no context)."
    )
    source_type: Literal["pdf", "web"] = Field(
        default="pdf",
        description="Whether the answer was grounded in the PDF ('pdf') or web research ('web').",
    )
    sources: list[RetrievedChunkResponse] = Field(
        default_factory=list,
        description="PDF chunks used as context (populated when source_type='pdf').",
    )
    web_sources: list[WebSourceResponse] = Field(
        default_factory=list,
        description="Web search results used as context (populated when source_type='web').",
    )
    has_sufficient_context: bool = Field(
        ..., description="Whether relevant context was found."
    )