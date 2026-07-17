"""Custom application exceptions."""


class StudyForgeException(Exception):
    """Base exception for all StudyForge AI application errors."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


# --- PDF / Upload errors -----------------------------------------------


class InvalidPDFError(StudyForgeException):
    """Raised when the uploaded file is not a valid PDF."""

    def __init__(self, message: str = "Uploaded file is not a valid PDF.") -> None:
        super().__init__(message, status_code=400)


class CorruptedPDFError(StudyForgeException):
    """Raised when the PDF cannot be opened or parsed."""

    def __init__(self, message: str = "Uploaded PDF is corrupted or unreadable.") -> None:
        super().__init__(message, status_code=400)


class EmptyPDFError(StudyForgeException):
    """Raised when the PDF has no extractable pages or content."""

    def __init__(self, message: str = "Uploaded PDF is empty.") -> None:
        super().__init__(message, status_code=400)


class PDFTooLargeError(StudyForgeException):
    """Raised when the PDF exceeds the maximum allowed size."""

    def __init__(self, message: str = "Uploaded PDF exceeds the maximum allowed size.") -> None:
        super().__init__(message, status_code=413)


class DocumentNotFoundError(StudyForgeException):
    """Raised when a requested document ID does not exist on disk."""

    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document with ID '{document_id}' not found.", status_code=404)


class TextExtractionError(StudyForgeException):
    """Raised when text cannot be meaningfully extracted from a PDF."""

    def __init__(
        self,
        message: str = "No meaningful text could be extracted from this PDF. "
        "It may be a scanned/image-only document.",
    ) -> None:
        super().__init__(message, status_code=422)


# --- Embedding / Vector store errors ------------------------------------


class EmbeddingGenerationError(StudyForgeException):
    """Raised when embedding generation fails."""

    def __init__(self, message: str = "Failed to generate embeddings for the document.") -> None:
        super().__init__(message, status_code=500)


class VectorStoreError(StudyForgeException):
    """Raised when FAISS index creation, saving, or loading fails."""

    def __init__(self, message: str = "Failed to build or access the vector store.") -> None:
        super().__init__(message, status_code=500)


class VectorStoreNotFoundError(StudyForgeException):
    """Raised when no vector store exists for a given document."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"No vector store found for document '{document_id}'. "
            "Run /store on this document first.",
            status_code=404,
        )


class SimilarityThresholdError(StudyForgeException):
    """Raised when retrieval parameters are invalid (e.g. malformed threshold)."""

    def __init__(
        self, message: str = "Invalid similarity threshold or retrieval parameters."
    ) -> None:
        super().__init__(message, status_code=400)


# --- LLM / Provider errors ----------------------------------------------


class AllProvidersFailedError(StudyForgeException):
    """Raised when every configured LLM provider fails after all retries."""

    def __init__(
        self,
        message: str = "All AI providers are currently unavailable. Please try again shortly.",
    ) -> None:
        super().__init__(message, status_code=503)


class LLMQuotaExceededError(StudyForgeException):
    """Raised when a provider reports quota/billing limits have been exceeded."""

    def __init__(
        self, message: str = "AI provider quota exceeded. Please try again later."
    ) -> None:
        super().__init__(message, status_code=429)


class LLMRateLimitError(StudyForgeException):
    """Raised when a provider reports too many requests in a short window."""

    def __init__(
        self, message: str = "AI provider rate limit reached. Please try again shortly."
    ) -> None:
        super().__init__(message, status_code=429)


class LLMTimeoutError(StudyForgeException):
    """Raised when an LLM provider does not respond within the configured timeout."""

    def __init__(self, message: str = "AI provider took too long to respond.") -> None:
        super().__init__(message, status_code=504)


class StreamingInterruptedError(StudyForgeException):
    """Raised when a streaming response is interrupted after partial output was sent."""

    def __init__(
        self, message: str = "The response stream was interrupted. Please try again."
    ) -> None:
        super().__init__(message, status_code=502)


class NetworkError(StudyForgeException):
    """Raised when a network-level failure occurs while contacting a provider."""

    def __init__(self, message: str = "A network error occurred. Please try again.") -> None:
        super().__init__(message, status_code=502)