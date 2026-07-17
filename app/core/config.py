"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings."""

    app_name: str = "StudyForge AI"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # LLM provider keys (used in later features)
    gemini_api_key: str = ""
    groq_api_key: str = ""
    grok_api_key: str = ""
    
    # Upload settings
    max_pdf_size_mb: int = 50
    

    # Chunking settings
    chunk_size_words: int = 500
    chunk_overlap_words: int = 100

    # Embedding settings
    embedding_model_name: str = "all-MiniLM-L6-v2"

# Retrieval settings
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.35

    # LLM settings
    gemini_model: str = "gemini/gemini-2.5-flash"
    groq_model: str = "groq/llama-3.3-70b-versatile"
    grok_model: str = "xai/grok-2-latest"
    llm_max_retries_per_provider: int = 2
    llm_retry_base_delay_seconds: float = 1.0
    llm_request_timeout_seconds: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()