"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings."""

    app_name: str = "StudyForge AI"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # LLM provider keys
    gemini_api_key: str = ""
    groq_api_key: str = ""
    grok_api_key: str = ""

    # Firecrawl MCP — web research integration
    # Get a free key at https://firecrawl.dev
    firecrawl_api_key: str = ""
    
    # Upload settings
    max_pdf_size_mb: int = 50

    # CORS — comma-separated list of allowed frontend origins.
    # Set ALLOWED_ORIGINS in your deployment env to your Vercel URL.
    # Example: "https://studyforge.vercel.app,http://localhost:3000"
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Supabase — Storage REST API (used by supabase-py for PDF blob storage)
    supabase_url: str = ""
    supabase_key: str = ""  # service role key
    supabase_storage_bucket: str = "pdfs"

    # Supabase — Direct Postgres connection string (used by asyncpg for pgvector)
    # Find it in: Supabase Dashboard → Project Settings → Database → Connection string → Direct
    # Format: postgresql://postgres.PROJECT_REF:PASSWORD@HOST:5432/postgres
    supabase_db_url: str = ""

    # Chunking settings
    chunk_size_words: int = 500
    chunk_overlap_words: int = 100

    # Embedding settings — embeddings are obtained via the Gemini hosted API
    # through langchain-google-genai; there is no local model to configure.

# Retrieval settings
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.35

    # LLM settings
    # Model names use the LiteLLM "provider/model" convention for readability;
    # the "provider/" prefix is stripped automatically in llm_service.py when
    # constructing LangChain chat model instances (which take bare model names).
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