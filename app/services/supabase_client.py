"""Supabase client singletons.

Two clients are needed:
  - psycopg ConnectionPool — direct Postgres connection pool used for all pgvector SQL
                             (INSERT chunks, cosine-similarity SELECT).
  - supabase-py            — REST-based Storage client used for PDF upload/download.

Both are module-level singletons, initialized lazily on first use.
"""

from __future__ import annotations

from psycopg_pool import ConnectionPool
from supabase import create_client, Client as SupabaseClient

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_db_pool: ConnectionPool | None = None
_storage_client: SupabaseClient | None = None


def get_db_pool() -> ConnectionPool:
    """Return (and lazily create) the psycopg ConnectionPool.

    The pool connects to Supabase's direct Postgres endpoint, which supports
    pgvector operators. The connection string is taken from
    ``settings.supabase_db_url``.

    Returns:
        ConnectionPool: A ready-to-use connection pool.

    Raises:
        RuntimeError: If ``SUPABASE_DB_URL`` is not configured.
    """
    global _db_pool
    if _db_pool is not None:
        return _db_pool

    settings = get_settings()
    if not settings.supabase_db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not configured. "
            "Set it to the direct Postgres connection string from Supabase Dashboard "
            "-> Project Settings -> Database -> Connection string -> Direct."
        )

    logger.info("Initializing psycopg connection pool")
    # psycopg connection pool is synchronous in construction but handles concurrent requests
    _db_pool = ConnectionPool(
        conninfo=settings.supabase_db_url,
        min_size=1,
        max_size=5,
        open=True,
    )
    logger.info("psycopg connection pool ready")
    return _db_pool


def get_storage_client() -> SupabaseClient:
    """Return (and lazily create) the supabase-py Storage client.

    Uses the project URL and service role key from settings.

    Returns:
        SupabaseClient: A configured Supabase client with Storage access.

    Raises:
        RuntimeError: If ``SUPABASE_URL`` or ``SUPABASE_KEY`` are not set.
    """
    global _storage_client
    if _storage_client is not None:
        return _storage_client

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be configured. "
            "Set them to your Supabase project URL and service role key."
        )

    logger.info("Initializing Supabase Storage client")
    _storage_client = create_client(settings.supabase_url, settings.supabase_key)
    logger.info("Supabase Storage client ready")
    return _storage_client


def close_db_pool() -> None:
    """Gracefully close the psycopg pool. Call from app shutdown handler."""
    global _db_pool
    if _db_pool is not None:
        _db_pool.close()
        _db_pool = None
        logger.info("psycopg connection pool closed")
