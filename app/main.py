"""StudyForge AI - FastAPI application entrypoint."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chunk, extract, upload, embed, store, retrieve, ask
from app.core.config import get_settings
from app.core.exceptions import StudyForgeException
from app.core.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Retrieval-Augmented Generation study assistant backend.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the deployed Vercel frontend and localhost for dev.
# In production, narrow ALLOWED_ORIGINS to your exact Vercel URL via the env
# var ALLOWED_ORIGINS (comma-separated list).  The wildcard fallback below
# is safe for development only.
# ---------------------------------------------------------------------------
_raw_origins = settings.allowed_origins  # e.g. "https://myapp.vercel.app,http://localhost:3000"
ALLOWED_ORIGINS: list[str] = [
    o.strip() for o in _raw_origins.split(",") if o.strip()
] or ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    """Log application startup and preload the embedding model."""
    logger.info(f"{settings.app_name} starting up | env={settings.app_env}")
    from app.services.embedding_service import load_embedding_model

    load_embedding_model()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Log application shutdown and close database connections."""
    logger.info(f"{settings.app_name} shutting down")
    from app.services.supabase_client import close_db_pool
    await close_db_pool()


@app.exception_handler(StudyForgeException)
async def studyforge_exception_handler(request: Request, exc: StudyForgeException) -> JSONResponse:
    """
    Handle all custom application exceptions with a clean JSON response.

    Args:
        request: The incoming request.
        exc: The raised StudyForgeException.

    Returns:
        JSONResponse: Structured error response.
    """
    logger.error(f"Handled exception on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler to prevent stack traces from leaking to clients.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        JSONResponse: Generic 500 error response.
    """
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "detail": "An unexpected error occurred."},
    )


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Application status information.
    """
    logger.info("Health check requested")
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
    }


app.include_router(upload.router)
app.include_router(extract.router)
app.include_router(chunk.router)
app.include_router(embed.router)
app.include_router(store.router)
app.include_router(retrieve.router)
app.include_router(ask.router)

# ```

# Also create:

# **`backend/storage/vector_store/.gitkeep`**
# ```
# ```

# Also create an empty file so the storage folder is tracked:

# **`backend/storage/uploads/.gitkeep`**
#