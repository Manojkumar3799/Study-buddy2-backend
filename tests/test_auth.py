"""Tests for JWT authentication dependency (app/core/auth.py).

Covers all five verification scenarios from the implementation spec:
(a) Valid JWT for user's own document — auth succeeds, user_id extracted
(b) Cross-user access protection — retrieval raises VectorStoreNotFoundError (404)
(c) Missing Authorization header — 401 AuthenticationError
(d) Expired / tampered JWT — 401 AuthenticationError
(e) Unsupported algorithm — 401 AuthenticationError

Token generation uses PyJWT directly to create signed test tokens; no Supabase
instance is required (all network calls are mocked).
"""

import time
import pytest
import jwt

from unittest.mock import AsyncMock, MagicMock, patch
from app.core.auth import get_current_user
from app.core.exceptions import AuthenticationError, VectorStoreNotFoundError
from fastapi.security import HTTPAuthorizationCredentials


# ────────────────────────────────────────────────────────────────────────────
# Helpers — build test JWTs
# ────────────────────────────────────────────────────────────────────────────

_TEST_SECRET = "super-secret-jwt-secret-for-testing-only"
_USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _make_hs256_token(user_id: str, exp_offset: int = 3600) -> str:
    """Return a signed HS256 JWT for the given user_id."""
    payload = {
        "sub": user_id,
        "role": "authenticated",
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_offset,
    }
    return jwt.encode(payload, _TEST_SECRET, algorithm="HS256")


def _make_expired_hs256_token(user_id: str) -> str:
    """Return an already-expired HS256 token."""
    return _make_hs256_token(user_id, exp_offset=-60)


def _make_tampered_token() -> str:
    """Return a token with a valid header/payload but wrong signature."""
    valid = _make_hs256_token(_USER_A_ID)
    parts = valid.split(".")
    # Corrupt the signature
    return f"{parts[0]}.{parts[1]}.invalidsignatureXYZ"


# ────────────────────────────────────────────────────────────────────────────
# Auth dependency unit tests
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_hs256_token_extracts_user_id():
    """(a) Valid HS256 token — get_current_user returns the sub claim."""
    token = _make_hs256_token(_USER_A_ID)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with patch("app.core.auth.get_settings") as mock_get_settings:
        mock_settings = MagicMock()
        mock_settings.supabase_jwt_secret = _TEST_SECRET
        mock_settings.supabase_jwks_url = ""
        mock_get_settings.return_value = mock_settings

        # Also patch _get_jwks_client to return None (no JWKS for HS256 test)
        with patch("app.core.auth._get_jwks_client", return_value=None):
            result = await get_current_user(credentials)

    assert result == _USER_A_ID


@pytest.mark.asyncio
async def test_missing_authorization_header_raises_401():
    """(c) No Authorization header — get_current_user raises AuthenticationError (401)."""
    with pytest.raises(AuthenticationError) as exc_info:
        await get_current_user(None)

    assert exc_info.value.status_code == 401
    assert "No authentication token provided" in exc_info.value.message


@pytest.mark.asyncio
async def test_expired_token_raises_401():
    """(d) Expired token — get_current_user raises AuthenticationError (401)."""
    token = _make_expired_hs256_token(_USER_A_ID)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with patch("app.core.auth.get_settings") as mock_get_settings:
        mock_settings = MagicMock()
        mock_settings.supabase_jwt_secret = _TEST_SECRET
        mock_settings.supabase_jwks_url = ""
        mock_get_settings.return_value = mock_settings

        with patch("app.core.auth._get_jwks_client", return_value=None):
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(credentials)

    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_tampered_token_raises_401():
    """(d) Token with invalid signature — get_current_user raises AuthenticationError (401)."""
    token = _make_tampered_token()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with patch("app.core.auth.get_settings") as mock_get_settings:
        mock_settings = MagicMock()
        mock_settings.supabase_jwt_secret = _TEST_SECRET
        mock_settings.supabase_jwks_url = ""
        mock_get_settings.return_value = mock_settings

        with patch("app.core.auth._get_jwks_client", return_value=None):
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(credentials)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_malformed_token_raises_401():
    """(d) Completely malformed token string — raises AuthenticationError (401)."""
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not.a.jwt")

    with pytest.raises(AuthenticationError) as exc_info:
        await get_current_user(credentials)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_unsupported_algorithm_raises_401():
    """(d) Token signed with an unsupported algorithm raises 401."""
    # Create a token with a non-standard alg by crafting the header manually
    import base64, json
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": _USER_A_ID}).encode()).rstrip(b"=")
    fake_token = f"{header.decode()}.{payload.decode()}.fakesig"

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=fake_token)

    with pytest.raises(AuthenticationError) as exc_info:
        await get_current_user(credentials)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_hs256_missing_secret_raises_401():
    """Attempting HS256 verification without SUPABASE_JWT_SECRET raises 401."""
    token = _make_hs256_token(_USER_A_ID)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with patch("app.core.auth.get_settings") as mock_get_settings:
        mock_settings = MagicMock()
        mock_settings.supabase_jwt_secret = ""  # not configured
        mock_settings.supabase_jwks_url = ""
        mock_get_settings.return_value = mock_settings

        with patch("app.core.auth._get_jwks_client", return_value=None):
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(credentials)

    assert exc_info.value.status_code == 401
    assert "SUPABASE_JWT_SECRET" in exc_info.value.message


# ────────────────────────────────────────────────────────────────────────────
# Cross-user access isolation tests (via retrieval service)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cross_user_document_access_returns_404():
    """
    (b) User B tries to query User A's document_id.
    The retrieval service sees count=0 for (document_id, user_b_id)
    and raises VectorStoreNotFoundError (mapped to HTTP 404).
    """
    from app.services.retrieval_service import retrieve_relevant_chunks

    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # User B's query for User A's document_id returns count=0
    mock_cur.fetchone.return_value = [0]

    with (
        patch("app.services.retrieval_service.get_db_pool", return_value=mock_pool),
        patch("app.services.retrieval_service.embed_question", return_value=[0.1, 0.2, 0.3]),
    ):
        with pytest.raises(VectorStoreNotFoundError) as exc_info:
            await retrieve_relevant_chunks(
                document_id="user-a-doc-id",
                user_id=_USER_B_ID,    # User B is NOT the owner
                question="What is this about?",
            )

    assert exc_info.value.status_code == 404
    # Verify the query was called with user_b_id (enforcing the isolation)
    count_calls = [
        call for call in mock_cur.execute.call_args_list
        if "COUNT" in str(call)
    ]
    assert len(count_calls) >= 1
    called_args = count_calls[0].args
    assert _USER_B_ID in called_args[1]


@pytest.mark.asyncio
async def test_user_can_access_own_document():
    """
    (a) User A accesses their own document — retrieval succeeds normally.
    """
    from app.services.retrieval_service import retrieve_relevant_chunks

    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # User A's document exists: count = 3
    mock_cur.fetchone.return_value = [3]
    # Hybrid search returns 2 passing chunks
    mock_cur.fetchall.return_value = [
        (1, "Chunk text A", 1, 2, 0.80, 0.75, 1, 1),
        (2, "Chunk text B", 3, 4, 0.60, 0.55, 1, 0),
    ]

    with (
        patch("app.services.retrieval_service.get_db_pool", return_value=mock_pool),
        patch("app.services.retrieval_service.embed_question", return_value=[0.1, 0.2, 0.3]),
    ):
        results = await retrieve_relevant_chunks(
            document_id="user-a-doc-id",
            user_id=_USER_A_ID,
            question="What is this about?",
            similarity_threshold=0.35,
        )

    assert len(results) == 2
    assert results[0].chunk_id == 1
    assert results[1].chunk_id == 2


# ────────────────────────────────────────────────────────────────────────────
# Health endpoint stays public
# ────────────────────────────────────────────────────────────────────────────

def test_health_endpoint_requires_no_auth():
    """
    (e) /health should remain publicly accessible without authentication.
    Verify it is not in any auth-protected router.
    """
    from app.main import app
    health_routes = [r for r in app.routes if getattr(r, "path", None) == "/health"]
    assert len(health_routes) == 1
    # The health route should have no dependencies that include get_current_user
    health_route = health_routes[0]
    dep_names = [
        dep.dependency.__name__
        for dep in getattr(health_route, "dependencies", [])
        if hasattr(dep.dependency, "__name__")
    ]
    assert "get_current_user" not in dep_names, (
        "/health must not require authentication — it needs to be accessible "
        "by infrastructure health checkers without a token."
    )
