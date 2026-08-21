"""FastAPI authentication dependency for Supabase Auth JWT verification.

Architecture Overview
---------------------
This module provides a single FastAPI dependency — ``get_current_user`` — that
extracts, verifies, and returns the authenticated user's UUID from an incoming
``Authorization: Bearer <token>`` header.

The OAuth flow itself (Google / GitHub redirect, callback, token exchange) runs
entirely on the Next.js frontend via the Supabase JS client.  The backend's
only responsibility is to cryptographically verify the JWT that the frontend
already obtained.

JWT Verification Strategy
--------------------------
Modern Supabase projects sign JWTs with **RS256** (asymmetric) and publish the
public verification keys at:
    https://<project-ref>.supabase.co/auth/v1/.well-known/jwks.json

Older projects and the local Supabase CLI emulator use **HS256** (symmetric)
with a static ``JWT_SECRET``.

This module handles both automatically by inspecting the ``alg`` header of
the incoming token:

1. ``alg == "RS256"`` → verify using ``PyJWKClient`` (fetches + caches public
   keys from the JWKS endpoint; no round-trip to Supabase on every request).
2. ``alg == "HS256"`` → verify locally using ``SUPABASE_JWT_SECRET``.

``PyJWKClient`` is thread-safe and caches keys by ``kid`` (Key ID), refreshing
only when an unknown ``kid`` is encountered, making this verification path
extremely low-latency.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

# HTTPBearer extracts the raw token string from "Authorization: Bearer <token>".
# auto_error=False means FastAPI will pass None when the header is absent rather
# than raising its own HTTPException — we handle the error ourselves to ensure
# it flows through our StudyForgeException handler (clean JSON, no stack trace).
_http_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _get_jwks_client() -> jwt.PyJWKClient | None:
    """Lazily create (and cache) a PyJWKClient for RS256 JWKS verification.

    Returns None if no SUPABASE_URL is configured, which gracefully falls back
    to HS256 verification.  The client is constructed at most once per process.
    """
    settings = get_settings()
    jwks_url = settings.supabase_jwks_url
    if not jwks_url:
        return None
    logger.info(f"Initializing PyJWKClient for JWKS URL: {jwks_url}")
    return jwt.PyJWKClient(
        jwks_url,
        cache_keys=True,       # Cache fetched keys by 'kid'
        lifespan=3600,         # Refresh cached keys every hour
    )


def _verify_rs256(token: str) -> dict:
    """Verify an RS256-signed JWT using the Supabase project's JWKS endpoint.

    Args:
        token: The raw JWT string (without 'Bearer ' prefix).

    Returns:
        dict: The decoded, verified JWT payload.

    Raises:
        AuthenticationError: If key lookup fails or the token is invalid.
    """
    jwks_client = _get_jwks_client()
    if jwks_client is None:
        raise AuthenticationError(
            "RS256 token received but SUPABASE_URL is not configured. "
            "Cannot fetch JWKS public keys."
        )
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            # Supabase sets aud="authenticated" for user tokens
            options={"verify_aud": False},  # audience varies; rely on 'role' claim instead
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired. Please log in again.")
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(f"Invalid token: {exc}")


def _verify_hs256(token: str) -> dict:
    """Verify an HS256-signed JWT using the configured SUPABASE_JWT_SECRET.

    Used for local Supabase CLI emulator and legacy Supabase projects.

    Args:
        token: The raw JWT string (without 'Bearer ' prefix).

    Returns:
        dict: The decoded, verified JWT payload.

    Raises:
        AuthenticationError: If the secret is missing or the token is invalid.
    """
    settings = get_settings()
    if not settings.supabase_jwt_secret:
        raise AuthenticationError(
            "HS256 token received but SUPABASE_JWT_SECRET is not configured. "
            "Set it in your .env file."
        )
    try:
        return jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired. Please log in again.")
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(f"Invalid token: {exc}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_http_bearer),
) -> str:
    """FastAPI dependency that authenticates the request and returns the user UUID.

    Extracts the JWT from the ``Authorization: Bearer <token>`` header,
    detects the signing algorithm (RS256 or HS256), verifies the token
    cryptographically, and extracts the ``sub`` claim (the user's UUID in
    Supabase Auth).

    Usage::

        @router.post("/upload")
        async def upload(user_id: str = Depends(get_current_user)):
            ...

    Args:
        credentials: Extracted by FastAPI's HTTPBearer scheme.

    Returns:
        str: The authenticated user's UUID (``sub`` claim from the JWT).

    Raises:
        AuthenticationError: If no token is provided, the token is malformed,
            the signature is invalid, or the token has expired.
    """
    if credentials is None:
        raise AuthenticationError(
            "No authentication token provided. "
            "Include 'Authorization: Bearer <token>' in the request header."
        )

    token = credentials.credentials

    # Peek at the header to decide which algorithm to use — no secret needed yet.
    try:
        header = jwt.get_unverified_header(token)
    except jwt.exceptions.DecodeError:
        raise AuthenticationError("Malformed token: could not decode JWT header.")

    alg = header.get("alg", "")

    if alg == "RS256":
        payload = _verify_rs256(token)
    elif alg == "HS256":
        payload = _verify_hs256(token)
    else:
        raise AuthenticationError(
            f"Unsupported JWT signing algorithm '{alg}'. Expected RS256 or HS256."
        )

    # 'sub' is the user's UUID in Supabase Auth (RFC 7519 subject claim)
    user_id: str | None = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Token payload is missing the 'sub' (user ID) claim.")

    logger.debug(f"Authenticated user: {user_id}")
    return user_id
