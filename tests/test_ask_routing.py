"""Tests for the /ask routing integration in the FastAPI endpoints.

Tests verify:
  - mode='pdf' always routes to PDF RAG
  - mode='web' always routes to Firecrawl
  - mode='auto' routes based on LLM / keyword classifier
  - Web path emits correct source_type='web' in the sources event
  - PDF path emits correct source_type='pdf' in the sources event
  - LLM provider fallback still works end-to-end on both paths
  - MCP failure on web path returns graceful no-context response

These are integration tests that mock at the service boundaries
(classify_route, firecrawl_search, retrieve_relevant_chunks, stream_answer)
so no real LLM calls or database queries are made.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.auth import get_current_user
from app.services.mcp_client import WebResult
from app.services.retrieval_service import RetrievedChunk


# ---------------------------------------------------------------------------
# Auth override — bypass JWT verification for routing tests
# ---------------------------------------------------------------------------

TEST_USER_ID = "test-user-aaaaaaaa-1111-aaaa-1111-aaaaaaaaaaaa"


def _override_get_current_user():
    """Stub dependency: returns a fixed test user_id without verifying any JWT."""
    return TEST_USER_ID


# Apply the override for all tests in this module
app.dependency_overrides[get_current_user] = _override_get_current_user

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCUMENT_ID = "test-doc-123"

FAKE_CHUNKS = [
    RetrievedChunk(chunk_id=0, text="Photosynthesis converts sunlight.", start_page=1, end_page=1, score=0.9),
    RetrievedChunk(chunk_id=1, text="Chlorophyll absorbs light.", start_page=2, end_page=2, score=0.85),
]

FAKE_WEB_RESULTS = [
    WebResult(title="AI 2025 News", url="https://example.com/ai", domain="example.com", content="Latest AI content"),
    WebResult(title="ML Blog", url="https://blog.ml/post", domain="blog.ml", content="Machine learning content"),
]


def _collect_stream_events(lines: str) -> list[dict]:
    """Parse all NDJSON events from a streaming response body."""
    events = []
    for line in lines.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _make_fake_stream(tokens: list[str], provider: str = "gemini"):
    """Return a generator that yields tokens then the __provider__ sentinel."""
    def _gen(messages):
        for t in tokens:
            yield t
        yield ("__provider__", provider)
    return _gen


# ---------------------------------------------------------------------------
# Non-streaming endpoint: mode tests
# ---------------------------------------------------------------------------

class TestAskEndpointMode:
    def test_mode_pdf_always_uses_rag(self):
        """mode='pdf' should call retrieve_relevant_chunks, not firecrawl_search."""
        with (
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=FAKE_CHUNKS),
            patch("app.api.ask.generate_answer", return_value=("PDF answer", "gemini")),
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock) as mock_web,
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}",
                json={"question": "What is photosynthesis?", "mode": "pdf"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] == "pdf"
        assert data["answer"] == "PDF answer"
        assert data["has_sufficient_context"] is True
        mock_web.assert_not_called()

    def test_mode_web_always_uses_firecrawl(self):
        """mode='web' should call firecrawl_search, not retrieve_relevant_chunks."""
        with (
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock, return_value=FAKE_WEB_RESULTS),
            patch("app.api.ask.generate_answer", return_value=("Web answer", "gemini")),
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock) as mock_rag,
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}",
                json={"question": "latest AI news", "mode": "web"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] == "web"
        assert data["answer"] == "Web answer"
        assert len(data["web_sources"]) == 2
        assert data["web_sources"][0]["url"] == "https://example.com/ai"
        mock_rag.assert_not_called()

    def test_mode_auto_with_pdf_keyword(self):
        """mode='auto' + PDF keyword → routed to PDF RAG without LLM call."""
        with (
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=FAKE_CHUNKS),
            patch("app.api.ask.generate_answer", return_value=("PDF answer", "gemini")),
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock) as mock_web,
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}",
                json={"question": "according to my document, explain this", "mode": "auto"},
            )
        assert resp.status_code == 200
        assert resp.json()["source_type"] == "pdf"
        mock_web.assert_not_called()

    def test_mode_auto_with_web_keyword(self):
        """mode='auto' + web keyword → routed to Firecrawl without LLM call."""
        with (
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock, return_value=FAKE_WEB_RESULTS),
            patch("app.api.ask.generate_answer", return_value=("Web answer", "gemini")),
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock) as mock_rag,
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}",
                json={"question": "latest news on AI 2025", "mode": "auto"},
            )
        assert resp.status_code == 200
        assert resp.json()["source_type"] == "web"
        mock_rag.assert_not_called()

    def test_web_mcp_failure_returns_no_context(self):
        """When Firecrawl returns [], the response says no context found."""
        with (
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock, return_value=[]),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}",
                json={"question": "latest news", "mode": "web"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_sufficient_context"] is False
        assert data["source_type"] == "web"
        assert data["web_sources"] == []

    def test_pdf_no_context_returns_message(self):
        """When PDF retrieval returns [], the response says no context found."""
        with (
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=[]),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}",
                json={"question": "something not in the PDF", "mode": "pdf"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_sufficient_context"] is False
        assert data["source_type"] == "pdf"


# ---------------------------------------------------------------------------
# Streaming endpoint: source_type in events
# ---------------------------------------------------------------------------

class TestAskStreamRouting:
    def test_stream_pdf_mode_emits_pdf_source_type(self):
        with (
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=FAKE_CHUNKS),
            patch("app.api.ask.stream_answer", side_effect=_make_fake_stream(["Hello ", "world"])),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}/stream",
                json={"question": "What is photosynthesis?", "mode": "pdf"},
            )
        assert resp.status_code == 200
        events = _collect_stream_events(resp.text)
        sources_event = next(e for e in events if e["type"] == "sources")
        assert sources_event["source_type"] == "pdf"
        assert sources_event["has_sufficient_context"] is True
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "Hello world"
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["provider_used"] == "gemini"

    def test_stream_web_mode_emits_web_source_type(self):
        with (
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock, return_value=FAKE_WEB_RESULTS),
            patch("app.api.ask.stream_answer", side_effect=_make_fake_stream(["Web ", "answer"])),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}/stream",
                json={"question": "latest news", "mode": "web"},
            )
        assert resp.status_code == 200
        events = _collect_stream_events(resp.text)
        sources_event = next(e for e in events if e["type"] == "sources")
        assert sources_event["source_type"] == "web"
        assert len(sources_event["sources"]) == 2
        assert sources_event["sources"][0]["url"] == "https://example.com/ai"
        assert sources_event["sources"][0]["domain"] == "example.com"

    def test_stream_web_no_results_emits_no_context(self):
        with (
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock, return_value=[]),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}/stream",
                json={"question": "latest news", "mode": "web"},
            )
        assert resp.status_code == 200
        events = _collect_stream_events(resp.text)
        sources_event = next(e for e in events if e["type"] == "sources")
        assert sources_event["has_sufficient_context"] is False
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["provider_used"] is None

    def test_stream_pdf_no_context(self):
        with (
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=[]),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}/stream",
                json={"question": "something not in doc", "mode": "pdf"},
            )
        assert resp.status_code == 200
        events = _collect_stream_events(resp.text)
        sources_event = next(e for e in events if e["type"] == "sources")
        assert sources_event["has_sufficient_context"] is False

    def test_stream_llm_error_emits_error_event(self):
        """When all LLM providers fail, an 'error' event is emitted in the stream."""
        from app.core.exceptions import AllProvidersFailedError

        with (
            patch("app.api.ask.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=FAKE_CHUNKS),
            patch("app.api.ask.stream_answer", side_effect=AllProvidersFailedError()),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}/stream",
                json={"question": "test", "mode": "pdf"},
            )
        assert resp.status_code == 200
        events = _collect_stream_events(resp.text)
        error_event = next((e for e in events if e["type"] == "error"), None)
        assert error_event is not None
        assert "AllProvidersFailedError" in error_event["error"]

    def test_stream_default_mode_is_auto(self):
        """Omitting mode should default to 'auto' — keyword 'latest' → web."""
        with (
            patch("app.api.ask.firecrawl_search", new_callable=AsyncMock, return_value=FAKE_WEB_RESULTS),
            patch("app.api.ask.stream_answer", side_effect=_make_fake_stream(["ans"])),
        ):
            client = TestClient(app)
            resp = client.post(
                f"/ask/{DOCUMENT_ID}/stream",
                # No mode field — defaults to "auto"
                json={"question": "latest news 2025"},
            )
        assert resp.status_code == 200
        events = _collect_stream_events(resp.text)
        sources_event = next(e for e in events if e["type"] == "sources")
        assert sources_event["source_type"] == "web"
