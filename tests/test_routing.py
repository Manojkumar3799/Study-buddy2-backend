"""Tests for the LLM-controlled routing service.

Covers:
  - Keyword pre-filter: PDF signals, web signals, ambiguous (None)
  - Full classify_route: keyword fast paths
  - Full classify_route: LLM path (mocked generate_answer)
  - Full classify_route: fallback to pdf_rag on LLM failure
"""

import pytest
from unittest.mock import patch

from app.services.routing_service import (
    RouteDecision,
    _keyword_prefilter,
    classify_route,
)


# ---------------------------------------------------------------------------
# _keyword_prefilter — unit tests (no LLM, no IO)
# ---------------------------------------------------------------------------

class TestKeywordPrefilter:
    def test_pdf_signal_document(self):
        assert _keyword_prefilter("according to my document, what is this?") == "pdf_rag"

    def test_pdf_signal_pdf(self):
        assert _keyword_prefilter("in the pdf, what does chapter 3 say?") == "pdf_rag"

    def test_pdf_signal_on_page(self):
        assert _keyword_prefilter("what is mentioned on page 5?") == "pdf_rag"

    def test_pdf_signal_author(self):
        assert _keyword_prefilter("The author says this is correct") == "pdf_rag"

    def test_web_signal_latest(self):
        assert _keyword_prefilter("what is the latest news on AI?") == "web_search"

    def test_web_signal_recent(self):
        assert _keyword_prefilter("Recent developments in quantum computing") == "web_search"

    def test_web_signal_today(self):
        assert _keyword_prefilter("What happened today in the stock market?") == "web_search"

    def test_web_signal_year_2025(self):
        assert _keyword_prefilter("best AI tools in 2025") == "web_search"

    def test_ambiguous_returns_none(self):
        """Questions with no keywords should return None so LLM is consulted."""
        assert _keyword_prefilter("explain machine learning") is None

    def test_ambiguous_concept(self):
        assert _keyword_prefilter("what is photosynthesis?") is None

    def test_pdf_takes_priority_over_web(self):
        """PDF signals beat web signals — prevents routing away from PDF for
        questions like 'latest results in my document'."""
        result = _keyword_prefilter("according to my document, what are the latest results?")
        assert result == "pdf_rag"


# ---------------------------------------------------------------------------
# classify_route — integration-style with mocked LLM
# ---------------------------------------------------------------------------

class TestClassifyRoute:
    def test_web_keyword_no_llm_call(self):
        """Keyword match should short-circuit before any LLM call."""
        with patch("app.services.routing_service._classify_via_llm") as mock_llm:
            result = classify_route("latest news on climate change 2025")
        assert result == "web_search"
        mock_llm.assert_not_called()

    def test_pdf_keyword_no_llm_call(self):
        with patch("app.services.routing_service._classify_via_llm") as mock_llm:
            result = classify_route("according to my PDF, summarise the methodology")
        assert result == "pdf_rag"
        mock_llm.assert_not_called()

    def test_ambiguous_calls_llm_returns_web(self):
        """Ambiguous question delegates to LLM and returns its decision."""
        with patch("app.services.routing_service._classify_via_llm", return_value="web_search"):
            result = classify_route("explain reinforcement learning")
        assert result == "web_search"

    def test_ambiguous_calls_llm_returns_pdf(self):
        with patch("app.services.routing_service._classify_via_llm", return_value="pdf_rag"):
            result = classify_route("explain reinforcement learning")
        assert result == "pdf_rag"

    def test_fallback_on_llm_failure(self):
        """When LLM classification raises, classify_route must return pdf_rag."""
        with patch(
            "app.services.llm_service.generate_answer",
            side_effect=Exception("provider down"),
        ):
            result = classify_route("explain quantum entanglement")
        assert result == "pdf_rag"

    def test_fallback_on_invalid_llm_json(self):
        """When LLM returns non-JSON, must still return pdf_rag safely."""
        with patch(
            "app.services.llm_service.generate_answer",
            return_value=("I am not sure", "gemini"),
        ):
            result = classify_route("explain quantum entanglement")
        assert result == "pdf_rag"

    def test_fallback_on_unknown_route_value(self):
        """When LLM returns unexpected route, must fall back to pdf_rag."""
        with patch(
            "app.services.llm_service.generate_answer",
            return_value=('{"route": "unknown_value"}', "gemini"),
        ):
            result = classify_route("explain something ambiguous")
        assert result == "pdf_rag"

    def test_valid_llm_web_json(self):
        with patch(
            "app.services.llm_service.generate_answer",
            return_value=('{"route": "web_search"}', "gemini"),
        ):
            result = classify_route("something ambiguous")
        assert result == "web_search"

    def test_valid_llm_pdf_json_with_surrounding_text(self):
        """LLMs often wrap JSON in prose — must still parse correctly."""
        response_text = 'Sure! Based on the question: {"route": "pdf_rag"} Hope this helps!'
        with patch(
            "app.services.llm_service.generate_answer",
            return_value=(response_text, "gemini"),
        ):
            result = classify_route("something ambiguous")
        assert result == "pdf_rag"
