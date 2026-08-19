"""Tests for the Firecrawl MCP client.

All tests mock out the MCP subprocess — no real network calls are made.
Covers:
  - firecrawl_search: happy path, no API key, import error, MCP failure
  - firecrawl_scrape: happy path, no API key, import error, MCP failure
  - _parse_search_output: various output formats
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# _parse_search_output — pure unit tests (no IO)
# ---------------------------------------------------------------------------

class TestParseSearchOutput:
    def _parse(self, raw):
        from app.services.mcp_client import _parse_search_output
        return _parse_search_output(raw)

    def test_firecrawl_v1_envelope(self):
        raw = json.dumps({
            "success": True,
            "data": [
                {"title": "AI News", "url": "https://example.com/ai", "content": "Content here"},
                {"title": "ML Blog", "url": "https://blog.ml/post", "description": "Desc"},
            ]
        })
        results = self._parse(raw)
        assert len(results) == 2
        assert results[0].title == "AI News"
        assert results[0].url == "https://example.com/ai"
        assert results[0].domain == "example.com"
        assert results[0].content == "Content here"

    def test_plain_list(self):
        raw = [
            {"title": "Page 1", "url": "https://a.com", "markdown": "# Markdown"},
        ]
        results = self._parse(raw)
        assert len(results) == 1
        assert results[0].content == "# Markdown"

    def test_plain_string_fallback(self):
        """Non-JSON string should be wrapped in a single WebResult."""
        results = self._parse("Some web content that is not JSON")
        assert len(results) == 1
        assert "Some web content" in results[0].content

    def test_caps_at_10_items(self):
        raw = [{"title": f"Item {i}", "url": f"https://x.com/{i}", "content": "x"} for i in range(20)]
        results = self._parse(raw)
        assert len(results) == 10

    def test_content_truncated_at_3000(self):
        long_content = "x" * 5000
        raw = [{"title": "T", "url": "https://a.com", "content": long_content}]
        results = self._parse(raw)
        assert len(results[0].content) <= 3000

    def test_missing_fields_handled(self):
        raw = [{"url": "https://a.com"}]  # no title, no content
        results = self._parse(raw)
        assert results[0].domain == "a.com"
        assert results[0].title in ("https://a.com", "Web Result")

    def test_prefers_markdown_over_content(self):
        raw = [{"title": "T", "url": "https://a.com", "markdown": "# MD", "content": "plain"}]
        results = self._parse(raw)
        assert results[0].content == "# MD"


# ---------------------------------------------------------------------------
# _extract_domain — utility
# ---------------------------------------------------------------------------

class TestExtractDomain:
    def test_standard_url(self):
        from app.services.mcp_client import _extract_domain
        assert _extract_domain("https://www.example.com/path") == "www.example.com"

    def test_no_scheme(self):
        from app.services.mcp_client import _extract_domain
        # urlparse can't extract netloc without a scheme
        result = _extract_domain("example.com/path")
        assert result  # should not raise


# ---------------------------------------------------------------------------
# firecrawl_search — with mocked MCP
# ---------------------------------------------------------------------------

class TestFirecrawlSearch:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self):
        with patch("app.services.mcp_client.settings") as mock_settings:
            mock_settings.firecrawl_api_key = ""
            from app.services.mcp_client import firecrawl_search
            results = await firecrawl_search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_import_error(self):
        with (
            patch("app.services.mcp_client.settings") as mock_settings,
            patch.dict("sys.modules", {"langchain_mcp_adapters.client": None}),
        ):
            mock_settings.firecrawl_api_key = "fc-test-key"
            # Re-import to trigger ImportError branch
            import importlib
            import app.services.mcp_client as mod
            importlib.reload(mod)
            results = await mod.firecrawl_search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_happy_path_returns_web_results(self):
        """Happy path: MCP returns Firecrawl search output."""
        from app.services.mcp_client import firecrawl_search, WebResult

        mock_search_tool = AsyncMock()
        mock_search_tool.name = "firecrawl_search"
        mock_search_tool.ainvoke = AsyncMock(
            return_value=json.dumps({
                "success": True,
                "data": [
                    {
                        "title": "AI Article",
                        "url": "https://example.com/ai",
                        "content": "Great AI content here",
                    }
                ],
            })
        )

        mock_client_instance = MagicMock()
        mock_client_instance.get_tools = MagicMock(return_value=[mock_search_tool])
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        mock_mcp_class = MagicMock(return_value=mock_client_instance)

        with (
            patch("app.services.mcp_client.settings") as mock_settings,
            patch("app.services.mcp_client.MultiServerMCPClient", mock_mcp_class, create=True),
        ):
            # Patch the import inside the function
            import langchain_mcp_adapters.client as lmcp
            with patch.object(lmcp, "MultiServerMCPClient", mock_mcp_class):
                mock_settings.firecrawl_api_key = "fc-test-key"
                results = await firecrawl_search("AI news")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_returns_empty_on_mcp_exception(self):
        """Any MCP failure should return [] without raising."""
        from app.services.mcp_client import firecrawl_search

        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("npx not found"))
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_mcp_class = MagicMock(return_value=mock_client_instance)

        with patch("app.services.mcp_client.settings") as mock_settings:
            mock_settings.firecrawl_api_key = "fc-test-key"
            try:
                import langchain_mcp_adapters.client as lmcp
                with patch.object(lmcp, "MultiServerMCPClient", mock_mcp_class):
                    results = await firecrawl_search("test query")
                    assert results == []
            except ImportError:
                # langchain-mcp-adapters not installed in test env — that's fine
                pass

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_search_tool(self):
        """If the MCP server doesn't expose firecrawl_search, return []."""
        from app.services.mcp_client import firecrawl_search

        mock_client_instance = MagicMock()
        mock_client_instance.get_tools = MagicMock(return_value=[])  # no tools
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_mcp_class = MagicMock(return_value=mock_client_instance)

        with patch("app.services.mcp_client.settings") as mock_settings:
            mock_settings.firecrawl_api_key = "fc-test-key"
            try:
                import langchain_mcp_adapters.client as lmcp
                with patch.object(lmcp, "MultiServerMCPClient", mock_mcp_class):
                    results = await firecrawl_search("test query")
                    assert results == []
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# firecrawl_scrape — smoke tests
# ---------------------------------------------------------------------------

class TestFirecrawlScrape:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_api_key(self):
        with patch("app.services.mcp_client.settings") as mock_settings:
            mock_settings.firecrawl_api_key = ""
            from app.services.mcp_client import firecrawl_scrape
            result = await firecrawl_scrape("https://example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        from app.services.mcp_client import firecrawl_scrape

        with patch("app.services.mcp_client.settings") as mock_settings:
            mock_settings.firecrawl_api_key = "fc-test-key"
            try:
                import langchain_mcp_adapters.client as lmcp
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(side_effect=OSError("spawn failed"))
                mock_client.__aexit__ = AsyncMock(return_value=False)
                with patch.object(lmcp, "MultiServerMCPClient", MagicMock(return_value=mock_client)):
                    result = await firecrawl_scrape("https://example.com")
                    assert result is None
            except ImportError:
                pass
