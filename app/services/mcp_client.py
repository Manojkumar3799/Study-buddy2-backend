"""Firecrawl MCP client — calls the firecrawl-mcp MCP server via stdio transport.

The firecrawl-mcp server is run as a subprocess (``npx -y firecrawl-mcp``) using
the MCP stdio transport as supported by ``langchain-mcp-adapters``.  Each public
function opens a fresh context-manager session so there is no shared subprocess
state between requests.

Environment requirement:
  FIRECRAWL_API_KEY must be set; if missing, all functions return empty results
  with a warning rather than raising.

Node.js requirement:
  ``npx`` must be on PATH.  If it is not, the subprocess will fail and the
  exception will be caught, returning empty results cleanly.
"""

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WebResult:
    """A single web search or scrape result with source metadata."""

    title: str
    url: str
    domain: str
    content: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Return the bare hostname from a URL, or the URL itself on failure."""
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _build_mcp_config() -> dict:
    """
    Build the MultiServerMCPClient server-config dict for Firecrawl.

    The subprocess inherits the current environment and additionally receives
    the FIRECRAWL_API_KEY override so that the MCP server can authenticate.
    """
    env = {**os.environ, "FIRECRAWL_API_KEY": settings.firecrawl_api_key}
    return {
        "firecrawl": {
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"],
            "env": env,
            "transport": "stdio",
        }
    }


def _parse_search_output(raw) -> list[WebResult]:
    """
    Parse the raw output of the ``firecrawl_search`` MCP tool into a list of
    ``WebResult`` objects.

    Firecrawl MCP may return:
    - A JSON string: ``'{"success": true, "data": [...]}'``
    - A plain-text string (fallback)
    - A Python dict / list (when langchain-mcp-adapters deserialises the tool
      content before returning it)

    We handle all three cases defensively.
    """
    # Normalise to a Python object
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Treat the whole string as a single result snippet
            return [WebResult(title="Web Result", url="", domain="", content=raw[:3000])]
    else:
        data = raw

    # Firecrawl v1 envelope: {"success": true, "data": [...]}
    if isinstance(data, dict):
        items = data.get("data", data.get("results", [data]))
    elif isinstance(data, list):
        items = data
    else:
        return []

    results: list[WebResult] = []
    for item in items[:10]:  # cap at 10 raw items
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        # Prefer 'markdown' content if present (richest form)
        content = (
            item.get("markdown")
            or item.get("content")
            or item.get("description")
            or item.get("snippet")
            or ""
        )
        results.append(
            WebResult(
                title=item.get("title") or item.get("name") or url or "Web Result",
                url=url,
                domain=_extract_domain(url),
                content=content[:3000],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def firecrawl_search(query: str, num_results: int = 5) -> list[WebResult]:
    """
    Search the web using the Firecrawl MCP ``firecrawl_search`` tool.

    Args:
        query: The natural-language search query.
        num_results: Maximum number of results to request (default 5).

    Returns:
        list[WebResult]: Parsed search results; empty list on any failure.
    """
    if not settings.firecrawl_api_key:
        logger.warning(
            "FIRECRAWL_API_KEY is not set — web search unavailable. "
            "Add it to .env to enable Firecrawl."
        )
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore[import]
    except ImportError:
        logger.error(
            "langchain-mcp-adapters is not installed. "
            "Run: pip install langchain-mcp-adapters"
        )
        return []

    try:
        logger.info(f"Firecrawl MCP search: query='{query[:80]}'")
        async with MultiServerMCPClient(_build_mcp_config()) as client:
            tools = client.get_tools()

            search_tool = next(
                (t for t in tools if t.name == "firecrawl_search"), None
            )
            if search_tool is None:
                logger.error("Firecrawl MCP: 'firecrawl_search' tool not found in tool list")
                return []

            raw = await search_tool.ainvoke(
                {"query": query, "limit": num_results}
            )

        results = _parse_search_output(raw)
        logger.info(f"Firecrawl MCP search returned {len(results)} result(s)")
        return results

    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"Firecrawl MCP search failed: {exc.__class__.__name__}: {exc}"
        )
        return []


async def firecrawl_scrape(url: str) -> WebResult | None:
    """
    Scrape a single URL using the Firecrawl MCP ``firecrawl_scrape`` tool.

    Args:
        url: The URL to scrape.

    Returns:
        WebResult: Scraped content; None on any failure.
    """
    if not settings.firecrawl_api_key:
        logger.warning("FIRECRAWL_API_KEY is not set — web scrape unavailable.")
        return None

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore[import]
    except ImportError:
        logger.error("langchain-mcp-adapters is not installed.")
        return None

    try:
        logger.info(f"Firecrawl MCP scrape: url='{url}'")
        async with MultiServerMCPClient(_build_mcp_config()) as client:
            tools = client.get_tools()

            scrape_tool = next(
                (t for t in tools if t.name == "firecrawl_scrape"), None
            )
            if scrape_tool is None:
                logger.error("Firecrawl MCP: 'firecrawl_scrape' tool not found")
                return None

            raw = await scrape_tool.ainvoke({"url": url})

        content = raw if isinstance(raw, str) else json.dumps(raw)
        logger.info(f"Firecrawl MCP scrape returned {len(content)} chars")
        return WebResult(
            title=url,
            url=url,
            domain=_extract_domain(url),
            content=content[:4000],
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"Firecrawl MCP scrape failed: {exc.__class__.__name__}: {exc}"
        )
        return None
