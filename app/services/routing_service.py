"""LLM-controlled routing: decides whether to use PDF RAG or Firecrawl web search.

Routing strategy (layered for speed and cost efficiency):
  1. Keyword pre-filter  — O(1), no LLM cost.
     Explicit PDF signals  → ``pdf_rag``
     Explicit web signals  → ``web_search``
  2. LLM classification  — used only for ambiguous questions.
     A fast, structured prompt sent to the primary LLM (via the existing
     provider fallback chain) returns ``{"route": "pdf_rag"}`` or
     ``{"route": "web_search"}``.
  3. Fallback            — any LLM failure defaults to ``pdf_rag`` so that
     the existing document-answering path is never broken.
"""

import json
import re
from typing import Literal

from app.core.logging_config import get_logger

logger = get_logger(__name__)

RouteDecision = Literal["pdf_rag", "web_search"]

# ---------------------------------------------------------------------------
# Keyword pre-filter sets
# ---------------------------------------------------------------------------

# Any of these phrases in the (lowercased) question → force web_search
_WEB_SIGNALS: frozenset[str] = frozenset({
    "latest", "recent", "current events", "today", "right now", "breaking",
    "news", "search online", "look up", "find online", "search for",
    "trending", "real-time", "real time", "live", "this week", "this month",
    "this year", "stock price", "weather", "score", "results", "update",
    "2024", "2025", "2026", "who won", "just happened", "happening now",
})

# Any of these phrases in the (lowercased) question → force pdf_rag
_PDF_SIGNALS: frozenset[str] = frozenset({
    "according to my pdf", "according to my document", "according to the document",
    "according to the paper", "in the pdf", "in my pdf", "in the document",
    "in my document", "from my document", "from the document", "from the pdf",
    "in the paper", "the author says", "the author states", "the paper says",
    "on page", "on pages", "chapter ", "section ", "the study says",
    "as mentioned in", "as stated in", "as described in",
})

# ---------------------------------------------------------------------------
# Routing prompt
# ---------------------------------------------------------------------------

_ROUTING_SYSTEM_PROMPT = """\
You are a routing assistant for an AI study tool. \
Classify the user question into ONE of two categories:

- "pdf_rag"   : The question is about content in an uploaded document/PDF \
(specific topics, summaries, definitions, page references, etc.)
- "web_search": The question requires current, external, or real-world \
information that may not be in the PDF (news, prices, recent events, \
general knowledge lookups, etc.)

Key web_search signals: latest, recent, current, today, news, trending, \
search for, look up, 2025, live data, stock price, who won.
Key pdf_rag signals: according to my document/PDF, on page, the author says, \
from the paper, in the study.

Reply ONLY with valid JSON — no other text:
  {\"route\": \"pdf_rag\"}
  OR
  {\"route\": \"web_search\"}\
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _keyword_prefilter(question: str) -> RouteDecision | None:
    """
    Fast O(1) keyword check — no LLM cost.

    Returns a ``RouteDecision`` if the question contains an unambiguous signal,
    or ``None`` if the question needs LLM classification.

    PDF signals take priority over web signals so that questions like
    "what does the document say about the latest results?" still route to PDF.
    """
    q_lower = question.lower()

    for kw in _PDF_SIGNALS:
        if kw in q_lower:
            logger.debug(f"PDF keyword match: '{kw}'")
            return "pdf_rag"

    for kw in _WEB_SIGNALS:
        if kw in q_lower:
            logger.debug(f"Web keyword match: '{kw}'")
            return "web_search"

    return None


def _classify_via_llm(question: str) -> RouteDecision:
    """
    Ask the LLM to classify the question.  Falls back to ``pdf_rag`` on any
    failure (import error, provider failure, JSON parse error, invalid value).

    The function is deliberately synchronous so it can be handed off to a
    thread-pool executor by the async caller without additional complexity.
    """
    try:
        from app.services.llm_service import generate_answer  # lazy import avoids circulars

        messages = [
            {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify: {question}"},
        ]
        answer, provider = generate_answer(messages)
        logger.debug(f"LLM routing answer (provider={provider}): {answer!r}")

        # Extract JSON — LLMs sometimes add surrounding prose
        match = re.search(r'\{[^}]+\}', answer)
        if match:
            data = json.loads(match.group())
            route = data.get("route", "")
            if route in ("pdf_rag", "web_search"):
                return route  # type: ignore[return-value]
            logger.warning(f"LLM routing returned unknown route value: {route!r}")

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"LLM routing classification failed — defaulting to pdf_rag: "
            f"{exc.__class__.__name__}: {exc}"
        )

    return "pdf_rag"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_route(question: str) -> RouteDecision:
    """
    Classify a user question to determine the best answering strategy.

    The function is synchronous and safe to call from a thread-pool executor
    (``asyncio.get_event_loop().run_in_executor(None, classify_route, q)``).

    Args:
        question: The user's raw question text.

    Returns:
        RouteDecision: ``'pdf_rag'`` or ``'web_search'``.
        Guaranteed never to raise — all failures default to ``'pdf_rag'``.
    """
    logger.info(f"Routing classification for: '{question[:80]}'")

    prefilter = _keyword_prefilter(question)
    if prefilter is not None:
        logger.info(f"Keyword pre-filter → {prefilter}")
        return prefilter

    route = _classify_via_llm(question)
    logger.info(f"LLM classification → {route}")
    return route
