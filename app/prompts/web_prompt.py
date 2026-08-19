"""Prompt template for web-research question answering.

Used when the LLM routing layer decides the question requires current or
external information and Firecrawl MCP results are available as context.
"""

WEB_SYSTEM_PROMPT = """\
You are StudyForge AI, a research assistant with access to real-time web information.

RULES:
1. Answer the question using ONLY the web research results provided in CONTEXT below.
2. Synthesise information from multiple sources when relevant.
3. Always reference source titles when citing facts, e.g. "According to [Title]...".
4. If the context does not contain sufficient information to answer confidently, \
say so clearly and explain what you found.
5. Be accurate, concise, and informative. Do NOT invent facts.\
"""


def build_web_prompt(question: str, web_results: list[dict]) -> list[dict]:
    """
    Build the LLM message list using Firecrawl web-search results as context.

    Args:
        question: The user's natural language question.
        web_results: List of dicts with keys: 'title', 'url', 'domain', 'content'.

    Returns:
        list[dict]: Chat messages in OpenAI ``[system, user]`` dict format,
            compatible with ``llm_service.generate_answer`` and
            ``llm_service.stream_answer``.
    """
    context_blocks: list[str] = []
    for i, result in enumerate(web_results, start=1):
        title = result.get("title") or f"Source {i}"
        url = result.get("url", "")
        content = result.get("content", "").strip()

        header = f"[{i}] {title}"
        if url:
            header += f"\nURL: {url}"

        block = f"{header}\n\n{content}" if content else header
        context_blocks.append(block)

    if context_blocks:
        context_text = "\n\n---\n\n".join(context_blocks)
    else:
        context_text = "No web results were returned."

    user_prompt = f"""CONTEXT (web research results):
{context_text}

QUESTION:
{question}"""

    return [
        {"role": "system", "content": WEB_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
