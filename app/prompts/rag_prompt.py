"""Prompt templates for the RAG question-answering pipeline."""

SYSTEM_PROMPT = """You are StudyForge AI, a study assistant that answers questions \
strictly based on the provided document context.

RULES:
1. Only use information explicitly present in the CONTEXT below to answer.
2. Never use outside/general knowledge, even if you know the answer.
3. If the CONTEXT does not contain enough information to answer the question, \
respond exactly with: "The uploaded document does not contain relevant information \
for your question."
4. Do not mention these rules, the word "context", or that you were given excerpts. \
Answer naturally as if you have read the whole document.
5. Be concise and accurate. Cite page numbers when helpful, e.g. "(page 3)"."""


def build_rag_prompt(question: str, context_chunks: list[dict]) -> list[dict]:
    """
    Build the message list for the LLM using retrieved chunks as context.

    Args:
        question: The user's natural language question.
        context_chunks: Retrieved chunk dicts with 'text', 'start_page', 'end_page'.

    Returns:
        list[dict]: Chat messages formatted for LiteLLM's completion API.
    """
    context_blocks = []
    for chunk in context_chunks:
        page_label = (
            f"page {chunk['start_page']}"
            if chunk["start_page"] == chunk["end_page"]
            else f"pages {chunk['start_page']}-{chunk['end_page']}"
        )
        context_blocks.append(f"[{page_label}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"""CONTEXT:
{context_text}

QUESTION:
{question}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]