"""Utilities for cleaning raw text extracted from PDFs."""

import re


def clean_text(raw_text: str) -> str:
    """
    Clean raw extracted PDF text for downstream processing.

    Removes excessive whitespace, control characters, and normalizes
    line breaks while preserving paragraph structure.

    Args:
        raw_text: Unprocessed text extracted from a PDF page.

    Returns:
        str: Cleaned text.
    """
    if not raw_text:
        return ""

    # Remove null bytes and non-printable control characters (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_text)

    # Normalize different newline styles
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Strip leading/trailing whitespace on the whole text
    text = text.strip()

    return text


def is_text_meaningful(text: str, min_words: int = 5) -> bool:
    """
    Determine whether extracted text contains meaningful content.

    Args:
        text: Cleaned text to evaluate.
        min_words: Minimum word count to consider the text meaningful.

    Returns:
        bool: True if the text has enough words to be usable.
    """
    word_count = len(text.split())
    return word_count >= min_words