"""Service layer for splitting extracted PDF text into overlapping chunks."""

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.text_extraction_service import PageText

logger = get_logger(__name__)
settings = get_settings()


class Chunk:
    """Represents a single text chunk with source page metadata."""

    def __init__(
        self,
        chunk_id: int,
        text: str,
        word_count: int,
        start_page: int,
        end_page: int,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.word_count = word_count
        self.start_page = start_page
        self.end_page = end_page


def _flatten_words_with_page_tags(pages: list[PageText]) -> list[tuple[str, int]]:
    """
    Flatten all pages into a single list of (word, page_number) tuples.

    Args:
        pages: List of extracted page texts.

    Returns:
        list[tuple[str, int]]: Word-to-page mapping preserving document order.
    """
    tagged_words: list[tuple[str, int]] = []
    for page in pages:
        for word in page.text.split():
            tagged_words.append((word, page.page_number))
    return tagged_words


def chunk_pages(
    pages: list[PageText],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Split page text into fixed-size overlapping word chunks.

    Args:
        pages: Extracted and cleaned per-page text.
        chunk_size: Number of words per chunk (defaults to configured value).
        overlap: Number of overlapping words between consecutive chunks
            (defaults to configured value).

    Returns:
        list[Chunk]: Ordered list of chunks with page-range metadata.
    """
    chunk_size = chunk_size or settings.chunk_size_words
    overlap = overlap or settings.chunk_overlap_words

    if overlap >= chunk_size:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    tagged_words = _flatten_words_with_page_tags(pages)
    total_words = len(tagged_words)

    if total_words == 0:
        logger.warning("No words available to chunk")
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    chunk_id = 0
    start_index = 0

    while start_index < total_words:
        end_index = min(start_index + chunk_size, total_words)
        window = tagged_words[start_index:end_index]

        words = [w for w, _ in window]
        page_numbers = [p for _, p in window]

        chunk_text = " ".join(words)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                word_count=len(words),
                start_page=min(page_numbers),
                end_page=max(page_numbers),
            )
        )

        chunk_id += 1

        if end_index == total_words:
            break

        start_index += step

    logger.info(
        f"Chunking complete: total_words={total_words}, "
        f"chunk_size={chunk_size}, overlap={overlap}, chunks_created={len(chunks)}"
    )

    return chunks