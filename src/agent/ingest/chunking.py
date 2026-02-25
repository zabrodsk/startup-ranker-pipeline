"""Split extracted text items into overlapping chunks with IDs."""

from agent.ingest.store import Chunk


def chunk_texts(
    raw_items: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Break raw extracted items into fixed-size overlapping chunks.

    Each raw item is expected to have:
        text: str
        source_file: str
        page | slide | page_or_slide: int | str  (location within the source)

    Args:
        raw_items: Output from any ingest extractor.
        chunk_size: Target number of characters per chunk.
        overlap: Character overlap between consecutive chunks from the same item.

    Returns:
        List of Chunk objects with globally unique chunk_ids.
    """
    chunks: list[Chunk] = []
    global_idx = 0

    for item in raw_items:
        text: str = item.get("text", "")
        source_file: str = item.get("source_file", "unknown")
        page_or_slide = (
            item.get("page")
            or item.get("slide")
            or item.get("page_or_slide", "N/A")
        )

        if not text.strip():
            continue

        if len(text) <= chunk_size:
            chunks.append(Chunk(
                chunk_id=f"chunk_{global_idx}",
                text=text,
                source_file=source_file,
                page_or_slide=page_or_slide,
            ))
            global_idx += 1
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"chunk_{global_idx}",
                    text=chunk_text,
                    source_file=source_file,
                    page_or_slide=page_or_slide,
                ))
                global_idx += 1
            start += chunk_size - overlap

    return chunks
