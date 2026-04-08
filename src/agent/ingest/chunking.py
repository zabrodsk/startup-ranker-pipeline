"""Split extracted text items into chunks with IDs.

Provides two chunking strategies:
- chunk_texts(): Legacy fixed-size overlapping window (500-char default).
- smart_chunk_texts(): Semantic-boundary-aware chunking (no overlap).
"""

from __future__ import annotations

import re

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


# ---------------------------------------------------------------------------
# Semantic-boundary-aware chunking
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_text_hierarchical(text: str, max_size: int) -> list[str]:
    """Split *text* into pieces ≤ *max_size* using semantic boundaries.

    Tries splits in priority order:
      1. Double newlines (paragraph boundaries)
      2. Single newlines (line boundaries)
      3. Sentence endings (.!? followed by whitespace)
      4. Word boundaries (spaces)
      5. Hard cut (last resort)
    """
    if len(text) <= max_size:
        return [text]

    # Try each split strategy in order of semantic quality.
    splitters: list[tuple[str, re.Pattern[str] | str]] = [
        ("para", "\n\n"),
        ("line", "\n"),
        ("sentence", _SENTENCE_RE),
        ("word", " "),
    ]

    for _name, sep in splitters:
        if isinstance(sep, re.Pattern):
            parts = sep.split(text)
        else:
            parts = text.split(sep)

        # Filter empties but keep whitespace-only parts for merge accuracy.
        parts = [p for p in parts if p]

        if len(parts) <= 1:
            # This separator didn't help — try next level.
            continue

        # Greedily merge consecutive parts up to max_size.
        merged: list[str] = []
        buf = parts[0]
        joiner = "\n\n" if sep == "\n\n" else ("\n" if sep == "\n" else " ")

        for part in parts[1:]:
            candidate = buf + joiner + part
            if len(candidate) <= max_size:
                buf = candidate
            else:
                merged.append(buf)
                buf = part
        if buf:
            merged.append(buf)

        # If every merged piece is ≤ max_size we're done.
        all_ok = all(len(m) <= max_size for m in merged)
        if all_ok:
            return merged

        # Some pieces still too large — recursively split those.
        result: list[str] = []
        for m in merged:
            if len(m) <= max_size:
                result.append(m)
            else:
                result.extend(_split_text_hierarchical(m, max_size))
        return result

    # Last resort: hard-cut at max_size boundaries.
    pieces: list[str] = []
    for i in range(0, len(text), max_size):
        piece = text[i : i + max_size].strip()
        if piece:
            pieces.append(piece)
    return pieces


def _merge_small_pieces(pieces: list[str], min_size: int, target_size: int) -> list[str]:
    """Merge consecutive small pieces (< *min_size*) as long as result ≤ *target_size*."""
    if not pieces:
        return pieces

    merged: list[str] = []
    buf = pieces[0]

    for piece in pieces[1:]:
        if len(buf) < min_size and len(buf) + 1 + len(piece) <= target_size:
            buf = buf + "\n" + piece
        elif len(piece) < min_size and len(buf) + 1 + len(piece) <= target_size:
            buf = buf + "\n" + piece
        else:
            merged.append(buf)
            buf = piece
    merged.append(buf)

    return merged


def smart_chunk_texts(
    raw_items: list[dict],
    target_size: int = 1000,
    max_size: int = 2000,
    min_size: int = 150,
) -> list[Chunk]:
    """Split extracted items into chunks using semantic boundaries.

    Unlike :func:`chunk_texts`, this function:
    - Respects paragraph, line, and sentence boundaries.
    - Produces **no overlap** between chunks (no duplicated content).
    - Merges tiny consecutive pieces to avoid orphan chunks.

    Each raw item is expected to have:
        text: str
        source_file: str
        page | slide | page_or_slide: int | str

    Args:
        raw_items: Output from any ingest extractor.
        target_size: Preferred maximum characters per chunk.
        max_size: Hard ceiling — chunks will never exceed this.
        min_size: Pieces smaller than this are merged with neighbours.

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

        text = text.strip()
        if not text:
            continue

        # Phase 1: keep small items intact.
        if len(text) <= max_size:
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{global_idx}",
                    text=text,
                    source_file=source_file,
                    page_or_slide=page_or_slide,
                )
            )
            global_idx += 1
            continue

        # Phase 2: split on semantic boundaries.
        pieces = _split_text_hierarchical(text, max_size)

        # Phase 3: merge tiny consecutive pieces.
        pieces = _merge_small_pieces(pieces, min_size, target_size)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{global_idx}",
                    text=piece,
                    source_file=source_file,
                    page_or_slide=page_or_slide,
                )
            )
            global_idx += 1

    return chunks
