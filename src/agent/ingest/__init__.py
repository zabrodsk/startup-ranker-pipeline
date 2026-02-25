"""File ingestion orchestrator for startup document folders."""

from pathlib import Path

from agent.ingest.chunking import chunk_texts
from agent.ingest.pdf_ingest import extract_pdf
from agent.ingest.pptx_ingest import extract_pptx
from agent.ingest.store import Chunk, EvidenceStore
from agent.ingest.tabular_ingest import extract_tabular

_EXTENSION_MAP = {
    ".pdf": extract_pdf,
    ".pptx": extract_pptx,
    ".csv": extract_tabular,
    ".xlsx": extract_tabular,
    ".xls": extract_tabular,
}

_TEXT_EXTENSIONS = {".txt", ".md"}


def _extract_text_file(path: Path) -> list[dict]:
    """Read a plain-text or markdown file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return []

    if not text:
        return []

    return [{
        "text": text,
        "page_or_slide": "N/A",
        "source_file": path.name,
    }]


def ingest_startup_folder(
    folder: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> EvidenceStore:
    """Walk a startup folder, extract and chunk all supported files.

    Args:
        folder: Path to a startup's document folder.
        chunk_size: Target characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        An EvidenceStore populated with chunks from all files.
    """
    folder = Path(folder)
    slug = folder.name

    raw_items: list[dict] = []

    for file_path in sorted(folder.iterdir()):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        suffix = file_path.suffix.lower()

        extractor = _EXTENSION_MAP.get(suffix)
        if extractor is not None:
            try:
                raw_items.extend(extractor(file_path))
            except Exception as exc:
                print(f"  Warning: failed to extract {file_path.name}: {exc}")
        elif suffix in _TEXT_EXTENSIONS:
            raw_items.extend(_extract_text_file(file_path))

    chunks = chunk_texts(raw_items, chunk_size=chunk_size, overlap=overlap)
    return EvidenceStore(startup_slug=slug, chunks=chunks)


__all__ = ["ingest_startup_folder", "EvidenceStore", "Chunk"]
