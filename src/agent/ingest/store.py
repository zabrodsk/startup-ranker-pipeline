"""In-memory evidence store for ingested document chunks."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk extracted from a startup document."""

    chunk_id: str
    text: str
    source_file: str
    page_or_slide: str | int


@dataclass
class EvidenceStore:
    """Holds all chunks for a single startup, with lookup helpers."""

    startup_slug: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None
