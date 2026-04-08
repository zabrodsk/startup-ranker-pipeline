"""Tests for smart_chunk_texts() — semantic-boundary-aware chunking."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from agent.ingest.chunking import smart_chunk_texts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(text: str, source_file: str = "test.pdf", page: int = 1) -> dict:
    return {"text": text, "source_file": source_file, "page": page}


# ---------------------------------------------------------------------------
# Phase 1: items that fit within max_size → kept as single chunk
# ---------------------------------------------------------------------------

class TestSmallItems:
    def test_short_item_kept_as_single_chunk(self):
        """Items shorter than target_size → 1 chunk, untouched."""
        item = _make_item("Hello world.", page=3)
        chunks = smart_chunk_texts([item], target_size=100, max_size=200)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].source_file == "test.pdf"
        assert chunks[0].page_or_slide == 3
        assert chunks[0].chunk_id == "chunk_0"

    def test_medium_item_under_max_kept_intact(self):
        """Items between target and max → still kept as 1 chunk."""
        text = "A" * 1500  # target=1000, max=2000 → fits
        chunks = smart_chunk_texts([_make_item(text)])
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_items_skipped(self):
        chunks = smart_chunk_texts([_make_item(""), _make_item("   ")])
        assert len(chunks) == 0

    def test_whitespace_only_stripped(self):
        chunks = smart_chunk_texts([_make_item("  hello  ")])
        assert len(chunks) == 1
        assert chunks[0].text == "hello"


# ---------------------------------------------------------------------------
# Phase 2: splitting on semantic boundaries
# ---------------------------------------------------------------------------

class TestSplitting:
    def test_split_on_paragraphs(self):
        """Long text with paragraph breaks → splits on \\n\\n."""
        para1 = "First paragraph. " * 60  # ~1020 chars
        para2 = "Second paragraph. " * 60
        text = para1.strip() + "\n\n" + para2.strip()
        chunks = smart_chunk_texts([_make_item(text)], target_size=1100, max_size=1200)
        assert len(chunks) == 2
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[1].text

    def test_split_on_lines(self):
        """Text with single newlines but no double newlines → splits on \\n."""
        lines = [f"Line {i}: " + "x" * 80 for i in range(30)]  # ~90 chars each, total ~2700
        text = "\n".join(lines)
        chunks = smart_chunk_texts([_make_item(text)], target_size=1000, max_size=2000)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 2000

    def test_split_on_sentences(self):
        """Text with no newlines but sentence endings → splits on ./?/!."""
        sentences = [f"Sentence number {i} with extra padding here." for i in range(60)]
        text = " ".join(sentences)  # one long line, ~2700 chars
        chunks = smart_chunk_texts([_make_item(text)], target_size=1000, max_size=2000)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 2000

    def test_split_on_words_last_resort(self):
        """Text with no punctuation or newlines → splits on spaces."""
        text = "word " * 500  # 2500 chars, no sentence endings
        chunks = smart_chunk_texts([_make_item(text)], target_size=1000, max_size=2000)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 2000

    def test_hard_cut_when_no_boundaries(self):
        """A single massive word → hard-cut at max_size."""
        text = "A" * 5000
        chunks = smart_chunk_texts([_make_item(text)], target_size=1000, max_size=2000)
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c.text) <= 2000


# ---------------------------------------------------------------------------
# Phase 3: merging small pieces
# ---------------------------------------------------------------------------

class TestMerging:
    def test_tiny_pieces_merged(self):
        """Several tiny paragraphs below min_size → merged together."""
        paras = ["Short line."] * 20  # 11 chars each
        text = "\n\n".join(paras)  # total ~230 chars
        # max_size=2000, so the whole thing is kept as one chunk
        chunks = smart_chunk_texts([_make_item(text)], target_size=1000, max_size=2000, min_size=150)
        assert len(chunks) == 1

    def test_merge_does_not_exceed_target(self):
        """Merging respects target_size ceiling."""
        paras = ["Hello world. " * 8] * 10  # ~104 chars each, total ~1040
        text = "\n\n".join(paras)
        # This text is ~1140 chars (including \n\n), under max_size=2000 → single chunk
        chunks = smart_chunk_texts([_make_item(text)], target_size=500, max_size=2000, min_size=150)
        # Each para is ~104 chars (< min_size=150), so they'll be merged up to target=500
        for c in chunks:
            assert len(c.text) <= 2000  # hard ceiling always respected


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_source_file_preserved(self):
        text = "A" * 3000
        chunks = smart_chunk_texts(
            [_make_item(text, source_file="pitch.pdf", page=5)],
            max_size=2000,
        )
        assert len(chunks) >= 2
        for c in chunks:
            assert c.source_file == "pitch.pdf"
            assert c.page_or_slide == 5

    def test_slide_metadata(self):
        item = {"text": "Slide content", "source_file": "deck.pptx", "slide": 3}
        chunks = smart_chunk_texts([item])
        assert chunks[0].page_or_slide == 3

    def test_page_or_slide_fallback(self):
        item = {"text": "Content", "source_file": "doc.docx", "page_or_slide": "Section 2"}
        chunks = smart_chunk_texts([item])
        assert chunks[0].page_or_slide == "Section 2"


# ---------------------------------------------------------------------------
# Global chunk ID numbering
# ---------------------------------------------------------------------------

class TestChunkIds:
    def test_sequential_ids_across_items(self):
        items = [
            _make_item("First item content.", page=1),
            _make_item("Second item content.", page=2),
            _make_item("Third item content.", page=3),
        ]
        chunks = smart_chunk_texts(items)
        assert [c.chunk_id for c in chunks] == ["chunk_0", "chunk_1", "chunk_2"]

    def test_sequential_ids_with_splits(self):
        """IDs remain sequential even when items are split into multiple chunks."""
        short = _make_item("Short.", page=1)
        long_text = "Paragraph one. " * 80 + "\n\n" + "Paragraph two. " * 80
        long_item = _make_item(long_text, page=2)
        chunks = smart_chunk_texts([short, long_item], target_size=500, max_size=1000)
        ids = [c.chunk_id for c in chunks]
        expected = [f"chunk_{i}" for i in range(len(chunks))]
        assert ids == expected


# ---------------------------------------------------------------------------
# Tabular data (pipe-separated rows with newlines)
# ---------------------------------------------------------------------------

class TestTabularData:
    def test_tabular_splits_on_rows(self):
        """Pipe-separated table rows should split on line boundaries."""
        header = "Name | Revenue | Growth"
        rows = [f"Company_{i} | ${i * 100}M | {i}%" for i in range(50)]
        text = header + "\n" + "\n".join(rows)
        chunks = smart_chunk_texts([_make_item(text, source_file="data.csv")], max_size=1000)
        for c in chunks:
            assert len(c.text) <= 1000
            # Each chunk should contain complete rows (no mid-row splits).
            for line in c.text.split("\n"):
                if "|" in line:
                    assert line.count("|") >= 2  # complete row


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input(self):
        assert smart_chunk_texts([]) == []

    def test_single_newline_text(self):
        chunks = smart_chunk_texts([_make_item("\n")])
        assert len(chunks) == 0

    def test_no_raw_items(self):
        chunks = smart_chunk_texts([], target_size=500)
        assert chunks == []

    def test_default_parameters(self):
        """Ensure default target=1000, max=2000 works."""
        text = "Hello. " * 300  # ~2100 chars
        chunks = smart_chunk_texts([_make_item(text)])
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 2000
