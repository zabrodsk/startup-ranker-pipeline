"""Extract text from PDF files with page numbers.

Uses PyMuPDF (fitz) with layout-aware extraction to preserve table
structure and column order. Falls back to pdfplumber if PyMuPDF is
unavailable.

When a page's plain-text extraction yields garbage — typical for PDFs
exported from Notion / macOS Quartz where text is rasterized as vector
paths instead of selectable glyphs — the page is rendered to an image
and transcribed via Gemini Vision (cheap, no extra infra).

Set ``PDF_OCR_ENABLED=false`` to disable the OCR fallback entirely.
"""

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

# Heuristic thresholds for detecting "vector text" pages that need OCR.
# A page is considered to need OCR if:
#   - the alphabetic character count is below MIN_ALPHA_CHARS, AND
#   - the page contains more than MIN_DRAWINGS vector drawings
#     (i.e. there is clearly visible content but no extractable text), OR
#   - the page contains at least one image and produced no text at all.
_MIN_ALPHA_CHARS = 60
_MIN_DRAWINGS_FOR_OCR = 80

# Gemini model used for OCR transcription. Override via env var.
_OCR_MODEL = os.getenv("PDF_OCR_MODEL", "gemini-2.5-flash")
_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "200"))


def extract_pdf(path: str | Path) -> list[dict]:
    """Extract text from a PDF, returning one item per page.

    Tries PyMuPDF (fitz) first; falls back to pdfplumber if unavailable.

    Returns:
        List of dicts with keys: text, page (1-indexed), source_file.
    """
    path = Path(path)
    try:
        return _extract_with_fitz(path)
    except ImportError:
        return _extract_with_pdfplumber(path)


def _extract_with_fitz(path: Path) -> list[dict]:
    import fitz  # PyMuPDF

    ocr_enabled = os.getenv("PDF_OCR_ENABLED", "true").lower() not in {"false", "0", "no"}
    results: list[dict] = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = _extract_page_text(page).strip()

            if ocr_enabled and _page_needs_ocr(page, text):
                _log.info(
                    "pdf_ingest: page %d of %s has unextractable text "
                    "(alpha=%d, drawings=%d, images=%d) — running Gemini OCR",
                    page_num, path.name,
                    _alpha_count(text), len(page.get_drawings()),
                    len(page.get_images()),
                )
                ocr_text = _ocr_page_with_gemini(page, page_num, path.name)
                if ocr_text and _alpha_count(ocr_text) > _alpha_count(text):
                    text = ocr_text.strip()

            if text:
                results.append({
                    "text": text,
                    "page": page_num,
                    "source_file": path.name,
                })
    return results


def _alpha_count(text: str) -> int:
    """Count alphabetic characters in *text*."""
    return sum(1 for ch in text if ch.isalpha())


def _page_needs_ocr(page, extracted_text: str) -> bool:
    """Heuristic: does this page contain content rendered as vector paths
    rather than as selectable text?

    Triggers when:
      * The extracted text has fewer than ``_MIN_ALPHA_CHARS`` alphabetic
        characters AND the page has more than ``_MIN_DRAWINGS_FOR_OCR``
        vector drawings (clearly visible content, but no real text), OR
      * The page produced zero text but contains at least one image.
    """
    alpha = _alpha_count(extracted_text)
    if alpha >= _MIN_ALPHA_CHARS:
        return False

    try:
        n_drawings = len(page.get_drawings())
    except Exception:
        n_drawings = 0
    if n_drawings > _MIN_DRAWINGS_FOR_OCR:
        return True

    if alpha == 0:
        try:
            n_images = len(page.get_images())
        except Exception:
            n_images = 0
        if n_images > 0:
            return True

    return False


def _ocr_page_with_gemini(page, page_num: int, source_file: str) -> str:
    """Render *page* to an image and ask Gemini Vision to transcribe it.

    Returns an empty string on any failure — callers should fall back to
    whatever plain extraction yielded.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _log.warning(
            "pdf_ingest: cannot OCR page %d of %s — GOOGLE_API_KEY not set",
            page_num, source_file,
        )
        return ""

    try:
        import base64

        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        pix = page.get_pixmap(dpi=_OCR_DPI)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        llm = ChatGoogleGenerativeAI(
            model=_OCR_MODEL,
            google_api_key=api_key,
            temperature=0.0,
            timeout=60,
            max_retries=2,
        )

        prompt = (
            "Transcribe ALL text from this PDF page exactly as it appears. "
            "Preserve structure: headings, bullet points, columns, tables, "
            "numbers, and labels. For tables, use a readable layout (rows on "
            "separate lines, columns separated by spaces or pipes). "
            "Output ONLY the transcribed text — no commentary, no markdown "
            "code fences, no introductory phrases."
        )
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
        ])
        response = llm.invoke([message])
        text = _coerce_response_text(response.content)
        return text.strip()
    except Exception as exc:
        _log.warning(
            "pdf_ingest: Gemini OCR failed for page %d of %s: %s",
            page_num, source_file, exc,
        )
        return ""


def _coerce_response_text(content) -> str:
    """Coerce a langchain message content (str or list of parts) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _extract_page_text(page) -> str:
    """Extract text from a single PyMuPDF page, preserving layout.

    Uses the "blocks" mode to get text blocks with bounding boxes, then
    reconstructs the reading order. For pages with table-like content
    (many small blocks arranged in columns), uses "words" mode to
    reconstruct rows by Y-coordinate grouping — this preserves the
    column alignment that plain get_text() destroys.
    """
    # Get blocks: each block is (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks", sort=True)  # sort=True = reading order
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]  # type 0 = text

    if not text_blocks:
        return ""

    # Detect if this page looks like a table/financial page:
    # heuristic — many short blocks spread across the page width.
    page_width = page.rect.width
    short_blocks = [b for b in text_blocks if (b[2] - b[0]) < page_width * 0.4]
    is_tabular = len(short_blocks) > len(text_blocks) * 0.5 and len(text_blocks) > 6

    if is_tabular:
        return _reconstruct_table_layout(page)
    else:
        # Non-tabular: join blocks with newlines, preserving paragraph breaks.
        lines: list[str] = []
        prev_y1 = None
        for b in text_blocks:
            y0 = b[1]
            text = b[4].strip()
            if not text:
                continue
            # Insert blank line for large vertical gaps (paragraph separation).
            if prev_y1 is not None and y0 - prev_y1 > 20:
                lines.append("")
            lines.append(text)
            prev_y1 = b[3]
        return "\n".join(lines)


def _reconstruct_table_layout(page) -> str:
    """Reconstruct tabular content by grouping words into rows by Y-coordinate."""
    # Get all words: (x0, y0, x1, y1, word, block_no, line_no, word_no)
    words = page.get_text("words", sort=True)
    if not words:
        return page.get_text().strip()

    # Group words into rows — words within Y_TOLERANCE of each other are on same row.
    Y_TOLERANCE = 4
    rows: list[list[tuple]] = []  # each row = list of (x0, word) tuples

    for word_info in words:
        x0, y0, x1, y1, word, *_ = word_info
        word = word.strip()
        if not word:
            continue
        # Find matching row by y0 proximity.
        placed = False
        for row in rows:
            row_y = row[0][0]  # first item stores (y0, x0, word)
            if abs(y0 - row_y) <= Y_TOLERANCE:
                row.append((y0, x0, word))
                placed = True
                break
        if not placed:
            rows.append([(y0, x0, word)])

    # Sort rows top-to-bottom, words left-to-right within each row.
    rows.sort(key=lambda r: r[0][0])
    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda w: w[1])  # sort by x0
        lines.append("  ".join(w[2] for w in row))

    return "\n".join(lines)


def _extract_with_pdfplumber(path: Path) -> list[dict]:
    import pdfplumber

    results: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = _extract_pdfplumber_page(page).strip()
            if text:
                results.append({
                    "text": text,
                    "page": page_num,
                    "source_file": path.name,
                })
    return results


def _extract_pdfplumber_page(page) -> str:
    """Extract text from a pdfplumber page, handling tables explicitly."""
    lines: list[str] = []

    # Extract tables first and convert to readable text.
    tables = page.extract_tables() or []
    for table in tables:
        # Render table as aligned text.
        table_lines = []
        for row in table:
            cells = [str(c or "").strip() for c in row]
            table_lines.append("  |  ".join(cells))
        lines.append("\n".join(table_lines))

    # Get remaining text outside table areas.
    remaining = (page.extract_text() or "").strip()
    if remaining:
        lines.append(remaining)

    return "\n\n".join(lines)
