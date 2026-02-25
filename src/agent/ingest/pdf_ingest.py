"""Extract text from PDF files with page numbers."""

from pathlib import Path


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

    results: list[dict] = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                results.append({
                    "text": text,
                    "page": page_num,
                    "source_file": path.name,
                })
    return results


def _extract_with_pdfplumber(path: Path) -> list[dict]:
    import pdfplumber

    results: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                results.append({
                    "text": text,
                    "page": page_num,
                    "source_file": path.name,
                })
    return results
