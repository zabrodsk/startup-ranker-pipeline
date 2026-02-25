"""Convert CSV / XLSX tabular files into text summaries."""

from pathlib import Path

import pandas as pd


def extract_tabular(path: str | Path) -> list[dict]:
    """Read a CSV or XLSX file and convert each sheet to a text summary.

    Returns:
        List of dicts with keys: text, page_or_slide, source_file.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return _read_excel(path)
    else:
        return []


def _dataframe_to_text(df: pd.DataFrame, sheet_name: str = "") -> str:
    """Render a DataFrame as a readable text block."""
    lines: list[str] = []
    if sheet_name:
        lines.append(f"Sheet: {sheet_name}")

    shape_info = f"{len(df)} rows x {len(df.columns)} columns"
    lines.append(shape_info)
    lines.append("Columns: " + ", ".join(str(c) for c in df.columns))

    desc = df.describe(include="all").to_string()
    lines.append("Summary statistics:\n" + desc)

    preview_rows = min(20, len(df))
    lines.append(f"First {preview_rows} rows:\n" + df.head(preview_rows).to_string())

    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict]:
    try:
        df = pd.read_csv(str(path))
    except Exception:
        return []

    text = _dataframe_to_text(df)
    if not text.strip():
        return []

    return [{
        "text": text,
        "page_or_slide": "N/A",
        "source_file": path.name,
    }]


def _read_excel(path: Path) -> list[dict]:
    try:
        sheets = pd.read_excel(str(path), sheet_name=None)
    except Exception:
        return []

    results: list[dict] = []
    for sheet_name, df in sheets.items():
        text = _dataframe_to_text(df, sheet_name=str(sheet_name))
        if text.strip():
            results.append({
                "text": text,
                "page_or_slide": str(sheet_name),
                "source_file": path.name,
            })
    return results
