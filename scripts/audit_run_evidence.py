#!/usr/bin/env python
"""Audit one analysis run: where did documents/Specter suffice, and where did
Perplexity web search actually add content to the answer?

For each decomposed question of a single job it reports:
  - what the documents/Specter had (chunks_preview),
  - whether a web search ran (web_search_query / web_search_used / decision),
  - whether the final answer cited web ([web]) or only documents ([chunk_XX]),
and buckets every question so you can see how many web calls earned their cost.

It reads the persisted qa_provenance for the job, so it must run where staging
Supabase credentials are available:

    railway run -- .venv/bin/python scripts/audit_run_evidence.py --job-id <JOB_ID>

A compact summary prints to the screen; the full per-question detail is written
to exports/audit_<job-id>.txt (exports/ is gitignored). Evidence text is shown
in full only for a PUBLIC company (e.g. Anthropic); pass --redact for a private
deal to suppress the chunk/web evidence text.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from web.replay_web_planner import (  # noqa: E402
    WEB_CITE,
    load_benchmark_rows,
    web_added,
)

CHUNK_CITE = re.compile(r"\[chunk[_-]?\d+\]", re.IGNORECASE)

SPECTER_ONLY = "specter_only"
WEB_ADDED = "web_added"
WEB_NO_VALUE = "web_no_value"


def classify(row: dict) -> str:
    """Bucket one question by how web search contributed to the final answer.

    specter_only  — no web call ran; documents/Specter answered it.
    web_added     — a web call ran AND the answer cites [web] (web made it in).
    web_no_value  — a web call ran but the answer doesn't use it (or is thin).

    The web_added test is the shared ``web_added`` from web.replay_web_planner,
    so this audit and the replay merge-gate score "real web wins" identically.
    """
    if not bool(row.get("web_search_query")):
        return SPECTER_ONLY
    return WEB_ADDED if web_added(row) else WEB_NO_VALUE


def _snippet(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("…" if len(text) > n else "")


def _format_detail(row: dict, bucket: str, *, redact: bool) -> str:
    answer = row.get("answer", "") or ""
    lines = [
        "=" * 88,
        f"[{bucket}] aspect={row.get('aspect')}",
        f"Q: {row.get('question')}",
    ]
    if not redact:
        lines += [
            "",
            "What documents/Specter had (chunks_preview):",
            _snippet(row.get("chunks_preview", ""), 600) or "(none)",
            "",
            f"Web query: {row.get('web_search_query') or '(none)'}",
            "Web results:",
            _snippet(row.get("web_search_results", ""), 600) or "(none)",
        ]
    lines += [
        "",
        "Final answer:",
        _snippet(answer, 900),
        "",
        f"cites: chunk={bool(CHUNK_CITE.search(answer))} web={bool(WEB_CITE.search(answer))} "
        f"| web_search_used={bool(row.get('web_search_used'))} "
        f"| decision={row.get('web_search_decision')!r}",
        "",
    ]
    return "\n".join(lines)


def build_summary(rows: list[dict]) -> str:
    buckets = Counter(classify(r) for r in rows)
    total = len(rows)
    searched = buckets[WEB_ADDED] + buckets[WEB_NO_VALUE]
    by_aspect: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_aspect[str(row.get("aspect") or "?")][classify(row)] += 1

    out = [
        "=== Evidence audit ===",
        f"Questions analysed:              {total}",
        f"Answered from Specter/docs only: {buckets[SPECTER_ONLY]}  (no web call)",
        f"Web call ran:                    {searched}",
        f"  -> web ADDED content ([web]):  {buckets[WEB_ADDED]}",
        f"  -> web added NO value:         {buckets[WEB_NO_VALUE]}",
    ]
    if searched:
        pct = round(100 * buckets[WEB_NO_VALUE] / searched)
        out.append(f"Wasted web calls:                {buckets[WEB_NO_VALUE]}/{searched} ({pct}%)")
    out.append("")
    out.append("By aspect/topic  (specter_only / web_added / web_no_value):")
    for aspect in sorted(by_aspect):
        c = by_aspect[aspect]
        out.append(f"  {aspect:<28} {c[SPECTER_ONLY]:>3} / {c[WEB_ADDED]:>3} / {c[WEB_NO_VALUE]:>3}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--company", default=None, help="Filter to one company name (optional)")
    ap.add_argument("--redact", action="store_true", help="Hide chunk/web evidence text (private deals)")
    ap.add_argument("--out", type=Path, default=ROOT / "exports")
    args = ap.parse_args()

    rows = load_benchmark_rows(args.job_id)
    if args.company:
        want = args.company.strip().lower()
        rows = [r for r in rows if str(r.get("company_name", "")).strip().lower() == want]
    if not rows:
        print(
            f"No QA rows for job {args.job_id!r}"
            + (f" / company {args.company!r}" if args.company else "")
        )
        return 1

    summary = build_summary(rows)
    print(summary)

    # Full per-question detail, web_added first so the value cases are easy to read.
    order = {WEB_ADDED: 0, WEB_NO_VALUE: 1, SPECTER_ONLY: 2}
    detailed = sorted(rows, key=lambda r: order[classify(r)])
    body = "\n".join(_format_detail(r, classify(r), redact=args.redact) for r in detailed)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"audit_{args.job_id}.txt"
    out_path.write_text(summary + "\n\n" + body)
    print(f"\nFull per-question detail written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
