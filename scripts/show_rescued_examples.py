#!/usr/bin/env python
"""Surface the strongest WebEvidencePlanner rescues for manual quality review.

Reads exports/web_planner_replay/<job>/stage2_answers.csv (written by the live
stage) and prints the rescued rows ranked by how much the web actually added:
rows where the document answer was thin but the web answer is substantive, with
the most web evidence accepted, come first. Company names are tokenised
(Company A, B, ...) so deal identity stays local.

LOCAL-ONLY: run this in your own terminal. It reads confidential deal data
(company names + document-derived answers). The tokenised side-by-sides are safe
to skim and to paste back for a second opinion; the underlying CSV is not.

    .venv/bin/python scripts/show_rescued_examples.py --job-id f20aa510 --top 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from web.replay_web_planner import DEFAULT_OUT_DIR, is_thin_answer  # noqa: E402

OLD_PREVIEW_CHARS = 400


def anonymise(rows: list[dict[str, str]]) -> dict[str, str]:
    """Map each distinct company_name to a stable 'Company A/B/...' token."""
    tokens: dict[str, str] = {}
    for row in rows:
        name = row.get("company_name", "")
        if name not in tokens:
            n = len(tokens)
            tokens[name] = f"Company {chr(ord('A') + n)}" if n < 26 else f"Company #{n + 1}"
    return tokens


def improvement_key(row: dict[str, str]) -> tuple[int, int, int]:
    """Rank by: document answer was thin, then most web evidence, then richest answer."""
    old_was_thin = is_thin_answer(row.get("old_answer", ""))
    accepted = int(row.get("results_accepted") or 0)
    return (int(old_was_thin), accepted, len(row.get("new_answer", "")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default="f20aa510")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--full-old", action="store_true", help="Show the full document answer, not a preview")
    args = parser.parse_args()

    csv_path = args.out / args.job_id / "stage2_answers.csv"
    if not csv_path.exists():
        print(f"Not found: {csv_path} — run the live stage first.")
        return 1

    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    rescued = [r for r in rows if str(r.get("rescued", "")).strip().lower() == "true"]
    if not rescued:
        print("No rescued rows in this run.")
        return 0

    tokens = anonymise(rows)
    rescued.sort(key=improvement_key, reverse=True)
    shown = rescued[: args.top]
    print(f"{len(rescued)} rescued rows; showing top {len(shown)} by web evidence added.\n")

    for i, row in enumerate(shown, start=1):
        old = row.get("old_answer", "")
        old_shown = old if args.full_old else old[:OLD_PREVIEW_CHARS] + ("…" if len(old) > OLD_PREVIEW_CHARS else "")
        print("=" * 88)
        print(
            f"[{i}] {tokens[row.get('company_name', '')]} · route={row.get('route')} · "
            f"web_blocks_used={row.get('results_accepted')} · doc_answer_was_thin={is_thin_answer(old)}"
        )
        print(f"\nQ: {row.get('question')}")
        print(f"\nOLD (from the documents):\n{old_shown or '(empty)'}")
        print(f"\nNEW (web-rescued):\n{row.get('new_answer')}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
