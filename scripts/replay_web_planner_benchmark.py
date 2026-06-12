#!/usr/bin/env python
"""CLI for the WebEvidencePlanner replay harness (benchmark job f20aa510).

Run under `railway run` (staging Supabase + provider keys):

    uv run python scripts/replay_web_planner_benchmark.py --job-id f20aa510 --stage tag
    uv run python scripts/replay_web_planner_benchmark.py --job-id f20aa510 --stage replay
    uv run python scripts/replay_web_planner_benchmark.py --job-id f20aa510 --stage live --sample-size 100
    uv run python scripts/replay_web_planner_benchmark.py --job-id f20aa510 --stage report

Stages: tag (one-off LLM tagging) -> replay (free, hard 0-false-positive
check) -> live (sampled re-search + re-answer) -> report (gate-review JSON).
Exit code is non-zero when the false-positive check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from web.replay_web_planner import (  # noqa: E402
    DEFAULT_JOB_ID,
    DEFAULT_LIVE_SAMPLE_SIZE,
    DEFAULT_OUT_DIR,
    TOPIC_ROUTES_FOR_LIVE,
    build_replay_report,
    find_skip_route_false_positives,
    load_benchmark_rows,
    load_route_tags,
    run_live_stage,
    run_tag_stage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--stage", required=True, choices=["tag", "replay", "live", "report"])
    parser.add_argument("--sample-size", type=int, default=DEFAULT_LIVE_SAMPLE_SIZE)
    parser.add_argument(
        "--routes", nargs="*", default=list(TOPIC_ROUTES_FOR_LIVE),
        help="Routes eligible for the live stage sample",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out / args.job_id
    rows = load_benchmark_rows(args.job_id)
    print(f"Loaded {len(rows)} QA rows for job {args.job_id}")

    if args.stage == "tag":
        tags = run_tag_stage(rows, out_dir)
        print(f"[tag] total tagged questions: {len(tags)}")
        return 0

    route_tags = load_route_tags(out_dir)
    if not route_tags:
        print("WARNING: no route_tags.json found — run --stage tag first; "
              "falling back to keyword routing only.")

    if args.stage == "live":
        summary = run_live_stage(
            rows, route_tags, out_dir,
            sample_size=args.sample_size,
            routes=tuple(args.routes),
            seed=args.seed,
        )
        (out_dir / "live_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    # replay + report both run the hard false-positive check.
    live_results = None
    live_summary_path = out_dir / "live_summary.json"
    if args.stage == "report" and live_summary_path.exists():
        live_results = json.loads(live_summary_path.read_text())

    report = build_replay_report(rows, route_tags, live_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "replay_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "route_distribution"}, indent=2)[:4000])
    print(f"Wrote {report_path}")

    offenders = find_skip_route_false_positives(rows, route_tags)
    if offenders:
        print(f"FAIL: {len(offenders)} rescued answers routed to a skip route "
              f"(hard gate is 0). See replay_report.json false_positive_check.")
        return 1
    print("PASS: 0 rescued answers routed to skip routes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
