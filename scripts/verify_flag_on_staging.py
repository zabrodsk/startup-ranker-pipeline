#!/usr/bin/env python
"""Read-only PASS/FAIL check for a cost-observable Sprint 3 flag on staging.

Run under `railway run` (staging Supabase):

    railway run --service startup-ranker-web -- .venv/bin/python \\
        scripts/verify_flag_on_staging.py --feature w3 --job-id <feature_job>
    railway run --service startup-ranker-web -- .venv/bin/python \\
        scripts/verify_flag_on_staging.py --feature w9 --job-id <feature_job> --baseline-job-id <off_job>

Cost-observable features only: w3, w9, w11, w13. w7 (Specter identity reuse) and
dup_gate are verified from MCP logs / the reused badge — see
docs/handoffs/sprint4-validation-runbook.md. Read-only; exit code is non-zero
when the check fails (or 2 when cost data is missing).
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

import web.db as db  # noqa: E402
from web.verify_flag_staging import COST_OBSERVABLE_FEATURES, verify_feature  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True, choices=list(COST_OBSERVABLE_FEATURES))
    parser.add_argument("--job-id", required=True, help="Job run with the feature flag ON")
    parser.add_argument("--baseline-job-id", default=None, help="Flag-off baseline job (w9/w11/w13)")
    args = parser.parse_args()

    feature_cost = db.get_job_cost_by_stage(args.job_id)
    if feature_cost is None:
        print(f"ERROR: no cost data for job {args.job_id!r} (Supabase unconfigured or job missing).")
        return 2
    baseline_cost = db.get_job_cost_by_stage(args.baseline_job_id) if args.baseline_job_id else None
    if args.baseline_job_id and baseline_cost is None:
        print(f"ERROR: no cost data for baseline job {args.baseline_job_id!r}.")
        return 2

    passed, evidence = verify_feature(args.feature, feature_cost, baseline_cost)
    print(json.dumps({"feature": args.feature, "job_id": args.job_id,
                      "baseline_job_id": args.baseline_job_id, "passed": passed,
                      "evidence": evidence}, indent=2))
    print(f"[{args.feature}] {'PASS' if passed else 'FAIL'}: {evidence}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
