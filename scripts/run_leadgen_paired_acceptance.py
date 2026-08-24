#!/usr/bin/env python3
"""Evaluate frozen paired RDI search-count and blinded-quality evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

PASSED = "passed"
FAILED = "failed"
BLOCKED = "blocked"
_EXPECTED_CASE_FIELDS = {
    "case_id",
    "baseline_search_calls",
    "proposed_search_calls",
    "quality_verdict",
}
_QUALITY_VERDICTS = {"equal_or_better", "worse", "pending"}


def evaluate_paired_cases(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen paired acceptance contract without provider calls."""
    if set(payload) != {"schema_version", "cases"}:
        raise ValueError("paired acceptance fields do not match the contract")
    if payload.get("schema_version") != "rdi-coverage-paired-acceptance-v1":
        raise ValueError("paired acceptance schema_version is invalid")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= 500:
        raise ValueError("paired acceptance cases must be a bounded non-empty array")
    cases = [_validated_case(item) for item in raw_cases]
    pending = sum(case["quality_verdict"] == "pending" for case in cases)
    equal_or_better = sum(
        case["quality_verdict"] == "equal_or_better" for case in cases
    )
    reviewed = len(cases) - pending
    proposed_median = float(median(case["proposed_search_calls"] for case in cases))
    baseline_median = float(median(case["baseline_search_calls"] for case in cases))
    quality_rate = None if not reviewed else equal_or_better / reviewed
    search_status = PASSED if proposed_median <= 8 else FAILED
    quality_status = (
        BLOCKED
        if pending or quality_rate is None
        else PASSED
        if quality_rate >= 0.8
        else FAILED
    )
    overall = (
        FAILED
        if FAILED in {search_status, quality_status}
        else BLOCKED
        if BLOCKED in {search_status, quality_status}
        else PASSED
    )
    return {
        "schema_version": "rdi-coverage-paired-acceptance-report-v1",
        "overall_status": overall,
        "case_count": len(cases),
        "baseline_median_search_calls": baseline_median,
        "proposed_median_search_calls": proposed_median,
        "reviewed_case_count": reviewed,
        "pending_review_count": pending,
        "equal_or_better_count": equal_or_better,
        "equal_or_better_rate": quality_rate,
        "gates": [
            {
                "name": "median_search_calls",
                "status": search_status,
                "required": "proposed median <= 8",
            },
            {
                "name": "blinded_evidence_quality",
                "status": quality_status,
                "required": "all cases reviewed and equal-or-better rate >= 0.8",
            },
        ],
    }


def _validated_case(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EXPECTED_CASE_FIELDS:
        raise ValueError("paired acceptance case fields do not match the contract")
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 128:
        raise ValueError("case_id must be bounded text")
    baseline = _count(value.get("baseline_search_calls"), "baseline_search_calls")
    proposed = _count(value.get("proposed_search_calls"), "proposed_search_calls")
    verdict = value.get("quality_verdict")
    if verdict not in _QUALITY_VERDICTS:
        raise ValueError("quality_verdict is invalid")
    return {
        "case_id": case_id,
        "baseline_search_calls": baseline,
        "proposed_search_calls": proposed,
        "quality_verdict": verdict,
    }


def _count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be an integer within 0..100")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the finite offline paired-acceptance command."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("paired acceptance input must be an object")
        report = evaluate_paired_cases(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sys.stdout.write(
            json.dumps(
                report,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        if report["overall_status"] == PASSED:
            return 0
        return 4 if report["overall_status"] == BLOCKED else 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        del exc
        sys.stderr.write("paired acceptance input failed safe validation\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
