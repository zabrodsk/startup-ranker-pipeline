from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path("scripts/run_leadgen_paired_acceptance.py")
SPEC = importlib.util.spec_from_file_location("run_leadgen_paired_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)


def _payload(*, proposed: list[int], verdicts: list[str]) -> dict[str, object]:
    return {
        "schema_version": "rdi-coverage-paired-acceptance-v1",
        "cases": [
            {
                "case_id": f"case-{index}",
                "baseline_search_calls": 12,
                "proposed_search_calls": calls,
                "quality_verdict": verdicts[index],
            }
            for index, calls in enumerate(proposed)
        ],
    }


def test_paired_acceptance_passes_search_and_blinded_quality_gates() -> None:
    report = paired.evaluate_paired_cases(
        _payload(
            proposed=[8, 7, 6, 8, 8],
            verdicts=["equal_or_better"] * 4 + ["worse"],
        )
    )

    assert report["overall_status"] == "passed"
    assert report["proposed_median_search_calls"] == 8.0
    assert report["equal_or_better_rate"] == 0.8


def test_pending_blinded_review_blocks_instead_of_passing() -> None:
    report = paired.evaluate_paired_cases(
        _payload(proposed=[7, 8], verdicts=["equal_or_better", "pending"])
    )

    assert report["overall_status"] == "blocked"
    assert report["pending_review_count"] == 1


def test_search_or_quality_regression_fails() -> None:
    report = paired.evaluate_paired_cases(
        _payload(proposed=[9, 9, 10], verdicts=["worse"] * 3)
    )

    assert report["overall_status"] == "failed"


def test_malformed_case_fails_safe() -> None:
    payload = _payload(proposed=[8], verdicts=["equal_or_better"])
    payload["cases"][0]["unknown"] = True

    with pytest.raises(ValueError):
        paired.evaluate_paired_cases(payload)
