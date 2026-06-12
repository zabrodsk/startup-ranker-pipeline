"""Tests for the display-only evidence-coverage metric in summary rows (S1).

HARD CONSTRAINT: scoring computation must be untouched — these tests pin
total_score/avg_pro/avg_contra to the same values with and without QA pairs.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.batch import build_summary_rows
from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company

THIN = "Insufficient information available."
RICH = "The company raised a 2M seed round in 2025 [chunk_3]."


def _result(qa_pairs, slug="mantic"):
    return {
        "slug": slug,
        "company": Company(name="Mantic"),
        "final_state": {
            "final_arguments": [
                Argument(content="Strong team", argument_type="pro", qa_indices=[], score=8),
                Argument(content="No revenue", argument_type="contra", qa_indices=[], score=5),
            ],
            "final_decision": "invest",
            "all_qa_pairs": qa_pairs,
        },
    }


def test_evidence_coverage_ratio_from_thin_and_answered_mix():
    qa_pairs = [
        {"question": "Q1", "answer": RICH},
        {"question": "Q2", "answer": THIN},
        {"question": "Q3", "answer": RICH},
        {"question": "Q4", "answer": ""},  # empty counts as thin
    ]
    row = build_summary_rows([_result(qa_pairs)])[0]
    assert row["evidence_coverage"] == 0.5
    assert row["answered_questions"] == 2
    assert row["total_questions"] == 4


def test_evidence_coverage_none_without_qa_pairs():
    row = build_summary_rows([_result([])])[0]
    assert row["evidence_coverage"] is None
    assert row["answered_questions"] == 0
    assert row["total_questions"] == 0


def test_summary_row_contains_coverage_keys():
    row = build_summary_rows([_result([{"question": "Q", "answer": RICH}])])[0]
    for key in ("evidence_coverage", "answered_questions", "total_questions"):
        assert key in row


def test_scoring_fields_unchanged_by_coverage_metric():
    # Identical arguments, different QA pairs -> identical scoring outputs.
    row_no_qa = build_summary_rows([_result([])])[0]
    row_with_qa = build_summary_rows(
        [_result([{"question": "Q", "answer": THIN} for _ in range(10)])]
    )[0]
    for key in ("total_score", "avg_pro", "avg_contra", "decision"):
        assert row_no_qa[key] == row_with_qa[key]
    assert row_no_qa["total_score"] == 3.0  # 8 - 5, pinned
    assert row_with_qa["evidence_coverage"] == 0.0


def test_non_dict_qa_pairs_are_ignored():
    row = build_summary_rows([_result(["garbage", {"question": "Q", "answer": RICH}])])[0]
    assert row["total_questions"] == 1
    assert row["evidence_coverage"] == 1.0
