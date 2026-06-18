"""Hermetic unit tests for the run-evidence audit classifier (no network)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "audit_run_evidence", ROOT / "scripts" / "audit_run_evidence.py"
)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_no_web_call_is_specter_only():
    row = {"answer": "Founding team is strong [chunk_3].", "web_search_query": ""}
    assert audit.classify(row) == audit.SPECTER_ONLY


def test_web_cited_and_substantive_is_web_added():
    row = {
        "answer": "The market is ~$5B per [web] and growing [chunk_2].",
        "web_search_query": "market size",
        "web_search_used": True,
    }
    assert audit.classify(row) == audit.WEB_ADDED


def test_web_ran_but_answer_does_not_cite_web_is_no_value():
    row = {
        "answer": "Founders are experienced operators [chunk_1].",
        "web_search_query": "founder background",
        "web_search_used": True,
    }
    assert audit.classify(row) == audit.WEB_NO_VALUE


def test_summary_counts_each_bucket():
    rows = [
        {"answer": "X [chunk_1].", "web_search_query": "", "aspect": "team"},
        {"answer": "Y [web].", "web_search_query": "q", "aspect": "market"},
        {"answer": "Z [chunk_2].", "web_search_query": "q", "aspect": "market"},
    ]
    summary = audit.build_summary(rows)
    assert "Questions analysed:              3" in summary
    assert "Answered from Specter/docs only: 1" in summary
    assert "Web call ran:                    2" in summary
