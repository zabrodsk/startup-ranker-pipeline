"""Tests for the replay harness's pure logic (no network, synthetic fixtures)."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from web.replay_web_planner import (
    build_plan_for_row,
    build_replay_report,
    extract_qa_rows,
    find_skip_route_false_positives,
    gate_confusion_matrix,
    is_rescued,
    merge_route_tags,
    question_hash,
    route_distribution,
    select_live_sample,
)

THIN = "Insufficient information available."
RICH = "The company raised a 2M seed round in 2025 led by a regional fund [web]."
SECTOR_TEXT = (
    "The European wildfire detection sensor market is growing rapidly with "
    "strong adoption by utilities and forest agencies, and analysts forecast "
    "demand in the hundreds of millions of euros across the region."
)


def _qa(question, answer=THIN, used=False, query="Mantic q", results=""):
    return {
        "question": question,
        "answer": answer,
        "aspect": "market",
        "chunks_preview": "[chunk_0]: deck text",
        "web_search_query": query,
        "web_search_results": results,
        "web_search_used": used,
        "web_search_decision": "used" if used else "skipped",
    }


def _payload(qa_pairs):
    return {
        "results": {
            "results": [
                {
                    "slug": "mantic",
                    "company": {
                        "name": "Mantic",
                        "industry": "wildfire detection sensors",
                        "domain": "mantic.ai",
                        "geo": "Europe",
                    },
                    "final_state": {"all_qa_pairs": qa_pairs},
                },
                {"slug": "skipped-co", "skipped": True},
            ]
        }
    }


def test_extract_qa_rows_flattens_payload_and_hashes_questions():
    rows = extract_qa_rows(_payload([_qa("Q1?"), _qa("Q2?")]))
    assert len(rows) == 2
    assert rows[0]["company_name"] == "Mantic"
    assert rows[0]["industry"] == "wildfire detection sensors"
    assert rows[0]["geo"] == "Europe"
    assert rows[0]["question_hash"] == question_hash("Q1?")


def test_extract_qa_rows_rejects_unknown_shape():
    with pytest.raises(ValueError):
        extract_qa_rows({"nope": True})


def test_question_hash_is_stable_and_normalized():
    assert question_hash("What is X?") == question_hash("  what is x?  ")
    assert question_hash("What is X?") != question_hash("What is Y?")


def test_is_rescued_requires_web_used_and_substantive_answer():
    assert is_rescued(_qa("Q?", answer=RICH, used=True)) is True
    assert is_rescued(_qa("Q?", answer=THIN, used=True)) is False
    assert is_rescued(_qa("Q?", answer=RICH, used=False)) is False


def test_build_plan_for_row_uses_tags_with_priority():
    rows = extract_qa_rows(_payload([_qa("Has the company announced funding?")]))
    row = rows[0]
    untagged = build_plan_for_row(row, {})
    assert untagged.route.value == "company_specific"
    tagged = build_plan_for_row(row, {row["question_hash"]: "sector_market"})
    assert tagged.route.value == "sector_market"
    assert tagged.rationale == "tag:sector_market"


def test_false_positive_checker_flags_rescued_skip_routes():
    rescued_arr = _qa(
        "What is the company's current ARR?", answer=RICH, used=True
    )  # keyword fallback would skip this — and it was actually rescued
    rescued_ok = _qa("Has the company announced funding?", answer=RICH, used=True)
    rows = extract_qa_rows(_payload([rescued_arr, rescued_ok]))

    offenders = find_skip_route_false_positives(rows, {})
    assert len(offenders) == 1
    assert "ARR" in offenders[0]["question"]
    assert offenders[0]["route"] == "skip_public_web"

    # A tag overriding the skip resolves the false positive.
    arr_hash = question_hash("What is the company's current ARR?")
    assert find_skip_route_false_positives(rows, {arr_hash: "company_specific"}) == []


def test_gate_confusion_matrix_counts_newly_accepted_sector_evidence():
    # Sector text without company mention: legacy gate rejects, planner accepts.
    sector_q = _qa(
        "Is the sector attractive given market size and growth?",
        used=False,
        results=SECTOR_TEXT,
    )
    rows = extract_qa_rows(_payload([sector_q]))
    tags = {rows[0]["question_hash"]: "sector_market"}
    matrix = gate_confusion_matrix(rows, tags)
    assert matrix["newly_accepted"] == 1
    assert matrix["both_accept"] == 0
    assert matrix["newly_accepted_samples"][0]["route"] == "sector_market"


def test_gate_confusion_matrix_skips_unsearched_rows():
    rows = extract_qa_rows(_payload([_qa("Q?", query="", results="")]))
    matrix = gate_confusion_matrix(rows, {})
    assert matrix["both_accept"] + matrix["both_reject"] == 0
    assert matrix["newly_accepted"] + matrix["newly_rejected"] == 0


def test_route_distribution_reports_sources_and_skips():
    rows = extract_qa_rows(
        _payload(
            [
                _qa("What is the company's current ARR?"),
                _qa("Has the company announced funding?"),
                _qa("Who are the main competitors?"),
            ]
        )
    )
    dist = route_distribution(rows, {})
    assert dist["total_questions"] == 3
    assert dist["by_route"]["skip_public_web"] == 1
    assert dist["by_route"]["company_specific"] == 1
    assert dist["by_route"]["competitors"] == 1
    assert dist["by_routing_source"] == {"keyword_fallback": 3}
    assert dist["skip_route_questions"] == 1


def test_build_replay_report_pass_and_fail():
    ok_rows = extract_qa_rows(
        _payload([_qa("Has the company announced funding?", answer=RICH, used=True)])
    )
    report = build_replay_report(ok_rows, {})
    assert report["false_positive_check"]["pass"] is True
    assert report["benchmark"]["rescued_answers"] == 1
    assert report["benchmark"]["baseline_rescue_rate"] == 1.0

    bad_rows = extract_qa_rows(
        _payload([_qa("What is the company's current ARR?", answer=RICH, used=True)])
    )
    bad_report = build_replay_report(bad_rows, {})
    assert bad_report["false_positive_check"]["pass"] is False
    assert bad_report["false_positive_check"]["rescued_routed_to_skip"] == 1


def test_merge_route_tags_is_idempotent_and_existing_wins():
    existing = {"h1": "sector_market"}
    new = {"h1": "competitors", "h2": "regulation"}
    merged = merge_route_tags(existing, new)
    assert merged == {"h1": "sector_market", "h2": "regulation"}
    assert merge_route_tags(merged, new) == merged


def test_select_live_sample_is_seeded_and_filters_routes():
    qa_pairs = [
        _qa(f"Is the market segment {i} attractive given market size?", results=SECTOR_TEXT)
        for i in range(20)
    ]
    qa_pairs.append(_qa("Has the company announced funding?"))  # company_specific: excluded
    rows = extract_qa_rows(_payload(qa_pairs))
    tags = {
        r["question_hash"]: "sector_market"
        for r in rows
        if "market segment" in r["question"]
    }

    sample_a = select_live_sample(rows, tags, sample_size=5, seed=7)
    sample_b = select_live_sample(rows, tags, sample_size=5, seed=7)
    assert [r["question"] for r in sample_a] == [r["question"] for r in sample_b]
    assert len(sample_a) == 5
    assert all("market segment" in r["question"] for r in sample_a)
