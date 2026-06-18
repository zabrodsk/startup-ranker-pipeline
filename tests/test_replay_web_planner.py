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
    predict_skip_for_row,
    predict_targeted_skip_for_row,
    question_hash,
    route_distribution,
    select_live_sample,
    skip_prediction_stats,
    skip_predictor_false_positives,
    synthesize_is_root,
    targeted_skip_false_positives_by_aspect,
    targeted_skip_stats,
    web_added,
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


# --- Sprint 4 C: skip-unanswerable predictor over the benchmark -----------------


def test_predict_skip_for_row_skips_arr_not_funding():
    rows = extract_qa_rows(_payload([
        _qa("What is the company's ARR?"),
        _qa("Has the company announced funding?"),
    ]))
    skip_arr, _ = predict_skip_for_row(rows[0])
    skip_funding, _ = predict_skip_for_row(rows[1])
    assert skip_arr is True
    assert skip_funding is False


def test_skip_predictor_false_positives_flags_rescued_skip_route():
    # A rescued ARR answer would be wrongly suppressed -> flagged by the hard gate.
    rows = extract_qa_rows(_payload([_qa("What is the company's ARR?", answer=RICH, used=True)]))
    offenders = skip_predictor_false_positives(rows)
    assert len(offenders) == 1
    assert offenders[0]["company_name"] == "Mantic"


def test_skip_predictor_false_positive_resolved_by_company_specific_tag():
    rows = extract_qa_rows(_payload([_qa("What is the company's ARR?", answer=RICH, used=True)]))
    tags = {rows[0]["question_hash"]: "company_specific"}
    assert skip_predictor_false_positives(rows, tags) == []


def test_skip_predictor_equivalent_to_planner_skip_routes():
    rows = extract_qa_rows(_payload([
        _qa("What is the company's ARR?", answer=RICH, used=True),
        _qa("Has the company announced funding?", answer=RICH, used=True),
        _qa("What is the market size?", answer=RICH, used=True),
    ]))
    predictor = {(o["company_name"], o["question"]) for o in skip_predictor_false_positives(rows)}
    planner = {(o["company_name"], o["question"]) for o in find_skip_route_false_positives(rows)}
    assert predictor == planner


def test_skip_prediction_stats_reports_rates_and_pass():
    rows = extract_qa_rows(_payload([
        _qa("What is the company's ARR?", answer=THIN, used=True),  # skippable, not rescued
        _qa("What is the market size?", answer=THIN, used=True),     # answerable
        _qa("Has the company announced funding?", answer=RICH, used=True),  # rescued, answerable
    ]))
    stats = skip_prediction_stats(rows)
    assert stats["total_rows"] == 3
    assert stats["rescued_predicted_skip"] == 0
    assert stats["pass"] is True
    assert 0.0 < stats["skip_rate_documents_incomplete"] <= 1.0


# --- persisted payload shape (real load_job_results output) ---------------------


def _persisted_payload(qa_provenance_rows):
    """The worker-composed shape load_job_results actually returns: flat
    qa_provenance_rows (build_qa_provenance_rows), NOT a company list with
    final_state. industry/domain/geo are not persisted at the row level."""
    return {
        "results": {
            "summary_rows": [{"company_name": "Mantic", "startup_slug": "mantic"}],
            "qa_provenance_rows": qa_provenance_rows,
            "job_status": "done",
        }
    }


def _prov(question, answer=THIN, used=False):
    return {
        "company_name": "Mantic",
        "startup_slug": "mantic",
        "aspect": "market",
        "question": question,
        "answer": answer,
        "chunks_preview": "[chunk_0]: deck text",
        "web_search_query": "Mantic q",
        "web_search_results": "",
        "web_search_used": used,
        "web_search_decision": "used" if used else "skipped",
    }


def test_extract_qa_rows_reads_persisted_qa_provenance_shape():
    payload = _persisted_payload([
        _prov("What is the company's ARR?", used=True),
        _prov("Has the company announced funding?", answer=RICH, used=True),
    ])
    rows = extract_qa_rows(payload)
    assert len(rows) == 2
    assert rows[0]["company_name"] == "Mantic"
    assert rows[0]["question"] == "What is the company's ARR?"
    assert rows[0]["web_search_used"] is True
    assert rows[1]["answer"] == RICH
    # Downstream consumers work on these rows (this is what the benchmark needs).
    assert predict_skip_for_row(rows[0])[0] is True  # ARR -> skip route
    assert is_rescued(rows[1]) is True


def test_extract_qa_rows_still_reads_in_memory_company_list():
    """Backward-compat: the synthetic in-memory shape must still parse."""
    rows = extract_qa_rows(_payload([_qa("Q1?"), _qa("Q2?")]))
    assert len(rows) == 2
    assert rows[0]["company_name"] == "Mantic"


# --- PR1: shared web_added + synthesize_is_root + Targeted skip set --------------


def _qa_aspect(question, aspect, answer=THIN, used=False, query="Mantic q", results=""):
    row = _qa(question, answer=answer, used=used, query=query, results=results)
    row["aspect"] = aspect
    return row


def test_web_added_is_the_shared_real_win_definition():
    # Searched + [web] cite + substantive -> a real win.
    assert web_added(_qa("Q?", answer=RICH, used=True)) is True
    # Searched but thin -> not a win.
    assert web_added(_qa("Q?", answer=THIN, used=True)) is False
    # Searched but no [web] cite -> not a win.
    assert web_added({"answer": "Founders are strong [chunk_1].", "web_search_query": "q"}) is False
    # Not searched -> not a win even if the text mentions [web].
    assert web_added({"answer": "mentions [web]", "web_search_query": ""}) is False


def test_synthesize_is_root_matches_canonical_questions():
    from agent.pipeline.stages.constants import INVESTMENT_QUESTIONS

    team_q = INVESTMENT_QUESTIONS["team"]
    assert synthesize_is_root({"question": team_q}) is True
    assert synthesize_is_root({"question": f"  {team_q.upper()}  "}) is True  # normalized
    assert synthesize_is_root({"question": "Who are the founders?"}) is False
    assert synthesize_is_root({"question": ""}) is False


def test_predict_targeted_skip_for_row_skips_team_keeps_market():
    rows = extract_qa_rows(_payload([
        _qa_aspect("Who are the key founders and their track record?", "team", answer=RICH, used=True),
        _qa_aspect("What is the market size and TAM?", "market", answer=RICH, used=True),
    ]))
    skip_team, reason_team = predict_targeted_skip_for_row(rows[0])
    skip_market, _ = predict_targeted_skip_for_row(rows[1])
    assert skip_team is True
    assert "team" in reason_team
    assert skip_market is False


def test_targeted_false_positives_market_product_empty_on_rescued_set():
    rows = extract_qa_rows(_payload([
        _qa_aspect("Who are the founders?", "team", answer=RICH, used=True),
        _qa_aspect("What is the market size and TAM?", "market", answer=RICH, used=True),
        _qa_aspect("What are the product's core features?", "product", answer=RICH, used=True),
    ]))
    by_aspect = targeted_skip_false_positives_by_aspect([r for r in rows if is_rescued(r)])
    assert by_aspect.get("market", []) == []      # the theorem
    assert by_aspect.get("product", []) == []     # the theorem
    assert len(by_aspect.get("team", [])) == 1    # allowed, intended drop


def test_targeted_skip_stats_shape_and_band_scoped_pass():
    rows = extract_qa_rows(_payload([
        _qa_aspect("Who are the founders?", "team", answer=THIN, used=True),            # team -> suppressed
        _qa_aspect("What is the market size and TAM?", "market", answer=RICH, used=True),  # market rescue, kept
        _qa_aspect("What is the company's ARR?", "general_company", answer=THIN, used=True),  # private metric
    ]))
    stats = targeted_skip_stats(rows)
    assert stats["total_rows"] == 3
    assert stats["pass"] is True                          # no market/product false-skips
    assert stats["false_skips"]["market"] == 0
    assert stats["false_skips"]["product"] == 0
    assert stats["saving"]["searches_suppressed_total"] >= 2   # team + ARR
    # team arm buys at least one extra suppression over the shipped predictor.
    assert stats["saving"]["delta_vs_current_skip_set"] >= 1
    assert stats["tag_coverage"]["total"] == 3
    assert "web_added_denominator" in stats["false_skips"]


def test_targeted_skip_stats_fails_when_market_rescue_would_be_dropped():
    # Force the residual-risk cell: a rescued market answer mistagged internal_fit.
    rows = extract_qa_rows(_payload([
        _qa_aspect("What is the market size and TAM?", "market", answer=RICH, used=True),
    ]))
    tags = {rows[0]["question_hash"]: "internal_fit"}
    stats = targeted_skip_stats(rows, tags)
    assert stats["false_skips"]["market"] == 1
    assert stats["pass"] is False
