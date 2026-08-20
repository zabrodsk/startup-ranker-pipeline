"""PR-B1: pure PASS/FAIL assertion logic for Sprint 3 flag staging verification.

Operates on the get_job_cost_by_stage shape {job_id, stages, totals}. Cost-
observable features only (w3, w9, w11, w13).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from web.verify_flag_staging import COST_OBSERVABLE_FEATURES, verify_feature


def _cost(stages, perplexity_requests=0, serper_requests=0, web_requests=None):
    totals = {
        "perplexity_search": {"requests": perplexity_requests},
        "serper_search": {"requests": serper_requests},
    }
    if web_requests is not None:
        totals["web_search"] = {"requests": web_requests}
    return {
        "job_id": "j",
        "stages": stages,
        "totals": totals,
    }


def test_cost_observable_feature_set():
    assert set(COST_OBSERVABLE_FEATURES) == {"w3", "w9", "w11", "w13"}


def test_w3_digest_stage_present_passes():
    feature = _cost([{"stage": "evidence_digest", "llm_calls": 1, "prompt_tokens": 800}])
    passed, _ = verify_feature("w3", feature)
    assert passed is True


def test_w3_no_digest_stage_fails():
    feature = _cost([{"stage": "answering", "llm_calls": 5}])
    passed, _ = verify_feature("w3", feature)
    assert passed is False


def test_w3_with_baseline_reports_token_drop():
    baseline = _cost([
        {"stage": "critique", "prompt_tokens": 500000},
        {"stage": "refinement", "prompt_tokens": 268000},
    ])
    feature = _cost([
        {"stage": "evidence_digest", "llm_calls": 1, "prompt_tokens": 800},
        {"stage": "critique", "prompt_tokens": 50000},
        {"stage": "refinement", "prompt_tokens": 30000},
    ])
    passed, evidence = verify_feature("w3", feature, baseline)
    assert passed is True
    assert "delta" in evidence


def test_w9_evaluation_calls_drop_passes():
    baseline = _cost([{"stage": "evaluation", "llm_calls": 14}])
    feature = _cost([{"stage": "evaluation", "llm_calls": 7}])
    passed, _ = verify_feature("w9", feature, baseline)
    assert passed is True


def test_w9_requires_baseline():
    passed, evidence = verify_feature("w9", _cost([{"stage": "evaluation", "llm_calls": 7}]))
    assert passed is False
    assert "baseline" in evidence.lower()


def test_w11_decomposition_calls_drop():
    baseline = _cost([{"stage": "decomposition", "llm_calls": 8}])
    feature = _cost([{"stage": "decomposition", "llm_calls": 0}])
    passed, _ = verify_feature("w11", feature, baseline)
    assert passed is True


def test_w13_perplexity_requests_drop():
    passed, _ = verify_feature("w13", _cost([], perplexity_requests=12), _cost([], perplexity_requests=40))
    assert passed is True


def test_w13_no_drop_fails():
    passed, _ = verify_feature("w13", _cost([], perplexity_requests=10), _cost([], perplexity_requests=10))
    assert passed is False


def test_w13_counts_serper_and_prefers_provider_aggregate():
    passed, evidence = verify_feature(
        "w13",
        _cost([], serper_requests=4, web_requests=4),
        _cost([], serper_requests=15, web_requests=15),
    )

    assert passed is True
    assert "web-search requests" in evidence


def test_unknown_feature_fails():
    passed, _ = verify_feature("w99", _cost([]))
    assert passed is False
