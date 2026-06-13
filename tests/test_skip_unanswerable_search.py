"""PR-C1: predict_search_unanswerable — skip predictably-unanswerable searches.

The predictor delegates its skip judgment to build_web_search_plan, so it skips
exactly the planner's two skip routes (private internal metrics / VC-internal-fit)
and nothing else. Every other route and all ambiguity -> do-not-skip (status quo),
so it can never skip a question the planner would search. Pure/local, confidentiality-
bounded inputs only.
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import agent.web_search.unanswerable as un
from agent.web_search.unanswerable import predict_search_unanswerable, skip_unanswerable_mode


def test_mode_defaults_off(monkeypatch):
    monkeypatch.delenv("RDI_SKIP_UNANSWERABLE_SEARCH", raising=False)
    assert skip_unanswerable_mode() == "off"


def test_invalid_mode_warns_once_and_defaults_off(monkeypatch, caplog):
    monkeypatch.setenv("RDI_SKIP_UNANSWERABLE_SEARCH", "banana")
    un._WARNED_INVALID_SKIP_VALUES.clear()
    with caplog.at_level("WARNING"):
        assert skip_unanswerable_mode() == "off"
        skip_unanswerable_mode()
    warnings = [r for r in caplog.records if "RDI_SKIP_UNANSWERABLE_SEARCH" in r.getMessage()]
    assert len(warnings) == 1


def test_arr_question_predicts_skip():
    skip, reason = predict_search_unanswerable(question="What is the company's current ARR?", company_name="Mantic")
    assert skip is True
    assert reason.startswith("skip_public_web")


def test_burn_rate_question_predicts_skip():
    skip, _ = predict_search_unanswerable(question="What is their monthly burn rate and cash runway?", company_name="Mantic")
    assert skip is True


def test_vc_thesis_question_predicts_internal_fit_skip():
    skip, reason = predict_search_unanswerable(
        question="Does this match the fund's investment thesis and stage focus?",
        company_name="Mantic",
    )
    assert skip is True
    assert reason.startswith("internal_fit")


def test_general_company_root_predicts_skip():
    skip, reason = predict_search_unanswerable(
        question="Overall assessment of the company.",
        company_name="Mantic",
        aspect="general_company",
        is_root=True,
    )
    assert skip is True
    assert reason.startswith("internal_fit")


def test_market_size_question_not_skipped():
    skip, _ = predict_search_unanswerable(question="What is the market size and TAM?", company_name="Mantic")
    assert skip is False


def test_funding_question_not_skipped():
    skip, _ = predict_search_unanswerable(question="Has the company announced funding?", company_name="Mantic")
    assert skip is False


def test_competitor_question_not_skipped():
    skip, _ = predict_search_unanswerable(question="Who are the main competitors?", company_name="Mantic")
    assert skip is False


def test_route_tag_company_specific_overrides_metric_wording():
    """An explicit company_specific tag prevents a skip even on ARR-worded text."""
    skip, _ = predict_search_unanswerable(
        question="What is the company's ARR?",
        route_tag="company_specific",
        company_name="Mantic",
    )
    assert skip is False


def test_confidentiality_signature_uses_only_public_fields():
    params = set(inspect.signature(predict_search_unanswerable).parameters)
    allowed = {"question", "route_tag", "aspect", "company_name", "company_domain", "industry", "is_root"}
    assert params <= allowed
    forbidden = ("chunk", "answer", "memo", "about", "tagline", "specter", "doc")
    for name in params:
        assert not any(bad in name.lower() for bad in forbidden), name


def test_determinism():
    args = dict(question="What is the company's ARR?", company_name="Mantic")
    assert predict_search_unanswerable(**args) == predict_search_unanswerable(**args)
