"""PR1: predict_targeted_skip + resolve_web_search_mode (the Targeted intensity).

Targeted is the broader, OPT-IN web-search predictor behind the per-run
Off/Targeted/Full knob. It skips the planner's private-metric route, re-adds the
internal_fit route, and additionally skips the ``team`` aspect band. The team arm
is an exact ``aspect == "team"`` match, mutually exclusive with market/product, so
"Targeted never drops a market or product question via the aspect arm" is a
theorem — pinned by the market/product tests below. Pure/local, confidentiality-
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
from agent.web_search.unanswerable import predict_targeted_skip, resolve_web_search_mode


# --- the team aspect arm --------------------------------------------------------


def test_team_aspect_skips_regardless_of_route():
    skip, reason = predict_targeted_skip(
        question="Who are the key members of the founding team?",
        aspect="team",
        company_name="Mantic",
    )
    assert skip is True
    assert "team aspect" in reason


def test_team_aspect_skips_even_when_tagged_a_topic_route():
    # The aspect arm fires independently of the planner route.
    skip, reason = predict_targeted_skip(
        question="What is the team's background?",
        route_tag="sector_market",
        aspect="team",
        company_name="Mantic",
    )
    assert skip is True
    assert "team aspect" in reason


# --- the theorem: market/product are never dropped by the aspect arm ------------


def test_market_aspect_not_skipped():
    skip, _ = predict_targeted_skip(
        question="What is the market size and TAM?",
        aspect="market",
        company_name="Mantic",
    )
    assert skip is False


def test_product_aspect_not_skipped():
    skip, _ = predict_targeted_skip(
        question="What are the product's core features and technology?",
        aspect="product",
        company_name="Mantic",
        is_root=True,
    )
    assert skip is False


def test_none_aspect_is_null_safe_not_skipped():
    # A benign company_specific question with no aspect must not skip.
    skip, _ = predict_targeted_skip(
        question="Has the company announced funding?",
        aspect=None,
        company_name="Mantic",
    )
    assert skip is False


# --- the route arm --------------------------------------------------------------


def test_internal_fit_route_skips():
    skip, reason = predict_targeted_skip(
        question="Does this match the fund's investment thesis and stage focus?",
        route_tag="internal_fit",
        aspect="general_company",
        company_name="Mantic",
    )
    assert skip is True
    assert "internal_fit route" in reason


def test_private_metric_route_skips():
    skip, reason = predict_targeted_skip(
        question="What is the company's current ARR and burn rate?",
        company_name="Mantic",
    )
    assert skip is True
    assert "skip_public_web route" in reason


def test_sector_market_route_not_skipped():
    skip, _ = predict_targeted_skip(
        question="What is the market size and TAM?",
        route_tag="sector_market",
        aspect="market",
        company_name="Mantic",
    )
    assert skip is False


def test_company_specific_route_not_skipped():
    skip, _ = predict_targeted_skip(
        question="Has the company announced funding?",
        route_tag="company_specific",
        aspect="general_company",
        company_name="Mantic",
    )
    assert skip is False


# --- the one residual risk cell (R2): product mistagged internal_fit ------------


def test_product_mistagged_internal_fit_is_skipped_pins_current_behavior():
    """A product question wrongly tagged internal_fit is skipped via the route arm.

    This is the single cell where Targeted can drop a product win (Design R2).
    The replay merge-gate measures it on the f20aa510 benchmark; if it is nonzero
    on market/product there, PR1 drops INTERNAL_FIT from TARGETED_SKIP_ROUTES.
    Until then, this pins the chosen behavior so the regression is visible.
    """
    skip, _ = predict_targeted_skip(
        question="What are the product's core features?",
        route_tag="internal_fit",
        aspect="product",
        company_name="Mantic",
    )
    assert skip is True


# --- confidentiality + determinism ----------------------------------------------


def test_confidentiality_signature_uses_only_public_fields():
    params = set(inspect.signature(predict_targeted_skip).parameters)
    allowed = {"question", "route_tag", "aspect", "company_name", "company_domain", "industry", "is_root"}
    assert params <= allowed
    forbidden = ("chunk", "answer", "memo", "about", "tagline", "specter", "doc")
    for name in params:
        assert not any(bad in name.lower() for bad in forbidden), name


def test_determinism():
    args = dict(question="Who is the founding team?", aspect="team", company_name="Mantic")
    assert predict_targeted_skip(**args) == predict_targeted_skip(**args)


# --- resolve_web_search_mode (precedence: explicit > env > full) ----------------


def test_resolve_mode_explicit_wins(monkeypatch):
    monkeypatch.setenv("RDI_WEB_SEARCH_MODE", "off")
    assert resolve_web_search_mode("targeted") == "targeted"
    assert resolve_web_search_mode("full") == "full"
    assert resolve_web_search_mode("off") == "off"


def test_resolve_mode_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("RDI_WEB_SEARCH_MODE", "targeted")
    assert resolve_web_search_mode(None) == "targeted"


def test_resolve_mode_defaults_full_when_unset(monkeypatch):
    monkeypatch.delenv("RDI_WEB_SEARCH_MODE", raising=False)
    assert resolve_web_search_mode(None) == "full"
    assert resolve_web_search_mode("") == "full"


def test_resolve_mode_invalid_explicit_falls_to_env(monkeypatch):
    monkeypatch.setenv("RDI_WEB_SEARCH_MODE", "targeted")
    un._WARNED_INVALID_WEB_SEARCH_VALUES.clear()
    assert resolve_web_search_mode("banana") == "targeted"


def test_resolve_mode_invalid_explicit_warns_once(monkeypatch, caplog):
    monkeypatch.delenv("RDI_WEB_SEARCH_MODE", raising=False)
    un._WARNED_INVALID_WEB_SEARCH_VALUES.clear()
    with caplog.at_level("WARNING"):
        assert resolve_web_search_mode("banana") == "full"
        resolve_web_search_mode("banana")
    warnings = [r for r in caplog.records if "web_search_mode" in r.getMessage()]
    assert len(warnings) == 1


def test_resolve_mode_invalid_env_warns_once(monkeypatch, caplog):
    monkeypatch.setenv("RDI_WEB_SEARCH_MODE", "banana")
    un._WARNED_INVALID_WEB_SEARCH_VALUES.clear()
    with caplog.at_level("WARNING"):
        assert resolve_web_search_mode(None) == "full"
        resolve_web_search_mode(None)
    warnings = [r for r in caplog.records if "RDI_WEB_SEARCH_MODE" in r.getMessage()]
    assert len(warnings) == 1
