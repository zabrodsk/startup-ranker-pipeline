"""Integration tests for RDI_WEB_EVIDENCE_PLANNER=off|shadow|on in evidence answering.

Hermetic: LLM and web-search provider are monkeypatched; no network.
"""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import evidence_answering as ea
from agent.dataclasses.company import Company
from agent.ingest.store import Chunk

THIN_ANSWER = "Insufficient information available."
HYBRID_ANSWER = "Hybrid answer citing [chunk_0] and [web]."

COMPANY_RESULT = (
    "Mantic, a Greek wildfire detection startup, announced a seed funding round "
    "to expand its sensor network and grow its team across southern Europe."
)
SECTOR_RESULT = (
    "The European market for wildfire detection sensors is projected to grow "
    "rapidly, with adoption by utilities and forest agencies and rising demand "
    "for early-warning monitoring systems across the region."
)
COMPETITOR_RESULT = (
    "Several venture-backed wildfire detection startups compete in Europe, "
    "including established sensor companies and newer satellite-based "
    "competitors offering alternatives for utilities and forest agencies."
)
COMPANY_COMPETITOR_RESULT = (
    "Mantic faces several wildfire detection competitors in Europe, with "
    "alternatives ranging from satellite monitoring providers to established "
    "sensor companies serving utilities and forest agencies."
)


class FakeLLM:
    """Grounded calls return a thin answer; hybrid calls return a rich answer."""

    def __init__(self, recorder):
        self.recorder = recorder

    async def ainvoke(self, messages):
        user_text = messages[-1].content
        self.recorder.append(user_text)
        if "Web Search Results" in user_text:
            return SimpleNamespace(content=HYBRID_ANSWER)
        return SimpleNamespace(content=THIN_ANSWER)


@pytest.fixture()
def harness(monkeypatch):
    """Patch LLM + retrieval + provider; return a recorder namespace."""
    rec = SimpleNamespace(prompts=[], searches=[], search_result=COMPANY_RESULT)

    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: FakeLLM(rec.prompts))
    monkeypatch.setattr(
        ea, "retrieve_chunks",
        lambda question, store, k=8: [
            Chunk(chunk_id="chunk_0", text="Pitch deck text.", source_file="deck", page_or_slide="1")
        ],
    )

    def fake_run_web_search(query, domain_filter=None, trigger_reason=None,
                            gating_mode=None, route=None, planner_mode=None,
                            query_purpose=None, cache_info=None):
        rec.searches.append({
            "query": query,
            "domain_filter": domain_filter,
            "trigger_reason": trigger_reason,
            "route": route,
            "planner_mode": planner_mode,
            "query_purpose": query_purpose,
        })
        result = rec.search_result
        return result(query) if callable(result) else result

    monkeypatch.setattr(ea, "_run_web_search", fake_run_web_search)
    monkeypatch.delenv("RDI_WEB_EVIDENCE_PLANNER", raising=False)
    monkeypatch.delenv("WEB_SEARCH_HEAVY_OVERRIDE", raising=False)
    monkeypatch.delenv("RDI_SKIP_UNANSWERABLE_SEARCH", raising=False)
    return rec


def _company(**overrides) -> Company:
    base = dict(
        name="Mantic",
        industry="wildfire detection sensors",
        domain="mantic.ai",
        geo="Athens, Greece",
    )
    base.update(overrides)
    return Company(**base)


def _state(max_calls: int = 10) -> dict:
    return {"count": [0], "lock": asyncio.Lock(), "max": max_calls}


def _answer(question: str, company: Company, state: dict, **kwargs):
    return asyncio.run(
        ea.answer_question_from_evidence(
            question, company, store=None, use_web_search=True,
            web_search_state=state, **kwargs,
        )
    )


# --- off mode: byte-identical legacy behavior -----------------------------------


def test_off_mode_is_byte_identical_legacy(harness):
    company = _company()
    question = "Has the company announced funding?"
    answer, prov = _answer(question, company, _state())

    assert answer == HYBRID_ANSWER
    assert len(harness.searches) == 1
    call = harness.searches[0]
    assert call["query"] == ea._build_web_search_query(company, question)
    assert call["domain_filter"] == ea._web_search_domain_filter(company, question)
    assert call["trigger_reason"] == "documents incomplete"
    assert call["route"] is None and call["planner_mode"] is None
    expected_useful, expected_reason = ea._web_results_add_value(
        question, company.name, COMPANY_RESULT
    )
    assert expected_useful is True
    assert prov["web_search_decision"] == f"used: {expected_reason}"
    assert prov["web_search_used"] is True
    assert "web_search_plan" not in prov
    assert "web_search_route" not in prov


def test_off_mode_explicit_env_matches_default(harness, monkeypatch):
    company = _company()
    question = "Has the company announced funding?"
    _, prov_default = _answer(question, company, _state())
    default_searches = list(harness.searches)

    harness.searches.clear()
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "off")
    _, prov_off = _answer(question, company, _state())

    assert harness.searches == default_searches
    assert prov_off == prov_default


def test_invalid_flag_value_warns_and_defaults_off(harness, monkeypatch, caplog):
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "banana")
    ea._WARNED_INVALID_PLANNER_MODE.clear()
    with caplog.at_level("WARNING"):
        _, prov = _answer("Has the company announced funding?", _company(), _state())
    assert "RDI_WEB_EVIDENCE_PLANNER" in caplog.text
    assert "web_search_plan" not in prov


# --- shadow mode: plan logged, behavior unchanged --------------------------------


def test_shadow_mode_runs_one_legacy_search_and_logs_plan(harness, monkeypatch):
    company = _company()
    question = "Who are the main competitors and alternatives?"
    harness.search_result = COMPANY_COMPETITOR_RESULT

    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "shadow")
    answer, prov = _answer(question, company, _state())

    # Exactly one provider call, with the LEGACY query/filter.
    assert len(harness.searches) == 1
    call = harness.searches[0]
    assert call["query"] == ea._build_web_search_query(company, question)
    assert call["domain_filter"] == ea._web_search_domain_filter(company, question)
    # Plan persisted in provenance.
    assert prov["web_search_route"] == "competitors"
    plan = prov["web_search_plan"]
    assert plan["mode"] == "shadow"
    assert plan["route"] == "competitors"
    assert 1 <= len(plan["queries"]) <= 3
    for q in plan["queries"]:
        assert "mantic" not in q["query"].lower()
        assert q["domain_filter"] is None
    # Answer identical to off mode.
    assert answer == HYBRID_ANSWER


def test_shadow_mode_skip_route_still_searches(harness, monkeypatch):
    """Shadow must not change behavior even when the plan says skip."""
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "shadow")
    _, prov = _answer("What is the company's current ARR?", _company(), _state())
    assert len(harness.searches) == 1  # legacy search still ran
    assert prov["web_search_route"] == "skip_public_web"
    assert prov["web_search_plan"]["queries"] == []


# --- on mode: planner controls behavior ------------------------------------------


def test_on_mode_competitors_multi_query_broad_web(harness, monkeypatch):
    company = _company()
    harness.search_result = COMPETITOR_RESULT
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")

    answer, prov = _answer(
        "Who are the main competitors and alternatives?", company, _state()
    )

    assert 1 <= len(harness.searches) <= 3
    for call in harness.searches:
        assert "mantic" not in call["query"].lower()
        assert call["domain_filter"] is None
        assert call["route"] == "competitors"
        assert call["planner_mode"] == "on"
        assert call["query_purpose"]
    n = len(harness.searches)
    assert prov["web_search_decision"] == f"used: {n}/{n} results passed topic_keywords gate"
    assert prov["web_search_used"] is True
    # Concatenated headers reach the hybrid prompt.
    hybrid_prompts = [p for p in harness.prompts if "Web Search Results" in p]
    assert len(hybrid_prompts) == 1
    assert "=== Web results 1/" in hybrid_prompts[0]
    assert answer == HYBRID_ANSWER


def test_on_mode_internal_fit_searches_company_anchored(harness, monkeypatch):
    # internal_fit no longer skips — it searches company-anchored (Gate-1 fix).
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    state = _state()
    _, prov = _answer(
        "Does the company's stage align with the VC's investment thesis?",
        _company(), state,
    )
    assert len(harness.searches) >= 1
    assert prov["web_search_route"] == "internal_fit"
    assert not prov["web_search_decision"].startswith("skipped: planner internal_fit")


def test_on_mode_cap_counts_each_provider_call(harness, monkeypatch):
    harness.search_result = SECTOR_RESULT
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    state = _state(max_calls=2)

    _answer("Who are the main competitors and alternatives?", _company(), state)

    # 3-query competitors plan, cap=2 -> exactly 2 provider calls.
    assert len(harness.searches) == 2
    assert state["count"][0] == 2


def test_on_mode_sector_result_accepted_without_company_mention(harness, monkeypatch):
    harness.search_result = SECTOR_RESULT  # never mentions Mantic
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")

    _, prov = _answer(
        "Is the sector attractive given market size, growth and regulation?",
        _company(), _state(),
    )
    # The fix in miniature: sector evidence used despite no company mention.
    assert prov["web_search_used"] is True
    assert prov["web_search_route"] == "sector_market"
    assert prov["web_search_decision"].startswith("used:")


def test_on_mode_company_specific_keeps_restrictive_filter_and_gate(harness, monkeypatch):
    harness.search_result = SECTOR_RESULT  # generic, no company mention
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")

    answer, prov = _answer("Has the company announced funding?", _company(), _state())

    assert len(harness.searches) == 1
    call = harness.searches[0]
    assert "mantic" in call["query"].lower()
    assert call["domain_filter"] == ["mantic.ai", "crunchbase.com", "linkedin.com"]
    # Generic sector text must NOT pass the company_specific gate.
    assert prov["web_search_used"] is False
    assert prov["web_search_decision"] == "skipped: 0/1 results passed company_mention gate"
    assert answer == THIN_ANSWER


def test_on_mode_route_tag_threads_through(harness, monkeypatch):
    harness.search_result = SECTOR_RESULT
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")

    _, prov = _answer(
        "Has the company announced funding?", _company(), _state(),
        route="sector_market",
    )
    assert prov["web_search_route"] == "sector_market"
    assert prov["web_search_plan"]["rationale"] == "tag:sector_market"
    assert all("mantic" not in c["query"].lower() for c in harness.searches)


def test_node_route_attribute_reaches_planner(harness, monkeypatch):
    harness.search_result = SECTOR_RESULT
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")

    node = SimpleNamespace(
        question="Has the company announced funding?",
        sub_nodes=[], answer=None, provenance=None,
        route="sector_market",
    )
    asyncio.run(
        ea._answer_node_from_evidence(
            node, _company(), store=None, use_web_search=True,
            web_search_state=_state(),
        )
    )
    assert node.provenance["web_search_route"] == "sector_market"
    assert node.provenance["web_search_plan"]["rationale"] == "tag:sector_market"


def test_node_without_route_attribute_still_works(harness, monkeypatch):
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    node = SimpleNamespace(
        question="Has the company announced funding?",
        sub_nodes=[], answer=None, provenance=None,
    )
    asyncio.run(
        ea._answer_node_from_evidence(
            node, _company(), store=None, use_web_search=True,
            web_search_state=_state(),
        )
    )
    assert node.provenance["web_search_route"] == "company_specific"


# --- Company.geo ingest -----------------------------------------------------------


def test_company_model_accepts_geo_default_none():
    assert Company(name="X").geo is None
    assert Company(name="X", geo="Berlin, Germany").geo == "Berlin, Germany"


def test_specter_ingest_populates_geo_from_hq_location(tmp_path):
    import pandas as pd
    from agent.ingest.specter_ingest import ingest_specter

    companies_path = tmp_path / "specter-export.csv"
    pd.DataFrame(
        [
            {"Company Name": "Alpha", "Industry": "SaaS", "HQ Location": "Prague, Czechia"},
            {"Company Name": "Beta", "Industry": "Fintech"},
        ]
    ).to_csv(companies_path, index=False)

    results = ingest_specter(companies_path)
    by_name = {company.name: company for company, _ in results}
    assert by_name["Alpha"].geo == "Prague, Czechia"
    assert by_name["Beta"].geo is None


# --- Sprint 4 C: skip predictably-unanswerable searches -------------------------
# The skip-gate consults predict_search_unanswerable (which delegates to the
# planner router) only for the ungated "documents incomplete" reason.
SKIP_QUESTION = "What is the company's current ARR and burn rate?"


def test_skip_gate_off_is_byte_identical(harness, monkeypatch):
    company = _company()
    _, prov_default = _answer(SKIP_QUESTION, company, _state())
    default_searches = list(harness.searches)
    assert "skip_unanswerable" not in prov_default

    harness.searches.clear()
    monkeypatch.setenv("RDI_SKIP_UNANSWERABLE_SEARCH", "off")
    _, prov_off = _answer(SKIP_QUESTION, company, _state())

    assert harness.searches == default_searches
    assert prov_off == prov_default


def test_skip_gate_shadow_logs_but_still_searches(harness, monkeypatch):
    monkeypatch.setenv("RDI_SKIP_UNANSWERABLE_SEARCH", "shadow")
    _, prov = _answer(SKIP_QUESTION, _company(), _state())

    assert len(harness.searches) == 1  # search still ran (behavior == off)
    assert prov["skip_unanswerable"]["mode"] == "shadow"
    assert prov["skip_unanswerable"]["predicted_skip"] is True
    # Shadow records the prediction but does NOT act on it (no early skip).
    assert not prov["web_search_decision"].startswith("skipped: predicted")


def test_skip_gate_on_skips_search_and_consumes_no_cap(harness, monkeypatch):
    monkeypatch.setenv("RDI_SKIP_UNANSWERABLE_SEARCH", "on")
    state = _state()
    answer, prov = _answer(SKIP_QUESTION, _company(), state)

    assert len(harness.searches) == 0
    assert state["count"][0] == 0  # no cap slot consumed
    assert answer == THIN_ANSWER  # grounded answer kept
    assert prov["web_search_used"] is False
    assert prov["web_search_decision"].startswith("skipped: predicted unanswerable")
    assert prov["skip_unanswerable"]["predicted_skip"] is True


def test_skip_gate_on_does_not_skip_answerable(harness, monkeypatch):
    monkeypatch.setenv("RDI_SKIP_UNANSWERABLE_SEARCH", "on")
    _, prov = _answer("Has the company announced funding?", _company(), _state())

    assert len(harness.searches) == 1  # answerable -> search runs
    assert prov["skip_unanswerable"]["predicted_skip"] is False
    assert prov["web_search_used"] is True


def test_skip_gate_on_with_planner_on_skips_once_gate_wins(harness, monkeypatch):
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setenv("RDI_SKIP_UNANSWERABLE_SEARCH", "on")
    state = _state()
    _, prov = _answer(SKIP_QUESTION, _company(), state)

    assert len(harness.searches) == 0  # single skip, no double-anything
    assert state["count"][0] == 0
    assert prov["web_search_decision"].startswith("skipped: predicted unanswerable")
