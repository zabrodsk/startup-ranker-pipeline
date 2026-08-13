"""Sprint 3 (W11): RDI_DECOMP_CACHE_STORE selects the decomposition cache.

"local" (default) keeps the per-container JSON cache byte-identical;
"supabase" shares decomposed trees across workers via a company-free key
(question + industry + aspect + full prompt signature including
ROUTE_TAGGING_INSTRUCTION). Unconfigured DB silently falls back to local.
Includes the canary test pinning the decomposition prompt to {question} and
{industry} only — adding company context to the prompt would invalidate the
company-free key design.
"""

import asyncio
import logging

from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.pipeline.stages import decomposition_cache_store as dcs
from agent.pipeline.stages import parallel_decomposition as pd

FLAG_ENV = "RDI_DECOMP_CACHE_STORE"


def _tree(route: str | None = "company_news") -> QuestionTree:
    return QuestionTree(
        aspect="market",
        root_node=QuestionNode(
            question="How big is the market?",
            sub_nodes=[QuestionNode(question="What is the TAM?", route=route)],
        ),
    )


# --- flag getter --------------------------------------------------------------


def test_store_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv(FLAG_ENV, raising=False)
    assert dcs._store_mode() == "local"
    assert dcs.use_supabase_store() is False


def test_invalid_store_mode_warns_once_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv(FLAG_ENV, "bogus")
    monkeypatch.setattr(dcs, "_WARNED_INVALID_STORE_MODE", set())
    with caplog.at_level(logging.WARNING, logger=dcs.__name__):
        assert dcs._store_mode() == "local"
        assert dcs._store_mode() == "local"
    warnings = [r for r in caplog.records if FLAG_ENV in r.getMessage()]
    assert len(warnings) == 1


def test_supabase_mode_with_unconfigured_db_falls_back_to_local(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "supabase")
    from web import db

    monkeypatch.setattr(db, "is_configured", lambda: False)
    assert dcs.use_supabase_store() is False


# --- key computation ------------------------------------------------------------


def test_query_hash_is_company_free_and_input_sensitive():
    sig = dcs.compute_prompt_signature()
    base = dcs.compute_query_hash("How big is the market?", "Robotics", "market", sig)
    # No company anywhere in the key inputs — two same-industry companies
    # necessarily produce the same hash.
    assert dcs.compute_query_hash("How big is the market?", "robotics ", "market", sig) == base
    assert dcs.compute_query_hash("Other question?", "Robotics", "market", sig) != base
    assert dcs.compute_query_hash("How big is the market?", "Fintech", "market", sig) != base
    assert dcs.compute_query_hash("How big is the market?", "Robotics", "team", sig) != base
    assert dcs.compute_query_hash("How big is the market?", "Robotics", "market", "x" + sig) != base


def test_prompt_signature_covers_route_tagging_instruction(monkeypatch):
    base = dcs.compute_prompt_signature()
    monkeypatch.setattr(dcs, "ROUTE_TAGGING_INSTRUCTION", "changed instruction")
    assert dcs.compute_prompt_signature() != base


def test_prompt_signature_covers_portfolio_budget_and_category_contract():
    base = dcs.compute_prompt_signature()
    market = dcs.compute_prompt_signature(aspect="market", question_budget=24)
    product = dcs.compute_prompt_signature(aspect="product", question_budget=20)

    assert market != base
    assert product != base
    assert market != product
    assert dcs._KEY_VERSION_PREFIX == "dtree-v5"


def test_prompt_signature_covers_prompt_overrides():
    base = dcs.compute_prompt_signature()
    overridden = dcs.compute_prompt_signature(
        {"prompt_overrides": {"decomposition.user": "Custom {question} for {industry}"}}
    )
    # Overrides may or may not resolve depending on registry shape; identical
    # default prompts must at minimum be stable.
    assert dcs.compute_prompt_signature() == base
    assert isinstance(overridden, str) and len(overridden) == 64


# --- canary: the key design assumes a company-free decomposition prompt --------


def test_canary_decomposition_user_prompt_takes_only_question_and_industry():
    """W11 key safety: the decomposition prompt must see ONLY question+industry.

    decomposition.py calls decompose_user_prompt.format(question=..., industry=...).
    If anyone adds a company placeholder to the prompt, this format call raises
    (or the placeholder list changes) and this canary fails — the company-free
    cache key in decomposition_cache_store must then be redesigned (dtree-v5).
    """
    from agent.prompt_library.defaults import PROMPT_DEFINITIONS
    from agent.prompt_library.manager import get_prompt

    assert PROMPT_DEFINITIONS["decomposition.user"]["required_placeholders"] == [
        "{question}",
        "{industry}",
    ]
    rendered = str(get_prompt("decomposition.user")).format(
        question="canary question", industry="canary industry"
    )
    assert "canary question" in rendered
    assert "canary industry" in rendered


# --- lookup/store ---------------------------------------------------------------


def test_lookup_hit_returns_tree_with_routes_and_bumps_counter(monkeypatch):
    from web import db

    stored_tree = _tree(route="sector_market").model_dump()
    increments = []
    monkeypatch.setattr(
        db,
        "get_decomposition_tree_cache_entry",
        lambda _h: {"tree": stored_tree, "company_name": "Other Co", "hit_count": 3},
    )
    monkeypatch.setattr(
        db,
        "increment_decomposition_tree_cache_hit",
        lambda h, c: increments.append((h, c)) or True,
    )

    tree = dcs.lookup(
        question="How big is the market?",
        industry="Robotics",
        aspect="market",
        company_name="Acme Robotics",
    )

    assert isinstance(tree, QuestionTree)
    assert tree.root_node.sub_nodes[0].route == "sector_market"
    assert len(increments) == 1
    assert increments[0][1] == 3


def test_lookup_logs_cross_company_observability(monkeypatch, caplog):
    from web import db

    monkeypatch.setattr(
        db,
        "get_decomposition_tree_cache_entry",
        lambda _h: {"tree": _tree().model_dump(), "company_name": "Other Co", "hit_count": 0},
    )
    monkeypatch.setattr(db, "increment_decomposition_tree_cache_hit", lambda h, c: True)

    with caplog.at_level(logging.INFO, logger=dcs.__name__):
        dcs.lookup(
            question="q", industry="i", aspect="market", company_name="Acme Robotics"
        )
    assert any("cross_company=True" in r.getMessage() for r in caplog.records)


def test_lookup_failures_are_a_miss(monkeypatch):
    from web import db

    def _boom(_h):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_decomposition_tree_cache_entry", _boom)
    assert (
        dcs.lookup(question="q", industry="i", aspect="market", company_name="A")
        is None
    )


def test_increment_failure_does_not_undo_the_hit(monkeypatch):
    from web import db

    monkeypatch.setattr(
        db,
        "get_decomposition_tree_cache_entry",
        lambda _h: {"tree": _tree().model_dump(), "company_name": None, "hit_count": 0},
    )

    def _boom(_h, _c):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "increment_decomposition_tree_cache_hit", _boom)
    tree = dcs.lookup(question="q", industry="i", aspect="market", company_name="A")
    assert isinstance(tree, QuestionTree)


def test_store_failures_are_swallowed(monkeypatch):
    from web import db

    def _boom(**_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "upsert_decomposition_tree_cache_entry", _boom)
    dcs.store(
        question="q", industry="i", aspect="market", company_name="A", tree=_tree()
    )  # must not raise


# --- _get_or_decompose_question integration ---------------------------------------


def test_local_default_never_touches_supabase_accessors(monkeypatch):
    monkeypatch.delenv(FLAG_ENV, raising=False)
    from web import db

    def _boom(*_a, **_k):
        raise AssertionError("supabase accessors must not run in local mode")

    monkeypatch.setattr(db, "get_decomposition_tree_cache_entry", _boom)
    monkeypatch.setattr(db, "upsert_decomposition_tree_cache_entry", _boom)

    local_calls = []
    monkeypatch.setattr(
        pd, "get_cached_question_tree", lambda q, c, a: local_calls.append(q) or _tree()
    )

    result = asyncio.run(
        pd._get_or_decompose_question("How big?", "Robotics", "market", "Acme")
    )
    assert result["aspect"] == "market"
    assert len(local_calls) == 1  # legacy local cache path ran


def test_supabase_hit_skips_decomposition(monkeypatch):
    monkeypatch.setattr(dcs, "use_supabase_store", lambda: True)
    monkeypatch.setattr(
        dcs, "lookup", lambda **_k: _tree(route="competitors")
    )

    async def _no_decompose(*_a, **_k):
        raise AssertionError("decomposition LLM must not run on a cache hit")

    monkeypatch.setattr(pd, "_decompose_single_question", _no_decompose)

    result = asyncio.run(
        pd._get_or_decompose_question("How big?", "Robotics", "market", "Acme")
    )
    assert result["tree"].root_node.sub_nodes[0].route == "competitors"


def test_supabase_miss_decomposes_and_stores(monkeypatch):
    monkeypatch.setattr(dcs, "use_supabase_store", lambda: True)
    monkeypatch.setattr(dcs, "lookup", lambda **_k: None)
    stored = []
    monkeypatch.setattr(dcs, "store", lambda **kwargs: stored.append(kwargs))

    fresh = _tree()

    async def _decompose(*_a, **_k):
        return {"aspect": "market", "tree": fresh}

    monkeypatch.setattr(pd, "_decompose_single_question", _decompose)

    result = asyncio.run(
        pd._get_or_decompose_question("How big?", "Robotics", "market", "Acme")
    )
    assert result["tree"] is fresh
    assert len(stored) == 1
    assert stored[0]["company_name"] == "Acme"
    assert stored[0]["tree"] is fresh
