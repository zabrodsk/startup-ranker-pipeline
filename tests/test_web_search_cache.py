"""Sprint 3 (W13): RDI_WEB_SEARCH_CACHE gates the durable web-search cache.

Off (default): no lookups, no stores, telemetry byte-identical. On: cache hits
return stored text without a provider call, mark telemetry with cache_hit, and
refund the per-company cap slot the caller acquired. Lookup/store are
fail-open (miss/no-op on any error or unconfigured DB).
"""

import asyncio
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import agent.evidence_answering as ea
import agent.web_search as web_search_module
from agent.dataclasses.company import Company
from agent.ingest.store import Chunk
from agent.web_search import result_cache

THIN_ANSWER = "Insufficient information available."
HYBRID_ANSWER = "Hybrid answer citing [chunk_0] and [web]."
COMPANY_RESULT = (
    "Mantic, a Greek wildfire detection startup, announced a seed funding round "
    "to expand its sensor network across southern Europe."
)


# --- key computation ---------------------------------------------------------


def test_query_hash_is_stable_and_filter_order_insensitive():
    h1 = result_cache.compute_query_hash("sonar", "wildfire market", ["a.com", "b.com"])
    h2 = result_cache.compute_query_hash("sonar", "wildfire market", ["b.com", "a.com"])
    assert h1 == h2


def test_query_hash_distinguishes_provider_query_and_filter():
    base = result_cache.compute_query_hash("sonar", "wildfire market", None)
    assert result_cache.compute_query_hash("brave", "wildfire market", None) != base
    assert result_cache.compute_query_hash("sonar", "other query", None) != base
    assert result_cache.compute_query_hash("sonar", "wildfire market", ["a.com"]) != base


def test_none_and_empty_domain_filter_share_a_key():
    assert result_cache.compute_query_hash("sonar", "q", None) == (
        result_cache.compute_query_hash("sonar", "q", [])
    )


# --- flag getters ------------------------------------------------------------


def test_cache_defaults_to_off(monkeypatch):
    monkeypatch.delenv(result_cache.RDI_WEB_SEARCH_CACHE_ENV, raising=False)
    assert result_cache.is_web_search_cache_enabled() is False


def test_invalid_mode_and_ttl_warn_once_and_fall_back(monkeypatch, caplog):
    monkeypatch.setenv(result_cache.RDI_WEB_SEARCH_CACHE_ENV, "bogus")
    monkeypatch.setenv(result_cache.RDI_WEB_SEARCH_CACHE_TTL_DAYS_ENV, "-3")
    monkeypatch.setattr(result_cache, "_WARNED_INVALID_CACHE_VALUES", set())

    with caplog.at_level(logging.WARNING, logger=result_cache.__name__):
        assert result_cache.is_web_search_cache_enabled() is False
        assert result_cache.is_web_search_cache_enabled() is False
        assert result_cache.cache_ttl_days() == 14
        assert result_cache.cache_ttl_days() == 14

    warnings = [r for r in caplog.records if "falling back" in r.getMessage()]
    assert len(warnings) == 2  # one per env var, despite repeated calls


# --- lookup/store fail-open behavior ----------------------------------------


def test_lookup_off_never_touches_db(monkeypatch):
    monkeypatch.delenv(result_cache.RDI_WEB_SEARCH_CACHE_ENV, raising=False)
    from web import db

    def _boom(*_args, **_kwargs):
        raise AssertionError("db must not be consulted when cache is off")

    monkeypatch.setattr(db, "get_web_search_cache_entry", _boom)
    assert result_cache.lookup("sonar", "q", None) is None


def test_lookup_on_returns_db_value_and_swallows_errors(monkeypatch):
    monkeypatch.setenv(result_cache.RDI_WEB_SEARCH_CACHE_ENV, "on")
    from web import db

    monkeypatch.setattr(
        db, "get_web_search_cache_entry", lambda _h, ttl_days: "cached text"
    )
    assert result_cache.lookup("sonar", "q", None) == "cached text"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_web_search_cache_entry", _boom)
    assert result_cache.lookup("sonar", "q", None) is None


def test_store_on_upserts_with_computed_hash_and_swallows_errors(monkeypatch):
    monkeypatch.setenv(result_cache.RDI_WEB_SEARCH_CACHE_ENV, "on")
    from web import db

    stored = []
    monkeypatch.setattr(
        db,
        "upsert_web_search_cache_entry",
        lambda **kwargs: stored.append(kwargs) or True,
    )
    result_cache.store("sonar", "q", ["b.com", "a.com"], "text")
    assert stored == [
        {
            "query_hash": result_cache.compute_query_hash("sonar", "q", ["a.com", "b.com"]),
            "provider": "sonar",
            "query": "q",
            "domain_filter": ["a.com", "b.com"],
            "results": "text",
        }
    ]

    def _boom(**_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "upsert_web_search_cache_entry", _boom)
    result_cache.store("sonar", "q", None, "text")  # must not raise


def test_store_off_never_touches_db(monkeypatch):
    monkeypatch.delenv(result_cache.RDI_WEB_SEARCH_CACHE_ENV, raising=False)
    from web import db

    def _boom(**_kwargs):
        raise AssertionError("db must not be consulted when cache is off")

    monkeypatch.setattr(db, "upsert_web_search_cache_entry", _boom)
    result_cache.store("sonar", "q", None, "text")


def test_db_accessor_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    from web import db

    assert db.get_web_search_cache_entry("some-hash", ttl_days=14) is None
    assert (
        db.upsert_web_search_cache_entry(
            query_hash="h", provider="sonar", query="q", domain_filter=[], results="r"
        )
        is False
    )


# --- _run_web_search integration ---------------------------------------------


class _FakeCollector:
    def __init__(self):
        self.metadata: list[dict] = []

    def record_perplexity_search(self, *, metadata=None, **_kwargs):
        self.metadata.append(metadata or {})


def _patch_provider(monkeypatch, calls: list):
    class _FakeProvider:
        def search(self, query, domain_filter=None):
            calls.append(query)
            return COMPANY_RESULT

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "sonar")
    monkeypatch.setenv("PPLX_API_KEY", "test-key")
    monkeypatch.setattr(web_search_module, "get_provider", lambda **_k: _FakeProvider())


def test_run_web_search_hit_skips_provider_and_marks_telemetry(monkeypatch):
    provider_calls: list = []
    _patch_provider(monkeypatch, provider_calls)
    collector = _FakeCollector()
    monkeypatch.setattr(ea, "get_current_collector", lambda: collector)
    monkeypatch.setattr(result_cache, "lookup", lambda *a: "CACHED RESULT")
    stored = []
    monkeypatch.setattr(result_cache, "store", lambda *a: stored.append(a))

    cache_info: dict = {}
    result = ea._run_web_search("some query", None, cache_info=cache_info)

    assert result == "CACHED RESULT"
    assert provider_calls == []
    assert cache_info == {"hit": True}
    assert stored == []  # hits are not re-stored
    assert len(collector.metadata) == 1
    assert collector.metadata[0]["cache_hit"] is True


def test_run_web_search_miss_calls_provider_stores_and_keeps_legacy_metadata(monkeypatch):
    provider_calls: list = []
    _patch_provider(monkeypatch, provider_calls)
    collector = _FakeCollector()
    monkeypatch.setattr(ea, "get_current_collector", lambda: collector)
    monkeypatch.setattr(result_cache, "lookup", lambda *a: None)
    stored = []
    monkeypatch.setattr(result_cache, "store", lambda *a: stored.append(a))

    cache_info: dict = {}
    result = ea._run_web_search(
        "some query", None, "documents incomplete", "root_only", cache_info=cache_info
    )

    assert result == COMPANY_RESULT
    assert provider_calls == ["some query"]
    assert cache_info == {"hit": False}
    assert stored == [("sonar", "some query", None, COMPANY_RESULT)]
    # Off-mode/miss telemetry stays byte-identical: no cache_hit key.
    assert collector.metadata == [
        {
            "query": "some query",
            "domain_filter": [],
            "trigger_reason": "documents incomplete",
            "gating_mode": "root_only",
        }
    ]


def test_run_web_search_does_not_store_failed_results(monkeypatch):
    class _FailingProvider:
        def search(self, query, domain_filter=None):
            return "Web search failed: provider exploded"

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "sonar")
    monkeypatch.setenv("PPLX_API_KEY", "test-key")
    monkeypatch.setattr(web_search_module, "get_provider", lambda **_k: _FailingProvider())
    monkeypatch.setattr(ea, "get_current_collector", lambda: None)
    monkeypatch.setattr(result_cache, "lookup", lambda *a: None)
    stored = []
    monkeypatch.setattr(result_cache, "store", lambda *a: stored.append(a))

    ea._run_web_search("some query")
    assert stored == []


# --- cap-slot refund through answer_question_from_evidence -------------------


class _FakeLLM:
    """Grounded calls return a thin answer; hybrid calls return a rich answer."""

    async def ainvoke(self, messages):
        if "Web Search Results" in messages[-1].content:
            return SimpleNamespace(content=HYBRID_ANSWER)
        return SimpleNamespace(content=THIN_ANSWER)


def _setup_answer_harness(monkeypatch, *, cached: bool):
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: _FakeLLM())
    monkeypatch.setattr(
        ea,
        "retrieve_chunks",
        lambda question, store, k=8: [
            Chunk(chunk_id="chunk_0", text="Deck text.", source_file="deck", page_or_slide="1")
        ],
    )
    provider_calls: list = []
    _patch_provider(monkeypatch, provider_calls)
    monkeypatch.setattr(ea, "get_current_collector", lambda: None)
    monkeypatch.setattr(
        result_cache, "lookup", lambda *a: COMPANY_RESULT if cached else None
    )
    monkeypatch.setattr(result_cache, "store", lambda *a: None)
    monkeypatch.delenv("RDI_WEB_EVIDENCE_PLANNER", raising=False)
    monkeypatch.delenv("WEB_SEARCH_HEAVY_OVERRIDE", raising=False)
    return provider_calls


def _answer(question: str, state: dict):
    company = Company(
        name="Mantic",
        industry="wildfire detection sensors",
        domain="mantic.ai",
        geo="Athens, Greece",
    )
    return asyncio.run(
        ea.answer_question_from_evidence(
            question,
            company,
            store=None,
            use_web_search=True,
            web_search_state=state,
        )
    )


def test_legacy_cache_hit_refunds_cap_slot_and_labels_decision(monkeypatch):
    provider_calls = _setup_answer_harness(monkeypatch, cached=True)
    state = {"count": [0], "lock": asyncio.Lock(), "max": 10}

    _, provenance = _answer("What funding has the company announced?", state)

    assert provider_calls == []
    assert state["count"][0] == 0  # slot refunded
    assert provenance["web_search_decision"] == "used: cache hit (no cap slot consumed)"


def test_legacy_cache_miss_consumes_cap_slot(monkeypatch):
    provider_calls = _setup_answer_harness(monkeypatch, cached=False)
    state = {"count": [0], "lock": asyncio.Lock(), "max": 10}

    _, provenance = _answer("What funding has the company announced?", state)

    assert provider_calls != []
    assert state["count"][0] == 1
    assert provenance["web_search_decision"].startswith("used: ")
    assert "cache hit" not in provenance["web_search_decision"]


def test_planner_cache_hits_refund_slots_and_label_decision(monkeypatch):
    provider_calls = _setup_answer_harness(monkeypatch, cached=True)
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    state = {"count": [0], "lock": asyncio.Lock(), "max": 10}

    _, provenance = _answer(
        "What is the market size and growth for wildfire detection sensors in Europe?",
        state,
    )

    assert provider_calls == []
    assert state["count"][0] == 0  # every acquired slot was refunded
    assert "cache hits, slots refunded" in provenance["web_search_decision"]
