from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
import pytest

import agent.evidence_answering as ea
import agent.web_search as web_search_module
from agent.web_search import result_cache
from agent.web_search.providers import HybridSearchProvider
from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree


def test_hybrid_resolves_to_serper_primary_with_perplexity_fallback(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")

    assert ea._resolve_web_search_provider_name() == "hybrid"


def test_hybrid_degrades_to_available_provider(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")

    assert ea._resolve_web_search_provider_name() == "sonar"


def test_default_sonar_degrades_to_serper_when_it_is_the_only_key(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "sonar")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    assert ea._resolve_web_search_provider_name() == "serper"


def test_run_web_search_uses_serper_for_hybrid_and_records_provider(monkeypatch) -> None:
    provider_names: list[str] = []
    recorded: list[tuple[str, dict]] = []

    class _Provider:
        last_provider_name = "serper"

        def search(self, query, domain_filter=None):  # noqa: ANN001
            return "Apaleo hotel software market evidence with relevant demand data."

    class _Collector:
        def record_web_search(self, *, provider, metadata):  # noqa: ANN001
            recorded.append((provider, metadata))

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(
        web_search_module,
        "get_provider",
        lambda *, provider_name, **_kwargs: provider_names.append(provider_name) or _Provider(),
    )
    monkeypatch.setattr(result_cache, "lookup", lambda *_args: None)
    monkeypatch.setattr(result_cache, "store", lambda *_args: None)
    monkeypatch.setattr(ea, "get_current_collector", lambda: _Collector())

    result = ea._run_web_search(
        "Apaleo market demand",
        trigger_reason="portfolio core",
        route="sector_market",
    )

    assert "Apaleo" in result
    assert provider_names == ["hybrid"]
    assert recorded == [
        (
            "serper",
            {
                "query": "Apaleo market demand",
                "domain_filter": [],
                "trigger_reason": "portfolio core",
                "route": "sector_market",
            },
        )
    ]


def test_run_web_search_uses_shared_hybrid_fallback_for_direct_callers(monkeypatch) -> None:
    attempts: list[str] = []
    recorded: list[str] = []

    class _Provider:
        last_provider_name = "sonar"
        attempted_provider_names = ["serper", "sonar"]

        def search(self, *_args, **_kwargs):
            attempts.extend(["serper", "sonar"])
            return "Search Results for: q\n\n1. Perplexity fallback evidence"

    class _Collector:
        def record_web_search(self, *, provider, **_kwargs):  # noqa: ANN001
            recorded.append(provider)

        def record_perplexity_search(self, **_kwargs):
            recorded.append("perplexity")

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(
        web_search_module,
        "get_provider",
        lambda *, provider_name, **_kwargs: (
            attempts.append(f"factory:{provider_name}") or _Provider()
        ),
    )
    monkeypatch.setattr(result_cache, "lookup", lambda *_args: None)
    monkeypatch.setattr(result_cache, "store", lambda *_args: None)
    monkeypatch.setattr(ea, "get_current_collector", lambda: _Collector())

    result = ea._run_web_search("q")

    assert "Perplexity fallback evidence" in result
    assert attempts == ["factory:hybrid", "serper", "sonar"]
    assert recorded == ["serper", "perplexity"]


def test_failed_hybrid_attempts_remain_in_cap_metadata_and_telemetry(monkeypatch) -> None:
    recorded: list[str] = []

    class _FailingProvider:
        attempted_provider_names = ["serper", "sonar"]
        last_provider_name = "sonar"

        def search(self, *_args, **_kwargs):
            raise RuntimeError("all providers failed")

    class _Collector:
        def record_web_search(self, *, provider, **_kwargs):  # noqa: ANN001
            recorded.append(provider)

        def record_perplexity_search(self, **_kwargs):
            recorded.append("perplexity")

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(
        web_search_module,
        "get_provider",
        lambda **_kwargs: _FailingProvider(),
    )
    monkeypatch.setattr(result_cache, "lookup", lambda *_args: None)
    monkeypatch.setattr(ea, "get_current_collector", lambda: _Collector())
    cache_info: dict = {}

    result = ea._run_web_search("Apaleo funding", cache_info=cache_info)

    assert result.startswith("Web search failed:")
    assert cache_info["attempted_providers"] == ["serper", "sonar"]
    assert cache_info["provider"] == "sonar"
    assert recorded == ["serper", "perplexity"]


def test_failed_concrete_provider_attempt_remains_in_metadata_and_telemetry(
    monkeypatch,
) -> None:
    recorded: list[str] = []

    class _FailingProvider:
        def search(self, *_args, **_kwargs):
            raise TimeoutError("serper timed out")

    class _Collector:
        def record_web_search(self, *, provider, **_kwargs):  # noqa: ANN001
            recorded.append(provider)

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(
        web_search_module,
        "get_provider",
        lambda **_kwargs: _FailingProvider(),
    )
    monkeypatch.setattr(result_cache, "lookup", lambda *_args: None)
    monkeypatch.setattr(ea, "get_current_collector", lambda: _Collector())
    cache_info: dict = {}

    result = ea._run_web_search(
        "Apaleo funding",
        cache_info=cache_info,
        provider_override="serper",
    )

    assert result.startswith("Web search failed:")
    assert cache_info["attempted_providers"] == ["serper"]
    assert cache_info["provider"] == "serper"
    assert recorded == ["serper"]


def test_hybrid_falls_back_when_primary_has_only_one_thin_result(monkeypatch) -> None:
    attempts: list[str] = []

    class _ThinPrimary:
        def search(self, *_args, **_kwargs):
            attempts.append("serper")
            return "Search Results for: q\n\n1. Unrelated — https://example.com\n   Thin."

    class _UsefulFallback:
        def search(self, *_args, **_kwargs):
            attempts.append("sonar")
            return (
                "Search Results for: q\n\n"
                "1. Relevant market evidence — https://example.com/one\n"
                "   Detailed market demand, adoption, customer, and growth evidence.\n\n"
                "2. Independent benchmark — https://example.com/two\n"
                "   Additional industry sizing, competition, and customer evidence."
            )

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    provider = HybridSearchProvider("2026-08-13")
    provider._providers = [("serper", _ThinPrimary()), ("sonar", _UsefulFallback())]

    result = provider.search("q")

    assert attempts == ["serper", "sonar"]
    assert provider.attempted_provider_names == ["serper", "sonar"]
    assert provider.last_provider_name == "sonar"
    assert "Independent benchmark" in result


def test_hybrid_respects_provider_attempt_limit(monkeypatch) -> None:
    attempts: list[str] = []

    class _Provider:
        def __init__(self, name: str):
            self.name = name

        def search(self, *_args, **_kwargs):
            attempts.append(self.name)
            return "Search Results for: q\n\nNo search results returned."

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    provider = HybridSearchProvider("2026-08-13")
    provider._providers = [("serper", _Provider("serper")), ("sonar", _Provider("sonar"))]

    with pytest.raises(RuntimeError, match="unusable"):
        provider.search("q", max_provider_attempts=1)

    assert attempts == ["serper"]
    assert provider.attempted_provider_names == ["serper"]


def test_hybrid_does_not_count_fallback_that_deadline_prevents(monkeypatch) -> None:
    attempts: list[str] = []
    clock = iter([0.0, 0.0, 2.0])

    class _ThinPrimary:
        def search(self, *_args, **_kwargs):
            attempts.append("serper")
            return "Search Results for: q\n\nNo search results returned."

    class _Fallback:
        def search(self, *_args, **_kwargs):
            attempts.append("sonar")
            return "Search Results for: q\n\n1. Useful fallback evidence"

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr("agent.web_search.providers.time.monotonic", lambda: next(clock))
    provider = HybridSearchProvider("2026-08-13")
    provider._providers = [("serper", _ThinPrimary()), ("sonar", _Fallback())]

    with pytest.raises(RuntimeError, match="unusable"):
        provider.search("q", deadline_seconds=1.0)

    assert attempts == ["serper"]
    assert provider.attempted_provider_names == ["serper"]


def test_hybrid_does_not_return_unusable_primary_after_fallback_failure(
    monkeypatch,
) -> None:
    class _ThinPrimary:
        def search(self, *_args, **_kwargs):
            return "Search Results for: Acme funding\n\nNo search results returned."

    class _FailingFallback:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("fallback unavailable")

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    provider = HybridSearchProvider("2026-08-13")
    provider._providers = [
        ("serper", _ThinPrimary()),
        ("sonar", _FailingFallback()),
    ]

    with pytest.raises(RuntimeError, match="failed or returned unusable"):
        provider.search("Acme funding")

    assert provider.last_provider_name is None
    assert provider.attempted_provider_names == ["serper", "sonar"]


def test_portfolio_primary_and_fallback_share_one_deadline(monkeypatch) -> None:
    attempts: list[tuple[str | None, float | None]] = []

    def fake_search(*_args, provider_override=None, provider_deadline_seconds=None, **_kwargs):
        attempts.append((provider_override, provider_deadline_seconds))
        time.sleep(0.02)
        return "No search results returned."

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(ea, "WEB_SEARCH_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(ea, "_run_web_search", fake_search)
    monkeypatch.setattr(
        ea,
        "evaluate_web_result_relevance",
        lambda **_kwargs: (False, "not useful"),
    )
    state = {"count": [0], "lock": asyncio.Lock(), "max": 2}

    result = asyncio.run(
        ea._run_quality_routed_search(
            query="Acme funding",
            domain_filter=None,
            purpose="funding",
            route=SimpleNamespace(value="company_specific"),
            relevance_policy=object(),
            representative_question="What funding has Acme raised?",
            company=Company(name="Acme"),
            trigger_reason="portfolio core",
            web_search_state=state,
        )
    )

    assert result is None
    assert [provider for provider, _deadline in attempts] == ["serper"]
    assert state["count"] == [1]


def test_planner_does_not_reserve_fallback_after_deadline(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_search(*_args, provider_override=None, **_kwargs):
        attempts.append(provider_override or "configured")
        time.sleep(0.02)
        return "No search results returned."

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setattr(ea, "WEB_SEARCH_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(ea, "_run_web_search", fake_search)
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: _HybridAnswerLLM())
    monkeypatch.setattr(ea, "retrieve_chunks", lambda *_args, **_kwargs: [])
    state = {"count": [0], "lock": asyncio.Lock(), "max": 2}

    asyncio.run(
        ea.answer_question_from_evidence(
            "What funding has Acme raised?",
            Company(name="Acme"),
            store=object(),
            use_web_search=True,
            web_search_state=state,
            route="company_specific",
        )
    )

    assert attempts == ["serper"]
    assert state["count"] == [1]


def test_tool_search_passes_bounded_deadline(monkeypatch) -> None:
    deadlines: list[float | None] = []

    class FakeProvider:
        def search(self, _query, *, deadline_seconds=None, **_kwargs):
            deadlines.append(deadline_seconds)
            return "Search Results for: q\n\nNo search results returned."

    from agent.pipeline.stages.answering import with_tool

    monkeypatch.setattr(with_tool, "get_provider", lambda **_kwargs: FakeProvider())
    tool = with_tool.IntelligentWebSearchTool(search_end_date="2026-08-07")

    tool._run("Apaleo market")

    assert deadlines == [with_tool.WEB_SEARCH_TIMEOUT_SEC]


class _HybridAnswerLLM:
    async def ainvoke(self, messages):  # noqa: ANN001
        from types import SimpleNamespace

        if "Web Search Results" in messages[-1].content:
            return SimpleNamespace(content="Apaleo has relevant public evidence [web].")
        return SimpleNamespace(content="Unknown from provided documents.")


def _run_planner_hybrid(monkeypatch, serper_result: str) -> tuple[dict, list[str]]:
    provider_attempts: list[str] = []

    def fake_search(
        *_args,
        provider_override=None,
        provider_deadline_seconds=None,
        **_kwargs,
    ):  # noqa: ANN001
        assert provider_deadline_seconds is not None
        assert provider_deadline_seconds > 0
        provider = provider_override or "configured"
        provider_attempts.append(provider)
        if provider == "serper":
            return serper_result
        return (
            "Search Results for: Apaleo funding\n\n"
            "1. Apaleo funding round — https://example.com/apaleo\n"
            "   Apaleo raised funding to expand its hotel software platform."
        )

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setenv("WEB_SEARCH_HEAVY_OVERRIDE", "always")
    monkeypatch.setattr(ea, "_run_web_search", fake_search)
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: _HybridAnswerLLM())
    monkeypatch.setattr(ea, "retrieve_chunks", lambda *_args, **_kwargs: [])

    _answer, provenance = asyncio.run(
        ea.answer_question_from_evidence(
            "What funding has Apaleo raised?",
            Company(name="Apaleo", domain="apaleo.com", industry="hospitality software"),
            store=object(),
            use_web_search=True,
            route="company_specific",
            aspect="general_company",
        )
    )
    return provenance, provider_attempts


def test_planner_uses_perplexity_only_when_serper_fails_quality_gate(monkeypatch) -> None:
    provenance, attempts = _run_planner_hybrid(
        monkeypatch,
        "No search results returned.",
    )

    assert attempts == ["serper", "sonar"]
    assert provenance["web_search_used"] is True
    assert "perplexity fallback" in provenance["web_search_decision"]


def test_empty_serper_response_triggers_perplexity_fallback(monkeypatch) -> None:
    provenance, attempts = _run_planner_hybrid(
        monkeypatch,
        (
            "Search Results for: Apaleo funding investors financing growth round "
            "strategic partnership expansion company facts\n\n"
            "No search results returned."
        ),
    )

    assert attempts == ["serper", "sonar"]
    assert provenance["web_search_used"] is True
    assert "perplexity fallback" in provenance["web_search_decision"]


def test_planner_does_not_call_perplexity_when_serper_passes_quality_gate(monkeypatch) -> None:
    provenance, attempts = _run_planner_hybrid(
        monkeypatch,
        (
            "Search Results for: Apaleo funding\n\n"
            "1. Apaleo funding — https://apaleo.com/news\n"
            "   Apaleo raised a growth funding round for its hotel software platform."
        ),
    )

    assert attempts == ["serper"]
    assert provenance["web_search_used"] is True
    assert "via serper" in provenance["web_search_decision"]


def test_shared_portfolio_bounds_search_objectives_before_answering(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_search(query, *_args, provider_override=None, **_kwargs):  # noqa: ANN001
        attempts.append(provider_override or "configured")
        return (
            f"Search Results for: {query}\n\n"
            f"1. Hospitality market evidence — https://example.com/{len(attempts)}\n"
            f"   Apaleo hospitality software {query} market size growth adoption demand "
            "customers opportunity evidence benchmark."
        )

    children = [
        QuestionNode(
            question=f"What evidence supports hospitality market segment {index} growth?",
            route="sector_market",
        )
        for index in range(18)
    ]
    trees = {
        "market": QuestionTree(
            aspect="market",
            root_node=QuestionNode(
                question="How attractive is the hospitality software market?",
                route="sector_market",
                sub_nodes=children,
            ),
        )
    }

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setenv("RDI_SHARED_WEB_EVIDENCE", "on")
    monkeypatch.setenv("WEB_SEARCH_CORE_BUDGET", "10")
    monkeypatch.setenv("WEB_SEARCH_RESERVE_BUDGET", "2")
    monkeypatch.setenv("WEB_SEARCH_HEAVY_OVERRIDE", "always")
    monkeypatch.setattr(ea, "_run_web_search", fake_search)
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: _HybridAnswerLLM())
    monkeypatch.setattr(ea, "retrieve_chunks", lambda *_args, **_kwargs: [])

    rows = asyncio.run(
        ea.answer_all_trees_from_evidence(
            trees,
            Company(name="Apaleo", domain="apaleo.com", industry="hospitality software"),
            store=object(),
            use_web_search=True,
        )
    )

    assert len(rows) == 19
    assert 1 <= len(attempts) <= 12
    assert set(attempts) == {"serper"}


def test_shared_portfolio_honors_targeted_mode_before_prefetch(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_search(query, *_args, provider_override=None, **_kwargs):  # noqa: ANN001
        attempts.append(provider_override or "configured")
        return (
            f"Search Results for: {query}\n\n"
            "1. Apaleo leadership — https://example.com/team\n"
            "   Apaleo founders and executives have hospitality software experience."
        )

    trees = {
        "team": QuestionTree(
            aspect="team",
            root_node=QuestionNode(
                question="Who are the founders and executives?",
                route="company_specific",
            ),
        )
    }
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setenv("RDI_SHARED_WEB_EVIDENCE", "on")
    monkeypatch.setattr(ea, "_run_web_search", fake_search)
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: _HybridAnswerLLM())
    monkeypatch.setattr(ea, "retrieve_chunks", lambda *_args, **_kwargs: [])

    asyncio.run(
        ea.answer_all_trees_from_evidence(
            trees,
            Company(name="Apaleo", domain="apaleo.com", industry="hospitality software"),
            store=object(),
            use_web_search=True,
            web_search_mode="targeted",
        )
    )

    assert attempts == []
    assert trees["team"].root_node.provenance["web_search_decision"].startswith(
        "skipped: targeted"
    )


def test_shared_portfolio_honors_global_provider_call_cap(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_search(query, *_args, provider_override=None, **_kwargs):  # noqa: ANN001
        attempts.append(provider_override or "configured")
        return (
            f"Search Results for: {query}\n\n"
            "1. Hospitality software evidence — https://example.com/market\n"
            "   Hospitality software market growth competitors customer adoption demand evidence."
        )

    trees = {
        "market": QuestionTree(
            aspect="market",
            root_node=QuestionNode(
                question="How large is the hospitality software market?",
                route="sector_market",
                sub_nodes=[
                    QuestionNode(question="Who are the competitors?", route="competitors"),
                    QuestionNode(question="What drives customer demand?", route="customer_need"),
                ],
            ),
        )
    }
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setenv("RDI_SHARED_WEB_EVIDENCE", "on")
    monkeypatch.setattr(ea, "MAX_PPLX_CALLS_PER_COMPANY", 1)
    monkeypatch.setattr(ea, "_run_web_search", fake_search)
    monkeypatch.setattr(ea, "create_llm", lambda temperature=0.2: _HybridAnswerLLM())
    monkeypatch.setattr(ea, "retrieve_chunks", lambda *_args, **_kwargs: [])

    asyncio.run(
        ea.answer_all_trees_from_evidence(
            trees,
            Company(name="Apaleo", domain="apaleo.com", industry="hospitality software"),
            store=object(),
            use_web_search=True,
        )
    )

    assert attempts == ["serper"]


def test_shared_portfolio_checks_job_control_before_paid_prefetch(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_search(*_args, **_kwargs):
        attempts.append("search")
        return "Search Results for: q\n\n1. Evidence — https://example.com\n   Evidence."

    async def stop_requested() -> None:
        raise RuntimeError("job stopped")

    trees = {
        "market": QuestionTree(
            aspect="market",
            root_node=QuestionNode(
                question="How large is the hospitality software market?",
                route="sector_market",
            ),
        )
    }
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "hybrid")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setenv("RDI_WEB_EVIDENCE_PLANNER", "on")
    monkeypatch.setenv("RDI_SHARED_WEB_EVIDENCE", "on")
    monkeypatch.setattr(ea, "_run_web_search", fake_search)

    with pytest.raises(RuntimeError, match="job stopped"):
        asyncio.run(
            ea.answer_all_trees_from_evidence(
                trees,
                Company(name="Apaleo", domain="apaleo.com"),
                store=object(),
                use_web_search=True,
                on_cooperate=stop_requested,
            )
        )

    assert attempts == []


def test_shared_portfolio_concurrency_never_exceeds_provider_throttle(monkeypatch) -> None:
    active = 0
    peak = 0

    async def fake_routed_search(**_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return None

    trees = {
        "market": QuestionTree(
            aspect="market",
            root_node=QuestionNode(
                question="How large is the hospitality software market?",
                route="sector_market",
                sub_nodes=[
                    QuestionNode(question="Who are the competitors?", route="competitors"),
                    QuestionNode(question="What drives demand?", route="customer_need"),
                ],
            ),
        )
    }
    monkeypatch.setenv("WEB_SEARCH_PREFETCH_CONCURRENCY", "4")
    monkeypatch.setenv("WEB_SEARCH_MAX_CONCURRENT", "1")
    monkeypatch.setattr(ea, "_run_quality_routed_search", fake_routed_search)

    async def run() -> None:
        await ea._prepare_shared_web_evidence(
            trees,
            Company(name="Apaleo", domain="apaleo.com", industry="hospitality software"),
            {"count": [0], "lock": asyncio.Lock(), "max": 12},
        )

    asyncio.run(run())

    assert peak == 1
