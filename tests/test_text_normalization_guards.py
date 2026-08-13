from agent.evidence_answering import (
    _answer_indicates_no_evidence,
    _coerce_text,
    _question_prefers_web_search,
    _web_results_add_value,
)
from agent.pipeline.stages.ranking import _normalize_text


def test_coerce_text_handles_list_content_blocks() -> None:
    value = [
        {"type": "text", "text": "Unknown from provided documents."},
        {"type": "other", "content": "fallback"},
    ]
    assert _coerce_text(value) == "Unknown from provided documents. fallback"


def test_coerce_text_drops_openai_reasoning_items() -> None:
    value = [
        {
            "id": "rs_0f429877ec46728d0069da280b92e881978e837bcd881f21d2",
            "summary": [],
            "type": "reasoning",
        },
        {"type": "text", "text": "Apaleo is a cloud-native hotel platform."},
    ]
    result = _coerce_text(value)
    assert result == "Apaleo is a cloud-native hotel platform."
    assert "reasoning" not in result
    assert "rs_" not in result


def test_coerce_text_drops_unknown_dict_shapes() -> None:
    # Unknown dict with no text/content field must not stringify into prose.
    value = [
        {"type": "unknown", "metadata": {"foo": "bar"}},
        {"type": "text", "text": "Hello world."},
    ]
    assert _coerce_text(value) == "Hello world."


def test_coerce_text_returns_empty_for_bare_reasoning_dict() -> None:
    value = {"type": "reasoning", "id": "rs_x", "summary": []}
    assert _coerce_text(value) == ""


def test_answer_indicates_no_evidence_accepts_non_string_payload() -> None:
    value = [{"text": "Unknown from provided documents."}]
    assert _answer_indicates_no_evidence(value) is True


def test_answer_indicates_no_evidence_handles_named_competitor_gap() -> None:
    text = (
        "The provided evidence does not name specific direct competitors to Apaleo. "
        "It does not identify these legacy providers by name."
    )
    assert _answer_indicates_no_evidence(text) is True


def test_answer_indicates_no_evidence_handles_missing_tam_specifics() -> None:
    text = (
        "The provided evidence does not contain a specific Total Addressable Market (TAM) figure "
        "or a formal market sizing analysis for Apaleo."
    )
    assert _answer_indicates_no_evidence(text) is True


def test_ranking_normalize_text_handles_list() -> None:
    assert _normalize_text(["Seed", "B2B SaaS"]) == "Seed B2B SaaS"


def test_question_prefers_web_search_for_competitor_and_market_sizing() -> None:
    assert _question_prefers_web_search("Who are the main competitors?") is True
    assert _question_prefers_web_search("What is the TAM / SAM / SOM for this company?") is True
    assert _question_prefers_web_search("What is the investment thesis?") is False


def test_web_search_domain_filter_is_broad_for_market_questions() -> None:
    from agent.dataclasses.company import Company
    from agent.evidence_answering import _web_search_domain_filter

    company = Company(
        name="Apaleo",
        industry="Hospitality software",
        tagline="",
        about="",
        domain="https://apaleo.com",
    )

    assert _web_search_domain_filter(company, "What is the TAM / SAM / SOM?") is None
    assert _web_search_domain_filter(company, "What integrations does Apaleo support?") == [
        "apaleo.com",
        "crunchbase.com",
        "linkedin.com",
    ]


def test_web_results_add_value_accepts_relevant_content() -> None:
    useful, reason = _web_results_add_value(
        question="What integrations does Apify support?",
        company_name="Apify",
        web_results=(
            "Apify offers integrations via API, Zapier, Make, and webhooks. "
            "The Apify platform connects to external SaaS tools."
        ),
    )
    assert useful is True
    assert "relevant" in reason


def test_web_results_add_value_rejects_noisy_failure_text() -> None:
    useful, reason = _web_results_add_value(
        question="What integrations does Apify support?",
        company_name="Apify",
        web_results="Web search failed: 429 rate limit exceeded.",
    )
    assert useful is False
    assert "failure" in reason


def test_build_web_search_query_enriches_competitor_intent() -> None:
    from agent.dataclasses.company import Company
    from agent.evidence_answering import _build_web_search_query

    query = _build_web_search_query(
        Company(name="Apaleo", industry="Hospitality software", tagline="", about="", domain=""),
        "Who are the main competitors?",
    )

    assert "Apaleo" in query
    assert "competitors" in query
    assert "alternatives" in query
    assert "rivals" in query


def test_build_web_search_query_removes_redundant_company_phrasing() -> None:
    from agent.dataclasses.company import Company
    from agent.evidence_answering import _build_web_search_query

    query = _build_web_search_query(
        Company(name="Apaleo", industry="Hospitality software", tagline="", about="", domain=""),
        "Who are the competitors of Apaleo and what are their moats compare to Apaleo?",
    )

    assert query == "Apaleo competitors moats comparison alternatives rivals"


# ---------------------------------------------------------------------------
# WEB_SEARCH_HEAVY_OVERRIDE gating (Sprint 1, W1/W2)
# ---------------------------------------------------------------------------

_COMPLETE_GROUNDED_ANSWER = (
    "The total addressable market is approximately $50B based on the market "
    "sizing slide in the deck, growing 12% annually."
)
_MARKET_QUESTION = "What is the TAM for this market segment?"


class _FakeAnswerLLM:
    """Minimal async LLM stub returning a fixed answer."""

    def __init__(self, answer_text: str) -> None:
        self._answer_text = answer_text

    async def ainvoke(self, _messages):  # noqa: ANN001, ANN202
        from types import SimpleNamespace

        return SimpleNamespace(content=self._answer_text)


def _run_answer_with_gating(
    monkeypatch,
    *,
    mode: str | None,
    is_root: bool,
    grounded_answer: str = _COMPLETE_GROUNDED_ANSWER,
    question: str = _MARKET_QUESTION,
) -> tuple[dict, list]:
    """Run answer_question_from_evidence with fakes; return (provenance, search calls)."""
    import asyncio

    import agent.evidence_answering as ea
    from agent.dataclasses.company import Company
    from agent.ingest.store import Chunk, EvidenceStore

    if mode is None:
        monkeypatch.delenv("WEB_SEARCH_HEAVY_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("WEB_SEARCH_HEAVY_OVERRIDE", mode)
    # Pin the import-time trigger constant so a local .env cannot flip it.
    monkeypatch.setattr(ea, "WEB_SEARCH_TRIGGER", "answer")

    chunk = Chunk(
        chunk_id="chunk_1",
        text="Market analysis: TAM $50B, growing 12% annually.",
        source_file="deck.pdf",
        page_or_slide=5,
    )
    monkeypatch.setattr(ea, "retrieve_chunks", lambda *a, **k: [chunk])
    monkeypatch.setattr(ea, "create_llm", lambda **k: _FakeAnswerLLM(grounded_answer))

    search_calls: list[tuple] = []

    def _fake_run_web_search(
        query,
        domain_filter=None,
        trigger_reason=None,
        gating_mode=None,
        cache_info=None,
        **_kwargs,
    ):  # noqa: ANN001
        search_calls.append((query, trigger_reason, gating_mode))
        return (
            "Apaleo operates in the hospitality software market. Analyst reports "
            "estimate the Apaleo-addressable TAM market segment at $50B."
        )

    monkeypatch.setattr(ea, "_run_web_search", _fake_run_web_search)

    company = Company(
        name="Apaleo", industry="Hospitality software", tagline="", about="", domain=""
    )
    store = EvidenceStore(startup_slug="apaleo")
    _answer, provenance = asyncio.run(
        ea.answer_question_from_evidence(
            question, company, store, use_web_search=True, is_root=is_root
        )
    )
    return provenance, search_calls


def test_root_only_blocks_heavy_search_for_child_with_complete_answer(monkeypatch) -> None:
    """Complete grounded answer + market question + root_only + child node -> no search."""
    provenance, search_calls = _run_answer_with_gating(
        monkeypatch, mode="root_only", is_root=False
    )
    assert search_calls == []
    assert provenance["web_search_used"] is False
    assert provenance["web_search_decision"] == "not needed"


def test_root_only_allows_heavy_search_for_root_node(monkeypatch) -> None:
    provenance, search_calls = _run_answer_with_gating(
        monkeypatch, mode="root_only", is_root=True
    )
    assert len(search_calls) == 1
    _query, trigger_reason, gating_mode = search_calls[0]
    assert trigger_reason == "question benefits from external web context"
    assert gating_mode == "root_only"
    assert provenance["web_search_used"] is True


def test_default_always_mode_fires_heavy_search_for_child_nodes(monkeypatch) -> None:
    """Env unset -> 'always' -> current behavior preserved (child nodes search too)."""
    _provenance, search_calls = _run_answer_with_gating(
        monkeypatch, mode=None, is_root=False
    )
    assert len(search_calls) == 1
    assert search_calls[0][2] == "always"


def test_never_mode_blocks_heavy_search_even_for_root(monkeypatch) -> None:
    provenance, search_calls = _run_answer_with_gating(
        monkeypatch, mode="never", is_root=True
    )
    assert search_calls == []
    assert provenance["web_search_decision"] == "not needed"


def test_never_mode_still_searches_when_documents_incomplete(monkeypatch) -> None:
    """The no-evidence trigger is independent of the heavy-override gate."""
    _provenance, search_calls = _run_answer_with_gating(
        monkeypatch,
        mode="never",
        is_root=False,
        grounded_answer="Unknown from provided documents.",
        question="What is the churn rate?",
    )
    assert len(search_calls) == 1
    assert search_calls[0][1] == "documents incomplete"


def test_invalid_heavy_override_falls_back_to_always(monkeypatch) -> None:
    _provenance, search_calls = _run_answer_with_gating(
        monkeypatch, mode="bogus-mode", is_root=False
    )
    assert len(search_calls) == 1
    assert search_calls[0][2] == "always"


def test_run_web_search_records_trigger_reason_metadata(monkeypatch) -> None:
    import agent.evidence_answering as ea
    import agent.web_search as web_search_module

    recorded: list[dict] = []

    class _FakeCollector:
        def record_perplexity_search(self, *, metadata=None, **_kwargs):  # noqa: ANN001
            recorded.append(metadata or {})

    class _FakeProvider:
        def search(self, query, domain_filter=None):  # noqa: ANN001
            return "Useful web results about Apaleo market size."

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "sonar")
    monkeypatch.setenv("PPLX_API_KEY", "test-key")
    monkeypatch.setattr(web_search_module, "get_provider", lambda **k: _FakeProvider())
    monkeypatch.setattr(ea, "get_current_collector", lambda: _FakeCollector())

    result = ea._run_web_search(
        "Apaleo TAM", None, "documents incomplete", "root_only"
    )

    assert "Apaleo" in result
    assert recorded == [
        {
            "query": "Apaleo TAM",
            "domain_filter": [],
            "trigger_reason": "documents incomplete",
            "gating_mode": "root_only",
        }
    ]


def test_run_costs_aggregate_perplexity_searches_by_trigger_reason() -> None:
    from agent.run_context import build_run_costs_from_model_executions

    rows = [
        {
            "service": "perplexity_search",
            "estimated_cost_usd": 0.005,
            "request_count": 1,
            "metadata": {"trigger_reason": "documents incomplete"},
        },
        {
            "service": "perplexity_search",
            "estimated_cost_usd": 0.005,
            "request_count": 1,
            "metadata": {"trigger_reason": "question benefits from external web context"},
        },
        {
            "service": "perplexity_search",
            "estimated_cost_usd": 0.005,
            "request_count": 1,
            "metadata": {},
        },
    ]

    costs = build_run_costs_from_model_executions(rows)

    assert costs["perplexity_search"]["requests"] == 3
    assert costs["perplexity_search"]["by_reason"] == {
        "documents incomplete": 1,
        "question benefits from external web context": 1,
        "unspecified": 1,
    }
