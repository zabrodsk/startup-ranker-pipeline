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
