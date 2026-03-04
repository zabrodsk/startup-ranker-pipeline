from agent.evidence_answering import (
    _answer_indicates_no_evidence,
    _coerce_text,
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


def test_ranking_normalize_text_handles_list() -> None:
    assert _normalize_text(["Seed", "B2B SaaS"]) == "Seed B2B SaaS"


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
