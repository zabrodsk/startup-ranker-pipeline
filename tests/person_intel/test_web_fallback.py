import asyncio
import threading

from agent.person_intel.models import PersonIntelSubject, PersonProfileJobRequest
from agent.person_intel.providers.web_fallback import WebFallbackProvider


def test_web_fallback_layered_queries_and_dedup(monkeypatch) -> None:
    monkeypatch.setenv("PERSON_INTEL_WEB_ENRICHMENT", "true")
    monkeypatch.setenv("PPLX_API_KEY", "dummy")

    calls: list[tuple[str, tuple[str, ...] | None, float | None, int]] = []

    class FakeProvider:
        def search(self, query, domain_filter=None, deadline_seconds=None):
            calls.append(
                (
                    query,
                    tuple(domain_filter) if isinstance(domain_filter, list) else None,
                    deadline_seconds,
                    threading.get_ident(),
                )
            )
            return "\n".join(
                [
                    "Search results for: foo",
                    "1. noise list item",
                    "https://example.com/direct-link-only",
                    "Jane Doe previously served as COO and scaled operations by 3x according to interviews https://example.com/a",
                    "Jane Doe previously served as COO and scaled operations by 3x according to interviews https://example.com/a",
                    "Discussed mission and values in a public keynote on sustainable finance https://example.com/b",
                ]
            )

    monkeypatch.setattr("agent.person_intel.providers.web_fallback.get_provider", lambda **kwargs: FakeProvider())

    provider = WebFallbackProvider()
    subject = PersonIntelSubject(
        primary_profile_url="https://www.linkedin.com/in/example",
        normalized_profile_url="https://www.linkedin.com/in/example",
        full_name="Jane Doe",
        current_company="Atomika",
        role="CEO",
    )
    req = PersonProfileJobRequest(primary_profile_url=subject.primary_profile_url)
    caller_thread = threading.get_ident()
    records = asyncio.run(provider.collect(req, subject))

    assert records
    assert any(domain_filter is None for _, domain_filter, _, _ in calls)
    assert any(
        domain_filter and "linkedin.com" in domain_filter
        for _, domain_filter, _, _ in calls
    )
    assert all(
        deadline_seconds and deadline_seconds <= 15
        for _, _, deadline_seconds, _ in calls
    )
    assert all(thread_id != caller_thread for _, _, _, thread_id in calls)
    assert all(r.snippet_or_field.startswith("web: ") for r in records)


def test_serper_fallback_preserves_organic_source_url(monkeypatch) -> None:
    monkeypatch.setenv("PERSON_INTEL_WEB_ENRICHMENT", "true")
    monkeypatch.setenv("SERPER_API_KEY", "dummy")
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    class FakeProvider:
        def search(self, *_args, **_kwargs):
            return (
                "Search Results for: Jane Doe\n\n"
                "1. Jane Doe profile — https://example.com/jane\n"
                "   Jane Doe previously led operations and scaled the business across international markets."
            )

    monkeypatch.setattr(
        "agent.person_intel.providers.web_fallback.get_provider",
        lambda **_kwargs: FakeProvider(),
    )
    subject = PersonIntelSubject(
        primary_profile_url="https://www.linkedin.com/in/example",
        normalized_profile_url="https://www.linkedin.com/in/example",
        full_name="Jane Doe",
        current_company="Atomika",
        role="CEO",
    )

    records = asyncio.run(
        WebFallbackProvider().collect(
            PersonProfileJobRequest(primary_profile_url=subject.primary_profile_url),
            subject,
        )
    )

    assert records
    assert records[0].url == "https://example.com/jane"
