import asyncio

from agent.person_intel.models import PersonIntelSubject, PersonProfileJobRequest
from agent.person_intel.providers.web_fallback import WebFallbackProvider


def test_web_fallback_layered_queries_and_dedup(monkeypatch) -> None:
    monkeypatch.setenv("PERSON_INTEL_WEB_ENRICHMENT", "true")
    monkeypatch.setenv("PPLX_API_KEY", "dummy")

    calls: list[tuple[str, tuple[str, ...] | None]] = []

    class FakeProvider:
        def search(self, query, domain_filter=None):
            calls.append((query, tuple(domain_filter) if isinstance(domain_filter, list) else None))
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
    records = asyncio.run(provider.collect(req, subject))

    assert records
    assert any(domain_filter is None for _, domain_filter in calls)
    assert any(domain_filter and "linkedin.com" in domain_filter for _, domain_filter in calls)
    assert all(r.snippet_or_field.startswith("web: ") for r in records)
