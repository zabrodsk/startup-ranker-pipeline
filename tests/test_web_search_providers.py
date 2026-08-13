from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.web_search import providers


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_serper_search_uses_api_contract_and_formats_evidence(monkeypatch) -> None:
    calls: list[dict] = []

    def post(url, **kwargs):  # noqa: ANN001
        calls.append({"url": url, **kwargs})
        return _Response(
            {
                "answerBox": {
                    "title": "Apaleo",
                    "answer": "An open hospitality platform.",
                    "link": "https://apaleo.com/",
                },
                "organic": [
                    {
                        "title": "Apaleo platform",
                        "link": "https://apaleo.com/platform",
                        "snippet": "Cloud property management and APIs for hotels.",
                        "date": "Aug 1, 2026",
                    },
                    {
                        "title": "Apaleo funding",
                        "link": "https://tech.eu/apaleo-funding",
                        "snippet": "The company announced a funding round.",
                    },
                ],
            }
        )

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(
        providers.importlib,
        "import_module",
        lambda name: SimpleNamespace(post=post) if name == "requests" else None,
    )

    provider = providers.SerperSearchProvider(
        search_end_date="2026-08-07",
        max_results=5,
    )
    result = provider.search(
        "Apaleo hotel software",
        domain_filter=["apaleo.com", "tech.eu"],
    )

    assert calls[0]["url"] == "https://google.serper.dev/search"
    assert calls[0]["headers"]["X-API-KEY"] == "test-key"
    assert calls[0]["json"]["q"].startswith("Apaleo hotel software (")
    assert "site:apaleo.com" in calls[0]["json"]["q"]
    assert "after:2025-08-07" in calls[0]["json"]["q"]
    assert "before:2026-08-08" in calls[0]["json"]["q"]
    assert calls[0]["json"]["gl"] == "us"
    assert calls[0]["json"]["hl"] == "en"
    assert calls[0]["json"]["num"] == 5
    assert "Answer box: Apaleo" in result
    assert "An open hospitality platform." in result
    assert "1. Apaleo platform — https://apaleo.com/platform" in result
    assert "2. Apaleo funding — https://tech.eu/apaleo-funding" in result


def test_serper_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SERPER_API_KEY"):
        providers.SerperSearchProvider(search_end_date="2026-08-07")


def test_serper_rejects_documented_placeholder_key(monkeypatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "your_serper_api_key_here")
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    assert providers.resolve_provider_name("hybrid") is None
    with pytest.raises(ValueError, match="SERPER_API_KEY"):
        providers.SerperSearchProvider(search_end_date="2026-08-07")


def test_provider_factory_supports_serper(monkeypatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(
        providers.importlib,
        "import_module",
        lambda name: SimpleNamespace(post=lambda *_args, **_kwargs: None)
        if name == "requests"
        else None,
    )

    provider = providers.get_provider(
        search_end_date="2026-08-07",
        provider_name="serper",
    )

    assert isinstance(provider, providers.SerperSearchProvider)


def test_provider_factory_resolves_default_sonar_to_only_available_serper(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setattr(
        providers.importlib,
        "import_module",
        lambda name: SimpleNamespace(post=lambda *_args, **_kwargs: None)
        if name == "requests"
        else None,
    )

    provider = providers.get_provider(
        search_end_date="2026-08-07",
        provider_name="sonar",
    )

    assert isinstance(provider, providers.SerperSearchProvider)


def test_hybrid_provider_falls_back_when_serper_fails(monkeypatch) -> None:
    attempts: list[str] = []

    class _FailingSerper:
        def __init__(self, **_kwargs):
            return None

        def search(self, *_args, **_kwargs):
            attempts.append("serper")
            raise RuntimeError("Serper unavailable")

    class _WorkingSonar:
        def __init__(self, **_kwargs):
            return None

        def search(self, *_args, **_kwargs):
            attempts.append("sonar")
            return (
                "Search Results for: q\n\n"
                "1. Useful fallback evidence q — https://example.com/one\n"
                "   Detailed q market demand, customer adoption, and growth evidence.\n\n"
                "2. Independent q benchmark — https://example.com/two\n"
                "   Additional q competition, funding, and customer evidence."
            )

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(providers, "SerperSearchProvider", _FailingSerper)
    monkeypatch.setattr(providers, "SonarSearchProvider", _WorkingSonar)

    provider = providers.get_provider(
        search_end_date="2026-08-07", provider_name="hybrid"
    )

    assert "Useful fallback evidence" in provider.search("q")
    assert attempts == ["serper", "sonar"]


def test_hybrid_provider_falls_back_when_primary_results_are_unrelated(monkeypatch) -> None:
    attempts: list[str] = []

    class _UnrelatedSerper:
        def search(self, *_args, **_kwargs):
            attempts.append("serper")
            return (
                "Search Results for: Apaleo funding investors\n\n"
                "1. Acme funding — https://example.com/acme\n"
                "   Acme announced funding from investors for international expansion.\n\n"
                "2. Beta investors — https://example.com/beta\n"
                "   Beta secured funding from investors to accelerate growth and hiring."
            )

    class _RelevantSonar:
        def search(self, *_args, **_kwargs):
            attempts.append("sonar")
            return (
                "Search Results for: Apaleo funding\n\n"
                "1. Apaleo funding round — https://example.com/apaleo-round\n"
                "   Apaleo raised growth funding for its hospitality platform.\n\n"
                "2. Apaleo investors — https://example.com/apaleo-investors\n"
                "   The financing included existing and new investors in Apaleo."
            )

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    provider = providers.HybridSearchProvider(search_end_date="2026-08-07")
    provider._providers = [("serper", _UnrelatedSerper()), ("sonar", _RelevantSonar())]

    result = provider.search("Apaleo funding investors")

    assert "Apaleo raised growth funding" in result
    assert provider.last_provider_name == "sonar"
    assert attempts == ["serper", "sonar"]


def test_hybrid_provider_survives_serper_initialization_failure(monkeypatch) -> None:
    class _BrokenSerper:
        def __init__(self, **_kwargs):
            raise RuntimeError("Serper client unavailable")

    class _WorkingSonar:
        def __init__(self, **_kwargs):
            return None

        def search(self, *_args, **_kwargs):
            return (
                "Search Results for: q\n\n"
                "1. Sonar evidence q — https://example.com/one\n"
                "   Detailed q market demand, customer adoption, and growth evidence.\n\n"
                "2. Independent q benchmark — https://example.com/two\n"
                "   Additional q competition, funding, and customer evidence."
            )

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("PPLX_API_KEY", "pplx-key")
    monkeypatch.setattr(providers, "SerperSearchProvider", _BrokenSerper)
    monkeypatch.setattr(providers, "SonarSearchProvider", _WorkingSonar)

    provider = providers.get_provider(
        search_end_date="2026-08-07", provider_name="hybrid"
    )

    assert "Sonar evidence" in provider.search("q")
