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
