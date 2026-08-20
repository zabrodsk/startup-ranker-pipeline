from __future__ import annotations

from io import BytesIO

import pytest

from agent.ingest.web_fallback import (
    HomepageFallbackError,
    fetch_company_homepage,
    quota_web_fallback_allowed,
)


class _Response:
    def __init__(self, html: str, *, url: str = "https://acme.example/") -> None:
        self._body = BytesIO(html.encode("utf-8"))
        self._url = url
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        return None


def _homepage(name: str = "Acme Robotics") -> str:
    detail = (
        "We provide production workflow software for industrial engineering teams. "
        "Customers use the platform to coordinate complex work, reduce downtime, "
        "and document measurable operating outcomes across European facilities. "
    )
    return f"<html><head><title>{name}</title><meta name='description' content='{detail}'></head><body><h1>{name}</h1><p>{detail * 3}</p></body></html>"


def test_quota_fallback_is_fail_closed_and_leadgen_machine_only(monkeypatch) -> None:
    config = {"source": "leadgen_machine"}
    monkeypatch.delenv("SPECTER_QUOTA_WEB_FALLBACK_ENABLED", raising=False)
    assert not quota_web_fallback_allowed(
        config,
        use_web_search=True,
        specter_url="acme.example",
        expected_name="Acme Robotics",
    )

    monkeypatch.setenv("SPECTER_QUOTA_WEB_FALLBACK_ENABLED", "true")
    assert quota_web_fallback_allowed(
        config,
        use_web_search=True,
        specter_url="acme.example",
        expected_name="Acme Robotics",
    )
    assert quota_web_fallback_allowed(
        config,
        use_web_search=True,
        specter_url="acme.example",
        expected_name=None,
    )
    assert not quota_web_fallback_allowed(
        {"source": "manual"},
        use_web_search=True,
        specter_url="acme.example",
        expected_name="Acme Robotics",
    )
    assert not quota_web_fallback_allowed(
        config,
        use_web_search=False,
        specter_url="acme.example",
        expected_name="Acme Robotics",
    )


def test_fetch_company_homepage_builds_identity_checked_evidence() -> None:
    def opener(_request, timeout):
        assert timeout == 15
        return _Response(_homepage())

    company, store = fetch_company_homepage(
        "https://acme.example",
        expected_name="Acme Robotics",
        opener=opener,
    )

    assert company.name == "Acme Robotics"
    assert company.domain == "acme.example"
    assert company.company_url == "https://acme.example/"
    assert store.startup_slug == "acme-robotics"
    assert store.chunks
    assert all(chunk.source_file == "https://acme.example/" for chunk in store.chunks)
    assert "production workflow software" in "\n".join(store.texts)


def test_fetch_company_homepage_uses_canonical_domain_when_name_is_unavailable() -> None:
    def opener(_request, timeout):
        return _Response(_homepage())

    company, store = fetch_company_homepage(
        "https://acme.example",
        opener=opener,
    )

    assert company.name == "acme.example"
    assert company.domain == "acme.example"
    assert store.startup_slug == "acme-example"


def test_fetch_company_homepage_rejects_identity_mismatch() -> None:
    def opener(_request, timeout):
        return _Response(_homepage("Different Holdings"))

    with pytest.raises(HomepageFallbackError, match="corroborate"):
        fetch_company_homepage(
            "https://acme.example",
            expected_name="Target Robotics",
            opener=opener,
        )


def test_fetch_company_homepage_rejects_cross_host_redirect() -> None:
    def opener(_request, timeout):
        return _Response(_homepage(), url="https://login.vendor.example/")

    with pytest.raises(HomepageFallbackError, match="redirected"):
        fetch_company_homepage(
            "https://acme.example",
            expected_name="Acme Robotics",
            opener=opener,
        )
