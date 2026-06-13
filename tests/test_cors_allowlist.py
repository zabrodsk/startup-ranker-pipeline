"""PR-A1: env-driven CORS allowlist (`_resolve_cors_settings`).

Default (``ALLOWED_ORIGINS`` unset) must stay byte-identical to the legacy
wildcard middleware. An explicit allowlist tightens origins, enables
credentials, and narrows methods/headers. Production with no allowlist keeps
the wildcard (byte-identical) but warns once.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import logging

from fastapi.testclient import TestClient

from web.app import _resolve_cors_settings, app


def test_default_unset_is_legacy_wildcard(monkeypatch):
    """ALLOWED_ORIGINS unset -> exactly the legacy wildcard middleware."""
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)

    origins, credentials, methods, headers = _resolve_cors_settings()

    assert origins == ["*"]
    assert credentials is False
    assert methods == ["*"]
    assert headers == ["*"]


def test_empty_or_whitespace_origins_treated_as_unset(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", " , ,  ")
    monkeypatch.delenv("APP_ENV", raising=False)

    origins, credentials, _methods, _headers = _resolve_cors_settings()

    assert origins == ["*"]
    assert credentials is False


def test_explicit_allowlist_tightens_and_enables_credentials(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com, https://foo.bar")

    origins, credentials, methods, headers = _resolve_cors_settings()

    assert origins == ["https://app.example.com", "https://foo.bar"]
    assert credentials is True
    assert methods == ["GET", "POST", "OPTIONS"]
    assert headers == ["Content-Type", "Authorization"]


def test_staging_unset_stays_wildcard_without_warning(monkeypatch, caplog):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("APP_ENV", "staging")

    with caplog.at_level(logging.WARNING):
        origins, credentials, _methods, _headers = _resolve_cors_settings()

    assert origins == ["*"]
    assert credentials is False
    assert not [r for r in caplog.records if "ALLOWED_ORIGINS" in r.getMessage()]


def test_production_unset_warns_once(monkeypatch, caplog):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    import web.app as app_module

    app_module._WARNED_CORS_WILDCARD_PROD.clear()
    with caplog.at_level(logging.WARNING):
        first = _resolve_cors_settings()
        _resolve_cors_settings()

    assert first[0] == ["*"]  # still wildcard (byte-identical)
    warnings = [r for r in caplog.records if "ALLOWED_ORIGINS" in r.getMessage()]
    assert len(warnings) == 1


def test_default_preflight_echoes_wildcard():
    """End-to-end: the wired middleware echoes the wildcard for the default build."""
    client = TestClient(app)

    resp = client.options(
        "/",
        headers={
            "Origin": "https://anything.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in resp.headers
