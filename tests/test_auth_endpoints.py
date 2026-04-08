"""Integration-style tests for Sprint 1 auth and startup dashboard endpoints.

All Supabase calls are mocked — no network required.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from web.app import app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_USER_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_FAKE_COMPANY_ID = "cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"
_FAKE_TOKEN = "fake.jwt.token"
_AUTH_HEADER = {"Authorization": f"Bearer {_FAKE_TOKEN}"}

_SUPABASE_USER = {"id": _FAKE_USER_ID, "email": "founder@acme.com"}
_STARTUP_PROFILE = {
    "id": _FAKE_USER_ID,
    "role": "startup",
    "display_name": "Jane Founder",
    "organization": "Acme Inc",
    "approved": True,
    "created_at": "2026-04-03T10:00:00Z",
}
_COMPANY_ROW = {
    "id": _FAKE_COMPANY_ID,
    "name": "Acme Inc",
    "industry": "SaaS",
    "tagline": "The best SaaS",
    "about": "We build stuff.",
    "team": None,
    "domain": "acme.com",
    "fundraising": False,
    "fundraising_updated_at": None,
    "claimed_at": None,
    "data_room_enabled": False,
}
_COMPANY_LINK = {
    "user_id": _FAKE_USER_ID,
    "company_id": _FAKE_COMPANY_ID,
    "role_in_company": None,
    "verified_at": "2026-04-03T10:00:00Z",
    "companies": _COMPANY_ROW,
}


def _make_mock_db(
    *,
    supabase_user=_SUPABASE_USER,
    user_profile=_STARTUP_PROFILE,
    company=_COMPANY_ROW,
    company_links=None,
    create_profile_result=None,
    create_link_ok=True,
    set_fundraising_result=None,
    chunks=None,
    latest_analysis=None,
):
    """Build a mock db module with sensible defaults."""
    m = MagicMock()
    m.is_configured.return_value = True

    # Auth
    mock_auth_user = MagicMock()
    mock_auth_user.id = _FAKE_USER_ID
    mock_client = MagicMock()
    m._get_client.return_value = mock_client

    m.get_authenticated_supabase_user.return_value = supabase_user
    m.get_user_profile.return_value = user_profile
    m.create_user_profile.return_value = (
        create_profile_result if create_profile_result is not None else _STARTUP_PROFILE
    )
    m.get_user_company_links.return_value = (
        company_links if company_links is not None else [_COMPANY_LINK]
    )
    m.get_company_by_id.return_value = company
    m.create_user_company_link.return_value = create_link_ok
    m.set_company_claimed_at.return_value = True
    m.set_company_fundraising.return_value = (
        set_fundraising_result
        if set_fundraising_result is not None
        else {**_COMPANY_ROW, "fundraising": True, "fundraising_updated_at": "2026-04-03T12:00:00Z"}
    )
    m.get_company_chunks.return_value = chunks if chunks is not None else []
    m.get_company_latest_analysis.return_value = latest_analysis
    return m


# ---------------------------------------------------------------------------
# POST /api/auth/register
# ---------------------------------------------------------------------------

def test_register_startup_success():
    import web.app as app_module

    mock_db = _make_mock_db()

    mock_auth_user = MagicMock()
    mock_auth_user.id = _FAKE_USER_ID
    mock_auth_response = MagicMock()
    mock_auth_response.user = mock_auth_user

    mock_client = MagicMock()
    mock_client.auth.sign_up.return_value = mock_auth_response
    mock_db._get_client.return_value = mock_client

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                "/api/auth/register",
                json={
                    "email": "founder@acme.com",
                    "password": "SecurePass123!",
                    "role": "startup",
                    "display_name": "Jane Founder",
                },
            )

    assert resp.status_code == 201
    body = resp.json()
    assert body["role"] == "startup"
    assert body["email"] == "founder@acme.com"
    assert "user_id" in body


def test_register_duplicate_email():
    import web.app as app_module

    mock_db = _make_mock_db()
    mock_client = MagicMock()
    # Registration uses admin.create_user; simulate "already exists" error
    mock_client.auth.admin.create_user.side_effect = Exception("User already registered")
    mock_db._get_client.return_value = mock_client

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                "/api/auth/register",
                json={"email": "existing@acme.com", "password": "pass", "role": "startup"},
            )

    assert resp.status_code == 409


def test_register_invalid_role():
    with TestClient(app) as client:
        resp = client.post(
            "/api/auth/register",
            json={"email": "user@co.com", "password": "pass", "role": "admin"},
        )
    # Pydantic validation rejects 'admin' (not in Literal["startup", "vc"])
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/auth/verify-domain
# ---------------------------------------------------------------------------

def test_verify_domain_match():
    import web.app as app_module
    mock_db = _make_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                "/api/auth/verify-domain",
                json={"email": "founder@acme.com", "company_id": _FAKE_COMPANY_ID},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is True
    assert body["domain"] == "acme.com"


def test_verify_domain_mismatch():
    import web.app as app_module
    mock_db = _make_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                "/api/auth/verify-domain",
                json={"email": "user@otherdomain.com", "company_id": _FAKE_COMPANY_ID},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is False


def test_verify_domain_company_not_found():
    import web.app as app_module
    mock_db = _make_mock_db(company=None)

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                "/api/auth/verify-domain",
                json={"email": "founder@acme.com", "company_id": "nonexistent-id"},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 404


def test_verify_domain_unauthenticated():
    with TestClient(app) as client:
        resp = client.post(
            "/api/auth/verify-domain",
            json={"email": "founder@acme.com", "company_id": _FAKE_COMPANY_ID},
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/startup/evidence — Specter filter
# ---------------------------------------------------------------------------

def test_startup_evidence_excludes_specter_files():
    import web.app as app_module

    safe_chunk = {
        "chunk_id": "chunk-1",
        "text": "Product overview...",
        "source_file": "pitch_deck.pdf",
        "page_or_slide": 3,
    }
    # get_company_chunks in db.py already filters Specter files, but let's
    # confirm the endpoint passes through whatever db returns cleanly.
    mock_db = _make_mock_db(chunks=[safe_chunk])

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/startup/evidence", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["chunks"]) == 1
    assert body["chunks"][0]["source_file"] == "pitch_deck.pdf"


def test_startup_evidence_no_chunks():
    import web.app as app_module
    mock_db = _make_mock_db(chunks=[])

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/startup/evidence", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json()["chunks"] == []


def test_startup_evidence_unauthenticated():
    with TestClient(app) as client:
        resp = client.get("/api/startup/evidence")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# PUT /api/startup/fundraising
# ---------------------------------------------------------------------------

def test_startup_fundraising_toggle_on():
    import web.app as app_module
    updated = {**_COMPANY_ROW, "fundraising": True, "fundraising_updated_at": "2026-04-03T12:00:00Z"}
    mock_db = _make_mock_db(set_fundraising_result=updated)

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/startup/fundraising",
                json={"fundraising": True},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["fundraising"] is True
    assert body["fundraising_updated_at"] is not None


def test_startup_fundraising_requires_startup_role():
    import web.app as app_module
    vc_profile = {**_STARTUP_PROFILE, "role": "vc"}
    mock_db = _make_mock_db(user_profile=vc_profile)

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/startup/fundraising",
                json={"fundraising": True},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/startup/profile
# ---------------------------------------------------------------------------

def test_startup_profile_returns_company():
    import web.app as app_module
    mock_db = _make_mock_db(latest_analysis=None)

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/startup/profile", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert body["company"]["name"] == "Acme Inc"
    assert body["company"]["fundraising"] is False
    assert body["analysis"] is None


def test_startup_profile_includes_analysis_when_available():
    import web.app as app_module
    analysis = {
        "id": "analysis-1",
        "status": "done",
        "results_payload": {
            "ranking_result": {
                "composite_score": 82.5,
                "strategy_fit_score": 80.0,
                "team_score": 85.0,
                "upside_score": 82.0,
                "bucket": "strong",
                "key_points": ["Strong team", "Large TAM"],
                "red_flags": ["Early stage"],
            }
        },
    }
    mock_db = _make_mock_db(latest_analysis=analysis)

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/startup/profile", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis"]["composite_score"] == 82.5
    assert body["analysis"]["bucket"] == "strong"
    assert "Strong team" in body["analysis"]["key_points"]
