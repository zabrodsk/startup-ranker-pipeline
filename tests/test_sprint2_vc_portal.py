"""Integration-style tests for Sprint 2 VC portal endpoints.

All Supabase calls and the matching engine are mocked — no network required.
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
_FAKE_VC_PROFILE_ID = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
_FAKE_COMPANY_ID = "cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"
_FAKE_MATCH_ID = "dddddddd-eeee-ffff-aaaa-bbbbbbbbbbbb"
_FAKE_TOKEN = "fake.jwt.token"
_AUTH_HEADER = {"Authorization": f"Bearer {_FAKE_TOKEN}"}

_SUPABASE_USER = {"id": _FAKE_USER_ID, "email": "investor@vcfirm.com"}
_VC_PROFILE_ROW = {
    "id": _FAKE_USER_ID,
    "role": "vc",
    "display_name": "Jane Investor",
    "organization": "Acme Capital",
    "approved": True,
    "created_at": "2026-04-04T10:00:00Z",
}
_VC_PROFILE_DB = {
    "id": _FAKE_VC_PROFILE_ID,
    "user_id": _FAKE_USER_ID,
    "firm_name": "Acme Capital",
    "investment_thesis": "B2B SaaS at Series A",
    "min_strategy_fit": 60,
    "min_team": 55,
    "min_potential": 60,
    "active": True,
    "created_at": "2026-04-04T10:00:00Z",
    "updated_at": "2026-04-04T10:00:00Z",
}
_STARTUP_PROFILE_ROW = {
    "id": _FAKE_USER_ID,
    "role": "startup",
    "display_name": "Jane Founder",
    "organization": "Acme Inc",
    "approved": True,
    "created_at": "2026-04-04T10:00:00Z",
}
_COMPANY_LINK = {
    "user_id": _FAKE_USER_ID,
    "company_id": _FAKE_COMPANY_ID,
    "role_in_company": None,
    "verified_at": "2026-04-04T10:00:00Z",
    "companies": {
        "id": _FAKE_COMPANY_ID,
        "name": "Acme Inc",
        "industry": "SaaS",
        "fundraising": True,
    },
}
_MATCH_ROW = {
    "id": _FAKE_MATCH_ID,
    "strategy_fit_score": 78.0,
    "team_score": 72.0,
    "potential_score": 80.0,
    "composite_score": 76.5,
    "bucket": "watchlist",
    "status": "new",
    "created_at": "2026-04-04T12:00:00Z",
    "companies": {"id": _FAKE_COMPANY_ID, "name": "Acme Inc", "industry": "SaaS",
                  "tagline": "The best SaaS", "about": "We build stuff."},
    "vc_profiles": {"id": _FAKE_VC_PROFILE_ID, "firm_name": "Acme Capital"},
}


def _make_vc_mock_db():
    m = MagicMock()
    m.is_configured.return_value = True
    m.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    m.get_user_profile.return_value = _VC_PROFILE_ROW
    m.get_vc_profile.return_value = _VC_PROFILE_DB
    m.create_vc_profile.return_value = _VC_PROFILE_DB
    m.update_vc_profile.return_value = _VC_PROFILE_DB
    m.get_matches_for_vc.return_value = [_MATCH_ROW]
    m.update_match_status.return_value = {**_MATCH_ROW, "status": "interested"}
    return m


def _make_startup_mock_db():
    m = MagicMock()
    m.is_configured.return_value = True
    m.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    m.get_user_profile.return_value = _STARTUP_PROFILE_ROW
    m.get_user_company_links.return_value = [_COMPANY_LINK]
    m.get_matches_for_company.return_value = [_MATCH_ROW]
    m.set_company_fundraising.return_value = {
        "fundraising": True,
        "fundraising_updated_at": "2026-04-04T12:00:00Z",
    }
    return m


# ---------------------------------------------------------------------------
# GET /api/vc/profile
# ---------------------------------------------------------------------------

def test_vc_get_profile_success():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/vc/profile", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert body["firm_name"] == "Acme Capital"
    assert body["investment_thesis"] == "B2B SaaS at Series A"
    assert body["min_strategy_fit"] == 60


def test_vc_get_profile_unauthenticated():
    with TestClient(app) as client:
        resp = client.get("/api/vc/profile")
    assert resp.status_code == 401


def test_vc_get_profile_blocked_for_startup():
    import web.app as app_module
    mock_db = _make_startup_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/vc/profile", headers=_AUTH_HEADER)

    assert resp.status_code == 403


def test_vc_get_profile_not_found():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    mock_db.get_vc_profile.return_value = None

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/vc/profile", headers=_AUTH_HEADER)

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/vc/profile
# ---------------------------------------------------------------------------

def test_vc_put_profile_update_existing():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    updated = {**_VC_PROFILE_DB, "firm_name": "Updated Capital"}
    mock_db.update_vc_profile.return_value = updated

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/vc/profile",
                json={"firm_name": "Updated Capital"},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    assert resp.json()["firm_name"] == "Updated Capital"


def test_vc_put_profile_create_new():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    mock_db.get_vc_profile.return_value = None  # no profile yet
    mock_db.create_vc_profile.return_value = _VC_PROFILE_DB

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/vc/profile",
                json={"firm_name": "Acme Capital", "investment_thesis": "B2B SaaS"},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    mock_db.create_vc_profile.assert_called_once()


# ---------------------------------------------------------------------------
# PUT /api/vc/thesis
# ---------------------------------------------------------------------------

def test_vc_put_thesis_success():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    updated = {**_VC_PROFILE_DB, "investment_thesis": "New thesis text"}
    mock_db.update_vc_profile.return_value = updated

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/vc/thesis",
                json={"investment_thesis": "New thesis text"},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    assert resp.json()["investment_thesis"] == "New thesis text"


def test_vc_put_thesis_404_if_no_profile():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    mock_db.get_vc_profile.return_value = None

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/vc/thesis",
                json={"investment_thesis": "test"},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/vc/thresholds
# ---------------------------------------------------------------------------

def test_vc_put_thresholds_success():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    updated = {**_VC_PROFILE_DB, "min_strategy_fit": 70, "min_team": 65, "min_potential": 70}
    mock_db.update_vc_profile.return_value = updated

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/vc/thresholds",
                json={"min_strategy_fit": 70, "min_team": 65, "min_potential": 70},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["min_strategy_fit"] == 70
    assert body["min_team"] == 65
    assert body["min_potential"] == 70


def test_vc_put_thresholds_rejects_out_of_range():
    import web.app as app_module
    # Auth must pass so Pydantic body validation (ge/le constraints) is reached
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.put(
                "/api/vc/thresholds",
                json={"min_strategy_fit": 150, "min_team": 0, "min_potential": 0},
                headers=_AUTH_HEADER,
            )
    # Pydantic rejects min_strategy_fit > 100
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/vc/matches
# ---------------------------------------------------------------------------

def test_vc_get_matches_success():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/vc/matches", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["matches"]) == 1
    m = body["matches"][0]
    assert m["company"]["name"] == "Acme Inc"
    assert m["composite_score"] == 76.5
    assert m["bucket"] == "watchlist"


def test_vc_get_matches_empty():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    mock_db.get_matches_for_vc.return_value = []

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/vc/matches", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json()["matches"] == []


# ---------------------------------------------------------------------------
# POST /api/vc/matches/{id}/action
# ---------------------------------------------------------------------------

def test_vc_match_action_interested():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                f"/api/vc/matches/{_FAKE_MATCH_ID}/action",
                json={"action": "interested"},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    assert resp.json()["status"] == "interested"


def test_vc_match_action_invalid():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                f"/api/vc/matches/{_FAKE_MATCH_ID}/action",
                json={"action": "delete"},  # not a valid status
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/startup/matches
# ---------------------------------------------------------------------------

def test_startup_get_matches_success():
    import web.app as app_module
    mock_db = _make_startup_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/startup/matches", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["matches"]) == 1
    m = body["matches"][0]
    # Startup sees firm_name, NOT thesis
    assert m["firm_name"] == "Acme Capital"
    assert "investment_thesis" not in m
    assert m["composite_score"] == 76.5


def test_startup_matches_unauthenticated():
    with TestClient(app) as client:
        resp = client.get("/api/startup/matches")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# PUT /api/startup/fundraising — matching_triggered flag
# ---------------------------------------------------------------------------

def test_fundraising_toggle_on_triggers_matching():
    import web.app as app_module
    mock_db = _make_startup_mock_db()

    # Patch the background engine so we don't actually run the graph
    with patch.object(app_module, "db", mock_db), \
         patch("web.app._run_matching_background") as mock_bg:
        with TestClient(app) as client:
            resp = client.put(
                "/api/startup/fundraising",
                json={"fundraising": True},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["fundraising"] is True
    assert body["matching_triggered"] is True


def test_fundraising_toggle_off_no_matching():
    import web.app as app_module
    mock_db = _make_startup_mock_db()
    mock_db.set_company_fundraising.return_value = {
        "fundraising": False,
        "fundraising_updated_at": "2026-04-04T12:00:00Z",
    }

    with patch.object(app_module, "db", mock_db), \
         patch("web.app._run_matching_background") as mock_bg:
        with TestClient(app) as client:
            resp = client.put(
                "/api/startup/fundraising",
                json={"fundraising": False},
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["fundraising"] is False
    assert body["matching_triggered"] is False
    mock_bg.assert_not_called()
