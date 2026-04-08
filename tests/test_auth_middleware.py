"""Unit tests for Sprint 1 auth middleware (get_current_user, role guards).

Supabase calls are mocked so these tests run without any network access.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from web.app import app, get_current_user, _require_startup, _require_vc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_USER_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_FAKE_TOKEN = "fake.jwt.token"
_AUTH_HEADER = {"Authorization": f"Bearer {_FAKE_TOKEN}"}

_SUPABASE_USER = {"id": _FAKE_USER_ID, "email": "founder@company.com"}
_STARTUP_PROFILE = {
    "id": _FAKE_USER_ID,
    "role": "startup",
    "display_name": "Jane Founder",
    "organization": "Acme Inc",
    "approved": True,
    "created_at": "2026-04-03T10:00:00Z",
}
_VC_PROFILE = {**_STARTUP_PROFILE, "role": "vc"}
_ADMIN_PROFILE = {**_STARTUP_PROFILE, "role": "admin"}


def _patch_db(supabase_user=_SUPABASE_USER, user_profile=_STARTUP_PROFILE):
    """Return a context manager that patches both DB calls used by get_current_user."""
    patches = [
        patch("web.app.db") if True else None,
    ]
    return patches


# ---------------------------------------------------------------------------
# get_current_user (direct async tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_current_user_missing_header():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(authorization=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    import web.app as app_module
    mock_db = MagicMock()
    mock_db.is_configured.return_value = True
    mock_db.get_authenticated_supabase_user.return_value = None
    mock_db.get_user_profile.return_value = _STARTUP_PROFILE

    with patch.object(app_module, "db", mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(authorization=f"Bearer {_FAKE_TOKEN}")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_no_profile():
    import web.app as app_module
    mock_db = MagicMock()
    mock_db.is_configured.return_value = True
    mock_db.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    mock_db.get_user_profile.return_value = None

    with patch.object(app_module, "db", mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(authorization=f"Bearer {_FAKE_TOKEN}")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_current_user_valid():
    import web.app as app_module
    mock_db = MagicMock()
    mock_db.is_configured.return_value = True
    mock_db.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    mock_db.get_user_profile.return_value = _STARTUP_PROFILE

    with patch.object(app_module, "db", mock_db):
        user = await get_current_user(authorization=f"Bearer {_FAKE_TOKEN}")

    assert user.id == _FAKE_USER_ID
    assert user.email == "founder@company.com"
    assert user.role == "startup"
    assert user.approved is True
    assert user.display_name == "Jane Founder"


# ---------------------------------------------------------------------------
# Role guards
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_require_startup_passes_for_startup():
    """_require_startup accepts a CurrentUser with role='startup'."""
    from web.app import CurrentUser
    startup_user = CurrentUser(
        id=_FAKE_USER_ID, email="f@co.com", role="startup", approved=True, display_name=None
    )
    result = await _require_startup(user=startup_user)
    assert result.role == "startup"


@pytest.mark.asyncio
async def test_require_startup_blocks_vc():
    """_require_startup rejects a CurrentUser with role='vc'."""
    from web.app import CurrentUser
    vc_user = CurrentUser(
        id=_FAKE_USER_ID, email="f@co.com", role="vc", approved=True, display_name=None
    )
    with pytest.raises(HTTPException) as exc_info:
        await _require_startup(user=vc_user)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_vc_passes_for_vc():
    """_require_vc accepts a CurrentUser with role='vc'."""
    from web.app import CurrentUser
    vc_user = CurrentUser(
        id=_FAKE_USER_ID, email="f@co.com", role="vc", approved=True, display_name=None
    )
    result = await _require_vc(user=vc_user)
    assert result.role == "vc"


@pytest.mark.asyncio
async def test_require_vc_blocks_startup():
    """_require_vc rejects a CurrentUser with role='startup'."""
    from web.app import CurrentUser
    startup_user = CurrentUser(
        id=_FAKE_USER_ID, email="f@co.com", role="startup", approved=True, display_name=None
    )
    with pytest.raises(HTTPException) as exc_info:
        await _require_vc(user=startup_user)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# /api/auth/me via TestClient (integration-style, DB mocked)
# ---------------------------------------------------------------------------

def test_auth_me_unauthenticated():
    with TestClient(app) as client:
        resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_auth_me_authenticated():
    import web.app as app_module
    mock_db = MagicMock()
    mock_db.is_configured.return_value = True
    mock_db.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    mock_db.get_user_profile.return_value = _STARTUP_PROFILE
    mock_db.get_user_company_links.return_value = []

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/auth/me", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == _FAKE_USER_ID
    assert body["role"] == "startup"
    assert body["companies"] == []
