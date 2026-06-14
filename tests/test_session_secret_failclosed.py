"""PR-A5: SESSION_SECRET fail-closed (close the default-secret auth bypass).

The stateless session token is ``raw_id.HMAC(SESSION_SECRET, raw_id)``. With a
hardcoded default secret, anyone could forge a valid token without the password.
After this change, an unset/empty SESSION_SECRET must NOT validate or issue
stateless signed sessions (fail closed); when it IS set, behavior is
byte-identical to today.

New-behavior drivers (fail on the old code): the empty-secret rejection and the
login 503. The configured-secret tests guard byte-identical-when-set.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import base64
import hashlib
import hmac

import pytest
from fastapi import HTTPException

import web.app as app_module
from web.app import LoginRequest, _check_session, login


def _sign(raw_id: str, secret: str) -> str:
    sig = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode("utf-8"), raw_id.encode("utf-8"), hashlib.sha256).digest()
        )
        .decode("utf-8")
        .rstrip("=")
    )
    return f"{raw_id}.{sig}"


def test_check_session_validates_token_when_secret_set(monkeypatch):
    monkeypatch.setattr(app_module, "SESSION_SECRET", "real-secret-value")
    token = _sign("rawid123", "real-secret-value")
    assert _check_session(token) is True


def test_check_session_rejects_hmac_token_when_secret_empty(monkeypatch):
    """NEW: an empty secret must never validate a signed token (fail closed)."""
    monkeypatch.setattr(app_module, "SESSION_SECRET", "")
    token = _sign("rawid123", "")  # "correctly" signed with the empty secret
    assert _check_session(token) is False


def test_check_session_rejects_token_forged_with_old_default(monkeypatch):
    """A token forged with the old hardcoded default is rejected when unset."""
    monkeypatch.setattr(app_module, "SESSION_SECRET", "")
    forged = _sign("attacker", "change-me-session-secret")
    assert _check_session(forged) is False


@pytest.mark.asyncio
async def test_login_fails_closed_when_secret_unset(monkeypatch):
    """NEW: login refuses to mint a session when signing is not configured."""
    monkeypatch.setattr(app_module, "APP_PASSWORD", "correct-pw")
    monkeypatch.setattr(app_module, "SESSION_SECRET", "")
    with pytest.raises(HTTPException) as exc:
        await login(LoginRequest(password="correct-pw"))
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_login_issues_valid_session_when_secret_set(monkeypatch):
    monkeypatch.setattr(app_module, "APP_PASSWORD", "correct-pw")
    monkeypatch.setattr(app_module, "SESSION_SECRET", "real-secret-value")
    monkeypatch.setattr(app_module, "_persist_sessions", lambda: None)
    result = await login(LoginRequest(password="correct-pw"))
    assert "session_id" in result
    assert _check_session(result["session_id"]) is True


@pytest.mark.asyncio
async def test_login_still_rejects_wrong_password(monkeypatch):
    monkeypatch.setattr(app_module, "APP_PASSWORD", "correct-pw")
    monkeypatch.setattr(app_module, "SESSION_SECRET", "real-secret-value")
    with pytest.raises(HTTPException) as exc:
        await login(LoginRequest(password="wrong-pw"))
    assert exc.value.status_code == 401
