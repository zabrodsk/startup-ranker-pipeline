"""Integration-style tests for Sprint 3 debate engine endpoints.

All Supabase calls and LLM calls are mocked — no network required.
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
from fastapi.testclient import TestClient

from web.app import app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_USER_ID   = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_FAKE_VC_ID     = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
_FAKE_COMPANY_ID= "cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"
_FAKE_MATCH_ID  = "dddddddd-eeee-ffff-aaaa-bbbbbbbbbbbb"
_FAKE_DEBATE_ID = "eeeeeeee-ffff-aaaa-bbbb-cccccccccccc"
_FAKE_TOKEN     = "fake.jwt.token"
_AUTH_HEADER    = {"Authorization": f"Bearer {_FAKE_TOKEN}"}

_SUPABASE_USER = {"id": _FAKE_USER_ID, "email": "investor@vcfirm.com"}
_VC_USER_PROFILE = {
    "id": _FAKE_USER_ID, "role": "vc", "display_name": "Jane Investor",
    "organization": "Acme Capital", "approved": True, "created_at": "2026-04-04T10:00:00Z",
}
_STARTUP_USER_PROFILE = {
    "id": _FAKE_USER_ID, "role": "startup", "display_name": "Jane Founder",
    "organization": "Acme Inc", "approved": True, "created_at": "2026-04-04T10:00:00Z",
}
_VC_PROFILE_DB = {
    "id": _FAKE_VC_ID, "user_id": _FAKE_USER_ID, "firm_name": "Acme Capital",
    "investment_thesis": "B2B SaaS at Series A",
    "min_strategy_fit": 30, "min_team": 30, "min_potential": 30,
    "active": True, "created_at": "2026-04-04T10:00:00Z", "updated_at": "2026-04-04T10:00:00Z",
}
_MATCH_ROW = {
    "id": _FAKE_MATCH_ID, "strategy_fit_score": 78.0, "team_score": 72.0,
    "potential_score": 80.0, "composite_score": 76.5, "bucket": "watchlist",
    "status": "interested", "created_at": "2026-04-04T12:00:00Z",
    "companies": {"id": _FAKE_COMPANY_ID, "name": "Acme Inc", "industry": "SaaS",
                  "tagline": "Best SaaS", "about": "We build stuff."},
}
_DEBATE_ROW = {
    "id": _FAKE_DEBATE_ID, "match_id": _FAKE_MATCH_ID,
    "company_id": _FAKE_COMPANY_ID, "vc_profile_id": _FAKE_VC_ID,
    "status": "active", "current_round": 1, "max_rounds": 3,
    "summary": None, "created_at": "2026-04-04T12:00:00Z", "updated_at": "2026-04-04T12:00:00Z",
}
_DEBATE_MESSAGE = {
    "id": "msg-1", "debate_id": _FAKE_DEBATE_ID, "round": 1,
    "speaker": "vc_agent", "content": "What is your ARR?", "citations": [],
    "created_at": "2026-04-04T12:01:00Z",
}
_COMPANY_ROW = {
    "id": _FAKE_COMPANY_ID, "name": "Acme Inc", "industry": "SaaS",
    "domain": "acme.com", "fundraising": True,
}
_COMPANY_LINK = {
    "user_id": _FAKE_USER_ID, "company_id": _FAKE_COMPANY_ID,
    "role_in_company": None, "verified_at": "2026-04-04T10:00:00Z",
    "companies": _COMPANY_ROW,
}


def _make_vc_mock_db(*, debate=None, debate_messages=None, existing_debate=None):
    m = MagicMock()
    m.is_configured.return_value = True
    m.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    m.get_user_profile.return_value = _VC_USER_PROFILE
    m.get_vc_profile.return_value = _VC_PROFILE_DB
    m.get_matches_for_vc.return_value = [_MATCH_ROW]
    m.update_match_status.return_value = {**_MATCH_ROW, "status": "in_debate"}
    m.get_debate_by_match.return_value = existing_debate
    m.create_debate.return_value = debate or _DEBATE_ROW
    m.get_debate_by_id.return_value = debate or _DEBATE_ROW
    m.get_debate_messages.return_value = debate_messages or [_DEBATE_MESSAGE]
    m.get_debates_for_vc.return_value = [_DEBATE_ROW]
    m.get_company_by_id.return_value = _COMPANY_ROW
    m.get_company_chunks.return_value = []
    m.get_company_latest_analysis.return_value = None
    m.pause_debate.return_value = True
    m.resume_debate.return_value = True
    return m


def _make_startup_mock_db():
    m = MagicMock()
    m.is_configured.return_value = True
    m.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    m.get_user_profile.return_value = _STARTUP_USER_PROFILE
    m.get_user_company_links.return_value = [_COMPANY_LINK]
    m.get_debates_for_company.return_value = [_DEBATE_ROW]
    return m


# ---------------------------------------------------------------------------
# POST /api/debates — create debate
# ---------------------------------------------------------------------------

def test_create_debate_success():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    # Patch background task so debate doesn't actually run
    with patch.object(app_module, "db", mock_db), \
         patch("web.app.asyncio.create_task") as mock_task:
        with TestClient(app) as client:
            resp = client.post(
                f"/api/debates?match_id={_FAKE_MATCH_ID}",
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == _FAKE_DEBATE_ID
    assert body["status"] == "active"


def test_create_debate_idempotent_returns_existing():
    """If a debate already exists for this match, return it without creating a new one."""
    import web.app as app_module
    mock_db = _make_vc_mock_db(existing_debate=_DEBATE_ROW)

    with patch.object(app_module, "db", mock_db), \
         patch("web.app.asyncio.create_task"):
        with TestClient(app) as client:
            resp = client.post(
                f"/api/debates?match_id={_FAKE_MATCH_ID}",
                headers=_AUTH_HEADER,
            )

    assert resp.status_code == 201
    mock_db.create_debate.assert_not_called()


def test_create_debate_unauthenticated():
    with TestClient(app) as client:
        resp = client.post(f"/api/debates?match_id={_FAKE_MATCH_ID}")
    assert resp.status_code == 401


def test_create_debate_blocked_for_startup():
    import web.app as app_module
    mock_db = _make_startup_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                f"/api/debates?match_id={_FAKE_MATCH_ID}",
                headers=_AUTH_HEADER,
            )
    assert resp.status_code == 403


def test_create_debate_match_not_found():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    mock_db.get_matches_for_vc.return_value = []  # no matches

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(
                f"/api/debates?match_id=nonexistent-id",
                headers=_AUTH_HEADER,
            )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/debates/{id}
# ---------------------------------------------------------------------------

def test_get_debate_success():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get(f"/api/debates/{_FAKE_DEBATE_ID}", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == _FAKE_DEBATE_ID
    assert len(body["messages"]) == 1
    assert body["messages"][0]["speaker"] == "vc_agent"


def test_get_debate_not_found():
    import web.app as app_module
    mock_db = _make_vc_mock_db()
    mock_db.get_debate_by_id.return_value = None

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get(f"/api/debates/nonexistent", headers=_AUTH_HEADER)

    assert resp.status_code == 404


def test_get_debate_unauthenticated():
    with TestClient(app) as client:
        resp = client.get(f"/api/debates/{_FAKE_DEBATE_ID}")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/debates/{id}/pause and /resume
# ---------------------------------------------------------------------------

def test_pause_debate():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(f"/api/debates/{_FAKE_DEBATE_ID}/pause", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"


def test_resume_debate():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db), \
         patch("web.app.asyncio.create_task"):
        with TestClient(app) as client:
            resp = client.post(f"/api/debates/{_FAKE_DEBATE_ID}/resume", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


def test_resume_completed_debate_rejected():
    import web.app as app_module
    completed = {**_DEBATE_ROW, "status": "completed"}
    mock_db = _make_vc_mock_db(debate=completed)

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.post(f"/api/debates/{_FAKE_DEBATE_ID}/resume", headers=_AUTH_HEADER)

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/vc/debates
# ---------------------------------------------------------------------------

def test_vc_list_debates():
    import web.app as app_module
    mock_db = _make_vc_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/vc/debates", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    assert len(resp.json()["debates"]) == 1


# ---------------------------------------------------------------------------
# GET /api/startup/debates
# ---------------------------------------------------------------------------

def test_startup_list_debates():
    import web.app as app_module
    mock_db = _make_startup_mock_db()

    with patch.object(app_module, "db", mock_db):
        with TestClient(app) as client:
            resp = client.get("/api/startup/debates", headers=_AUTH_HEADER)

    assert resp.status_code == 200
    assert len(resp.json()["debates"]) == 1


def test_startup_debates_unauthenticated():
    with TestClient(app) as client:
        resp = client.get("/api/startup/debates")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Agent unit tests (no LLM calls — mock the LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_agent_returns_content():
    """Startup agent returns content and citations from retrieved chunks."""
    from agent.debate.agents import startup_agent_turn
    from agent.ingest.store import Chunk, EvidenceStore

    store = EvidenceStore(startup_slug="test")
    store.chunks = [
        Chunk(chunk_id="c1", text="ARR is €2M as of Q4 2025.", source_file="deck.pdf", page_or_slide=3),
        Chunk(chunk_id="c2", text="NRR is 120%.", source_file="deck.pdf", page_or_slide=4),
    ]

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Our ARR is €2M [c1] with strong 120% NRR [c2]."
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agent.debate.agents.create_llm", return_value=mock_llm):
        result = await startup_agent_turn(
            company_name="Acme Inc",
            vc_argument="What is your ARR and retention?",
            store=store,
        )

    assert "content" in result
    assert len(result["citations"]) > 0
    assert result["citations"][0]["chunk_id"] in ("c1", "c2")


@pytest.mark.asyncio
async def test_vc_agent_returns_content():
    """VC agent returns a challenge using the thesis and analysis summary."""
    from agent.debate.agents import vc_agent_turn

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Your Series B stage is outside our Seed mandate."
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agent.debate.agents.create_llm", return_value=mock_llm):
        result = await vc_agent_turn(
            company_name="Acme Inc",
            vc_thesis="We invest in Seed-stage B2B SaaS.",
            analysis_summary={
                "strategy_fit_score": 41.0,
                "team_score": 70.0,
                "upside_score": 68.0,
                "strategy_fit_summary": "Poor fit — Series B stage.",
                "team_summary": "Strong founders.",
                "potential_summary": "Good market.",
                "red_flags": ["Series B beyond Seed mandate"],
            },
            startup_argument="We are a strong fit for your thesis.",
            current_round=1,
            max_rounds=3,
        )

    assert "content" in result
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_debate_summary_generation():
    """Summary generator returns non-empty text."""
    from agent.debate.agents import generate_debate_summary

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Summary: VC raised stage concerns; startup defended with traction data."
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    messages = [
        {"round": 1, "speaker": "vc_agent", "content": "Stage mismatch concern."},
        {"round": 1, "speaker": "startup_agent", "content": "We have strong traction."},
    ]

    with patch("agent.debate.agents.create_llm", return_value=mock_llm):
        summary = await generate_debate_summary(
            company_name="Acme Inc",
            messages=messages,
            num_rounds=1,
        )

    assert isinstance(summary, str)
    assert len(summary) > 0
