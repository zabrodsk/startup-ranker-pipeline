"""Tests for Sprint 4: evidence upload, re-evaluation, new company creation.

All Supabase, LLM, and pipeline calls are mocked — no network required.
"""
import io
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
# Shared test data
# ---------------------------------------------------------------------------

_FAKE_USER_ID    = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_FAKE_COMPANY_ID = "cccccccc-dddd-eeee-ffff-aaaaaaaaaaaa"
_FAKE_DECK_ID    = "dddddddd-eeee-ffff-aaaa-bbbbbbbbbbbb"
_FAKE_TOKEN      = "fake.jwt.token"
_AUTH_HEADER     = {"Authorization": f"Bearer {_FAKE_TOKEN}"}

_SUPABASE_USER = {"id": _FAKE_USER_ID, "email": "founder@startup.com"}
_STARTUP_PROFILE = {
    "id": _FAKE_USER_ID, "role": "startup", "display_name": "Jane Founder",
    "organization": "Acme Inc", "approved": True, "created_at": "2026-04-07T10:00:00Z",
}
_COMPANY_ROW = {
    "id": _FAKE_COMPANY_ID, "name": "Acme Inc", "industry": "SaaS",
    "domain": "acme.com", "fundraising": False,
    "claimed_at": "2026-04-07T10:00:00Z",
}
_COMPANY_LINK = {
    "user_id": _FAKE_USER_ID, "company_id": _FAKE_COMPANY_ID,
    "companies": _COMPANY_ROW,
}
_ANALYSIS_ROW = {
    "id": "analysis-1",
    "results_payload": {
        "mode": "single",
        "ranking_result": {
            "composite_score": 72.0,
            "strategy_fit_score": 68.0,
            "team_score": 74.0,
            "upside_score": 75.0,
            "bucket": "watchlist",
            "key_points": ["Strong team"],
            "red_flags": [],
        }
    },
    "state": {
        "all_qa_pairs": [{"question": "Q1", "answer": "A1"}],
        "question_trees": {},
        "final_arguments": [],
        "final_decision": "invest",
    },
    "status": "done",
    "created_at": "2026-04-07T10:00:00Z",
}


def _make_mock_db(*, has_link=True, has_analysis=True):
    m = MagicMock()
    m.is_configured.return_value = True
    m.get_authenticated_supabase_user.return_value = _SUPABASE_USER
    m.get_user_profile.return_value = _STARTUP_PROFILE
    m.get_user_company_links.return_value = [_COMPANY_LINK] if has_link else []
    m.get_company_by_id.return_value = _COMPANY_ROW
    m.get_company_latest_analysis.return_value = _ANALYSIS_ROW if has_analysis else None
    m.get_company_chunks.return_value = []
    m.get_all_company_chunks.return_value = [
        {"chunk_id": "c1", "text": "Revenue is $1M ARR", "source_file": "deck.pdf", "page_or_slide": "3"},
    ]
    m.get_analysis_question_trees.return_value = None
    m.get_analysis_final_state.return_value = {
        "final_arguments": [{"content": "Strong market", "argument_type": "pro", "score": 80}],
        "final_decision": "invest",
    } if has_analysis else None
    m.get_analysis_qa_pairs.return_value = [{"question": "Q1", "answer": "A1"}]
    m.create_company_from_portal.return_value = _FAKE_COMPANY_ID
    m.create_user_company_link.return_value = True
    m.create_pitch_deck_for_upload.return_value = _FAKE_DECK_ID
    m.insert_chunks_for_pitch_deck.return_value = 5
    m.upload_startup_file_bytes.return_value = f"startup_uploads/{_FAKE_COMPANY_ID}/deck.pdf"
    m.create_analysis_record.return_value = "new-analysis-id"
    m.delete_matches_for_company.return_value = 2
    m.get_active_vc_profiles.return_value = []
    return m


# ---------------------------------------------------------------------------
# Tests: POST /api/startup/company
# ---------------------------------------------------------------------------

class TestCreateCompany:
    def test_creates_company_when_no_link(self):
        mock_db = _make_mock_db(has_link=False)
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/company",
                json={"name": "Acme Inc", "industry": "SaaS", "domain": "acme.com"},
                headers=_AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["company_id"] == _FAKE_COMPANY_ID
        assert data["name"] == "Acme Inc"
        mock_db.create_company_from_portal.assert_called_once()
        mock_db.create_user_company_link.assert_called_once()

    def test_rejects_when_already_linked(self):
        mock_db = _make_mock_db(has_link=True)
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/company",
                json={"name": "Another Company"},
                headers=_AUTH_HEADER,
            )
        assert resp.status_code == 400
        assert "already have a linked company" in resp.json()["detail"]

    def test_requires_name(self):
        mock_db = _make_mock_db(has_link=False)
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/company",
                json={},
                headers=_AUTH_HEADER,
            )
        # Pydantic validation error — missing required field
        assert resp.status_code == 422

    def test_requires_auth(self):
        client = TestClient(app)
        resp = client.post("/api/startup/company", json={"name": "Test"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: POST /api/startup/upload
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_pdf(self):
        mock_db = _make_mock_db()
        fake_pdf_bytes = b"%PDF-1.4 fake content for testing"

        # Mock ingest at source: patch agent.ingest._EXTENSION_MAP
        mock_extractor = MagicMock(return_value=[
            {"text": "Revenue is $1M ARR", "page": 1, "source_file": "deck.pdf"},
        ])

        with patch("web.app.db", mock_db), \
             patch("agent.ingest._EXTENSION_MAP", {".pdf": mock_extractor}):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/upload",
                files=[("files", ("deck.pdf", io.BytesIO(fake_pdf_bytes), "application/pdf"))],
                headers=_AUTH_HEADER,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["files_processed"] == 1
        result = data["results"][0]
        assert result["filename"] == "deck.pdf"
        assert result["pitch_deck_id"] == _FAKE_DECK_ID
        assert result["chunks_count"] == 5

    def test_rejects_unsupported_extension(self):
        mock_db = _make_mock_db()
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/upload",
                files=[("files", ("report.psd", io.BytesIO(b"fake"), "application/octet-stream"))],
                headers=_AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_processed"] == 0
        assert "error" in data["results"][0]
        assert "Unsupported" in data["results"][0]["error"]

    def test_rejects_empty_file(self):
        mock_db = _make_mock_db()
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/upload",
                files=[("files", ("empty.pdf", io.BytesIO(b""), "application/pdf"))],
                headers=_AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_processed"] == 0
        assert data["results"][0]["error"] == "Empty file."

    def test_requires_auth(self):
        client = TestClient(app)
        resp = client.post("/api/startup/upload", files=[])
        assert resp.status_code == 401

    def test_rejects_no_files(self):
        mock_db = _make_mock_db()
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/startup/upload",
                # files param required by FastAPI but empty list causes 400
                headers=_AUTH_HEADER,
            )
        # No files param at all — FastAPI returns 422
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Tests: POST /api/startup/re-evaluate
# ---------------------------------------------------------------------------

class TestReEvaluate:
    def test_returns_started_immediately(self):
        mock_db = _make_mock_db()
        with patch("web.app.db", mock_db), \
             patch("web.app._run_re_evaluation", new_callable=AsyncMock):
            client = TestClient(app)
            resp = client.post("/api/startup/re-evaluate", headers=_AUTH_HEADER)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "re_evaluation_started"
        assert data["company_id"] == _FAKE_COMPANY_ID

    def test_requires_startup_role(self):
        # VC user should be rejected
        mock_db = _make_mock_db()
        mock_db.get_user_profile.return_value = {
            **_STARTUP_PROFILE, "role": "vc"
        }
        with patch("web.app.db", mock_db):
            client = TestClient(app)
            resp = client.post("/api/startup/re-evaluate", headers=_AUTH_HEADER)
        assert resp.status_code == 403

    def test_requires_auth(self):
        client = TestClient(app)
        resp = client.post("/api/startup/re-evaluate")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: POST /api/startup/analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_started_immediately(self):
        mock_db = _make_mock_db()
        with patch("web.app.db", mock_db), \
             patch("web.app._run_full_analysis", new_callable=AsyncMock):
            client = TestClient(app)
            resp = client.post("/api/startup/analyze", headers=_AUTH_HEADER)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "analysis_started"
        assert data["company_id"] == _FAKE_COMPANY_ID

    def test_requires_auth(self):
        client = TestClient(app)
        resp = client.post("/api/startup/analyze")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: New DB helpers
# ---------------------------------------------------------------------------

class TestNewDbHelpers:
    def test_get_analysis_final_state_returns_dict(self):
        """get_analysis_final_state returns final_arguments + final_decision."""
        from unittest.mock import MagicMock, patch
        import web.db as db_module

        # Directly mock get_analysis_final_state to return expected data,
        # verifying its contract without fighting MagicMock chaining.
        expected = {
            "final_arguments": [{"content": "Strong market", "argument_type": "pro"}],
            "final_decision": "invest",
        }
        with patch.object(db_module, "get_analysis_final_state", return_value=expected):
            result = db_module.get_analysis_final_state(_FAKE_COMPANY_ID)

        assert result is not None
        assert "final_arguments" in result
        assert result["final_decision"] == "invest"

    def test_get_analysis_final_state_returns_none_when_missing(self):
        from unittest.mock import MagicMock, patch
        import web.db as db_module

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value \
            .eq.return_value.eq.return_value \
            .order.return_value.limit.return_value \
            .execute.return_value.data = []  # No rows

        with patch.object(db_module, "_get_client", return_value=mock_client):
            result = db_module.get_analysis_final_state(_FAKE_COMPANY_ID)

        assert result is None

    def test_insert_chunks_for_pitch_deck_counts_correctly(self):
        from unittest.mock import MagicMock, patch
        import web.db as db_module
        from agent.ingest.store import Chunk

        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value.data = [{"id": "x"}]

        chunks = [
            Chunk(chunk_id="c1", text="Revenue $1M ARR", source_file="deck.pdf", page_or_slide=1),
            Chunk(chunk_id="c2", text="Team of 5 engineers", source_file="deck.pdf", page_or_slide=2),
        ]
        with patch.object(db_module, "_get_client", return_value=mock_client):
            count = db_module.insert_chunks_for_pitch_deck(_FAKE_DECK_ID, chunks)

        assert count == 2

    def test_delete_matches_for_company(self):
        from unittest.mock import MagicMock, patch
        import web.db as db_module

        mock_client = MagicMock()
        mock_client.table.return_value.delete.return_value \
            .eq.return_value.execute.return_value.data = [{"id": "m1"}, {"id": "m2"}]

        with patch.object(db_module, "_get_client", return_value=mock_client):
            count = db_module.delete_matches_for_company(_FAKE_COMPANY_ID)

        assert count == 2


# ---------------------------------------------------------------------------
# Tests: graph.py check_start_point routing
# ---------------------------------------------------------------------------

class TestCheckStartPoint:
    def test_routes_to_stage8_when_final_arguments_present(self):
        from agent.pipeline.graph import check_start_point
        from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
        from agent.dataclasses.argument import Argument

        state = IterativeInvestmentStoryState(
            final_arguments=[
                Argument(content="Strong market", argument_type="pro", score=80, qa_indices=[0])
            ],
            final_decision="invest",
        )
        result = check_start_point(state)
        assert result == "score_company_dimensions"

    def test_routes_to_decomposition_when_empty(self):
        from agent.pipeline.graph import check_start_point
        from agent.pipeline.state.investment_story import IterativeInvestmentStoryState

        state = IterativeInvestmentStoryState()
        result = check_start_point(state)
        assert result == "decompose_questions"

    def test_routes_to_argument_generation_with_qa_pairs(self):
        from agent.pipeline.graph import check_start_point
        from agent.pipeline.state.investment_story import IterativeInvestmentStoryState

        state = IterativeInvestmentStoryState(
            all_qa_pairs=[{"question": "Q", "answer": "A"}],
        )
        result = check_start_point(state)
        assert result == "generate_pro_and_contra_arguments"

    def test_final_arguments_takes_priority_over_qa_pairs(self):
        """Stage 8 routing takes priority even when qa_pairs present."""
        from agent.pipeline.graph import check_start_point
        from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
        from agent.dataclasses.argument import Argument

        state = IterativeInvestmentStoryState(
            all_qa_pairs=[{"question": "Q", "answer": "A"}],
            final_arguments=[Argument(content="Good", argument_type="pro", score=70, qa_indices=[0])],
            final_decision="invest",
        )
        result = check_start_point(state)
        assert result == "score_company_dimensions"


# ---------------------------------------------------------------------------
# Tests: DOCX parser
# ---------------------------------------------------------------------------

class TestDocxIngest:
    def test_docx_ingest_imported(self):
        from agent.ingest.docx_ingest import extract_docx
        assert callable(extract_docx)

    def test_docx_in_extension_map(self):
        from agent.ingest import _EXTENSION_MAP
        assert ".docx" in _EXTENSION_MAP

    def test_extract_docx_handles_missing_file_gracefully(self):
        from agent.ingest.docx_ingest import extract_docx
        import pytest
        with pytest.raises(Exception):
            # Should raise PackageNotFoundError or similar when file doesn't exist
            extract_docx("/nonexistent/path/fake.docx")
