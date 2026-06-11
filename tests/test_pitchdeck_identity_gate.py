import asyncio
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skip(
    reason=(
        "Targets the pre-guardrails pitchdeck identity gate "
        "(identity_confirmation_required / specter_resolution_status / "
        "augment_with_specter_status), which no longer exists in web/app.py on any "
        "branch; superseded by the analysis-quality preflight (d77ac78, covered by "
        "test_analysis_quality_guardrails.py). Pending rewrite or removal — "
        "see Sprint 0 report (2026-06-11)."
    )
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.dataclasses.company import Company
from agent.ingest.store import Chunk, EvidenceStore
from web import app as web_app
from web.app import AnalysisStatus, app


def _login(client: TestClient) -> None:
    response = client.post("/api/login", json={"password": "9876"})
    assert response.status_code == 200
    client.cookies.set("session_id", response.json()["session_id"])


def _reset_state() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
    web_app._jobs.clear()
    web_app._job_controls.clear()
    web_app._results_cache.clear()
    web_app._jobs_overview_cache.update({"expires_at": 0.0, "payload": None})
    web_app._company_runs_cache.update({"expires_at": 0.0, "payload": None})


def test_pitchdeck_specter_gate_requires_url_confirmation(monkeypatch) -> None:
    _reset_state()
    monkeypatch.setattr(web_app, "db", SimpleNamespace(is_configured=lambda: False))

    class NoopThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self) -> None:
            raise AssertionError("analysis thread should not start before URL confirmation")

    monkeypatch.setattr(web_app.threading, "Thread", NoopThread)

    with TestClient(app) as client:
        _login(client)
        upload = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"no domains in this deck"), "text/plain")},
        )
        job_id = upload.json()["job_id"]

        response = client.post(
            f"/api/analyze/{job_id}",
            json={"input_mode": "pitchdeck", "use_specter_mcp": True},
        )

    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "identity_confirmation_required"
    assert payload["job_id"] == job_id
    assert payload["candidate_url"] is None
    assert web_app._jobs[job_id].status == "pending"
    assert web_app._results_cache[job_id]["specter_resolution_status"] == "missing_url"


def test_pitchdeck_confirmed_url_starts_analysis_and_updates_run_config(monkeypatch) -> None:
    _reset_state()
    monkeypatch.setattr(web_app, "db", SimpleNamespace(is_configured=lambda: False))
    started = {"called": False}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self) -> None:
            started["called"] = True

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    with TestClient(app) as client:
        _login(client)
        upload = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"no domains in this deck"), "text/plain")},
        )
        job_id = upload.json()["job_id"]

        response = client.post(
            f"/api/analyze/{job_id}",
            json={
                "input_mode": "pitchdeck",
                "use_specter_mcp": True,
                "confirmed_company_url": "https://www.dessia.tech/company",
            },
        )

    assert response.status_code == 200
    assert started["called"] is True
    run_config = web_app._results_cache[job_id]["run_config"]
    assert run_config["confirmed_company_url"] == "dessia.tech"
    assert run_config["identity_source"] == "user_confirmed"
    assert run_config["specter_resolution_status"] == "pending"


def test_pitchdeck_detected_url_starts_without_confirmation(monkeypatch) -> None:
    _reset_state()
    monkeypatch.setattr(web_app, "db", SimpleNamespace(is_configured=lambda: False))
    started = {"called": False}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self) -> None:
            started["called"] = True

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    with TestClient(app) as client:
        _login(client)
        upload = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"Website: acme.com"), "text/plain")},
        )
        job_id = upload.json()["job_id"]

        response = client.post(
            f"/api/analyze/{job_id}",
            json={"input_mode": "pitchdeck", "use_specter_mcp": True},
        )

    assert response.status_code == 200
    assert started["called"] is True
    run_config = web_app._results_cache[job_id]["run_config"]
    assert run_config["confirmed_company_url"] == "acme.com"
    assert run_config["identity_source"] == "deck_detected"


def test_deck_only_fallback_persists_confirmed_domain(monkeypatch, tmp_path: Path) -> None:
    _reset_state()
    captured: dict[str, object] = {}
    job_id = "job-identity"
    upload_dir = tmp_path / job_id
    upload_dir.mkdir()
    (upload_dir / "deck.txt").write_text("No URL here", encoding="utf-8")
    store = EvidenceStore(
        startup_slug=job_id,
        chunks=[Chunk(chunk_id="chunk_0", text="Deck text", source_file="deck.txt", page_or_slide="1")],
    )

    web_app._jobs[job_id] = AnalysisStatus(job_id=job_id, status="running", progress="Starting")
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [{"name": "deck.txt"}],
        "confirmed_company_url": "dessia.tech",
        "identity_source": "user_confirmed",
        "specter_resolution_status": "pending",
        "pitchdeck_identity_store": store,
        "run_config": {
            "input_mode": "pitchdeck",
            "confirmed_company_url": "dessia.tech",
            "identity_source": "user_confirmed",
            "specter_resolution_status": "pending",
        },
    }

    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.augment_with_specter_status",
        lambda *args, **kwargs: {
            "store": store,
            "company": None,
            "url": "dessia.tech",
            "status": "deck_only",
        },
    )

    async def fake_evaluate_startup(*args, **kwargs):
        company = Company(name="Dessia Technologies")
        return {
            "slug": job_id,
            "company": company,
            "company_name": company.name,
            "evidence_store": store,
            "final_state": {
                "final_arguments": [],
                "final_decision": "watch",
                "ranking_result": None,
                "all_qa_pairs": [],
            },
        }

    monkeypatch.setattr(web_app, "evaluate_startup", fake_evaluate_startup)
    monkeypatch.setattr(
        web_app,
        "_build_results_payload",
        lambda results, job_id_arg, upload_dir_arg, write_excel=True: web_app._results_cache[job_id_arg].update(
            {"results": {"job_status": "done"}}
        ),
    )
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            persist_company_result=lambda **kwargs: captured.update(kwargs) or True,
            persist_analysis=lambda *args, **kwargs: True,
            insert_job_status_history=lambda *args, **kwargs: True,
        ),
    )

    asyncio.run(
        web_app._run_document_analysis(
            job_id,
            upload_dir,
            use_web_search=False,
            use_specter_mcp=True,
        )
    )

    result_row = captured["result_row"]
    assert result_row["company"].domain == "dessia.tech"
    assert captured["run_config"]["specter_resolution_status"] == "deck_only"
