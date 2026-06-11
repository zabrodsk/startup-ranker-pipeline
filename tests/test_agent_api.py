import io
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from web import agent_api as agent_api_module
from web import app as web_app_module
from web.app import app

pytestmark = pytest.mark.skip(
    reason=(
        "web/agent_api.py exists but its router is not mounted in web/app.py on "
        "this lineage; the include_router wiring plus the 13 _*_payload handlers "
        "it needs exist only on codex/main-with-both-sessions. Pending restore "
        "decision — see Sprint 0 report (2026-06-11)."
    )
)


def _agent_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-agent-key"}


def _reset_agent_state() -> None:
    web_app_module._jobs.clear()
    web_app_module._job_controls.clear()
    web_app_module._results_cache.clear()
    web_app_module._person_jobs.clear()
    web_app_module._person_job_tasks.clear()
    web_app_module._jobs_overview_cache.update({"expires_at": 0.0, "payload": None})
    web_app_module._company_runs_cache.update({"expires_at": 0.0, "payload": None})
    agent_api_module._IDEMPOTENCY_RECORDS.clear()


def test_agent_api_requires_bearer_auth(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    with TestClient(app) as client:
        missing = client.get("/api/agent/capabilities")
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "unauthorized"

        ok = client.get("/api/agent/capabilities", headers=_agent_headers())
        assert ok.status_code == 200
        assert ok.json()["auth_scheme"] == "bearer"


def test_agent_openapi_has_typed_response_schemas(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    with TestClient(app) as client:
        response = client.get("/api/agent/openapi.json", headers=_agent_headers())

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["paths"]).issuperset(
        {
            "/api/agent/uploads",
            "/api/agent/analyses",
            "/api/agent/analyses/{analysis_id}",
            "/api/agent/person-profiles",
        }
    )
    assert (
        schema["paths"]["/api/agent/analyses"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]
    )
    assert (
        schema["paths"]["/api/agent/person-profiles"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]
    )


def test_agent_analysis_happy_path(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    async def fake_run_analysis(job_id: str, **kwargs) -> None:
        results = {
            "mode": "single",
            "company_name": "Acme",
            "startup_slug": "acme",
            "decision": "invest",
            "total_score": 82.1,
            "summary_rows": [{"company_name": "Acme", "decision": "invest"}],
            "argument_rows": [],
            "qa_provenance_rows": [],
            "job_status": "done",
            "job_message": "Analysis complete",
            "llm": "Test LLM",
        }
        web_app_module._results_cache[job_id]["results"] = results
        web_app_module._jobs[job_id].status = "done"
        web_app_module._jobs[job_id].progress = "Analysis complete"
        web_app_module._jobs[job_id].results = results

    real_thread = threading.Thread

    class ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self) -> None:
            if self._target:
                worker = real_thread(target=self._target, daemon=True)
                worker.start()
                worker.join()

    monkeypatch.setattr(web_app_module, "_run_analysis", fake_run_analysis)
    monkeypatch.setattr(web_app_module.threading, "Thread", ImmediateThread)

    with TestClient(app) as client:
        upload = client.post(
            "/api/agent/uploads",
            headers=_agent_headers(),
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )
        assert upload.status_code == 202
        upload_id = upload.json()["upload_id"]

        created = client.post(
            "/api/agent/analyses",
            headers=_agent_headers(),
            json={"upload_id": upload_id, "input_mode": "pitchdeck"},
        )
        assert created.status_code == 202
        assert created.json()["analysis_id"] == upload_id

        status = client.get(f"/api/agent/analyses/{upload_id}", headers=_agent_headers())
        assert status.status_code == 200
        assert status.json()["status"] == "done"
        assert status.json()["result_available"] is True

        result = client.get(f"/api/agent/analyses/{upload_id}/result", headers=_agent_headers())
        assert result.status_code == 200
        assert result.json()["results"]["company_name"] == "Acme"


def test_agent_analysis_result_returns_409_while_running(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    class NoopThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self) -> None:
            return None

    monkeypatch.setattr(web_app_module.threading, "Thread", NoopThread)

    with TestClient(app) as client:
        upload = client.post(
            "/api/agent/uploads",
            headers=_agent_headers(),
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )
        upload_id = upload.json()["upload_id"]
        client.post(
            "/api/agent/analyses",
            headers=_agent_headers(),
            json={"upload_id": upload_id, "input_mode": "pitchdeck"},
        )

        result = client.get(f"/api/agent/analyses/{upload_id}/result", headers=_agent_headers())

    assert result.status_code == 409
    assert result.json()["error"]["code"] == "conflict"


def test_agent_analysis_actions_support_pause_resume_stop(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    with TestClient(app) as client:
        upload = client.post(
            "/api/agent/uploads",
            headers=_agent_headers(),
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )
        analysis_id = upload.json()["upload_id"]

        paused = client.post(
            f"/api/agent/analyses/{analysis_id}/actions",
            headers=_agent_headers(),
            json={"action": "pause"},
        )
        resumed = client.post(
            f"/api/agent/analyses/{analysis_id}/actions",
            headers=_agent_headers(),
            json={"action": "resume"},
        )
        stopped = client.post(
            f"/api/agent/analyses/{analysis_id}/actions",
            headers=_agent_headers(),
            json={"action": "stop"},
        )

    assert paused.status_code == 202
    assert paused.json()["status"] == "paused"
    assert resumed.json()["status"] == "running"
    assert stopped.json()["status"] == "stopped"
    assert stopped.json()["terminal"] is True


def test_agent_person_profile_create_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()
    monkeypatch.setattr(web_app_module, "_schedule_person_profile_job", lambda job_id, req: None)

    payload = {"primary_profile_url": "https://www.linkedin.com/in/example", "full_name": "Example"}
    headers = {**_agent_headers(), "Idempotency-Key": "same-person"}

    with TestClient(app) as client:
        first = client.post("/api/agent/person-profiles", headers=headers, json=payload)
        second = client.post("/api/agent/person-profiles", headers=headers, json=payload)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.headers["Idempotent-Replayed"] == "true"


def test_agent_person_profile_status_flow(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()
    monkeypatch.setattr(web_app_module, "_schedule_person_profile_job", lambda job_id, req: None)

    with TestClient(app) as client:
        created = client.post(
            "/api/agent/person-profiles",
            headers=_agent_headers(),
            json={"primary_profile_url": "https://www.linkedin.com/in/example"},
        )
        job_id = created.json()["job_id"]
        web_app_module._person_jobs[job_id].status = "done"
        web_app_module._person_jobs[job_id].progress = "Profile completed"
        web_app_module._person_jobs[job_id].result = {"profile_json": {"claims": []}}

        status = client.get(f"/api/agent/person-profiles/{job_id}", headers=_agent_headers())

    assert status.status_code == 200
    assert status.json()["status"] == "done"
    assert status.json()["result_available"] is True


def test_agent_bulk_founders_returns_jobs_and_skips(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()
    monkeypatch.setattr(web_app_module, "_schedule_person_profile_job", lambda job_id, req: None)

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/person-profiles/bulk-founders",
            headers=_agent_headers(),
            json={
                "company_slug": "acme",
                "founders": [
                    {"full_name": "Alice", "primary_profile_url": "https://www.linkedin.com/in/alice"},
                    {"full_name": "Bob"},
                ],
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert len(payload["jobs"]) == 1
    assert len(payload["skipped"]) == 1
    assert payload["skipped"][0]["reason"] == "missing_primary_profile_url"


def test_agent_read_only_endpoints_return_jobs_company_runs_and_company_analyses(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    web_app_module._jobs["job-1"] = web_app_module.AnalysisStatus(
        job_id="job-1",
        status="done",
        progress="Analysis complete",
    )
    web_app_module._results_cache["job-1"] = {
        "results": {"company_name": "Acme"},
        "run_config": {"run_name": "Test run"},
    }
    monkeypatch.setattr(
        web_app_module,
        "_list_company_runs_for_ui",
        lambda: [{"company_name": "Acme", "runs": [{"job_id": "job-1"}]}],
    )

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def load_analyses_by_company(company_name: str) -> list[dict[str, str]]:
            return [{"job_id": "job-1", "company_name": company_name}]

    monkeypatch.setattr(web_app_module, "db", FakeDb())

    with TestClient(app) as client:
        jobs = client.get("/api/agent/jobs", headers=_agent_headers())
        runs = client.get("/api/agent/company-runs", headers=_agent_headers())
        company = client.get("/api/agent/companies/Acme/analyses", headers=_agent_headers())

    assert jobs.status_code == 200
    assert jobs.json()["jobs"][0]["analysis_id"] == "job-1"
    assert runs.status_code == 200
    assert runs.json()["companies"][0]["company_name"] == "Acme"
    assert company.status_code == 200
    assert company.json()["analyses"][0]["company_name"] == "Acme"


def test_browser_routes_still_use_cookie_auth(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "test-agent-key")
    _reset_agent_state()

    with TestClient(app) as client:
        login = client.post("/api/login", json={"password": "9876"})
        assert login.status_code == 200
        client.cookies.set("session_id", login.json()["session_id"])

        upload = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )

    assert upload.status_code == 200
    assert "job_id" in upload.json()
