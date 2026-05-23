from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from web import app as web_app
from web.app import app


@pytest.fixture
def leadgen_runtime(monkeypatch):
    original_worker_flag = web_app.ENABLE_SPECTER_WORKER_SERVICE
    web_app.ENABLE_SPECTER_WORKER_SERVICE = True
    web_app._jobs.clear()
    web_app._job_controls.clear()
    web_app._results_cache.clear()
    monkeypatch.setenv("LEADGEN_API_KEY", "leadgen-secret")
    monkeypatch.setattr(
        web_app,
        "build_default_phase_model_policy",
        lambda: SimpleNamespace(answering={"provider": "openai", "model": "gpt-5"}),
    )
    monkeypatch.setattr(web_app, "phase_model_defaults_payload", lambda: {"answering": "gpt-5"})
    monkeypatch.setattr(web_app, "resolve_effective_phase_models", lambda policy: {"answering": "gpt-5"})
    monkeypatch.setattr(web_app, "build_phase_policy_display_label", lambda choices: "GPT-5")
    monkeypatch.setattr(web_app, "_runtime_versions", lambda: {"app_version": "test"})

    state = {
        "statuses": {},
        "queued": [],
        "upserted": {},
        "events": [],
    }

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def load_job_status(job_id: str):
            return state["statuses"].get(job_id)

        @staticmethod
        def insert_analysis_event(job_id, *, message, event_type="progress", stage=None, payload=None):
            state["events"].append(
                {
                    "job_id": job_id,
                    "message": message,
                    "event_type": event_type,
                    "stage": stage,
                    "payload": payload,
                }
            )

        @staticmethod
        def insert_job_status_history(job_id, *, status, progress=None, source="app"):
            state["statuses"][job_id] = {"status": status, "progress": progress, "source": source}

        @staticmethod
        def upsert_job(job_id, **kwargs):
            state["upserted"][job_id] = kwargs
            return "job-uuid"

        @staticmethod
        def upsert_job_control(*args, **kwargs):
            return True

        @staticmethod
        def persist_source_files(*args, **kwargs):
            return True

        @staticmethod
        def queue_specter_worker_job(job_id, **kwargs):
            state["queued"].append({"job_id": job_id, **kwargs})
            state["statuses"][job_id] = {
                "status": "running",
                "progress": kwargs.get("progress"),
                "source": "worker_queue",
            }
            return True

    monkeypatch.setattr(web_app, "db", FakeDb())
    try:
        yield state
    finally:
        web_app.ENABLE_SPECTER_WORKER_SERVICE = original_worker_flag
        web_app._jobs.clear()
        web_app._job_controls.clear()
        web_app._results_cache.clear()


def _batch_payload() -> dict:
    return {
        "batch_id": "smoke-rdi-1",
        "generated_at": "2026-05-23T06:30:04+00:00",
        "scoring_version": "phase1-rubric-v1",
        "summary": {
            "lead_count": 4,
            "high_priority_count": 1,
            "review_count": 1,
            "archive_count": 1,
        },
        "leads": [
            {
                "lead": {
                    "name": "Alpha",
                    "website": "https://alpha.example",
                    "domain": "alpha.example",
                    "source": "specter",
                    "source_url": "https://source.example/alpha",
                },
                "score": {
                    "score": 100,
                    "bucket": "high_priority",
                    "rationale": "Strong fit.",
                    "version": "phase1-rubric-v1",
                },
            },
            {
                "lead": {
                    "name": "Beta",
                    "website": "not a url",
                    "domain": "beta.example",
                    "source": "specter",
                },
                "score": {"score": 72, "bucket": "review", "version": "phase1-rubric-v1"},
            },
            {
                "lead": {
                    "name": "Archive Co",
                    "website": "https://archive.example",
                    "domain": "archive.example",
                    "source": "specter",
                },
                "score": {"score": 25, "bucket": "archive", "version": "phase1-rubric-v1"},
            },
            {
                "lead": {
                    "name": "No URL",
                    "website": "",
                    "domain": "",
                    "source": "specter",
                },
                "score": {"score": 10, "bucket": "archive", "version": "phase1-rubric-v1"},
            },
        ],
    }


def test_leadgen_ingest_creates_one_forced_specter_url_job(leadgen_runtime) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "created"
    assert payload["accepted_count"] == 3
    assert payload["rejected_count"] == 1
    assert payload["accepted_urls"] == [
        "https://alpha.example",
        "beta.example",
        "https://archive.example",
    ]
    assert payload["errors"] == [
        {
            "lead_name": "No URL",
            "url": None,
            "reason": "Missing or invalid website/domain.",
        }
    ]

    job_id = payload["job_id"]
    cache = web_app._results_cache[job_id]
    run_config = cache["run_config"]
    assert cache["specter_urls"] == payload["accepted_urls"]
    assert run_config["source"] == "leadgen"
    assert run_config["run_name"] == "leadgen:smoke-rdi-1"
    assert run_config["input_mode"] == "specter"
    assert run_config["use_web_search"] is True
    assert run_config["use_specter_mcp"] is True
    assert run_config["fetch_full_team"] is True
    assert run_config["leadgen"]["batch_id"] == "smoke-rdi-1"
    assert run_config["leadgen"]["scoring_version"] == "phase1-rubric-v1"
    assert [lead["bucket"] for lead in run_config["leadgen"]["leads"]] == [
        "high_priority",
        "review",
        "archive",
    ]
    assert len(leadgen_runtime["queued"]) == 1
    assert leadgen_runtime["queued"][0]["job_id"] == job_id
    assert leadgen_runtime["queued"][0]["run_config"]["specter_urls"] == payload["accepted_urls"]


def test_leadgen_ingest_retry_returns_existing_job_without_requeue(leadgen_runtime) -> None:
    with TestClient(app) as client:
        first = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        )
        second = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["status"] == "existing"
    assert second.json()["job_id"] == first.json()["job_id"]
    assert len(leadgen_runtime["queued"]) == 1


def test_leadgen_ingest_rejects_invalid_urls_without_job(leadgen_runtime) -> None:
    payload = _batch_payload()
    payload["leads"] = [
        {"lead": {"name": "No URL", "website": "", "domain": ""}, "score": {"bucket": "archive"}},
        {"lead": {"name": "Bad URL", "website": "not a url"}, "score": {"bucket": "high_priority"}},
    ]

    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=payload,
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "rejected"
    assert body["job_id"] is None
    assert body["accepted_count"] == 0
    assert body["rejected_count"] == 2
    assert leadgen_runtime["queued"] == []


def test_leadgen_ingest_requires_configured_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LEADGEN_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        )

    assert response.status_code == 503


def test_leadgen_ingest_rejects_wrong_api_key(leadgen_runtime) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "wrong"},
            json=_batch_payload(),
        )

    assert response.status_code == 401


def test_leadgen_ingest_rejects_malformed_payload(leadgen_runtime) -> None:
    payload = _batch_payload()
    payload["leads"] = [{"lead": "not-an-object", "score": {}}]
    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=payload,
        )

    assert response.status_code == 422
