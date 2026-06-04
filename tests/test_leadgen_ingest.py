from __future__ import annotations

from copy import deepcopy
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
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)

    async def fake_require_identity(_authorization):
        return {
            "started_by_user_id": "user-1",
            "started_by_email": "operator@rockaway.vc",
            "started_by_display_name": "Operator",
            "started_by_label": "Operator",
        }

    monkeypatch.setattr(web_app, "_require_supabase_identity", fake_require_identity)
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
        "batches": {},
        "batch_by_id": {},
        "leads": {},
        "next_intake": 1,
        "next_lead": 1,
    }

    def detail(intake_id: str):
        batch = state["batches"].get(intake_id)
        if not batch:
            return None
        return {
            "batch": deepcopy(batch),
            "leads": deepcopy(state["leads"].get(intake_id, [])),
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

        @staticmethod
        def create_leadgen_intake(*, batch, leads):
            existing_id = state["batch_by_id"].get(batch["batch_id"])
            if existing_id:
                existing = state["batches"][existing_id]
                if existing["payload_hash"] != batch["payload_hash"]:
                    return {"status": "conflict", "batch": deepcopy(existing)}
                return {"status": "existing", "batch": deepcopy(existing)}

            intake_id = f"intake-{state['next_intake']}"
            state["next_intake"] += 1
            stored_batch = {**deepcopy(batch), "id": intake_id, "intake_id": intake_id, "created_at": "2026-05-25T10:00:00Z"}
            state["batches"][intake_id] = stored_batch
            state["batch_by_id"][stored_batch["batch_id"]] = intake_id
            stored_leads = []
            for lead in leads:
                lead_id = f"lead-{state['next_lead']}"
                state["next_lead"] += 1
                stored_leads.append({**deepcopy(lead), "id": lead_id, "lead_id": lead_id, "intake_id": intake_id})
            state["leads"][intake_id] = stored_leads
            return {"status": "created", "batch": deepcopy(stored_batch)}

        @staticmethod
        def list_leadgen_intake_batches(*, status="pending", limit=50):
            rows = list(state["batches"].values())
            if status != "all":
                rows = [row for row in rows if row.get("status") == status]
            return deepcopy(rows[:limit])

        @staticmethod
        def load_leadgen_intake(intake_id):
            return detail(intake_id)

        @staticmethod
        def mark_leadgen_intake_queued(*, intake_id, lead_ids, job_id_legacy, actor):
            batch = state["batches"][intake_id]
            batch["job_id_legacy"] = job_id_legacy
            batch["approved_by_email"] = actor.get("started_by_email")
            selected_ids = set(lead_ids or [])
            for lead in state["leads"][intake_id]:
                if lead["lead_id"] in selected_ids:
                    lead["approval_status"] = "approved"
            remaining_pending = [
                lead
                for lead in state["leads"][intake_id]
                if lead.get("eligible") and lead.get("approval_status") == "pending"
            ]
            batch["status"] = "partially_approved" if remaining_pending else "queued"
            return detail(intake_id)

        @staticmethod
        def reject_leadgen_intake(*, intake_id, actor=None, reason=None, lead_ids=None, reject_all=False):
            batch = state["batches"][intake_id]
            if reject_all:
                batch["status"] = "rejected"
            for lead in state["leads"][intake_id]:
                if reject_all or lead["lead_id"] in set(lead_ids or []):
                    lead["approval_status"] = "rejected"
                    lead["eligible"] = False
                    lead["rejection_reason"] = reason
            return detail(intake_id)

    monkeypatch.setattr(web_app, "db", FakeDb())
    try:
        yield state
    finally:
        web_app.ENABLE_SPECTER_WORKER_SERVICE = original_worker_flag
        web_app._jobs.clear()
        web_app._job_controls.clear()
        web_app._results_cache.clear()


def _batch_payload() -> dict:
    rockaway_pass = {"thesis_key": "rockaway", "thesis_status": "pass"}
    return {
        "batch_id": "smoke-rdi-1",
        "generated_at": "2026-05-23T06:30:04+00:00",
        "scoring_version": "phase1-rubric-v1",
        "summary": {
            "lead_count": 5,
            "rockaway_pass_count": 3,
            "review_count": 1,
        },
        "leads": [
            {
                "lead": {
                    "name": "Alpha",
                    "website": "https://alpha.example",
                    "domain": "alpha.example",
                    "source": "specter",
                    "source_url": "https://source.example/alpha",
                    **rockaway_pass,
                },
                "score": {
                    "score": 100,
                    "bucket": "high_priority",
                    "rationale": "Rockaway thesis PASS.",
                    "version": "phase1-rubric-v1",
                },
            },
            {
                "lead": {
                    "name": "Beta",
                    "website": "not a url",
                    "domain": "beta.example",
                    "source": "specter",
                    **rockaway_pass,
                },
                "score": {"score": 92, "bucket": "high_priority", "version": "phase1-rubric-v1"},
            },
            {
                "lead": {
                    "name": "Alpha Duplicate",
                    "website": "https://alpha.example",
                    "domain": "alpha.example",
                    "source": "specter",
                    **rockaway_pass,
                },
                "score": {"score": 88, "bucket": "high_priority", "version": "phase1-rubric-v1"},
            },
            {
                "lead": {
                    "name": "Review Co",
                    "website": "https://review.example",
                    "domain": "review.example",
                    "source": "specter",
                    "thesis_key": "rockaway",
                    "thesis_status": "review",
                },
                "score": {"score": 72, "bucket": "review", "version": "phase1-rubric-v1"},
            },
            {
                "lead": {
                    "name": "No URL",
                    "website": "",
                    "domain": "",
                    "source": "specter",
                    **rockaway_pass,
                },
                "score": {"score": 10, "bucket": "archive", "version": "phase1-rubric-v1"},
            },
        ],
    }


def test_leadgen_ingest_stores_pending_intake_without_queueing_job(leadgen_runtime) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "pending_approval"
    assert payload["intake_id"] == "intake-1"
    assert payload["job_id"] is None
    assert payload["accepted_count"] == 2
    assert payload["rejected_count"] == 3
    assert payload["accepted_urls"] == ["https://alpha.example", "beta.example"]
    assert leadgen_runtime["queued"] == []
    assert web_app._jobs == {}

    stored_leads = leadgen_runtime["leads"]["intake-1"]
    assert [lead["approval_status"] for lead in stored_leads] == [
        "pending",
        "pending",
        "duplicate",
        "ineligible",
        "invalid",
    ]


def test_leadgen_intake_alias_uses_same_pending_behavior(leadgen_runtime) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/intake",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        )

    assert response.status_code == 202
    assert response.json()["status"] == "pending_approval"
    assert leadgen_runtime["queued"] == []


def test_leadgen_ingest_accepts_current_legacy_rockaway_pass_rationale(leadgen_runtime) -> None:
    payload = _batch_payload()
    payload["leads"][0]["lead"].pop("thesis_key")
    payload["leads"][0]["lead"].pop("thesis_status")

    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=payload,
        )

    assert response.status_code == 202
    assert response.json()["accepted_urls"][0] == "https://alpha.example"
    stored_lead = leadgen_runtime["leads"]["intake-1"][0]
    assert stored_lead["thesis_key"] == "rockaway"
    assert stored_lead["thesis_status"] == "pass"


def test_leadgen_ingest_retry_is_idempotent_and_conflicts_on_changed_payload(leadgen_runtime) -> None:
    payload = _batch_payload()
    changed = deepcopy(payload)
    changed["leads"][0]["score"]["score"] = 99

    with TestClient(app) as client:
        first = client.post("/api/leadgen/ingest", headers={"X-API-Key": "leadgen-secret"}, json=payload)
        second = client.post("/api/leadgen/ingest", headers={"X-API-Key": "leadgen-secret"}, json=payload)
        conflict = client.post("/api/leadgen/ingest", headers={"X-API-Key": "leadgen-secret"}, json=changed)

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["status"] == "existing_pending"
    assert second.json()["intake_id"] == first.json()["intake_id"]
    assert conflict.status_code == 409
    assert leadgen_runtime["queued"] == []


def test_approval_queues_one_specter_job_for_selected_leads(leadgen_runtime) -> None:
    with TestClient(app) as client:
        intake = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        ).json()
        detail = client.get(f"/api/leadgen/intakes/{intake['intake_id']}").json()
        selected_id = detail["leads"][1]["lead_id"]
        response = client.post(
            f"/api/leadgen/intakes/{intake['intake_id']}/approve",
            json={"lead_ids": [selected_id]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["approved_count"] == 1
    assert body["approved_urls"] == ["beta.example"]
    assert body["batch"]["status"] == "partially_approved"
    assert len(leadgen_runtime["queued"]) == 1

    job_id = body["job_id"]
    cache = web_app._results_cache[job_id]
    run_config = cache["run_config"]
    assert cache["specter_urls"] == [{"url": "beta.example", "name": "Beta"}]
    assert run_config["specter_urls"] == [{"url": "beta.example", "name": "Beta"}]
    assert run_config["source"] == "leadgen"
    assert run_config["run_name"] == "leadgen:smoke-rdi-1"
    assert run_config["leadgen"]["intake_id"] == "intake-1"
    assert run_config["leadgen"]["approved_lead_ids"] == [selected_id]
    assert run_config["leadgen"]["leads"][0]["name"] == "Beta"
    assert run_config["started_by_email"] == "operator@rockaway.vc"


def test_approval_requires_selection_and_does_not_queue(leadgen_runtime) -> None:
    with TestClient(app) as client:
        intake = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        ).json()
        response = client.post(
            f"/api/leadgen/intakes/{intake['intake_id']}/approve",
            json={"lead_ids": []},
        )

    assert response.status_code == 400
    assert leadgen_runtime["queued"] == []


def test_approval_rechecks_stored_source_platform_domain(leadgen_runtime) -> None:
    with TestClient(app) as client:
        intake = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=_batch_payload(),
        ).json()

        stale_lead = leadgen_runtime["leads"][intake["intake_id"]][0]
        stale_lead["url"] = "github.com"
        stale_lead["domain"] = "github.com"
        stale_lead["eligible"] = True
        stale_lead["approval_status"] = "pending"

        response = client.post(
            f"/api/leadgen/intakes/{intake['intake_id']}/approve",
            json={"lead_ids": [stale_lead["lead_id"]]},
        )

    assert response.status_code == 400
    assert "source platform github.com" in response.json()["detail"]
    assert leadgen_runtime["queued"] == []


def test_leadgen_ingest_rejects_invalid_urls_without_job(leadgen_runtime) -> None:
    payload = _batch_payload()
    payload["leads"] = [
        {
            "lead": {"name": "No URL", "website": "", "domain": "", "thesis_key": "rockaway", "thesis_status": "pass"},
            "score": {"bucket": "high_priority"},
        },
        {
            "lead": {"name": "Bad URL", "website": "not a url", "thesis_key": "rockaway", "thesis_status": "pass"},
            "score": {"bucket": "high_priority"},
        },
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


def test_leadgen_ingest_rejects_source_platform_domains(leadgen_runtime) -> None:
    payload = _batch_payload()
    payload["leads"] = [
        {
            "lead": {
                "name": "Repo Signal Co",
                "website": "",
                "domain": "github.com",
                "source": "github_trends",
                "source_url": "https://github.com/example/repo",
                "thesis_key": "rockaway",
                "thesis_status": "pass",
            },
            "score": {"bucket": "high_priority"},
        },
        {
            "lead": {
                "name": "Domain Fallback Co",
                "website": "https://github.com/example/repo",
                "domain": "fallback.example",
                "source": "github_trends",
                "source_url": "https://github.com/example/repo",
                "thesis_key": "rockaway",
                "thesis_status": "pass",
            },
            "score": {"bucket": "high_priority"},
        },
    ]

    with TestClient(app) as client:
        response = client.post(
            "/api/leadgen/ingest",
            headers={"X-API-Key": "leadgen-secret"},
            json=payload,
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending_approval"
    assert body["accepted_urls"] == ["fallback.example"]
    assert body["errors"][0]["reason"] == "Website/domain points to source platform github.com, not a company website."

    stored_leads = leadgen_runtime["leads"]["intake-1"]
    assert stored_leads[0]["approval_status"] == "invalid"
    assert stored_leads[1]["approval_status"] == "pending"


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
