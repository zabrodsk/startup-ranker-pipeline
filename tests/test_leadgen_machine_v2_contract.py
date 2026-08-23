from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.leadgen_machine import MachineStartAccepted, MachineStartDefiniteRejection
from web.leadgen_machine_v2 import MachineV2Dependencies, build_leadgen_machine_v2_router


SERVICE_HEADERS = {"X-LeadGen-Service-Key": "unit-test-machine-key"}
PRAGUE_DATE = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Prague")).date().isoformat()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _bundle(*, requires_specter_mcp: bool) -> dict[str, Any]:
    component_payload = {"name": "Acme", "domain": "acme.example"}
    return {
        "schema_version": "frozen-leadgen-evidence-bundle-v1",
        "external_company_id": "a1f4e5d0-1111-4111-8111-111111111111",
        "canonical_domain": "acme.example",
        "specter_company_id": "0123456789abcdef01234567",
        "requires_specter_mcp": requires_specter_mcp,
        "parent_bundle_sha256": None,
        "created_at": "2026-08-21T08:00:00Z",
        "company": {
            "name": "Acme",
            "domain": "acme.example",
            "company_url": "https://acme.example",
            "specter_company_id": "0123456789abcdef01234567",
            "team": [{"name": "Ada Founder", "title": "CEO"}],
        },
        "evidence_chunks": [
            {
                "chunk_id": "company-profile",
                "text": "Acme builds industrial software.",
                "source_file": "specter:company_profile",
                "page_or_slide": "profile",
            }
        ],
        "components": [
            {
                "component": "company_profile",
                "provider": "specter_mcp",
                "retrieved_at": "2026-08-21T08:00:00Z",
                "fresh_until": "2026-11-19T08:00:00Z",
                "schema_version": "specter-company-profile-v1",
                "payload_sha256": hashlib.sha256(_canonical(component_payload)).hexdigest(),
                "errors": [],
            }
        ],
        "component_payloads": {"company_profile": component_payload},
        "specter_operations": [{
            "component": "company_profile",
            "operation": "get_company_profile",
            "trigger": "authoritative_lite",
            "consumer": "leadgen_lite",
            "cache_status": "miss",
            "called": True,
            "success": True,
            "occurred_at": "2026-08-21T08:00:00Z",
            "error": None,
        }],
        "authorization": {
            "company_id": "a1f4e5d0-1111-4111-8111-111111111111",
            "canonical_domain": "acme.example",
            "thesis_status": "pass",
            "thesis_version": "May 2026",
            "thesis_sha256": "2" * 64,
            "lite_tier": "send_to_rdi",
            "lite_scoring_version": "rockaway-lite-v2",
            "lite_sha256": "3" * 64,
            "rdi_ready": True,
            "route": "rockaway_rdi",
            "routing_version": "investor-routing-v1",
            "routing_sha256": "4" * 64,
            "source_run_id": "source-run-01",
            "source_sha256": "5" * 64,
            "frozen_lineage_sha256": "6" * 64,
        },
    }


def _intake(bundle_sha256: str) -> dict[str, Any]:
    return {
        "external_company_id": "a1f4e5d0-1111-4111-8111-111111111111",
        "canonical_domain": "acme.example",
        "campaign_id": "campaign-2026-08-21",
        "iteration_id": "iteration-01",
        "source_run_id": "source-run-01",
        "batch_id": "batch-01",
        "idempotency_key": "acme-rdi-v2",
        "leadgen_business_date": PRAGUE_DATE,
        "business_timezone": "Europe/Prague",
        "target_environment": "staging",
        "evidence_bundle_sha256": bundle_sha256,
    }


class FakeV2Store:
    def __init__(self) -> None:
        self.bundles: dict[str, dict[str, Any]] = {}
        self.intakes: dict[str, dict[str, Any]] = {}
        self.availability = {
            "state": "open",
            "accepting_new_analyses": True,
            "blocked_until": None,
            "reason_code": None,
        }
        self.started_count = 0

    def is_configured(self) -> bool:
        return True

    def put_machine_v2_evidence_bundle(self, **record: Any) -> dict[str, Any]:
        existing = self.bundles.get(record["bundle_sha256"])
        if existing is not None:
            return {"action": "existing", **deepcopy(existing)}
        self.bundles[record["bundle_sha256"]] = deepcopy(record)
        return {"action": "created", **deepcopy(record)}

    def create_machine_v2_intake(self, **record: Any) -> dict[str, Any]:
        existing = self.intakes.get(record["intake_id"])
        if existing is not None:
            return {"action": "existing", **deepcopy(existing)}
        bundle = self.bundles.get(record["evidence_bundle_sha256"])
        if bundle is None:
            return {"action": "bundle_missing"}
        row = {
            **deepcopy(record),
            "lifecycle_state": "intake_pending",
            "wait_reason": None,
            "blocked_until": None,
            "actual_start_business_date": None,
            "job_id": None,
        }
        self.intakes[row["intake_id"]] = row
        return {"action": "created", **deepcopy(row)}

    def reserve_machine_v2_start(self, **request: Any) -> dict[str, Any]:
        row = self.intakes[request["intake_id"]]
        bundle = self.bundles[row["evidence_bundle_sha256"]]
        if bundle["requires_specter_mcp"] and not self.availability["accepting_new_analyses"]:
            row.update(
                lifecycle_state="pending_provider_quota",
                wait_reason="specter_mcp_quota_exhausted",
                blocked_until=self.availability["blocked_until"],
            )
            return {"action": "pending_provider_quota", **deepcopy(row), "daily_started_count": self.started_count}
        self.started_count += 1
        row.update(
            lifecycle_state="start_reserved",
            wait_reason=None,
            blocked_until=None,
            actual_start_business_date=request["actual_start_business_date"],
            job_id=request["job_id"],
        )
        return {
            "action": "reserved",
            **deepcopy(row),
            "daily_start_limit": request["daily_start_limit"],
            "daily_started_count": self.started_count,
            "daily_remaining_capacity": request["daily_start_limit"] - self.started_count,
            "bundle_payload": deepcopy(bundle["payload"]),
        }

    def finalize_machine_v2_start(self, **request: Any) -> dict[str, Any]:
        row = self.intakes[request["intake_id"]]
        row["lifecycle_state"] = request["lifecycle_state"]
        return deepcopy(row)

    def release_machine_v2_start(self, **request: Any) -> dict[str, Any]:
        row = self.intakes[request["intake_id"]]
        self.started_count -= 1
        row.update(
            lifecycle_state=request["lifecycle_state"],
            wait_reason=request["wait_reason"],
            blocked_until=request["blocked_until"],
            actual_start_business_date=None,
            job_id=None,
        )
        return {"action": "released", **deepcopy(row)}

    def load_machine_v2_lifecycle(self, intake_id: str) -> dict[str, Any] | None:
        row = self.intakes.get(intake_id)
        return deepcopy(row) if row else None


def _client(store: FakeV2Store, starts: list[dict[str, Any]]) -> TestClient:
    async def start_adapter(job_id, url_items, context, actor):
        starts.append({"job_id": job_id, "context": deepcopy(context)})
        return MachineStartAccepted(job_id=job_id, status="running")

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_v2_router(
            MachineV2Dependencies(store=store, start_adapter=start_adapter)
        )
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_HEADERS["X-LeadGen-Service-Key"])
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_LEADGEN_MACHINE_V2_ENABLED", "true")
    monkeypatch.setenv("RDI_LEADGEN_TARGET_ENVIRONMENT", "staging")
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "20")
    monkeypatch.setenv("RDI_SCORING_VERSION", "ranking-v1")


def test_v2_writes_are_dormant_until_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RDI_LEADGEN_MACHINE_V2_ENABLED", "false")
    client = _client(FakeV2Store(), [])
    bundle = _bundle(requires_specter_mcp=False)
    digest = hashlib.sha256(_canonical(bundle)).hexdigest()

    response = client.put(
        f"/api/machine/leadgen/v2/evidence-bundles/{digest}",
        headers=SERVICE_HEADERS,
        json=bundle,
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "machine_v2_disabled"
    assert response.json()["detail"]["contract_version"] == "rdi.leadgen-machine.v2"


def _upload(client: TestClient, bundle: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical(bundle)).hexdigest()
    response = client.put(
        f"/api/machine/leadgen/v2/evidence-bundles/{digest}",
        headers=SERVICE_HEADERS,
        json=bundle,
    )
    assert response.status_code == 201
    return digest


def test_bundle_requires_authoritative_lite_and_routing_gate() -> None:
    client = _client(FakeV2Store(), [])
    bundle = _bundle(requires_specter_mcp=False)
    bundle["authorization"]["lite_tier"] = "hold"
    digest = hashlib.sha256(_canonical(bundle)).hexdigest()

    response = client.put(
        f"/api/machine/leadgen/v2/evidence-bundles/{digest}",
        headers=SERVICE_HEADERS,
        json=bundle,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "machine_v2_authorization_invalid"


def test_quota_blocked_intake_is_persisted_pending_without_consuming_start() -> None:
    store = FakeV2Store()
    starts: list[dict[str, Any]] = []
    client = _client(store, starts)
    digest = _upload(client, _bundle(requires_specter_mcp=True))
    created = client.post("/api/machine/leadgen/v2/intakes", headers=SERVICE_HEADERS, json=_intake(digest))
    assert created.status_code == 202
    intake_id = created.json()["intake_id"]
    store.availability.update(
        accepting_new_analyses=False,
        state="blocked",
        blocked_until="2026-08-22T00:05:00Z",
        reason_code="specter_mcp_quota_exhausted",
    )

    pending = client.post(
        f"/api/machine/leadgen/v2/intakes/{intake_id}/start",
        headers=SERVICE_HEADERS,
        json={"target_environment": "staging", "business_timezone": "Europe/Prague"},
    )

    assert pending.status_code == 202
    assert pending.json()["lifecycle_state"] == "pending_provider_quota"
    assert pending.json()["daily_started_count"] == 0
    assert starts == []


def test_complete_bundle_starts_while_gate_is_blocked_and_is_passed_to_worker() -> None:
    store = FakeV2Store()
    starts: list[dict[str, Any]] = []
    client = _client(store, starts)
    digest = _upload(client, _bundle(requires_specter_mcp=False))
    intake_id = client.post(
        "/api/machine/leadgen/v2/intakes", headers=SERVICE_HEADERS, json=_intake(digest)
    ).json()["intake_id"]
    store.availability.update(
        accepting_new_analyses=False,
        state="blocked",
        blocked_until="2026-08-22T00:05:00Z",
        reason_code="specter_mcp_quota_exhausted",
    )

    response = client.post(
        f"/api/machine/leadgen/v2/intakes/{intake_id}/start",
        headers=SERVICE_HEADERS,
        json={"target_environment": "staging", "business_timezone": "Europe/Prague"},
    )

    assert response.status_code == 202
    assert response.json()["lifecycle_state"] == "queued"
    assert response.json()["daily_started_count"] == 1
    assert starts[0]["context"]["leadgen_machine_v2"]["evidence_bundle_sha256"] == digest
    assert starts[0]["context"]["leadgen_machine_v2"]["canonical_domain"] == "acme.example"
    assert (
        starts[0]["context"]["leadgen_machine_v2"]["external_company_id"]
        == "a1f4e5d0-1111-4111-8111-111111111111"
    )
    assert starts[0]["context"]["rdi_scoring_version"] == "ranking-v1"
    assert "evidence_bundle" not in starts[0]["context"]["leadgen_machine_v2"]


def test_legacy_v2_terminal_result_recovers_bounded_pipeline_scoring_version() -> None:
    store = FakeV2Store()
    client = _client(store, [])
    digest = _upload(client, _bundle(requires_specter_mcp=False))
    intake_id = client.post(
        "/api/machine/leadgen/v2/intakes",
        headers=SERVICE_HEADERS,
        json=_intake(digest),
    ).json()["intake_id"]
    store.intakes[intake_id].update(
        lifecycle_state="succeeded",
        actual_start_business_date=PRAGUE_DATE,
        job_id="rdi-v2-job-" + "a" * 32,
        rdi_company_id="b1f4e5d0-1111-4111-8111-111111111111",
        completed_at="2026-08-23T01:00:00Z",
        terminal_result={
            "external_company_id": "a1f4e5d0-1111-4111-8111-111111111111",
            "rdi_company_id": "b1f4e5d0-1111-4111-8111-111111111111",
            "composite_score": "76.43",
            "strategy_fit_score": "69.12",
            "team_score": "74.18",
            "upside_score": "93.0",
            "rdi_bucket": "priority_review",
            "completed_at": "2026-08-23T01:00:00Z",
            "pipeline_version": "v1",
            "scoring_version": None,
        },
    )

    response = client.get(
        f"/api/machine/leadgen/v2/intakes/{intake_id}/result",
        headers=SERVICE_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["composite_score"] == "76.43"
    assert response.json()["scoring_version"] == "ranking-v1"


def test_quota_race_after_reservation_returns_to_pending_and_releases_start() -> None:
    store = FakeV2Store()

    async def reject_start(job_id, url_items, context, actor):
        return MachineStartDefiniteRejection(
            status_code=429,
            error_code="machine_specter_mcp_quota_exhausted",
            message="Specter quota is exhausted.",
            blocked_until="2026-08-22T00:05:00Z",
            retry_after_seconds=3600,
        )

    app = FastAPI()
    app.include_router(build_leadgen_machine_v2_router(
        MachineV2Dependencies(store=store, start_adapter=reject_start)
    ))
    client = TestClient(app)
    digest = _upload(client, _bundle(requires_specter_mcp=True))
    intake_id = client.post(
        "/api/machine/leadgen/v2/intakes", headers=SERVICE_HEADERS, json=_intake(digest)
    ).json()["intake_id"]

    response = client.post(
        f"/api/machine/leadgen/v2/intakes/{intake_id}/start",
        headers=SERVICE_HEADERS,
        json={"target_environment": "staging", "business_timezone": "Europe/Prague"},
    )

    assert response.status_code == 202
    assert response.json()["lifecycle_state"] == "pending_provider_quota"
    assert response.json()["actual_start_business_date"] is None
    assert store.started_count == 0


def test_bundle_hash_mismatch_fails_before_persistence() -> None:
    store = FakeV2Store()
    client = _client(store, [])

    response = client.put(
        f"/api/machine/leadgen/v2/evidence-bundles/{'0' * 64}",
        headers=SERVICE_HEADERS,
        json=_bundle(requires_specter_mcp=False),
    )

    assert response.status_code == 409
    assert store.bundles == {}
