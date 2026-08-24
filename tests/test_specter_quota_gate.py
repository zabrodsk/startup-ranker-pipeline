from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import pytest
from fastapi import UploadFile

from agent.ingest.specter_mcp_client import (
    SpecterCompanyNotFoundError,
    SpecterQuotaLimitError,
)
from agent.ingest.store import Chunk, EvidenceStore
from web import app as web_app
from web import db
from web.specter_quota_broker import (
    BROKER_MODE_ENV,
    LEGACY_MODE_ENV,
    SpecterQuotaPolicy,
    SpecterQuotaUsage,
    specter_quota_broker_configured,
    specter_quota_decision,
)
from web.specter_quota_gate import (
    SpecterQuotaGateUnavailable,
    get_specter_quota_availability,
    maybe_recover_specter_quota_gate,
    public_specter_quota_availability,
    requires_specter_mcp,
    trip_specter_quota_gate,
)


def test_requires_specter_mcp_covers_all_analysis_shapes() -> None:
    assert requires_specter_mcp(input_mode="specter", specter_urls=["acme.com"])
    assert requires_specter_mcp(input_mode="pitchdeck", use_specter_mcp=True)
    assert not requires_specter_mcp(input_mode="specter", specter_urls=[])
    assert not requires_specter_mcp(input_mode="pitchdeck", use_specter_mcp=False)
    assert not requires_specter_mcp(input_mode="original", use_specter_mcp=True)


def test_legacy_broker_mode_configures_all_callers(monkeypatch) -> None:
    monkeypatch.delenv(BROKER_MODE_ENV, raising=False)
    monkeypatch.setenv(LEGACY_MODE_ENV, "enforce")

    assert specter_quota_broker_configured() is True


def test_document_augmentation_broker_key_uses_detected_company_domain() -> None:
    store = EvidenceStore(
        startup_slug="acme",
        chunks=[
            Chunk(
                chunk_id="chunk_1",
                text="Company website: https://acme.example",
                source_file="deck.pdf",
                page_or_slide="1",
            )
        ],
    )

    assert web_app._document_augmentation_company_ref(store) == "domain:acme.example"


def test_policy_simulation_preserves_safety_reserve_at_observed_limit() -> None:
    policy = SpecterQuotaPolicy()
    usage = SpecterQuotaUsage()
    classes = (
        ["autonomous_campaign"] * 160
        + ["scheduled_import"] * 40
        + ["recovery_probe"] * 5
        + ["flexible_pool"] * 20
    )
    for quota_class in classes:
        allowed, reason = specter_quota_decision(
            policy=policy,
            usage=usage,
            quota_class=quota_class,
            operation="find_company",
            remaining_rdi_slots=20,
        )
        assert allowed, (quota_class, usage, reason)
        usage = SpecterQuotaUsage(
            total=usage.total + 1,
            scheduled_import=usage.scheduled_import + (quota_class == "scheduled_import"),
            recovery_probe=usage.recovery_probe + (quota_class == "recovery_probe"),
            autonomous_campaign=usage.autonomous_campaign
            + (quota_class == "autonomous_campaign"),
        )

    assert usage.total == 225
    assert policy.observed_limit - usage.total == policy.safety_reserve
    assert specter_quota_decision(
        policy=policy,
        usage=usage,
        quota_class="flexible_pool",
        operation="find_company",
        remaining_rdi_slots=0,
    ) == (False, "quota_estimate_exhausted")


def test_policy_simulation_enforces_company_and_founder_caps() -> None:
    policy = SpecterQuotaPolicy()
    assert specter_quota_decision(
        policy=policy,
        usage=SpecterQuotaUsage(company=8),
        quota_class="autonomous_campaign",
        operation="find_company",
        remaining_rdi_slots=1,
    ) == (False, "company_cap_exhausted")
    assert specter_quota_decision(
        policy=policy,
        usage=SpecterQuotaUsage(founder_profiles=3),
        quota_class="autonomous_campaign",
        operation="get_person_profile",
        remaining_rdi_slots=1,
    ) == (False, "founder_profile_cap_exhausted")


def test_serialized_concurrent_simulation_cannot_breach_daily_capacity() -> None:
    policy = SpecterQuotaPolicy()
    usage = [SpecterQuotaUsage()]
    accepted = [0]
    lock = Lock()
    classes = (
        ["flexible_pool"] * 100
        + ["scheduled_import"] * 60
        + ["recovery_probe"] * 20
        + ["autonomous_campaign"] * 200
    )

    def attempt(quota_class: str) -> None:
        with lock:
            current = usage[0]
            allowed, _ = specter_quota_decision(
                policy=policy,
                usage=current,
                quota_class=quota_class,
                operation="find_company",
                remaining_rdi_slots=20,
            )
            if not allowed:
                return
            usage[0] = SpecterQuotaUsage(
                total=current.total + 1,
                scheduled_import=current.scheduled_import
                + (quota_class == "scheduled_import"),
                recovery_probe=current.recovery_probe + (quota_class == "recovery_probe"),
                autonomous_campaign=current.autonomous_campaign
                + (quota_class == "autonomous_campaign"),
            )
            accepted[0] += 1

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(attempt, classes))

    assert accepted[0] == 225
    assert usage[0].total == policy.observed_limit - policy.safety_reserve


def test_open_gate_resumes_preserved_document_job_once(monkeypatch) -> None:
    job_id = "quota-resume-test"
    launched: list[str] = []
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="pending",
        progress="Waiting for Specter quota reset.",
    )
    web_app._results_cache[job_id] = {
        "specter_quota_waiting": True,
        "analysis_thread_active": False,
        "analysis_runtime_args": {"input_mode": "pitchdeck"},
    }

    def set_status(target: str, status: str, *_args: Any, **_kwargs: Any) -> None:
        web_app._jobs[target].status = status

    monkeypatch.setattr(web_app, "_set_job_status", set_status)
    monkeypatch.setattr(web_app, "_append_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(
        web_app,
        "_launch_cached_analysis_thread",
        lambda target: launched.append(target),
    )
    try:
        web_app._resume_waiting_specter_document_jobs()
        web_app._resume_waiting_specter_document_jobs()
        assert launched == [job_id]
        assert web_app._jobs[job_id].status == "running"
        assert web_app._results_cache[job_id]["specter_quota_waiting"] is False
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


class FakeGateStore:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.configured = True
        self.state = state or {
            "provider": "specter_mcp",
            "target_environment": "staging",
            "state": "open",
            "enforcement_enabled": True,
            "accepting_new_analyses": True,
            "quota_remaining": "unknown",
            "blocked_until": None,
            "next_probe_at": None,
            "retry_after_seconds": 0,
            "reason_code": None,
            "observed_at": "2026-08-21T10:00:00Z",
        }
        self.trip_calls: list[dict[str, Any]] = []
        self.acquire_calls: list[dict[str, Any]] = []
        self.finish_calls: list[dict[str, Any]] = []
        self.acquire_action = "acquired"

    def is_configured(self) -> bool:
        return self.configured

    def get_specter_mcp_quota_gate(self, **_kwargs: Any) -> dict[str, Any] | None:
        return deepcopy(self.state)

    def trip_specter_mcp_quota_gate(self, **kwargs: Any) -> dict[str, Any] | None:
        self.trip_calls.append(deepcopy(kwargs))
        self.state.update(
            {
                "state": "blocked",
                "accepting_new_analyses": not kwargs["enforcement_enabled"],
                "enforcement_enabled": kwargs["enforcement_enabled"],
                "blocked_until": "2026-08-22T00:05:00Z",
                "next_probe_at": "2026-08-22T00:05:00Z",
                "retry_after_seconds": 50_000,
                "reason_code": kwargs["reason_code"],
            }
        )
        return deepcopy(self.state)

    def acquire_specter_mcp_quota_probe(self, **kwargs: Any) -> dict[str, Any] | None:
        self.acquire_calls.append(deepcopy(kwargs))
        return {**deepcopy(self.state), "action": self.acquire_action}

    def finish_specter_mcp_quota_probe(self, **kwargs: Any) -> dict[str, Any] | None:
        self.finish_calls.append(deepcopy(kwargs))
        if kwargs["succeeded"]:
            self.state.update(
                {
                    "state": "open",
                    "accepting_new_analyses": True,
                    "blocked_until": None,
                    "next_probe_at": None,
                    "retry_after_seconds": 0,
                    "reason_code": None,
                }
            )
        return deepcopy(self.state)


def _blocked_state() -> dict[str, Any]:
    return {
        "provider": "specter_mcp",
        "target_environment": "staging",
        "state": "blocked",
        "enforcement_enabled": True,
        "accepting_new_analyses": False,
        "quota_remaining": "unknown",
        "blocked_until": "2026-08-22T00:05:00Z",
        "next_probe_at": "2026-08-21T10:00:00Z",
        "retry_after_seconds": 0,
        "reason_code": "specter_mcp_quota_exhausted",
        "observed_at": "2026-08-21T10:00:00Z",
    }


def test_enforced_gate_fails_closed_when_storage_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeGateStore()
    store.configured = False
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "enforce")

    with pytest.raises(SpecterQuotaGateUnavailable):
        get_specter_quota_availability(store)


def test_observe_mode_remains_open_when_storage_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeGateStore()
    store.configured = False
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "observe")

    availability = get_specter_quota_availability(store)

    assert availability["state"] == "open"
    assert availability["accepting_new_analyses"] is True
    assert availability["gate_storage_available"] is False


def test_trip_records_the_shared_gate_with_safe_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeGateStore()
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "enforce")
    error = SpecterQuotaLimitError(
        "You have used all your 100 MCP credits today. MCP calls are paused "
        "until the daily reset at 00:00 UTC."
    )

    availability = trip_specter_quota_gate(
        store,
        error=error,
        source_component="specter_company_worker",
        source_job_id="job-1",
    )

    assert availability["accepting_new_analyses"] is False
    assert store.trip_calls == [
        {
            "target_environment": "staging",
            "enforcement_enabled": True,
            "reason_code": "specter_mcp_quota_exhausted",
            "reset_hint": "00:00 UTC",
            "source_component": "specter_company_worker",
            "source_job_id": "job-1",
            "retry_after_seconds": None,
        }
    ]


def test_trip_records_temporary_provider_outage_with_short_recovery_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeGateStore()
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "enforce")

    availability = trip_specter_quota_gate(
        store,
        error=RuntimeError("provider transport unavailable"),
        source_component="specter_company_worker",
        source_job_id="job-1",
        retry_after_seconds=300,
        reason_code="specter_mcp_unavailable",
    )

    assert availability["accepting_new_analyses"] is False
    assert store.trip_calls[0]["reason_code"] == "specter_mcp_unavailable"
    assert store.trip_calls[0]["retry_after_seconds"] == 300
    assert store.trip_calls[0]["reset_hint"] is None


def test_recovery_probe_skips_same_day_probe_after_quota_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeGateStore(_blocked_state())
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "enforce")
    calls: list[str] = []

    availability = maybe_recover_specter_quota_gate(
        store,
        probe=lambda: calls.append("probe"),
    )

    assert calls == []
    assert availability["state"] == "blocked"
    assert store.finish_calls == []


def test_recovery_probe_does_not_run_not_found_probe_on_same_business_day(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeGateStore(_blocked_state())
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "enforce")

    def probe() -> None:
        raise SpecterCompanyNotFoundError("No company found")

    availability = maybe_recover_specter_quota_gate(store, probe=probe)

    assert availability["state"] == "blocked"
    assert store.finish_calls == []


def test_recovery_probe_does_not_run_quota_probe_on_same_business_day(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeGateStore(_blocked_state())
    monkeypatch.setenv("SPECTER_MCP_QUOTA_GATE_MODE", "enforce")

    def probe() -> None:
        raise SpecterQuotaLimitError("secret account detail: Daily MCP limit reached")

    maybe_recover_specter_quota_gate(store, probe=probe)

    assert store.finish_calls == []


def test_public_availability_drops_internal_source_and_probe_fields() -> None:
    payload = {
        **_blocked_state(),
        "source_component": "worker",
        "source_job_id": "job-secret",
        "probe_lease_token": "token-secret",
        "probe_lease_until": "2026-08-21T10:01:00Z",
    }

    public = public_specter_quota_availability(payload)

    assert public["state"] == "blocked"
    assert "source_component" not in public
    assert "source_job_id" not in public
    assert "probe_lease_token" not in public


def test_public_availability_exposes_new_circuit_state_while_preserving_legacy_state() -> None:
    payload = {
        "provider": "specter_mcp",
        "target_environment": "staging",
        "circuit_state": "closed",
        "enforcement_enabled": True,
        "accepting_new_analyses": True,
        "estimated_remaining": 144,
        "business_date": "2026-08-24",
        "retry_at": None,
        "reason": None,
        "observed_at": "2026-08-24T08:00:00Z",
    }

    public = public_specter_quota_availability(payload)

    assert public["circuit_state"] == "closed"
    assert public["state"] == "open"
    assert public["estimated_remaining"] == 144
    assert public["business_date"] == "2026-08-24"


class _FakePostgrestError(Exception):
    code = "P0001"
    message = "specter_mcp_quota_gate_blocked"
    details = (
        '{"blocked_until":"2026-08-22T00:05:00+00:00",'
        '"next_probe_at":"2026-08-22T00:05:00+00:00",'
        '"reason_code":"specter_mcp_quota_exhausted"}'
    )


def test_machine_trigger_rejection_maps_to_provider_blocked() -> None:
    mapped = db._specter_gate_block_from_exception(_FakePostgrestError())

    assert mapped == {
        "action": "provider_blocked",
        "reason_code": "specter_mcp_quota_exhausted",
        "blocked_until": "2026-08-22T00:05:00+00:00",
        "next_probe_at": "2026-08-22T00:05:00+00:00",
    }


class _FakeExecution:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class _FakeRequest:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def execute(self) -> _FakeExecution:
        return _FakeExecution(self.data)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(self, name: str, params: dict[str, Any]) -> _FakeRequest:
        self.calls.append((name, params))
        return _FakeRequest({"state": "open"})


def test_database_gate_adapter_uses_named_rpc_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(db, "_get_client", lambda: client)

    result = db.trip_specter_mcp_quota_gate(
        target_environment="staging",
        enforcement_enabled=True,
        reason_code="specter_mcp_quota_exhausted",
        reset_hint="00:00 UTC",
        source_component="api_preflight",
        source_job_id="job-1",
        retry_after_seconds=None,
    )

    assert result == {"state": "open"}
    assert client.calls == [
        (
            "trip_specter_mcp_quota_gate",
            {
                "p_target_environment": "staging",
                "p_enforcement_enabled": True,
                "p_reason_code": "specter_mcp_quota_exhausted",
                "p_reset_hint": "00:00 UTC",
                "p_source_component": "api_preflight",
                "p_source_job_id": "job-1",
                "p_retry_after_seconds": None,
            },
        )
    ]


def test_database_broker_reservation_adapter_uses_named_rpc_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(db, "_get_client", lambda: client)

    result = db.reserve_specter_quota_authorization(
        target_environment="staging",
        business_date="2026-08-24",
        business_timezone="Europe/Prague",
        consumer="rdi",
        company_ref="domain:acme.example",
        operation="get_company_profile",
        quota_class="autonomous_campaign",
        idempotency_key="auth-1",
        enforcement_enabled=True,
        remaining_rdi_slots=12,
        actor="service:rockaway-leadgen",
        metadata={"intake_id": "rdi-v2-intake-" + "a" * 32},
    )

    assert result == {"state": "open"}
    assert client.calls == [
        (
            "reserve_specter_quota_authorization",
            {
                "p_target_environment": "staging",
                "p_business_date": "2026-08-24",
                "p_business_timezone": "Europe/Prague",
                "p_consumer": "rdi",
                "p_company_ref": "domain:acme.example",
                "p_operation": "get_company_profile",
                "p_quota_class": "autonomous_campaign",
                "p_idempotency_key": "auth-1",
                "p_enforcement_enabled": True,
                "p_remaining_rdi_slots": 12,
                "p_actor": "service:rockaway-leadgen",
                "p_metadata": {"intake_id": "rdi-v2-intake-" + "a" * 32},
            },
        )
    ]


def test_url_upload_rejects_before_creating_job_when_gate_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_availability() -> dict[str, Any]:
        return _blocked_state()

    monkeypatch.setattr(web_app, "_check_session", lambda _session: True)
    monkeypatch.setattr(
        web_app,
        "_specter_mcp_availability_with_recovery",
        blocked_availability,
    )
    before = set(web_app._jobs)

    response = asyncio.run(
        web_app.upload_urls(
            {"urls": ["galtea.ai"]},
            session_id="session",
        )
    )

    assert response.status_code == 429
    assert response.headers["retry-after"]
    assert json.loads(response.body)["code"] == "specter_mcp_quota_exhausted"
    assert set(web_app._jobs) == before


def test_pitchdeck_upload_with_mcp_is_blocked_before_file_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_availability() -> dict[str, Any]:
        return _blocked_state()

    monkeypatch.setattr(web_app, "_check_session", lambda _session: True)
    monkeypatch.setattr(
        web_app,
        "_specter_mcp_availability_with_recovery",
        blocked_availability,
    )
    monkeypatch.setattr(
        web_app.tempfile,
        "mkdtemp",
        lambda: (_ for _ in ()).throw(AssertionError("blocked upload must not create storage")),
    )

    response = asyncio.run(
        web_app.upload_files(
            files=[UploadFile(filename="Galtea.pdf", file=BytesIO(b"deck"))],
            input_mode="pitchdeck",
            use_specter_mcp=True,
            session_id="session",
        )
    )

    assert response.status_code == 429
    assert json.loads(response.body)["code"] == "specter_mcp_quota_exhausted"


def test_csv_and_document_only_inputs_do_not_require_specter_gate() -> None:
    assert web_app._analysis_requires_specter_mcp_start_preflight(
        "missing-job",
        web_app.AnalyzeRequest(input_mode="pitchdeck", use_specter_mcp=False),
    ) is False
    web_app._results_cache["csv-job"] = {
        "specter": {"companies": "companies.csv"},
        "specter_urls": [],
    }
    try:
        assert web_app._analysis_requires_specter_mcp_start_preflight(
            "csv-job",
            web_app.AnalyzeRequest(input_mode="specter"),
        ) is False
    finally:
        web_app._results_cache.pop("csv-job", None)


def test_gate_migration_is_service_only_and_serializes_machine_races() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "supabase"
        / "migrations"
        / "20260821094721_specter_mcp_quota_gate.sql"
    ).read_text().lower()

    assert "create table if not exists public.specter_mcp_quota_gate" in migration
    assert "alter table public.specter_mcp_quota_gate enable row level security" in migration
    assert "security invoker" in migration
    assert "from public, anon, authenticated" in migration
    assert "to service_role" in migration
    assert "hashtextextended('specter_mcp_gate:' ||" in migration
    assert "where idempotency_identity = new.idempotency_identity" in migration
    assert "specter_mcp_quota_gate_blocked" in migration
    assert "create index" not in migration
