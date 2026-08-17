from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from web.leadgen_machine import (
    CONTRACT_VERSION,
    SERVICE_ACTOR,
    MachineIntakeRequest,
    MachineLifecycleDependencies,
    MachineStartAccepted,
    _daily_start_limit,
    build_leadgen_machine_router,
)

SERVICE_KEY = "unit-test-machine-key"
SERVICE_HEADERS = {"X-LeadGen-Service-Key": SERVICE_KEY}
BUSINESS_TIMEZONE = "Europe/Prague"
BUSINESS_DATE = datetime.now(timezone.utc).astimezone(
    ZoneInfo(BUSINESS_TIMEZONE)
).date().isoformat()


def _load_local_specter_batch_worker() -> Any:
    worker_path = Path("src/agent/specter_batch_worker.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "sprint_06a_specter_batch_worker",
        worker_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _configured_scoring_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDI_SCORING_VERSION", "ranking-v1")
    monkeypatch.setenv("RDI_LEADGEN_TARGET_ENVIRONMENT", "staging")


def _request(**overrides: Any) -> dict[str, Any]:
    payload = {
        "external_company_id": "lg-company-001",
        "canonical_domain": "acme.example",
        "campaign_id": "campaign-2026-07-31",
        "iteration_id": "iteration-01",
        "source_run_id": "source-run-01",
        "batch_id": "batch-01",
        "idempotency_key": "company-001-rdi-v1",
        "business_date": BUSINESS_DATE,
        "business_timezone": BUSINESS_TIMEZONE,
        "target_environment": "staging",
        "provenance_reference": "leadgen://source-run-01/company/lg-company-001",
    }
    payload.update(overrides)
    return payload


class FakeMachineStore:
    def __init__(self, events: list[str] | None = None) -> None:
        self.created: list[dict[str, Any]] = []
        self.intakes_by_identity: dict[str, dict[str, Any]] = {}
        self.intakes_by_id: dict[str, dict[str, Any]] = {}
        self.reserve_calls: list[dict[str, Any]] = []
        self.release_calls: list[dict[str, Any]] = []
        self.finalize_calls: list[dict[str, Any]] = []
        self.load_calls: list[str] = []
        self.events = events if events is not None else []
        self.fail_reservation = False
        self.fail_release = False
        self._lock = threading.Lock()

    def is_configured(self) -> bool:
        return True

    def create_machine_intake(self, **record: Any) -> dict[str, Any]:
        existing = self.intakes_by_identity.get(record["idempotency_identity"])
        if existing is not None:
            if existing["payload_hash"] != record["payload_hash"]:
                return {"action": "conflict", **deepcopy(existing)}
            return {"action": "existing", **deepcopy(existing)}
        self.created.append(deepcopy(record))
        self.intakes_by_identity[record["idempotency_identity"]] = deepcopy(record)
        self.intakes_by_id[record["intake_id"]] = deepcopy(record)
        self.events.append("intake_committed")
        return {"action": "created", **deepcopy(record)}

    def reserve_machine_start(self, **request: Any) -> dict[str, Any] | None:
        with self._lock:
            self.reserve_calls.append(deepcopy(request))
            if self.fail_reservation:
                return None
            record = self.intakes_by_id.get(request["intake_id"])
            if record is None:
                return {"action": "unknown"}
            if record["target_environment"] != request["target_environment"]:
                return {"action": "environment_mismatch", **deepcopy(record)}
            if (
                record["business_date"] != request["business_date"]
                or record["business_timezone"] != request["business_timezone"]
            ):
                return {"action": "scope_mismatch", **deepcopy(record)}
            daily_starts = sum(
                1
                for item in self.intakes_by_id.values()
                if item["target_environment"] == record["target_environment"]
                and item["business_date"] == record["business_date"]
                and item["lifecycle_state"]
                in {
                    "start_fenced",
                    "uncertain",
                    "queued",
                    "running",
                    "succeeded",
                    "failed",
                    "cancelled",
                }
            )
            capacity = {
                "daily_start_limit": request["daily_start_limit"],
                "daily_started_count": daily_starts,
                "daily_remaining_capacity": max(
                    request["daily_start_limit"] - daily_starts,
                    0,
                ),
            }
            state = record["lifecycle_state"]
            if state in {"start_fenced", "uncertain", "queued", "running"}:
                return {"action": "existing", **deepcopy(record), **capacity}
            if state in {"rejected", "failed", "cancelled", "succeeded"}:
                return {"action": "terminal_invalid", **deepcopy(record), **capacity}
            if daily_starts >= request["daily_start_limit"]:
                return {"action": "rate_limited", **deepcopy(record), **capacity}
            record.update(
                {
                    "lifecycle_state": "start_fenced",
                    "job_id": request["job_id"],
                    "start_actor": request["actor"],
                }
            )
            self.events.append("fence_committed")
            return {
                "action": "reserved",
                **deepcopy(record),
                "daily_start_limit": request["daily_start_limit"],
                "daily_started_count": daily_starts + 1,
                "daily_remaining_capacity": max(
                    request["daily_start_limit"] - daily_starts - 1,
                    0,
                ),
            }

    def finalize_machine_start(self, **request: Any) -> dict[str, Any] | None:
        with self._lock:
            self.finalize_calls.append(deepcopy(request))
            record = self.intakes_by_id.get(request["intake_id"])
            if record is None or record.get("job_id") != request["job_id"]:
                return None
            record.update(
                {
                    "lifecycle_state": request["lifecycle_state"],
                    "safe_error_code": request.get("safe_error_code"),
                    "safe_error_class": request.get("safe_error_class"),
                    "safe_error_message": request.get("safe_error_message"),
                }
            )
            self.events.append(f"start_finalized:{request['lifecycle_state']}")
            return deepcopy(record)

    def release_machine_start(self, **request: Any) -> dict[str, Any] | None:
        with self._lock:
            self.release_calls.append(deepcopy(request))
            if self.fail_release:
                return None
            record = self.intakes_by_id.get(request["intake_id"])
            if (
                record is None
                or record.get("job_id") != request["job_id"]
                or record.get("start_actor") != request["actor"]
                or record.get("lifecycle_state") != "start_fenced"
            ):
                return None
            record.update(
                {
                    "lifecycle_state": "accepted",
                    "job_id": None,
                    "start_actor": None,
                    "started_at": None,
                    "safe_error_code": None,
                    "safe_error_class": None,
                    "safe_error_message": None,
                }
            )
            self.events.append("start_released")
            return deepcopy(record)

    def load_machine_lifecycle(self, intake_id: str) -> dict[str, Any] | None:
        self.load_calls.append(intake_id)
        record = self.intakes_by_id.get(intake_id)
        return deepcopy(record) if record is not None else None


def _client(store: FakeMachineStore, starts: list[dict[str, Any]]) -> TestClient:
    async def start_adapter(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> MachineStartAccepted:
        starts.append(
            {
                "job_id": job_id,
                "url_items": deepcopy(url_items),
                "context": deepcopy(context),
                "actor": deepcopy(actor),
            }
        )
        return MachineStartAccepted(status="running", job_id=job_id)

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=start_adapter,
            )
        )
    )
    return TestClient(app)


def test_machine_intake_model_is_strict_bounded_and_company_owned() -> None:
    model = MachineIntakeRequest.model_validate(_request())

    assert model.canonical_domain == "acme.example"
    assert model.model_dump()["target_environment"] == "staging"

    invalid_payloads = [
        _request(unexpected="not allowed"),
        _request(external_company_id=""),
        _request(campaign_id=" leading-space"),
        _request(iteration_id="x" * 129),
        _request(canonical_domain="github.com"),
        _request(canonical_domain="acme example"),
        _request(provenance_reference="line-one\nline-two"),
        _request(target_environment="prod-ish"),
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            MachineIntakeRequest.model_validate(payload)


def test_shared_domain_normalizer_aligns_machine_human_and_persistence() -> None:
    from web import db
    from web.leadgen_domain import normalize_company_domain

    model = MachineIntakeRequest.model_validate(
        _request(canonical_domain="https://WWW.Acme.Example:443/path")
    )

    assert model.canonical_domain == "acme.example"
    assert normalize_company_domain("https://WWW.Acme.Example:443/path") == "acme.example"
    assert normalize_company_domain("deals.acme.example/path") == "deals.acme.example"
    assert db._normalize_company_key(None, "https://WWW.Acme.Example:443/path") == "domain:acme.example"
    for invalid in (
        "ftp://acme.example",
        "https://user@acme.example",
        "https://acme.example:bad/path",
        "https://acme example/path",
        "localhost",
    ):
        assert normalize_company_domain(invalid) is None


def test_every_shared_source_host_and_subdomain_is_rejected_by_machine_policy() -> None:
    from web.leadgen_domain import COMPANY_SOURCE_HOSTS, company_source_host

    assert "substack.com" in COMPANY_SOURCE_HOSTS
    for source_host in COMPANY_SOURCE_HOSTS:
        assert company_source_host(source_host) == source_host
        assert company_source_host(f"company.{source_host}") == source_host
        for candidate in (source_host, f"company.{source_host}"):
            with pytest.raises(ValidationError):
                MachineIntakeRequest.model_validate(
                    _request(canonical_domain=f"https://WWW.{candidate}/profile")
                )


def test_machine_auth_matrix_is_dedicated_redacted_and_constant_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    client = _client(store, starts)
    monkeypatch.delenv("RDI_LEADGEN_AUTOSTART_KEY", raising=False)

    missing_configuration = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    )

    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    missing = client.post("/api/machine/leadgen/v1/intakes", json=_request())
    wrong = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers={"X-LeadGen-Service-Key": "wrong-key"},
        json=_request(),
    )
    accepted = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    )

    assert missing_configuration.status_code == 503
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 202
    assert accepted.json()["contract_version"] == CONTRACT_VERSION
    assert accepted.json()["external_company_id"] == "lg-company-001"
    assert accepted.json()["approval_required"] is False
    assert SERVICE_KEY not in str(missing_configuration.json())
    assert SERVICE_KEY not in str(missing.json())
    assert SERVICE_KEY not in str(wrong.json())
    assert SERVICE_KEY not in str(accepted.json())
    assert len(store.created) == 1
    assert starts == []


@pytest.mark.parametrize("route_kind", ["intake", "start"])
def test_machine_auth_precedes_body_validation_and_sanitizes_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    route_kind: str,
) -> None:
    marker = "submitted-credential-like-marker-must-not-reflect"
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    client = _client(store, starts)
    if route_kind == "intake":
        path = "/api/machine/leadgen/v1/intakes"
        invalid_body = _request(external_company_id={"credential": marker})
    else:
        intake = _ingest_machine(client)
        path = f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}/start"
        invalid_body = {"target_environment": marker}

    monkeypatch.delenv("RDI_LEADGEN_AUTOSTART_KEY", raising=False)
    missing_server_key = client.post(path, headers=SERVICE_HEADERS, json=invalid_body)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    missing_header = client.post(path, json=invalid_body)
    wrong_header = client.post(
        path,
        headers={"X-LeadGen-Service-Key": "wrong-key"},
        json=invalid_body,
    )
    authenticated_invalid = client.post(path, headers=SERVICE_HEADERS, json=invalid_body)

    assert missing_server_key.status_code == 503
    assert missing_server_key.json()["detail"]["code"] == "machine_auth_not_configured"
    assert missing_header.status_code == 401
    assert wrong_header.status_code == 401
    assert authenticated_invalid.status_code == 422
    assert authenticated_invalid.json() == {
        "detail": {
            "code": "machine_request_invalid",
            "message": "Machine request is invalid.",
            "contract_version": CONTRACT_VERSION,
        }
    }
    for response in (
        missing_server_key,
        missing_header,
        wrong_header,
        authenticated_invalid,
    ):
        assert marker not in response.text
    assert store.reserve_calls == []
    assert starts == []


@pytest.mark.parametrize(
    ("server_environment", "request_environment", "expected_status"),
    [
        (None, "staging", 503),
        ("preview", "staging", 503),
        ("production", "staging", 409),
    ],
)
def test_intake_rejects_unbound_server_environment_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    server_environment: str | None,
    request_environment: str,
    expected_status: int,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    if server_environment is None:
        monkeypatch.delenv("RDI_LEADGEN_TARGET_ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("RDI_LEADGEN_TARGET_ENVIRONMENT", server_environment)
    client = _client(store, starts)

    response = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(target_environment=request_environment),
    )

    assert response.status_code == expected_status
    expected_code = (
        "machine_target_environment_not_configured"
        if expected_status == 503
        else "machine_target_environment_mismatch"
    )
    assert response.json()["detail"]["code"] == expected_code
    assert store.created == []
    assert store.reserve_calls == []
    assert starts == []


@pytest.mark.parametrize("server_environment", [None, "preview", "production"])
def test_start_rechecks_server_environment_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
    server_environment: str | None,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    client = _client(store, starts)
    intake = _ingest_machine(client)
    if server_environment is None:
        monkeypatch.delenv("RDI_LEADGEN_TARGET_ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("RDI_LEADGEN_TARGET_ENVIRONMENT", server_environment)

    response = _start_machine(client, intake["intake_id"])

    assert response.status_code in {409, 503}
    assert store.reserve_calls == []
    assert starts == []


def test_one_runtime_cannot_bypass_limit_with_alternate_environment_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "1")
    monkeypatch.setenv("RDI_LEADGEN_TARGET_ENVIRONMENT", "staging")
    client = _client(store, starts)
    first = _ingest_machine(client, campaign_id="campaign-staging")

    first_started = _start_machine(client, first["intake_id"])
    alternate = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(
            external_company_id="lg-company-002",
            canonical_domain="beta.example",
            campaign_id="campaign-production-label",
            idempotency_key="company-002-rdi-v1",
            target_environment="production",
        ),
    )

    assert first_started.status_code == 202
    assert alternate.status_code == 409
    assert alternate.json()["detail"]["code"] == "machine_target_environment_mismatch"
    assert len(store.created) == 1
    assert len(store.reserve_calls) == 1
    assert len(starts) == 1


def test_new_intake_for_closed_business_date_is_rejected_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])

    response = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(business_date="2020-01-01"),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "machine_business_date_closed"
    assert store.created == []


def test_deprecated_global_limit_alias_remains_compatible_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("RDI_LEADGEN_DAILY_START_LIMIT", raising=False)
    monkeypatch.setenv("RDI_LEADGEN_GLOBAL_START_LIMIT", "7")

    assert _daily_start_limit() == 7
    assert "deprecated" in caplog.text


def test_conflicting_daily_and_deprecated_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "7")
    monkeypatch.setenv("RDI_LEADGEN_GLOBAL_START_LIMIT", "8")

    with pytest.raises(HTTPException) as error:
        _daily_start_limit()

    assert error.value.status_code == 503
    assert error.value.detail["code"] == "invalid_start_configuration"


def test_configured_daily_limit_cannot_raise_absolute_twenty_start_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "21")
    monkeypatch.delenv("RDI_LEADGEN_GLOBAL_START_LIMIT", raising=False)

    with pytest.raises(HTTPException) as error:
        _daily_start_limit()

    assert error.value.status_code == 503
    assert error.value.detail["code"] == "invalid_start_configuration"


@pytest.mark.parametrize(
    ("flag_value", "expected_code"),
    [(None, "autonomous_start_disabled"), ("false", "autonomous_start_disabled"), ("yes", "invalid_start_configuration")],
)
def test_machine_start_defaults_closed_without_reservation_or_remote_call(
    monkeypatch: pytest.MonkeyPatch,
    flag_value: str | None,
    expected_code: str,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    client = _client(store, starts)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    if flag_value is None:
        monkeypatch.delenv("RDI_LEADGEN_AUTOSTART_ENABLED", raising=False)
    else:
        monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", flag_value)

    intake = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    ).json()
    response = client.post(
        f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}/start",
        headers=SERVICE_HEADERS,
        json={
            "target_environment": "staging",
            "business_date": BUSINESS_DATE,
            "business_timezone": BUSINESS_TIMEZONE,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == expected_code
    assert store.reserve_calls == []
    assert starts == []


def test_machine_start_requires_explicit_scoring_version_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.delenv("RDI_SCORING_VERSION", raising=False)
    client = _client(store, starts)
    intake = _ingest_machine(client)

    response = _start_machine(client, intake["intake_id"])

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "invalid_start_configuration"
    assert store.reserve_calls == []
    assert starts == []


def test_machine_start_carries_exact_scoring_version_into_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    contexts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_SCORING_VERSION", "ranking-2026-07-31")

    async def capture_context(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> MachineStartAccepted:
        del url_items, actor
        contexts.append(deepcopy(context))
        return MachineStartAccepted(job_id=job_id, status="running")

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=capture_context)
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    response = _start_machine(client, intake["intake_id"])

    assert response.status_code == 202, response.text
    assert contexts[0]["rdi_scoring_version"] == "ranking-2026-07-31"


def test_production_machine_start_persists_exact_scoring_version_in_job_run_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent.ingest import specter_mcp_client
    from web import app as web_app

    persisted: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []

    class RuntimeDatabase:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def insert_job_status_history(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        @staticmethod
        def persist_source_files(*args: Any, **kwargs: Any) -> bool:
            persisted.append({"args": deepcopy(args), "kwargs": deepcopy(kwargs)})
            return True

        @staticmethod
        def queue_specter_worker_job(*args: Any, **kwargs: Any) -> bool:
            queued.append({"args": deepcopy(args), "kwargs": deepcopy(kwargs)})
            return True

    class OfflineSpecterClient:
        @staticmethod
        def find_company(identifier: str) -> dict[str, str]:
            return {"domain": identifier}

    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_SCORING_VERSION", "ranking-2026-07-31")
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", RuntimeDatabase())
    monkeypatch.setattr(web_app.tempfile, "mkdtemp", lambda: str(tmp_path))
    monkeypatch.setattr(
        web_app,
        "build_default_phase_model_policy",
        lambda: SimpleNamespace(answering={"provider": "offline", "model": "test"}),
    )
    monkeypatch.setattr(web_app, "phase_model_defaults_payload", lambda: {})
    monkeypatch.setattr(web_app, "resolve_effective_phase_models", lambda _policy: {})
    monkeypatch.setattr(web_app, "build_phase_policy_display_label", lambda _models: "offline:test")
    monkeypatch.setattr(
        specter_mcp_client,
        "get_default_client",
        lambda: OfflineSpecterClient(),
    )
    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=web_app._start_leadgen_machine_url_job,
            )
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    try:
        response = _start_machine(client, intake["intake_id"])
    finally:
        job_id = store.intakes_by_id[intake["intake_id"]].get("job_id")
        if job_id:
            web_app._jobs.pop(job_id, None)
            web_app._results_cache.pop(job_id, None)

    assert response.status_code == 202, response.text
    assert persisted[0]["kwargs"]["run_config"]["rdi_scoring_version"] == "ranking-2026-07-31"
    assert queued[0]["kwargs"]["run_config"]["rdi_scoring_version"] == "ranking-2026-07-31"


def test_production_writer_claim_heartbeat_preserve_runtime_versions_for_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent.ingest import specter_mcp_client
    from web import app as web_app
    from web import db

    class MemoryQuery:
        def __init__(self, database: MemorySupabase, table_name: str) -> None:
            self.database = database
            self.table_name = table_name
            self.operation = "select"
            self.payload: dict[str, Any] | None = None
            self.filters: list[tuple[str, Any]] = []
            self.row_limit: int | None = None

        def select(self, _columns: str) -> MemoryQuery:
            self.operation = "select"
            return self

        def upsert(self, payload: dict[str, Any], **_kwargs: Any) -> MemoryQuery:
            self.operation = "upsert"
            self.payload = deepcopy(payload)
            return self

        def insert(self, payload: dict[str, Any]) -> MemoryQuery:
            self.operation = "insert"
            self.payload = deepcopy(payload)
            return self

        def delete(self) -> MemoryQuery:
            self.operation = "delete"
            return self

        def eq(self, key: str, value: Any) -> MemoryQuery:
            self.filters.append((key, value))
            return self

        def limit(self, value: int) -> MemoryQuery:
            self.row_limit = value
            return self

        def order(self, *_args: Any, **_kwargs: Any) -> MemoryQuery:
            return self

        def execute(self) -> SimpleNamespace:
            rows = self.database.rows.setdefault(self.table_name, [])
            matches = [
                row
                for row in rows
                if all(row.get(key) == value for key, value in self.filters)
            ]
            if self.operation == "upsert":
                assert self.payload is not None
                if self.table_name == "jobs":
                    existing = next(
                        (
                            row
                            for row in rows
                            if row.get("job_id_legacy") == self.payload.get("job_id_legacy")
                        ),
                        None,
                    )
                    if existing is None:
                        existing = {"id": "00000000-0000-4000-8000-000000000001"}
                        rows.append(existing)
                    existing.update(deepcopy(self.payload))
                    return SimpleNamespace(data=[deepcopy(existing)])
                rows.append(deepcopy(self.payload))
                return SimpleNamespace(data=[deepcopy(self.payload)])
            if self.operation == "insert":
                assert self.payload is not None
                rows.append(deepcopy(self.payload))
                return SimpleNamespace(data=[deepcopy(self.payload)])
            if self.operation == "delete":
                self.database.rows[self.table_name] = [row for row in rows if row not in matches]
                return SimpleNamespace(data=[])
            if self.row_limit is not None:
                matches = matches[: self.row_limit]
            return SimpleNamespace(data=deepcopy(matches))

    class MemorySupabase:
        def __init__(self) -> None:
            self.rows: dict[str, list[dict[str, Any]]] = {}

        def table(self, table_name: str) -> MemoryQuery:
            return MemoryQuery(self, table_name)

    class OfflineSpecterClient:
        @staticmethod
        def find_company(identifier: str) -> dict[str, str]:
            return {"domain": identifier}

    database = MemorySupabase()
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_SCORING_VERSION", "ranking-2026-07-31")
    monkeypatch.setenv("PIPELINE_VERSION", "pipeline-2026-07-31")
    monkeypatch.setattr(db, "_get_client", lambda: database)
    monkeypatch.setattr(db, "is_configured", lambda: True)
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", db)
    monkeypatch.setattr(web_app.tempfile, "mkdtemp", lambda: str(tmp_path))
    monkeypatch.setattr(
        web_app,
        "build_default_phase_model_policy",
        lambda: SimpleNamespace(answering={"provider": "offline", "model": "test"}),
    )
    monkeypatch.setattr(web_app, "phase_model_defaults_payload", lambda: {})
    monkeypatch.setattr(web_app, "resolve_effective_phase_models", lambda _policy: {})
    monkeypatch.setattr(web_app, "build_phase_policy_display_label", lambda _models: "offline:test")
    monkeypatch.setattr(specter_mcp_client, "get_default_client", lambda: OfflineSpecterClient())

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=web_app._start_leadgen_machine_url_job,
            )
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    try:
        started = _start_machine(client, intake["intake_id"])
        job_id = started.json()["job_id"]
        claimed = db.claim_specter_worker_job(job_id, worker_id="worker-test")
        heartbeat = db.heartbeat_specter_worker_job(
            job_id,
            status="running",
            progress="Worker running",
            worker_id="worker-test",
        )
    finally:
        persisted_job_id = store.intakes_by_id[intake["intake_id"]].get("job_id")
        if persisted_job_id:
            web_app._jobs.pop(persisted_job_id, None)
            web_app._results_cache.pop(persisted_job_id, None)

    assert started.status_code == 202, started.text
    assert claimed is not None
    assert heartbeat is True
    job_row = database.rows["jobs"][0]
    assert job_row["pipeline_version"] == "pipeline-2026-07-31"
    assert job_row["run_config"]["pipeline_version"] == "pipeline-2026-07-31"
    assert job_row["run_config"]["rdi_scoring_version"] == "ranking-2026-07-31"
    migration = Path(
        "supabase/migrations/20260731000000_leadgen_machine_lifecycle.sql"
    ).read_text(encoding="utf-8")
    assert "j.pipeline_version" in migration
    assert "j.run_config ->> 'rdi_scoring_version'" in migration


def test_machine_service_actor_is_a_fixed_auditable_identity() -> None:
    assert SERVICE_ACTOR == "service:rockaway-leadgen"


def test_machine_lifecycle_path_ids_are_bounded_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    client = _client(store, starts)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    invalid_id = "not-an-rdi-intake-reference"

    responses = [
        client.post(
            f"/api/machine/leadgen/v1/intakes/{invalid_id}/start",
            headers=SERVICE_HEADERS,
            json={
                "target_environment": "staging",
                "business_date": BUSINESS_DATE,
                "business_timezone": BUSINESS_TIMEZONE,
            },
        ),
        client.get(
            f"/api/machine/leadgen/v1/intakes/{invalid_id}/status",
            headers=SERVICE_HEADERS,
        ),
        client.get(
            f"/api/machine/leadgen/v1/intakes/{invalid_id}/result",
            headers=SERVICE_HEADERS,
        ),
        client.get(
            f"/api/machine/leadgen/v1/intakes/{invalid_id}/error",
            headers=SERVICE_HEADERS,
        ),
    ]

    assert [response.status_code for response in responses] == [422, 422, 422, 422]
    assert all(
        response.json()["detail"]["code"] == "machine_intake_id_invalid"
        for response in responses
    )
    assert store.reserve_calls == []
    assert store.load_calls == []
    assert starts == []


def test_machine_intake_exact_replay_is_byte_stable_across_router_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    first_client = _client(store, starts)

    created = first_client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    )
    replay_client = _client(store, starts)
    replayed = replay_client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    )

    assert created.status_code == 202
    assert replayed.status_code == 202
    assert replayed.content == created.content
    assert len(store.created) == 1
    assert len(store.intakes_by_identity) == 1
    assert starts == []


def test_machine_intake_same_identity_changed_material_payload_conflicts_without_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, starts)

    created = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    )
    changed_domain = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(canonical_domain="acme-changed.example"),
    )
    changed_provenance = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(
            provenance_reference="leadgen://source-run-01/company/changed"
        ),
    )

    assert created.status_code == 202
    assert changed_domain.status_code == 409
    assert changed_domain.json()["detail"]["code"] == "machine_intake_payload_conflict"
    assert changed_provenance.status_code == 409
    assert len(store.created) == 1
    assert len(store.intakes_by_identity) == 1
    assert starts == []


@pytest.mark.parametrize(
    "field",
    [
        "external_company_id",
        "campaign_id",
        "iteration_id",
        "source_run_id",
        "batch_id",
        "idempotency_key",
        "target_environment",
    ],
)
def test_machine_intake_identity_is_injective_across_stable_caller_fields(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, starts)
    first = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(),
    ).json()
    replacement = "production" if field == "target_environment" else f"changed-{field}"
    if field == "target_environment":
        monkeypatch.setenv("RDI_LEADGEN_TARGET_ENVIRONMENT", "production")

    second = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(**{field: replacement}),
    )

    assert second.status_code == 202
    assert second.json()["intake_id"] != first["intake_id"]
    assert second.json()["rdi_correlation_id"] != first["rdi_correlation_id"]
    assert len(store.created) == 2


def _ingest_machine(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post(
        "/api/machine/leadgen/v1/intakes",
        headers=SERVICE_HEADERS,
        json=_request(**overrides),
    )
    assert response.status_code == 202
    return response.json()


def _start_machine(
    client: TestClient,
    intake_id: str,
    *,
    target_environment: str = "staging",
):
    return client.post(
        f"/api/machine/leadgen/v1/intakes/{intake_id}/start",
        headers=SERVICE_HEADERS,
        json={
            "target_environment": target_environment,
            "business_date": BUSINESS_DATE,
            "business_timezone": BUSINESS_TIMEZONE,
        },
    )


def test_start_commits_durable_fence_before_exactly_one_remote_call_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    store = FakeMachineStore(events)
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    client = _client(store, starts)
    intake = _ingest_machine(client)

    original_adapter = client.app.routes[-1].endpoint  # keep a concrete route access assertion
    assert original_adapter is not None

    # The injected adapter observes already-durable state when it is entered.
    async def ordered_start(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> MachineStartAccepted:
        persisted = store.load_machine_lifecycle(intake["intake_id"])
        assert persisted is not None
        assert persisted["lifecycle_state"] == "start_fenced"
        assert persisted["job_id"] == job_id
        events.append("remote_entered")
        starts.append(
            {
                "job_id": job_id,
                "url_items": deepcopy(url_items),
                "context": deepcopy(context),
                "actor": deepcopy(actor),
            }
        )
        return MachineStartAccepted(status="running", job_id=job_id)

    ordered_app = FastAPI()
    ordered_app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=ordered_start)
        )
    )
    ordered_client = TestClient(ordered_app)

    started = _start_machine(ordered_client, intake["intake_id"])
    replayed = _start_machine(ordered_client, intake["intake_id"])

    assert started.status_code == 202
    assert replayed.status_code == 202
    assert replayed.content == started.content
    assert started.json()["lifecycle_state"] == "queued"
    assert started.json()["actor"] == SERVICE_ACTOR
    assert started.json()["uncertain"] is False
    assert len(starts) == 1
    assert starts[0]["url_items"] == ["https://acme.example"]
    assert starts[0]["actor"] == {
        "started_by_user_id": SERVICE_ACTOR,
        "started_by_display_name": SERVICE_ACTOR,
        "started_by_label": SERVICE_ACTOR,
    }
    assert store.reserve_calls[0]["target_environment"] == "staging"
    assert starts[0]["context"]["target_environment"] == "staging"
    assert starts[0]["context"]["leadgen_machine"]["target_environment"] == "staging"
    assert events == [
        "intake_committed",
        "fence_committed",
        "remote_entered",
        "start_finalized:queued",
    ]


def test_machine_start_does_not_turn_external_id_into_worker_expected_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.ingest.specter_mcp_client import _verify_match
    from agent.specter_batch_worker import _build_company_tasks

    store = FakeMachineStore()
    worker_tasks: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")

    async def worker_boundary(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> MachineStartAccepted:
        del context, actor
        worker_tasks.extend(_build_company_tasks({"specter_urls": url_items}, None))
        return MachineStartAccepted(status="running", job_id=job_id)

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=worker_boundary)
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    started = _start_machine(client, intake["intake_id"])

    assert started.status_code == 202
    assert len(worker_tasks) == 1
    assert worker_tasks[0]["url"] == "https://acme.example"
    assert worker_tasks[0]["domain"] == "acme.example"
    assert worker_tasks[0]["expected_name"] == ""
    _verify_match(
        "acme.example",
        None,
        {"name": "Acme", "domain": "acme.example"},
    )


def test_timeout_after_possible_acceptance_is_stably_uncertain_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    store = FakeMachineStore(events)
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")

    async def accepted_then_timeout(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> MachineStartAccepted:
        starts.append({"job_id": job_id})
        events.append("remote_accepted_then_timeout")
        raise TimeoutError("provider payload must never escape")

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=accepted_then_timeout,
            )
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    timed_out = _start_machine(client, intake["intake_id"])
    replayed = _start_machine(client, intake["intake_id"])

    assert timed_out.status_code == 202
    assert replayed.status_code == 202
    assert replayed.content == timed_out.content
    assert timed_out.json()["lifecycle_state"] == "uncertain"
    assert timed_out.json()["uncertain"] is True
    assert len(starts) == 1
    assert "provider payload" not in str(timed_out.json())
    assert events == [
        "intake_committed",
        "fence_committed",
        "remote_accepted_then_timeout",
        "start_finalized:uncertain",
    ]


def test_nonaccepting_remote_reply_is_uncertain_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")

    async def nonaccepting_reply(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> dict[str, Any]:
        starts.append({"job_id": job_id})
        return {"status": "rejected", "job_id": job_id}

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=nonaccepting_reply)
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    first = _start_machine(client, intake["intake_id"])
    replay = _start_machine(client, intake["intake_id"])

    assert first.status_code == 202
    assert first.json()["lifecycle_state"] == "uncertain"
    assert replay.content == first.content
    assert len(starts) == 1


def test_definite_no_start_releases_fence_and_allows_one_safe_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web.leadgen_machine import MachineStartDefiniteRejection

    events: list[str] = []
    store = FakeMachineStore(events)
    calls: list[str] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")

    async def reject_then_accept(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> Any:
        del url_items, context, actor
        calls.append(job_id)
        if len(calls) == 1:
            return MachineStartDefiniteRejection(
                status_code=503,
                error_code="machine_start_unavailable",
                message="The worker-backed start was definitely not accepted.",
            )
        return MachineStartAccepted(job_id=job_id, status="running")

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=reject_then_accept)
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    rejected = _start_machine(client, intake["intake_id"])
    retried = _start_machine(client, intake["intake_id"])

    assert rejected.status_code == 503
    assert rejected.json()["detail"] == {
        "code": "machine_start_unavailable",
        "message": "The worker-backed start was definitely not accepted.",
        "contract_version": CONTRACT_VERSION,
    }
    assert retried.status_code == 202
    assert retried.json()["lifecycle_state"] == "queued"
    assert len(calls) == 2
    assert len(store.release_calls) == 1
    assert events == [
        "intake_committed",
        "fence_committed",
        "start_released",
        "fence_committed",
        "start_finalized:queued",
    ]


def test_definite_no_start_release_failure_preserves_fence_and_blocks_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web.leadgen_machine import MachineStartDefiniteRejection

    store = FakeMachineStore()
    store.fail_release = True
    calls: list[str] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")

    async def definitely_rejected(*args: Any) -> Any:
        calls.append(str(args[0]))
        return MachineStartDefiniteRejection(
            status_code=429,
            error_code="machine_upstream_rate_limited",
            message="The upstream start limit is exhausted.",
        )

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=definitely_rejected)
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    rejected = _start_machine(client, intake["intake_id"])
    replayed = _start_machine(client, intake["intake_id"])

    assert rejected.status_code == 503
    assert rejected.json()["detail"]["code"] == "machine_start_release_failed"
    assert replayed.status_code == 202
    assert replayed.json()["lifecycle_state"] == "start_fenced"
    assert len(calls) == 1


@pytest.mark.parametrize("local_failure", ["worker_disabled", "storage_unavailable"])
def test_production_machine_adapter_classifies_only_local_preinvocation_rejections(
    monkeypatch: pytest.MonkeyPatch,
    local_failure: str,
) -> None:
    from web import app as web_app
    from web.leadgen_machine import MachineStartDefiniteRejection

    calls: list[str] = []

    async def underlying(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls.append("invoked")
        return {"status": "running", "job_id": "unexpected"}

    monkeypatch.setattr(web_app, "_start_leadgen_url_job", underlying)
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", local_failure != "worker_disabled")
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(is_configured=lambda: local_failure != "storage_unavailable"),
    )

    result = asyncio.run(
        web_app._start_leadgen_machine_url_job(
            "rdi-job-0123456789abcdef0123456789abcdef",
            ["https://acme.example"],
            {"source": "leadgen_machine"},
            {},
        )
    )

    assert isinstance(result, MachineStartDefiniteRejection)
    assert result.status_code == 503
    assert result.error_code == "machine_start_unavailable"
    assert calls == []


@pytest.mark.parametrize(
    "outcome",
    [
        JSONResponse(status_code=429, content={"detail": "secret quota payload"}),
        JSONResponse(status_code=503, content={"detail": "secret provider payload"}),
        HTTPException(status_code=400, detail="secret policy payload"),
        HTTPException(status_code=503, detail="secret persistence payload"),
    ],
)
def test_production_machine_adapter_never_proves_no_start_after_invocation(
    monkeypatch: pytest.MonkeyPatch,
    outcome: JSONResponse | HTTPException,
) -> None:
    from web import app as web_app
    from web.leadgen_machine import MachineStartDefiniteRejection

    async def underlying(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if isinstance(outcome, HTTPException):
            raise outcome
        return outcome

    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", SimpleNamespace(is_configured=lambda: True))
    monkeypatch.setattr(web_app, "_start_leadgen_url_job", underlying)

    call = web_app._start_leadgen_machine_url_job(
        "rdi-job-0123456789abcdef0123456789abcdef",
        ["https://acme.example"],
        {"source": "leadgen_machine"},
        {},
    )
    if isinstance(outcome, HTTPException):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(call)
        assert raised.value is outcome
    else:
        result = asyncio.run(call)
        assert result is outcome
        assert not isinstance(result, MachineStartDefiniteRejection)


def test_production_wrapper_post_invocation_429_is_uncertain_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web import app as web_app

    store = FakeMachineStore()
    calls: list[str] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", SimpleNamespace(is_configured=lambda: True))

    async def rate_limit_then_accept(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> Any:
        del url_items, context, actor
        calls.append(job_id)
        if len(calls) == 1:
            return JSONResponse(
                status_code=429,
                content={"detail": "secret upstream quota detail"},
            )
        return {"status": "running", "job_id": job_id}

    monkeypatch.setattr(web_app, "_start_leadgen_url_job", rate_limit_then_accept)
    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=web_app._start_leadgen_machine_url_job,
            )
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    uncertain = _start_machine(client, intake["intake_id"])
    replayed = _start_machine(client, intake["intake_id"])

    assert uncertain.status_code == 202
    assert uncertain.json()["lifecycle_state"] == "uncertain"
    assert "secret" not in uncertain.text.lower()
    assert replayed.content == uncertain.content
    assert len(calls) == 1
    assert store.release_calls == []


@pytest.mark.parametrize(
    ("provider_failure", "expected_status", "expected_code"),
    [
        ("quota", 429, "machine_upstream_rate_limited"),
        ("unavailable", 503, "machine_upstream_unavailable"),
    ],
)
def test_real_machine_preflight_rejection_releases_once_and_retries_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider_failure: str,
    expected_status: int,
    expected_code: str,
) -> None:
    from agent.ingest import specter_mcp_client
    from web import app as web_app

    runtime_events: list[str] = []

    class RuntimeDatabase:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def insert_job_status_history(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        @staticmethod
        def persist_source_files(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            runtime_events.append("persisted")
            return True

        @staticmethod
        def queue_specter_worker_job(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            runtime_events.append("queued")
            return True

    class RetryableSpecterClient:
        calls = 0

        @classmethod
        def find_company(cls, identifier: str) -> dict[str, str]:
            cls.calls += 1
            if cls.calls == 1:
                if provider_failure == "quota":
                    raise specter_mcp_client.SpecterQuotaLimitError("private quota detail")
                raise specter_mcp_client.SpecterMCPError("private provider detail")
            return {"domain": identifier}

    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", RuntimeDatabase())
    monkeypatch.setattr(web_app.tempfile, "mkdtemp", lambda: str(tmp_path))
    monkeypatch.setattr(
        web_app,
        "build_default_phase_model_policy",
        lambda: SimpleNamespace(answering={"provider": "offline", "model": "test"}),
    )
    monkeypatch.setattr(web_app, "phase_model_defaults_payload", lambda: {})
    monkeypatch.setattr(web_app, "resolve_effective_phase_models", lambda _policy: {})
    monkeypatch.setattr(web_app, "build_phase_policy_display_label", lambda _models: "offline:test")
    monkeypatch.setattr(specter_mcp_client, "get_default_client", lambda: RetryableSpecterClient())

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=web_app._start_leadgen_machine_url_job,
            )
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    try:
        rejected = _start_machine(client, intake["intake_id"])
        retried = _start_machine(client, intake["intake_id"])
        replayed = _start_machine(client, intake["intake_id"])
    finally:
        job_id = store.intakes_by_id[intake["intake_id"]].get("job_id")
        if job_id:
            web_app._jobs.pop(job_id, None)
            web_app._results_cache.pop(job_id, None)

    assert rejected.status_code == expected_status
    assert rejected.json()["detail"]["code"] == expected_code
    assert "private" not in rejected.text.lower()
    assert retried.status_code == 202
    assert retried.json()["lifecycle_state"] == "queued"
    assert replayed.content == retried.content
    assert RetryableSpecterClient.calls == 2
    assert runtime_events == ["persisted", "queued"]
    assert len(store.release_calls) == 1


@pytest.mark.parametrize(
    ("provider_failure", "expected_status", "expected_code"),
    [
        ("quota", 429, "specter_mcp_quota_exhausted"),
        ("unavailable", 503, "specter_mcp_unavailable"),
    ],
)
def test_human_preflight_provider_failures_remain_json_responses(
    monkeypatch: pytest.MonkeyPatch,
    provider_failure: str,
    expected_status: int,
    expected_code: str,
) -> None:
    from agent.ingest import specter_mcp_client
    from web import app as web_app

    class RuntimeDatabase:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def persist_source_files(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            raise AssertionError("provider preflight must not persist")

        @staticmethod
        def queue_specter_worker_job(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            raise AssertionError("provider preflight must not queue")

    class FailingSpecterClient:
        @staticmethod
        def find_company(_identifier: str) -> dict[str, str]:
            if provider_failure == "quota":
                raise specter_mcp_client.SpecterQuotaLimitError("human quota detail")
            raise specter_mcp_client.SpecterMCPError("human provider detail")

    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", RuntimeDatabase())
    monkeypatch.setattr(specter_mcp_client, "get_default_client", lambda: FailingSpecterClient())

    response = asyncio.run(
        web_app._start_leadgen_url_job(
            "lg-human-preflight",
            ["https://acme.example"],
            {"source": "leadgen"},
            {},
        )
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == expected_status
    assert json.loads(response.body)["code"] == expected_code


@pytest.mark.parametrize(
    "queue_outcome",
    ["negative_ack", "ack_lost", "claimed_then_negative_ack"],
)
def test_worker_visible_queue_failure_is_uncertain_and_replay_never_restarts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    queue_outcome: str,
) -> None:
    from agent.ingest import specter_mcp_client
    from web import app as web_app

    runtime_events: list[str] = []
    runtime_record: dict[str, Any] = {}

    class RuntimeDatabase:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def insert_job_status_history(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        @staticmethod
        def persist_source_files(*args: Any, **kwargs: Any) -> bool:
            del args
            runtime_record["worker_state"] = deepcopy(kwargs["worker_state"])
            runtime_events.append("queued_persistence_committed")
            return True

        @staticmethod
        def queue_specter_worker_job(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            runtime_events.append("queue_ack_entered")
            if queue_outcome == "ack_lost":
                raise TimeoutError("queue acknowledgement was lost")
            if queue_outcome == "claimed_then_negative_ack":
                def claim_committed_job() -> None:
                    assert runtime_record["worker_state"]["status"] == "queued"
                    runtime_record["worker_state"]["status"] = "running"
                    runtime_events.append("worker_claimed_committed_job")

                claim_thread = threading.Thread(target=claim_committed_job)
                claim_thread.start()
                claim_thread.join(timeout=2)
                assert not claim_thread.is_alive()
            return False

    class OfflineSpecterClient:
        @staticmethod
        def find_company(identifier: str) -> dict[str, str]:
            return {"domain": identifier}

    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_WORKER_SERVICE", True)
    monkeypatch.setattr(web_app, "db", RuntimeDatabase())
    monkeypatch.setattr(web_app.tempfile, "mkdtemp", lambda: str(tmp_path))
    monkeypatch.setattr(
        web_app,
        "build_default_phase_model_policy",
        lambda: SimpleNamespace(answering={"provider": "offline", "model": "test"}),
    )
    monkeypatch.setattr(web_app, "phase_model_defaults_payload", lambda: {})
    monkeypatch.setattr(web_app, "resolve_effective_phase_models", lambda _policy: {})
    monkeypatch.setattr(web_app, "build_phase_policy_display_label", lambda _models: "offline:test")
    monkeypatch.setattr(specter_mcp_client, "get_default_client", lambda: OfflineSpecterClient())

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(
                store=store,
                start_adapter=web_app._start_leadgen_machine_url_job,
            )
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    try:
        first = _start_machine(client, intake["intake_id"])
        exact_replay = _start_machine(client, intake["intake_id"])
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_replays = [
                future.result(timeout=2)
                for future in (
                    executor.submit(_start_machine, client, intake["intake_id"]),
                    executor.submit(_start_machine, client, intake["intake_id"]),
                )
            ]
    finally:
        job_id = store.intakes_by_id[intake["intake_id"]].get("job_id")
        if job_id:
            web_app._jobs.pop(job_id, None)
            web_app._results_cache.pop(job_id, None)

    assert first.status_code == 202, first.text
    assert first.json()["lifecycle_state"] == "uncertain"
    assert exact_replay.content == first.content
    assert all(replay.content == first.content for replay in concurrent_replays)
    assert runtime_events.count("queued_persistence_committed") == 1
    assert runtime_events.count("queue_ack_entered") == 1
    assert store.release_calls == []
    if queue_outcome == "claimed_then_negative_ack":
        assert runtime_record["worker_state"]["status"] == "running"


def test_reservation_persistence_failure_and_definite_guards_make_zero_remote_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    client = _client(store, starts)
    intake = _ingest_machine(client)

    unknown = _start_machine(client, "rdi-intake-" + "0" * 32)
    mismatch = _start_machine(
        client,
        intake["intake_id"],
        target_environment="production",
    )
    store.intakes_by_id[intake["intake_id"]]["lifecycle_state"] = "rejected"
    rejected = _start_machine(client, intake["intake_id"])
    store.intakes_by_id[intake["intake_id"]]["lifecycle_state"] = "accepted"
    store.fail_reservation = True
    persistence_failure = _start_machine(client, intake["intake_id"])

    assert unknown.status_code == 404
    assert mismatch.status_code == 409
    assert rejected.status_code == 409
    assert persistence_failure.status_code == 503
    assert starts == []


def test_daily_start_limit_is_atomic_and_replay_cannot_bypass_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "1")
    client = _client(store, starts)
    first = _ingest_machine(client)
    second = _ingest_machine(
        client,
        external_company_id="lg-company-002",
        canonical_domain="beta.example",
        idempotency_key="company-002-rdi-v1",
    )

    first_started = _start_machine(client, first["intake_id"])
    first_replay = _start_machine(client, first["intake_id"])
    limited = _start_machine(client, second["intake_id"])

    assert first_started.status_code == 202
    assert first_replay.status_code == 202
    assert first_replay.json()["job_id"] == first_started.json()["job_id"]
    assert limited.status_code == 429
    assert limited.json()["detail"]["code"] == "machine_start_rate_limited"
    assert len(starts) == 1


def test_daily_start_limit_applies_across_campaigns_in_one_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "1")
    client = _client(store, starts)
    first = _ingest_machine(client, campaign_id="campaign-a")
    second = _ingest_machine(
        client,
        external_company_id="lg-company-002",
        canonical_domain="beta.example",
        campaign_id="campaign-b",
        idempotency_key="company-002-rdi-v1",
    )

    first_started = _start_machine(client, first["intake_id"])
    second_limited = _start_machine(client, second["intake_id"])

    assert first_started.status_code == 202
    assert second_limited.status_code == 429
    assert second_limited.json()["detail"]["code"] == "machine_start_rate_limited"
    assert len(starts) == 1


def test_concurrent_cross_campaign_starts_share_one_daily_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")
    monkeypatch.setenv("RDI_LEADGEN_DAILY_START_LIMIT", "1")
    client = _client(store, starts)
    first = _ingest_machine(client, campaign_id="campaign-a")
    second = _ingest_machine(
        client,
        external_company_id="lg-company-002",
        canonical_domain="beta.example",
        campaign_id="campaign-b",
        idempotency_key="company-002-rdi-v1",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_start_machine, client, first["intake_id"]),
            executor.submit(_start_machine, client, second["intake_id"]),
        ]
        responses = [future.result(timeout=2) for future in futures]

    assert sorted(response.status_code for response in responses) == [202, 429]
    assert len(starts) == 1


def test_concurrent_start_replay_returns_one_reference_and_calls_remote_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    starts: list[dict[str, Any]] = []
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_ENABLED", "true")

    async def blocking_start(
        job_id: str,
        url_items: list[dict[str, str] | str],
        context: dict[str, Any],
        actor: dict[str, str | None],
    ) -> MachineStartAccepted:
        starts.append({"job_id": job_id})
        entered.set()
        assert release.wait(timeout=2)
        return MachineStartAccepted(status="running", job_id=job_id)

    app = FastAPI()
    app.include_router(
        build_leadgen_machine_router(
            MachineLifecycleDependencies(store=store, start_adapter=blocking_start)
        )
    )
    client = TestClient(app)
    intake = _ingest_machine(client)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(_start_machine, client, intake["intake_id"])
        assert entered.wait(timeout=2)
        second = _start_machine(client, intake["intake_id"])
        release.set()
        first = first_future.result(timeout=2)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.json()["lifecycle_state"] == "start_fenced"
    assert len(starts) == 1


def _machine_get(client: TestClient, path: str):
    return client.get(path, headers=SERVICE_HEADERS)


def test_nonterminal_status_is_distinct_and_terminal_views_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])
    intake = _ingest_machine(client)
    base = f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}"

    status = _machine_get(client, f"{base}/status")
    result = _machine_get(client, f"{base}/result")
    error = _machine_get(client, f"{base}/error")
    unauthenticated = client.get(f"{base}/status")

    assert status.status_code == 202
    assert status.json()["lifecycle_state"] == "accepted"
    assert status.json()["terminal"] is False
    assert status.json()["external_company_id"] == "lg-company-001"
    assert status.json()["rdi_company_id"] is None
    assert status.json()["intake_id"] == intake["intake_id"]
    assert status.json()["job_id"] is None
    assert status.json()["completed_at"] is None
    assert result.status_code == 409
    assert result.json()["detail"]["code"] == "machine_result_not_terminal"
    assert error.status_code == 409
    assert error.json()["detail"]["code"] == "machine_error_not_terminal"
    assert unauthenticated.status_code == 401


def test_company_identity_stays_unknown_until_authoritative_terminal_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])

    first = _ingest_machine(client, campaign_id="campaign-a")
    second = _ingest_machine(client, campaign_id="campaign-b")

    assert first["intake_id"] != second["intake_id"]
    assert first["rdi_correlation_id"] != second["rdi_correlation_id"]
    assert first["rdi_company_id"] is None
    assert second["rdi_company_id"] is None

    authoritative_company_id = "5d9018b8-ff3e-4f8e-a07d-0f7dbf7eb381"
    record = store.intakes_by_id[first["intake_id"]]
    record.update(
        {
            "lifecycle_state": "succeeded",
            "job_id": "rdi-job-0123456789abcdef0123456789abcdef",
            "completed_at": "2026-07-31T10:06:00Z",
            "terminal_result": {
                "rdi_company_id": authoritative_company_id,
                "composite_score": "74.00",
                "strategy_fit_score": "74.00",
                "team_score": "74.00",
                "upside_score": "74.00",
                "rdi_bucket": "standard_review",
                "completed_at": "2026-07-31T10:06:00Z",
                "pipeline_version": "pipeline-v1",
                "scoring_version": "ranking-v1",
            },
        }
    )

    result = _machine_get(
        client,
        f"/api/machine/leadgen/v1/intakes/{first['intake_id']}/result",
    )

    assert result.status_code == 200
    assert result.json()["rdi_company_id"] == authoritative_company_id
    identity_material = {
        "intake_id": first["intake_id"],
        "job_id": record["job_id"],
        "rdi_company_id": authoritative_company_id,
        "completed_at": "2026-07-31T10:06:00Z",
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            identity_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:32]
    assert result.json()["result_id"] == f"rdi-result-{expected_digest}"


@pytest.mark.parametrize("terminal_company_id", [None, "not-a-uuid"])
def test_terminal_result_requires_authoritative_company_uuid(
    monkeypatch: pytest.MonkeyPatch,
    terminal_company_id: str | None,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])
    intake = _ingest_machine(client)
    record = store.intakes_by_id[intake["intake_id"]]
    record.update(
        {
            "lifecycle_state": "succeeded",
            "job_id": "rdi-job-0123456789abcdef0123456789abcdef",
            "completed_at": "2026-07-31T10:06:00Z",
            "terminal_result": {
                "rdi_company_id": terminal_company_id,
                "composite_score": "74.00",
                "strategy_fit_score": "74.00",
                "team_score": "74.00",
                "upside_score": "74.00",
                "rdi_bucket": "standard_review",
                "completed_at": "2026-07-31T10:06:00Z",
                "pipeline_version": "pipeline-v1",
                "scoring_version": "ranking-v1",
            },
        }
    )

    response = _machine_get(
        client,
        f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}/result",
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "machine_terminal_contract_invalid"


def test_terminal_result_is_exact_checksummed_and_restart_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])
    intake = _ingest_machine(client)
    record = store.intakes_by_id[intake["intake_id"]]
    record.update(
        {
            "lifecycle_state": "succeeded",
            "job_id": "rdi-job-0123456789abcdef0123456789abcdef",
            "rdi_company_id": "5d9018b8-ff3e-4f8e-a07d-0f7dbf7eb381",
            "started_at": "2026-07-31T10:00:00Z",
            "updated_at": "2026-07-31T10:01:00Z",
            "completed_at": "2026-07-31T10:06:00Z",
            "terminal_result": {
                "rdi_company_id": "5d9018b8-ff3e-4f8e-a07d-0f7dbf7eb381",
                "composite_score": "74.000",
                "strategy_fit_score": "75.50",
                "team_score": "73.25",
                "upside_score": "73.2500",
                "rdi_bucket": "standard_review",
                "completed_at": "2026-07-31T10:06:00Z",
                "pipeline_version": "pipeline-2026-07-31",
                "scoring_version": "ranking-v1",
            },
        }
    )
    base = f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}"

    first = _machine_get(client, f"{base}/result")
    restarted = _machine_get(_client(store, []), f"{base}/result")
    wrong_view = _machine_get(client, f"{base}/error")
    status = _machine_get(client, f"{base}/status")

    assert first.status_code == 200
    assert restarted.content == first.content
    body = first.json()
    assert body["final_status"] == "succeeded"
    assert body["external_company_id"] == "lg-company-001"
    assert body["rdi_company_id"] == record["rdi_company_id"]
    assert body["composite_score"] == "74.000"
    assert body["strategy_fit_score"] == "75.50"
    assert body["team_score"] == "73.25"
    assert body["upside_score"] == "73.2500"
    checksum_payload = {key: value for key, value in body.items() if key != "checksum"}
    expected_checksum = hashlib.sha256(
        json.dumps(
            checksum_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert body["checksum"] == expected_checksum
    assert wrong_view.status_code == 409
    assert wrong_view.json()["detail"]["code"] == "machine_error_unavailable"
    assert status.status_code == 200
    assert status.json()["terminal"] is True
    assert status.json()["updated_at"] == "2026-07-31T10:06:00Z"
    assert status.json()["completed_at"] == "2026-07-31T10:06:00Z"


def test_terminal_error_is_bounded_redacted_and_restart_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])
    intake = _ingest_machine(client)
    record = store.intakes_by_id[intake["intake_id"]]
    record.update(
        {
            "lifecycle_state": "failed",
            "job_id": "rdi-job-fedcba9876543210fedcba9876543210",
            "updated_at": "2026-07-31T10:07:00Z",
            "completed_at": "2026-07-31T10:07:00Z",
            "safe_error_code": "analysis_failed",
            "safe_error_class": "terminal_analysis_failure",
            "safe_error_message": "Bearer secret-provider-token\nTraceback: internal",
            "raw_provider_payload": {"credential": "must-not-escape"},
            "stack_trace": "must-not-escape",
        }
    )
    base = f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}"

    first = _machine_get(client, f"{base}/error")
    restarted = _machine_get(_client(store, []), f"{base}/error")
    wrong_view = _machine_get(client, f"{base}/result")

    assert first.status_code == 200
    assert restarted.content == first.content
    body = first.json()
    assert body["final_status"] == "failed"
    assert body["error_code"] == "analysis_failed"
    assert body["error_class"] == "terminal_analysis_failure"
    assert body["message"] == "The RDI analysis ended without a publishable result."
    assert len(body["message"]) <= 240
    serialized = json.dumps(body).lower()
    assert "bearer" not in serialized
    assert "token" not in serialized
    assert "traceback" not in serialized
    assert "must-not-escape" not in serialized
    assert wrong_view.status_code == 409
    assert wrong_view.json()["detail"]["code"] == "machine_result_unavailable"


def test_terminal_result_rejects_mismatched_persisted_company_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])
    intake = _ingest_machine(client)
    record = store.intakes_by_id[intake["intake_id"]]
    record.update(
        {
            "lifecycle_state": "succeeded",
            "job_id": "rdi-job-0123456789abcdef0123456789abcdef",
            "rdi_company_id": "5d9018b8-ff3e-4f8e-a07d-0f7dbf7eb381",
            "completed_at": "2026-07-31T10:06:00Z",
            "terminal_result": {
                "rdi_company_id": "2db2f5c6-147d-4472-b5a2-17d9f1abe488",
                "composite_score": "74.00",
                "strategy_fit_score": "74.00",
                "team_score": "74.00",
                "upside_score": "74.00",
                "rdi_bucket": "standard_review",
                "completed_at": "2026-07-31T10:06:00Z",
                "pipeline_version": "pipeline-v1",
                "scoring_version": "ranking-v1",
            },
        }
    )

    response = _machine_get(
        client,
        f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}/result",
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "machine_result_identity_mismatch"


def test_terminal_result_rejects_mismatched_external_company_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMachineStore()
    monkeypatch.setenv("RDI_LEADGEN_AUTOSTART_KEY", SERVICE_KEY)
    client = _client(store, [])
    intake = _ingest_machine(client)
    record = store.intakes_by_id[intake["intake_id"]]
    record.update(
        {
            "lifecycle_state": "succeeded",
            "job_id": "rdi-job-0123456789abcdef0123456789abcdef",
            "rdi_company_id": "5d9018b8-ff3e-4f8e-a07d-0f7dbf7eb381",
            "completed_at": "2026-07-31T10:06:00Z",
            "terminal_result": {
                "external_company_id": "lg-company-other",
                "rdi_company_id": "5d9018b8-ff3e-4f8e-a07d-0f7dbf7eb381",
                "composite_score": "74.00",
                "strategy_fit_score": "74.00",
                "team_score": "74.00",
                "upside_score": "74.00",
                "rdi_bucket": "standard_review",
                "completed_at": "2026-07-31T10:06:00Z",
                "pipeline_version": "pipeline-v1",
                "scoring_version": "ranking-v1",
            },
        }
    )

    response = _machine_get(
        client,
        f"/api/machine/leadgen/v1/intakes/{intake['intake_id']}/result",
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "machine_result_identity_mismatch"


class _RpcCapture:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(self, name: str, params: dict[str, Any]):
        self.calls.append((name, deepcopy(params)))
        return SimpleNamespace(
            execute=lambda: SimpleNamespace(data=deepcopy(self.responses[name]))
        )

    def table(self, _name: str):
        raise AssertionError("machine lifecycle mutations must use atomic RPCs")


def test_machine_database_adapter_uses_only_atomic_rpc_request_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web import db

    capture = _RpcCapture(
        {
            "create_leadgen_machine_intake": {"action": "created", "intake_id": "rdi-intake-1"},
            "reserve_leadgen_machine_start": {"action": "reserved", "intake_id": "rdi-intake-1"},
            "release_leadgen_machine_start": {"action": "released", "intake_id": "rdi-intake-1"},
            "finalize_leadgen_machine_start": {"lifecycle_state": "queued", "intake_id": "rdi-intake-1"},
            "get_leadgen_machine_lifecycle": {"lifecycle_state": "queued", "intake_id": "rdi-intake-1"},
        }
    )
    monkeypatch.setattr(db, "_get_client", lambda: capture)

    created = db.create_machine_intake(intake_id="rdi-intake-1", payload_hash="a" * 64)
    reserved = db.reserve_machine_start(
        intake_id="rdi-intake-1",
        target_environment="staging",
        business_date=BUSINESS_DATE,
        business_timezone=BUSINESS_TIMEZONE,
        job_id="rdi-job-1",
        actor=SERVICE_ACTOR,
        daily_start_limit=20,
    )
    released = db.release_machine_start(
        intake_id="rdi-intake-1",
        job_id="rdi-job-1",
        actor=SERVICE_ACTOR,
    )
    finalized = db.finalize_machine_start(
        intake_id="rdi-intake-1",
        job_id="rdi-job-1",
        lifecycle_state="queued",
        safe_error_code=None,
        safe_error_class=None,
        safe_error_message=None,
        actor=SERVICE_ACTOR,
    )
    loaded = db.load_machine_lifecycle("rdi-intake-1")

    assert created == {"action": "created", "intake_id": "rdi-intake-1"}
    assert reserved == {"action": "reserved", "intake_id": "rdi-intake-1"}
    assert released == {"action": "released", "intake_id": "rdi-intake-1"}
    assert finalized == {"lifecycle_state": "queued", "intake_id": "rdi-intake-1"}
    assert loaded == {"lifecycle_state": "queued", "intake_id": "rdi-intake-1"}
    assert capture.calls == [
        (
            "create_leadgen_machine_intake",
            {"p_record": {"intake_id": "rdi-intake-1", "payload_hash": "a" * 64}},
        ),
        (
            "reserve_leadgen_machine_start",
            {
                "p_intake_id": "rdi-intake-1",
                "p_target_environment": "staging",
                "p_business_date": BUSINESS_DATE,
                "p_business_timezone": BUSINESS_TIMEZONE,
                "p_job_id": "rdi-job-1",
                "p_actor": SERVICE_ACTOR,
                "p_daily_start_limit": 20,
            },
        ),
        (
            "release_leadgen_machine_start",
            {
                "p_intake_id": "rdi-intake-1",
                "p_job_id": "rdi-job-1",
                "p_actor": SERVICE_ACTOR,
            },
        ),
        (
            "finalize_leadgen_machine_start",
            {
                "p_intake_id": "rdi-intake-1",
                "p_job_id": "rdi-job-1",
                "p_lifecycle_state": "queued",
                "p_safe_error_code": None,
                "p_safe_error_class": None,
                "p_safe_error_message": None,
                "p_actor": SERVICE_ACTOR,
            },
        ),
        (
            "get_leadgen_machine_lifecycle",
            {"p_intake_id": "rdi-intake-1"},
        ),
    ]


def test_machine_company_run_persists_requested_domain_separately_from_provider_alias() -> None:
    from web import db

    upserts: list[tuple[str, dict[str, Any], str | None]] = []

    class Query:
        def __init__(self, table_name: str) -> None:
            self.table_name = table_name
            self.operation = ""
            self.payload: dict[str, Any] = {}
            self.on_conflict: str | None = None

        def upsert(
            self,
            payload: dict[str, Any],
            on_conflict: str | None = None,
        ) -> "Query":
            self.operation = "upsert"
            self.payload = deepcopy(payload)
            self.on_conflict = on_conflict
            return self

        def insert(self, payload: dict[str, Any]) -> "Query":
            self.operation = "insert"
            self.payload = deepcopy(payload)
            return self

        def select(self, _columns: str) -> "Query":
            self.operation = "select"
            return self

        def delete(self) -> "Query":
            self.operation = "delete"
            return self

        def eq(self, _column: str, _value: Any) -> "Query":
            return self

        def limit(self, _count: int) -> "Query":
            return self

        def execute(self) -> SimpleNamespace:
            if self.operation == "upsert":
                upserts.append((self.table_name, deepcopy(self.payload), self.on_conflict))
            if self.table_name == "companies" and self.operation == "upsert":
                return SimpleNamespace(data=[{"id": "company-adspawn-uuid"}])
            return SimpleNamespace(data=[])

    class Client:
        @staticmethod
        def table(table_name: str) -> Query:
            return Query(table_name)

    persisted = db._persist_company_analysis_row(
        Client(),
        job_uuid="job-uuid",
        job_id_legacy="rdi-job-0123456789abcdef0123456789abcdef",
        result_row={
            "company": SimpleNamespace(
                name="AdSpawn",
                domain="adspawn.io",
                industry=None,
                tagline=None,
                about=None,
                team=[],
            ),
            "slug": "adspawn",
            "final_state": {},
            "analysis_status": "done",
        },
        company_payload={
            "decision": "invest",
            "total_score": 80,
            "ranking_result": {"composite_score": 80, "bucket": "priority"},
            "summary_rows": [{}],
        },
        run_config={
            "source": "leadgen_machine",
            "input_mode": "specter",
            "leadgen_machine": {"canonical_domain": "adspawn.com"},
        },
        excel_storage_path=None,
        replace_documents=False,
    )

    company_run = next(payload for table, payload, _conflict in upserts if table == "company_runs")
    assert persisted is True
    assert company_run["company_id"] == "company-adspawn-uuid"
    assert company_run["company_key"] == "domain:adspawn.io"
    assert company_run["source_company_key"] == "domain:adspawn.com"
    assert db._machine_source_company_key(
        {
            "source": "leadgen",
            "leadgen_machine": {"canonical_domain": "adspawn.com"},
        }
    ) is None


def test_url_worker_task_uses_requested_source_company_key() -> None:
    worker = _load_local_specter_batch_worker()

    tasks = worker._build_company_tasks(
        {
            "source": "leadgen_machine",
            "specter_urls": ["https://adspawn.com"],
        },
        None,
    )

    assert len(tasks) == 1
    assert tasks[0] == {
        **tasks[0],
        "mode": "url",
        "url": "https://adspawn.com",
        "domain": "adspawn.com",
        "expected_name": "",
        "slug": "adspawn-com",
        "name": "adspawn.com",
        "source_company_key": "domain:adspawn.com",
    }
    assert worker._task_company_key(tasks[0]) == "domain:adspawn.com"


def test_non_machine_url_task_preserves_legacy_shape_and_resume_key() -> None:
    worker = _load_local_specter_batch_worker()

    tasks = worker._build_company_tasks(
        {"source": "leadgen", "specter_urls": ["https://legacy.example"]},
        None,
    )

    assert len(tasks) == 1
    assert tasks[0] == {
        **tasks[0],
        "mode": "url",
        "url": "https://legacy.example",
        "domain": "legacy.example",
        "expected_name": "",
        "slug": "legacy-example",
        "name": "legacy.example",
    }
    assert "source_company_key" not in tasks[0]
    assert worker._task_company_key(tasks[0]) == "legacy-example"


def test_worker_restart_skips_persisted_requested_key_after_provider_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_local_specter_batch_worker()

    processed: list[str] = []

    class ResumeDatabase:
        def __init__(self) -> None:
            self.finished: dict[str, Any] | None = None

        @staticmethod
        def load_job_company_runs(_job_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "company_key": "domain:adspawn.io",
                    "source_company_key": "domain:adspawn.com",
                    "company_name": "AdSpawn",
                    "startup_slug": "adspawn",
                    "decision": "invest",
                }
            ]

        @staticmethod
        def insert_analysis_event(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        @staticmethod
        def heartbeat_specter_worker_job(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        @staticmethod
        def get_job_control(_job_id: str) -> dict[str, bool]:
            return {"stop_requested": False, "pause_requested": False}

        @staticmethod
        def load_job_results(
            _job_id: str,
            preferred_mode: str | None = None,
        ) -> dict[str, Any]:
            del preferred_mode
            return {"results": {}}

        @staticmethod
        def load_run_costs(_job_id: str) -> dict[str, Any]:
            return {}

        @staticmethod
        def persist_analysis_snapshot(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            return True

        def finish_specter_worker_job(self, _job_id: str, **kwargs: Any) -> None:
            self.finished = deepcopy(kwargs)

        @staticmethod
        def insert_analysis_error(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

    database = ResumeDatabase()
    monkeypatch.setattr(worker, "db", database)
    monkeypatch.setattr(
        worker,
        "_download_worker_inputs",
        lambda _job_id, _config: (None, None, None),
    )
    monkeypatch.setattr(
        worker.web_app,
        "_parse_max_startups_from_instructions",
        lambda _instructions: None,
    )

    async def run_company(
        *,
        company_descriptor: dict[str, Any],
        completed_companies: int,
        failed_companies: int,
        **_kwargs: Any,
    ) -> tuple[int, int]:
        processed.append(str(company_descriptor.get("source_company_key") or "missing"))
        return completed_companies + 1, failed_companies

    monkeypatch.setattr(worker, "_run_company_subprocess", run_company)

    asyncio.run(
        worker._process_job(
            {
                "job_id": "rdi-job-0123456789abcdef0123456789abcdef",
                "run_config": {
                    "source": "leadgen_machine",
                    "specter_urls": ["https://adspawn.com"],
                    "leadgen_machine": {"canonical_domain": "adspawn.com"},
                },
            },
            "worker-restart-test",
        )
    )

    assert processed == []
    assert database.finished is not None
    assert database.finished["status"] == "done"
    assert database.finished["completed_companies"] == 1


def test_worker_completed_key_loader_preserves_legacy_name_slug_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_local_specter_batch_worker()
    monkeypatch.setattr(
        worker,
        "db",
        SimpleNamespace(
            load_job_company_runs=lambda _job_id: [
                {
                    "company_name": "Legacy Company",
                    "startup_slug": "legacy-company",
                    "decision": "invest",
                }
            ]
        ),
    )

    completed, successful, failed = worker._load_completed_company_keys("legacy-job")

    assert completed == {"legacy-company"}
    assert successful == 1
    assert failed == 0


def test_machine_terminal_projection_uses_unique_requested_source_key() -> None:
    sql = Path(
        "supabase/migrations/20260731000000_leadgen_machine_lifecycle.sql"
    ).read_text(encoding="utf-8")
    lowered = " ".join(sql.lower().split())

    assert "alter table public.company_runs add column if not exists source_company_key text" in lowered
    assert "create index if not exists idx_company_runs_job_source_company" in lowered
    assert "cr.source_company_key = 'domain:' || v_row.canonical_domain" in lowered
    assert "v_company_run_count integer" in lowered
    assert "v_company_run_count = 1" in lowered
    assert "cr.company_key = 'domain:' || v_row.canonical_domain" not in lowered


def test_machine_forward_migration_has_atomic_security_and_audit_contract() -> None:
    migration = Path(
        "supabase/migrations/20260731000000_leadgen_machine_lifecycle.sql"
    )
    sql = migration.read_text(encoding="utf-8")
    lowered = " ".join(sql.lower().split())

    required_fragments = [
        "create table if not exists public.leadgen_machine_intakes",
        "create table if not exists public.leadgen_machine_events",
        "foreign key",
        "unique",
        "check",
        "enable row level security",
        "security definer set search_path = pg_catalog, public",
        "pg_advisory_xact_lock",
        "for update",
        "create or replace function public.create_leadgen_machine_intake",
        "create or replace function public.reserve_leadgen_machine_start",
        "create or replace function public.release_leadgen_machine_start",
        "create or replace function public.finalize_leadgen_machine_start",
        "create or replace function public.get_leadgen_machine_lifecycle",
        "revoke all on table public.leadgen_machine_intakes from public, anon, authenticated, service_role",
        "revoke all on table public.leadgen_machine_events from public, anon, authenticated, service_role",
        "revoke all on function",
        "grant execute on function",
        "to service_role",
    ]

    for fragment in required_fragments:
        assert fragment in lowered
    assert "grant all on table public.leadgen_machine" not in lowered
    assert "grant select" not in lowered
    assert "grant insert" not in lowered
    assert "grant update" not in lowered
    assert "grant execute on function" in lowered
    assert " to anon" not in lowered
    assert " to authenticated" not in lowered
    assert "coalesce(v_row.completed_at, now())" not in lowered
    assert "v_job_status_at" in lowered
    assert "v_analysis_status_at" in lowered
    assert "j.run_config ->> 'rdi_scoring_version'" in lowered
    assert "coalesce(j.prompt_version, j.schema_version)" not in lowered
    assert "rdi_company_id uuid" in lowered
    assert "v_company_run public.company_runs%rowtype" in lowered
    assert "'rdi_company_id', v_company_run.company_id::text" in lowered
    assert "and v_company_run.company_id is not null" in lowered
    assert "'rdi_company_id', v_row.rdi_company_id" not in lowered
    assert "hashtextextended(v_row.target_environment, 0)" in lowered
    assert "p_target_environment not in ('staging', 'production')" in lowered
    assert "v_row.target_environment || ':' || v_row.campaign_id" not in lowered
    assert "where target_environment = v_row.target_environment and campaign_id" not in lowered
    assert "set lifecycle_state = 'accepted', job_id = null, start_actor = null, started_at = null" in lowered
    assert "'start_released'" in lowered
    assert "canonical_domain !~ '^www\\.'" in lowered


def test_daily_scope_correction_replaces_unsafe_global_reservation_contract() -> None:
    sql = Path(
        "supabase/migrations/20260807102328_leadgen_machine_daily_start_scope.sql"
    ).read_text(encoding="utf-8")
    lowered = " ".join(sql.lower().split())

    required_fragments = [
        "create table if not exists public.leadgen_machine_daily_scopes",
        "add column if not exists business_date date",
        "add column if not exists business_timezone text",
        "at time zone 'europe/prague'",
        "foreign key (target_environment, business_date)",
        "create index if not exists idx_leadgen_machine_intakes_daily_scope_state",
        "enforce_leadgen_machine_scope_immutability",
        "drop function if exists public.reserve_leadgen_machine_start(",
        "text, text, text, text, integer",
        "p_business_date date",
        "p_business_timezone text",
        "p_daily_start_limit integer",
        "p_daily_start_limit > 20",
        "v_row.target_environment || ':' || v_row.business_date::text",
        "where target_environment = v_row.target_environment and business_date = v_row.business_date",
        "'daily_start_limit'",
        "'daily_started_count'",
        "'daily_remaining_capacity'",
        "create or replace function public.get_leadgen_machine_daily_capacity",
        "enable row level security",
        "revoke all on table public.leadgen_machine_daily_scopes",
        "grant execute on function",
        "rollback",
    ]
    for fragment in required_fragments:
        assert fragment in lowered
    assert "p_global_limit" not in lowered
    assert "grant select" not in lowered
    assert "grant insert" not in lowered
    assert "grant update" not in lowered


def test_protected_rdi_scoring_manifest_matches_authoritative_production() -> None:
    from scripts.verify_protected_scoring import main

    assert main() == 0


def test_web_app_mounts_machine_contract_without_replacing_recovered_routes() -> None:
    from web import app as web_app

    paths = {route.path for route in web_app.app.routes}

    assert "/api/machine/leadgen/v1/intakes" in paths
    assert "/api/machine/leadgen/v1/intakes/{intake_id}/start" in paths
    assert "/api/machine/leadgen/v1/intakes/{intake_id}/status" in paths
    assert "/api/machine/leadgen/v1/intakes/{intake_id}/result" in paths
    assert "/api/machine/leadgen/v1/intakes/{intake_id}/error" in paths
    assert "/api/leadgen/ingest" in paths
    assert "/api/leadgen/intakes/{intake_id}/approve" in paths
    assert "/api/status/{job_id}" in paths
    assert "/api/analyses/{job_id}" in paths
