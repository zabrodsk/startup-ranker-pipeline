"""Version-two LeadGen machine intake with immutable evidence reuse.

The v2 boundary is intentionally additive.  Version one remains available for
existing lifecycle reconciliation while new clients can persist a complete
authorization/evidence bundle before asking RDI to start.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Coroutine, Literal, Mapping, Protocol
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from web.leadgen_domain import company_source_host, normalize_company_domain
from web.leadgen_machine import (
    MachineStartAccepted,
    MachineStartDefiniteRejection,
    _daily_start_limit,
    _machine_actor,
    _require_service_key,
    _scoring_version,
    _result_payload as _v1_result_payload,
    _error_payload as _v1_error_payload,
    _timestamp,
    _require_start_enabled,
    _stable_reference,
    _target_environment,
)
from starlette.responses import Response


CONTRACT_VERSION = "rdi.leadgen-machine.v2"
SERVICE_ACTOR = "service:rockaway-leadgen"
BUNDLE_SCHEMA_VERSION = "frozen-leadgen-evidence-bundle-v1"
BUSINESS_TIMEZONE = "Europe/Prague"
V2_ENABLED_ENV = "RDI_LEADGEN_MACHINE_V2_ENABLED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPECTER_ID_RE = re.compile(r"^[0-9a-f]{24}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INTAKE_RE = re.compile(r"^rdi-v2-intake-[0-9a-f]{32}$")
_LEGACY_SCORING_VERSION_BY_PIPELINE = {"v1": "ranking-v1"}


def _problem(
    status_code: int,
    code: str,
    message: str,
    *,
    extra_detail: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    detail = {"code": code, "message": message, "contract_version": CONTRACT_VERSION}
    if extra_detail:
        detail.update(extra_detail)
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def canonical_bundle_sha256(value: object) -> str:
    """Return the v1 bundle content identity used at every trust boundary."""
    return _sha256(value)


def _require_v2_enabled() -> None:
    raw = os.getenv(V2_ENABLED_ENV)
    if raw != "true":
        code = "machine_v2_disabled" if raw in {None, "", "0", "false"} else "machine_v2_configuration_invalid"
        raise _problem(503, code, "LeadGen machine v2 is not enabled.")


def _company_id(value: str) -> str:
    try:
        normalized = str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("must be a canonical UUID") from exc
    if normalized != value:
        raise ValueError("must be a canonical lowercase UUID")
    return value


def _domain(value: Any) -> str:
    normalized = normalize_company_domain(value)
    if normalized is None or company_source_host(normalized):
        raise ValueError("must be a canonical company-owned domain")
    return normalized


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("must be a bounded safe identifier")
    return value


def _version_label(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("must be a bounded clean version label")
    return value


class BundleComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str
    provider: str
    retrieved_at: datetime
    fresh_until: datetime | None
    schema_version: str
    payload_sha256: str
    errors: list[str] = Field(max_length=32)

    @field_validator("component", "provider", "schema_version")
    @classmethod
    def _bounded_identifier(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("payload_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 digest")
        return value


class BundleChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(min_length=1, max_length=200_000)
    source_file: str = Field(min_length=1, max_length=2_048)
    page_or_slide: str | int

    @field_validator("chunk_id")
    @classmethod
    def _chunk_identifier(cls, value: str) -> str:
        return _identifier(value)


class LeadGenAuthorizationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str
    canonical_domain: str
    thesis_status: Literal["pass"]
    thesis_version: str
    thesis_sha256: str
    lite_tier: Literal["send_to_rdi"]
    lite_scoring_version: str
    lite_sha256: str
    rdi_ready: Literal[True]
    route: Literal["rockaway_rdi", "dual_review"]
    routing_version: str
    routing_sha256: str
    source_run_id: str
    source_sha256: str
    frozen_lineage_sha256: str

    @field_validator("company_id")
    @classmethod
    def _canonical_company_id(cls, value: str) -> str:
        return _company_id(value)

    @field_validator("canonical_domain", mode="before")
    @classmethod
    def _canonical_domain(cls, value: Any) -> str:
        return _domain(value)

    @field_validator(
        "lite_scoring_version",
        "routing_version",
        "source_run_id",
    )
    @classmethod
    def _safe_identifier(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("thesis_version")
    @classmethod
    def _human_version_label(cls, value: str) -> str:
        return _version_label(value)

    @field_validator(
        "thesis_sha256",
        "lite_sha256",
        "routing_sha256",
        "source_sha256",
        "frozen_lineage_sha256",
    )
    @classmethod
    def _hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 digest")
        return value


class BundleSpecterOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str = Field(min_length=1, max_length=128)
    operation: str = Field(min_length=1, max_length=128)
    trigger: str = Field(min_length=1, max_length=128)
    consumer: str = Field(min_length=1, max_length=128)
    cache_status: Literal["hit", "miss"]
    called: bool
    success: bool
    occurred_at: datetime
    error: str | None = Field(default=None, max_length=2048)


class FrozenLeadGenEvidenceBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["frozen-leadgen-evidence-bundle-v1"]
    external_company_id: str
    canonical_domain: str
    specter_company_id: str | None
    requires_specter_mcp: bool
    parent_bundle_sha256: str | None
    created_at: datetime
    company: dict[str, Any]
    evidence_chunks: list[BundleChunk] = Field(max_length=10_000)
    components: list[BundleComponent] = Field(max_length=256)
    component_payloads: dict[str, Any]
    specter_operations: list[BundleSpecterOperation] = Field(default_factory=list, max_length=200)
    authorization: LeadGenAuthorizationManifest

    @field_validator("external_company_id")
    @classmethod
    def _canonical_external_id(cls, value: str) -> str:
        return _company_id(value)

    @field_validator("canonical_domain", mode="before")
    @classmethod
    def _bundle_domain(cls, value: Any) -> str:
        return _domain(value)

    @field_validator("specter_company_id")
    @classmethod
    def _specter_id(cls, value: str | None) -> str | None:
        if value is not None and not _SPECTER_ID_RE.fullmatch(value):
            raise ValueError("must be a canonical Specter company ID")
        return value

    @field_validator("parent_bundle_sha256")
    @classmethod
    def _parent_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def _lineage_matches(self) -> "FrozenLeadGenEvidenceBundleV1":
        if self.authorization.company_id != self.external_company_id:
            raise ValueError("authorization company does not match bundle")
        if self.authorization.canonical_domain != self.canonical_domain:
            raise ValueError("authorization domain does not match bundle")
        company_domain = _domain(self.company.get("domain") or self.company.get("company_url"))
        if company_domain != self.canonical_domain:
            raise ValueError("company payload domain does not match bundle")
        if not self.requires_specter_mcp and not self.evidence_chunks:
            raise ValueError("complete reusable bundle requires evidence chunks")
        component_names = [item.component for item in self.components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("bundle components must be unique")
        if set(self.component_payloads) != set(component_names):
            raise ValueError("bundle component payloads do not match the manifest")
        if any(
            _sha256(self.component_payloads[item.component]) != item.payload_sha256
            for item in self.components
        ):
            raise ValueError("bundle component payload hash does not match")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        # Preserve the exact canonical shape accepted from older v2 producers:
        # newly optional audit fields must not silently change their hash.
        return self.model_dump(mode="json", exclude_none=False, exclude_unset=True)


class MachineV2IntakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    external_company_id: str
    canonical_domain: str
    campaign_id: str
    iteration_id: str
    source_run_id: str
    batch_id: str
    idempotency_key: str
    leadgen_business_date: date
    business_timezone: Literal["Europe/Prague"]
    target_environment: Literal["staging", "production"]
    evidence_bundle_sha256: str

    @field_validator("external_company_id")
    @classmethod
    def _company(cls, value: str) -> str:
        return _company_id(value)

    @field_validator("canonical_domain", mode="before")
    @classmethod
    def _domain(cls, value: Any) -> str:
        return _domain(value)

    @field_validator(
        "campaign_id", "iteration_id", "source_run_id", "batch_id", "idempotency_key"
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("evidence_bundle_sha256")
    @classmethod
    def _bundle_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 digest")
        return value


class MachineV2StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_environment: Literal["staging", "production"]
    business_timezone: Literal["Europe/Prague"]


class MachineV2Store(Protocol):
    def is_configured(self) -> bool: ...

    def put_machine_v2_evidence_bundle(self, **record: Any) -> dict[str, Any] | None: ...

    def create_machine_v2_intake(self, **record: Any) -> dict[str, Any] | None: ...

    def reserve_machine_v2_start(self, **record: Any) -> dict[str, Any] | None: ...

    def finalize_machine_v2_start(self, **record: Any) -> dict[str, Any] | None: ...

    def release_machine_v2_start(self, **record: Any) -> dict[str, Any] | None: ...

    def load_machine_v2_lifecycle(self, intake_id: str) -> dict[str, Any] | None: ...


StartAdapter = Callable[
    [str, list[dict[str, str] | str], dict[str, Any], dict[str, str | None]],
    Awaitable[MachineStartAccepted | MachineStartDefiniteRejection | Any],
]


class MachineV2Dependencies:
    def __init__(self, *, store: MachineV2Store | None, start_adapter: StartAdapter) -> None:
        self.store = store
        self.start_adapter = start_adapter


class MachineV2AuthenticatedRoute(APIRoute):
    """Authenticate before parsing and return stable v2 validation errors."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_handler = super().get_route_handler()

        async def authenticated_handler(request: Request) -> Response:
            try:
                _require_service_key(request.headers.get("X-LeadGen-Service-Key"))
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, Mapping) else {}
                raise _problem(
                    exc.status_code,
                    str(detail.get("code") or "machine_auth_invalid"),
                    str(detail.get("message") or "Machine authentication failed."),
                    headers=exc.headers,
                ) from exc
            try:
                return await original_handler(request)
            except RequestValidationError as exc:
                authorization_error = any(
                    "authorization" in tuple(str(part) for part in error.get("loc", ()))
                    for error in exc.errors()
                )
                code = (
                    "machine_v2_authorization_invalid"
                    if authorization_error
                    else "machine_v2_request_invalid"
                )
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "code": code,
                            "message": "Machine v2 authorization is invalid."
                            if authorization_error
                            else "Machine v2 request is invalid.",
                            "contract_version": CONTRACT_VERSION,
                        }
                    },
                )

        return authenticated_handler


def _require_store(dependencies: MachineV2Dependencies) -> MachineV2Store:
    if dependencies.store is None or not dependencies.store.is_configured():
        raise _problem(503, "machine_v2_storage_not_configured", "Machine v2 storage is unavailable.")
    return dependencies.store


def _intake_id(request: MachineV2IntakeRequest) -> str:
    identity = {
        "external_company_id": request.external_company_id,
        "campaign_id": request.campaign_id,
        "iteration_id": request.iteration_id,
        "idempotency_key": request.idempotency_key,
        "target_environment": request.target_environment,
    }
    return _stable_reference("rdi-v2-intake", identity)


def _job_id(intake_id: str) -> str:
    return f"rdi-v2-job-{hashlib.sha256(intake_id.encode()).hexdigest()[:32]}"


def _prague_date() -> str:
    return datetime.now(timezone.utc).astimezone(ZoneInfo(BUSINESS_TIMEZONE)).date().isoformat()


def _correlation_id(intake_id: str) -> str:
    return f"rdi-v2-correlation-{hashlib.sha256(intake_id.encode()).hexdigest()[:32]}"


def _status_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    state = str(record.get("lifecycle_state") or "intake_pending")
    created_at = _timestamp(record.get("created_at"), field="created_at")
    updated_at = _timestamp(record.get("updated_at") or record.get("created_at"), field="updated_at")
    return {
        "contract_version": CONTRACT_VERSION,
        "intake_id": record["intake_id"],
        "external_company_id": record["external_company_id"],
        "canonical_domain": record["canonical_domain"],
        "rdi_company_id": record.get("rdi_company_id"),
        "rdi_correlation_id": _correlation_id(str(record["intake_id"])),
        "evidence_bundle_sha256": record["evidence_bundle_sha256"],
        "requires_specter_mcp": bool(record.get("requires_specter_mcp", True)),
        "lifecycle_state": state,
        "wait_reason": record.get("wait_reason"),
        "blocked_until": record.get("blocked_until"),
        "leadgen_business_date": str(record["leadgen_business_date"]),
        "actual_start_business_date": (
            str(record["actual_start_business_date"])
            if record.get("actual_start_business_date")
            else None
        ),
        "job_id": record.get("job_id"),
        "terminal": state in {"succeeded", "failed", "cancelled"},
        "created_at": created_at,
        "updated_at": updated_at,
        "started_at": (
            _timestamp(record.get("started_at"), field="started_at")
            if record.get("started_at")
            else None
        ),
        "completed_at": (
            _timestamp(record.get("completed_at"), field="completed_at")
            if record.get("completed_at")
            else None
        ),
    }


def _terminal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    terminal = dict(record)
    terminal_result = terminal.get("terminal_result")
    if isinstance(terminal_result, dict):
        terminal_result = dict(terminal_result)
        if not terminal_result.get("scoring_version"):
            legacy_version = _LEGACY_SCORING_VERSION_BY_PIPELINE.get(
                str(terminal_result.get("pipeline_version") or "")
            )
            if legacy_version:
                terminal_result["scoring_version"] = legacy_version
        terminal["terminal_result"] = terminal_result
    return {
        **terminal,
        "external_company_id": str(record["external_company_id"]),
        "rdi_correlation_id": _correlation_id(str(record["intake_id"])),
    }


def _v2_result_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _v1_result_payload(_terminal_record(record))
    payload.update(
        contract_version=CONTRACT_VERSION,
        evidence_bundle_sha256=record["evidence_bundle_sha256"],
        leadgen_business_date=str(record["leadgen_business_date"]),
        actual_start_business_date=str(record["actual_start_business_date"]),
    )
    payload["checksum"] = _sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    )
    return payload


def _v2_error_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _v1_error_payload(_terminal_record(record))
    payload.update(
        contract_version=CONTRACT_VERSION,
        evidence_bundle_sha256=record["evidence_bundle_sha256"],
        leadgen_business_date=str(record["leadgen_business_date"]),
        actual_start_business_date=(
            str(record["actual_start_business_date"])
            if record.get("actual_start_business_date")
            else None
        ),
    )
    return payload


def build_leadgen_machine_v2_router(dependencies: MachineV2Dependencies) -> APIRouter:
    router = APIRouter(route_class=MachineV2AuthenticatedRoute)

    @router.put(
        "/api/machine/leadgen/v2/evidence-bundles/{bundle_sha256}",
        status_code=201,
    )
    async def put_bundle(
        bundle_sha256: str,
        request: FrozenLeadGenEvidenceBundleV1,
    ) -> dict[str, Any]:
        _require_v2_enabled()
        if not _SHA256_RE.fullmatch(bundle_sha256):
            raise _problem(422, "machine_v2_bundle_hash_invalid", "Bundle hash is invalid.")
        payload = request.canonical_payload()
        if _sha256(payload) != bundle_sha256:
            raise _problem(409, "machine_v2_bundle_hash_mismatch", "Bundle hash does not match its canonical payload.")
        store = _require_store(dependencies)
        result = store.put_machine_v2_evidence_bundle(
            bundle_sha256=bundle_sha256,
            schema_version=BUNDLE_SCHEMA_VERSION,
            external_company_id=request.external_company_id,
            canonical_domain=request.canonical_domain,
            requires_specter_mcp=request.requires_specter_mcp,
            parent_bundle_sha256=request.parent_bundle_sha256,
            authorization_sha256=_sha256(request.authorization.model_dump(mode="json")),
            byte_size=len(_canonical_bytes(payload)),
            payload=payload,
            created_at=request.created_at.isoformat().replace("+00:00", "Z"),
        )
        if result is None:
            raise _problem(503, "machine_v2_bundle_persistence_failed", "Evidence bundle could not be persisted.")
        if result.get("action") == "conflict":
            raise _problem(409, "machine_v2_bundle_conflict", "Existing bundle content conflicts with this payload.")
        return {
            "contract_version": CONTRACT_VERSION,
            "bundle_sha256": bundle_sha256,
            "status": "stored",
            "requires_specter_mcp": request.requires_specter_mcp,
        }

    @router.post("/api/machine/leadgen/v2/intakes", status_code=202)
    async def create_intake(request: MachineV2IntakeRequest) -> dict[str, Any]:
        _require_v2_enabled()
        target_environment = _target_environment(request.target_environment)
        payload = request.model_dump(mode="json")
        intake_id = _intake_id(request)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = {
            "contract_version": CONTRACT_VERSION,
            "intake_id": intake_id,
            "idempotency_identity": _sha256(
                {
                    "external_company_id": request.external_company_id,
                    "campaign_id": request.campaign_id,
                    "iteration_id": request.iteration_id,
                    "idempotency_key": request.idempotency_key,
                    "target_environment": target_environment,
                }
            ),
            "payload_hash": _sha256(payload),
            **payload,
            "target_environment": target_environment,
            "lifecycle_state": "intake_pending",
            "created_at": now,
            "updated_at": now,
        }
        result = _require_store(dependencies).create_machine_v2_intake(**record)
        if result is None:
            raise _problem(503, "machine_v2_intake_persistence_failed", "Machine intake could not be persisted.")
        if result.get("action") == "bundle_missing":
            raise _problem(409, "machine_v2_bundle_missing", "The referenced evidence bundle is unavailable.")
        if result.get("action") == "bundle_mismatch":
            raise _problem(409, "machine_v2_bundle_identity_mismatch", "Bundle identity does not match the intake.")
        if result.get("action") == "conflict":
            raise _problem(409, "machine_v2_intake_conflict", "The idempotent intake has different content.")
        return _status_payload(result)

    @router.post("/api/machine/leadgen/v2/intakes/{intake_id}/start", status_code=202)
    async def start_intake(intake_id: str, request: MachineV2StartRequest) -> dict[str, Any]:
        _require_v2_enabled()
        _require_start_enabled()
        if not _INTAKE_RE.fullmatch(intake_id):
            raise _problem(422, "machine_v2_intake_id_invalid", "Machine v2 intake ID is invalid.")
        target_environment = _target_environment(request.target_environment)
        scoring_version = _scoring_version()
        actual_start_date = _prague_date()
        store = _require_store(dependencies)
        reservation = store.reserve_machine_v2_start(
            intake_id=intake_id,
            target_environment=target_environment,
            actual_start_business_date=actual_start_date,
            business_timezone=BUSINESS_TIMEZONE,
            job_id=_job_id(intake_id),
            actor=SERVICE_ACTOR,
            daily_start_limit=_daily_start_limit(),
        )
        if reservation is None:
            raise _problem(503, "machine_v2_start_reservation_failed", "Machine v2 start could not be persisted.")
        action = reservation.get("action")
        if action == "pending_provider_quota":
            payload = _status_payload(reservation)
            payload["daily_started_count"] = int(reservation.get("daily_started_count") or 0)
            return payload
        if action == "rate_limited":
            raise _problem(429, "machine_start_rate_limited", "The machine-start limit is exhausted.")
        if action == "unknown":
            raise _problem(404, "machine_v2_intake_not_found", "Machine v2 intake was not found.")
        if action not in {"reserved", "existing"}:
            raise _problem(409, "machine_v2_start_invalid", "Machine v2 intake cannot start from its current state.")
        if action == "existing" and reservation.get("lifecycle_state") != "start_reserved":
            payload = _status_payload(reservation)
            for key in ("daily_start_limit", "daily_started_count", "daily_remaining_capacity"):
                if key in reservation:
                    payload[key] = int(reservation[key])
            return payload

        bundle = reservation.get("bundle_payload")
        if not isinstance(bundle, dict):
            raise _problem(503, "machine_v2_bundle_unavailable", "Reserved evidence bundle is unavailable.")
        context = {
            "source": "leadgen_machine_v2",
            "target_environment": target_environment,
            "rdi_scoring_version": scoring_version,
            "leadgen_machine_v2": {
                "contract_version": CONTRACT_VERSION,
                "intake_id": intake_id,
                "external_company_id": str(reservation["external_company_id"]),
                "canonical_domain": reservation["canonical_domain"],
                "campaign_id": reservation["campaign_id"],
                "iteration_id": reservation["iteration_id"],
                "source_run_id": reservation["source_run_id"],
                "batch_id": reservation["batch_id"],
                "evidence_bundle_sha256": reservation["evidence_bundle_sha256"],
                "requires_specter_mcp": bool(reservation.get("requires_specter_mcp")),
                "leadgen_business_date": str(reservation["leadgen_business_date"]),
                "actual_start_business_date": actual_start_date,
            },
        }
        try:
            remote = await dependencies.start_adapter(
                reservation["job_id"],
                [f"https://{reservation['canonical_domain']}"],
                context,
                _machine_actor(),
            )
        except Exception:
            finalized = store.finalize_machine_v2_start(
                intake_id=intake_id,
                job_id=reservation["job_id"],
                lifecycle_state="uncertain",
                actor=SERVICE_ACTOR,
                safe_error_code="remote_start_outcome_uncertain",
            )
            return _status_payload(finalized or {**reservation, "lifecycle_state": "uncertain"})
        if isinstance(remote, MachineStartDefiniteRejection):
            quota_blocked = remote.error_code == "machine_specter_mcp_quota_exhausted"
            released = store.release_machine_v2_start(
                intake_id=intake_id,
                job_id=reservation["job_id"],
                actor=SERVICE_ACTOR,
                lifecycle_state=("pending_provider_quota" if quota_blocked else "intake_pending"),
                wait_reason=remote.error_code if quota_blocked else None,
                blocked_until=remote.blocked_until if quota_blocked else None,
            )
            if released is None:
                raise _problem(503, "machine_v2_start_release_failed", "The rejected v2 start could not be safely released.")
            if quota_blocked:
                payload = _status_payload(released)
                payload["daily_started_count"] = max(int(reservation.get("daily_started_count") or 1) - 1, 0)
                return payload
            raise _problem(remote.status_code, remote.error_code, remote.message)
        if not isinstance(remote, MachineStartAccepted) or remote.job_id != reservation["job_id"]:
            finalized = store.finalize_machine_v2_start(
                intake_id=intake_id,
                job_id=reservation["job_id"],
                lifecycle_state="start_reserved",
                actor=SERVICE_ACTOR,
                safe_error_code="remote_start_outcome_uncertain",
            )
            return _status_payload(finalized or reservation)
        finalized = store.finalize_machine_v2_start(
            intake_id=intake_id,
            job_id=reservation["job_id"],
            lifecycle_state="queued",
            actor=SERVICE_ACTOR,
            safe_error_code=None,
        )
        payload = _status_payload(finalized or {**reservation, "lifecycle_state": "queued"})
        for key in ("daily_start_limit", "daily_started_count", "daily_remaining_capacity"):
            if key in reservation:
                payload[key] = int(reservation[key])
        return payload

    @router.get(
        "/api/machine/leadgen/v2/intakes/{intake_id}/status",
        response_model=None,
    )
    async def status(intake_id: str) -> dict[str, Any] | JSONResponse:
        if not _INTAKE_RE.fullmatch(intake_id):
            raise _problem(422, "machine_v2_intake_id_invalid", "Machine v2 intake ID is invalid.")
        record = _require_store(dependencies).load_machine_v2_lifecycle(intake_id)
        if record is None:
            raise _problem(404, "machine_v2_intake_not_found", "Machine v2 intake was not found.")
        payload = _status_payload(record)
        if not payload["terminal"]:
            return JSONResponse(status_code=202, content=payload)
        return payload

    @router.get("/api/machine/leadgen/v2/intakes/{intake_id}/result")
    async def result(intake_id: str) -> dict[str, Any]:
        if not _INTAKE_RE.fullmatch(intake_id):
            raise _problem(422, "machine_v2_intake_id_invalid", "Machine v2 intake ID is invalid.")
        record = _require_store(dependencies).load_machine_v2_lifecycle(intake_id)
        if record is None:
            raise _problem(404, "machine_v2_intake_not_found", "Machine v2 intake was not found.")
        return _v2_result_payload(record)

    @router.get("/api/machine/leadgen/v2/intakes/{intake_id}/error")
    async def error(intake_id: str) -> dict[str, Any]:
        if not _INTAKE_RE.fullmatch(intake_id):
            raise _problem(422, "machine_v2_intake_id_invalid", "Machine v2 intake ID is invalid.")
        record = _require_store(dependencies).load_machine_v2_lifecycle(intake_id)
        if record is None:
            raise _problem(404, "machine_v2_intake_not_found", "Machine v2 intake was not found.")
        return _v2_error_payload(record)

    return router


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "FrozenLeadGenEvidenceBundleV1",
    "LeadGenAuthorizationManifest",
    "MachineV2Dependencies",
    "build_leadgen_machine_v2_router",
    "canonical_bundle_sha256",
]
