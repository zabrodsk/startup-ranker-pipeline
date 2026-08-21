"""Stable machine boundary for one-company LeadGen-to-RDI lifecycle calls."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable, Coroutine, Literal, Protocol, cast
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import Response

from web.leadgen_domain import company_source_host, normalize_company_domain
from web.specter_quota_gate import (
    SpecterQuotaGateUnavailable,
    get_specter_quota_availability,
    public_specter_quota_availability,
)

CONTRACT_VERSION = "rdi.leadgen-machine.v1"
SERVICE_ACTOR = "service:rockaway-leadgen"
SERVICE_KEY_ENV = "RDI_LEADGEN_AUTOSTART_KEY"
START_ENABLED_ENV = "RDI_LEADGEN_AUTOSTART_ENABLED"
DAILY_START_LIMIT_ENV = "RDI_LEADGEN_DAILY_START_LIMIT"
LEGACY_GLOBAL_START_LIMIT_ENV = "RDI_LEADGEN_GLOBAL_START_LIMIT"
SCORING_VERSION_ENV = "RDI_SCORING_VERSION"
TARGET_ENVIRONMENT_ENV = "RDI_LEADGEN_TARGET_ENVIRONMENT"
SERVICE_KEY_HEADER = "X-LeadGen-Service-Key"
BUSINESS_TIMEZONE = "Europe/Prague"

_LOGGER = logging.getLogger(__name__)
_PRAGUE = ZoneInfo(BUSINESS_TIMEZONE)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INTAKE_REFERENCE_RE = re.compile(r"^rdi-intake-[0-9a-f]{32}$")
_SAFE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def _strict_identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            "must be 1-128 characters using letters, digits, dot, underscore, colon, or hyphen"
        )
    return value


def _validated_intake_reference(value: str) -> str:
    if not _INTAKE_REFERENCE_RE.fullmatch(value):
        raise _problem(422, "machine_intake_id_invalid", "Machine intake ID is invalid.")
    return value


class MachineIntakeRequest(BaseModel):
    """One canonical LeadGen company and its stable caller identity."""

    model_config = ConfigDict(extra="forbid", strict=True)

    external_company_id: str
    canonical_domain: str
    campaign_id: str
    iteration_id: str
    source_run_id: str
    batch_id: str
    idempotency_key: str
    business_date: str
    business_timezone: Literal["Europe/Prague"]
    target_environment: Literal["staging", "production"]
    provenance_reference: str = Field(min_length=1, max_length=512)

    @field_validator(
        "external_company_id",
        "campaign_id",
        "iteration_id",
        "source_run_id",
        "batch_id",
        "idempotency_key",
    )
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return _strict_identifier(value)

    @field_validator("canonical_domain", mode="before")
    @classmethod
    def _validate_domain(cls, value: Any) -> str:
        normalized = normalize_company_domain(value)
        if normalized is None:
            raise ValueError("must be a canonical lowercase company-owned domain")
        if company_source_host(normalized):
            raise ValueError("must not use a shared source or social host")
        return normalized

    @field_validator("provenance_reference")
    @classmethod
    def _validate_provenance(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("must be bounded printable text without surrounding whitespace")
        return value

    @field_validator("business_date")
    @classmethod
    def _validate_business_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("must be an ISO YYYY-MM-DD date") from exc
        if parsed.isoformat() != value:
            raise ValueError("must be an ISO YYYY-MM-DD date")
        return value


class MachineStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_environment: Literal["staging", "production"]
    business_date: str
    business_timezone: Literal["Europe/Prague"]

    @field_validator("business_date")
    @classmethod
    def _validate_business_date(cls, value: str) -> str:
        return MachineIntakeRequest._validate_business_date(value)


LifecycleState = Literal[
    "accepted",
    "rejected",
    "start_fenced",
    "uncertain",
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


class _MachineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["rdi.leadgen-machine.v1"]
    intake_id: str
    external_company_id: str
    rdi_company_id: str | None
    rdi_correlation_id: str


class MachineIntakeResponse(_MachineResponse):
    """Durable accepted or rejected intake mapping."""

    batch_id: str
    intake_status: Literal["accepted", "rejected"]
    rejection_code: str | None
    approval_required: bool
    job_id: str | None


class MachineStartResponse(_MachineResponse):
    """Stable start fence or replay outcome."""

    job_id: str
    lifecycle_state: LifecycleState
    actor: Literal["service:rockaway-leadgen"]
    uncertain: bool
    business_date: str
    business_timezone: Literal["Europe/Prague"]
    daily_start_limit: int = Field(ge=1, le=20)
    daily_started_count: int = Field(ge=0, le=20)
    daily_remaining_capacity: int = Field(ge=0, le=20)


class MachineStatusResponse(_MachineResponse):
    """Current persisted lifecycle state and timestamps."""

    job_id: str | None
    lifecycle_state: LifecycleState
    terminal: bool
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


class MachineResultResponse(_MachineResponse):
    """Authoritative successful terminal result."""

    result_id: str
    job_id: str
    final_status: Literal["succeeded"]
    composite_score: str
    strategy_fit_score: str
    team_score: str
    upside_score: str
    rdi_bucket: str
    completed_at: str
    pipeline_version: str
    scoring_version: str
    checksum: str


class MachineErrorResponse(_MachineResponse):
    """Redacted authoritative terminal failure."""

    job_id: str
    final_status: Literal["failed", "cancelled", "rejected"]
    error_code: str
    error_class: str
    message: str
    terminal_at: str


class MachineLifecycleStore(Protocol):
    def is_configured(self) -> bool: ...

    def create_machine_intake(self, **record: Any) -> dict[str, Any] | None: ...

    def reserve_machine_start(self, **record: Any) -> dict[str, Any] | None: ...

    def release_machine_start(self, **record: Any) -> dict[str, Any] | None: ...

    def finalize_machine_start(self, **record: Any) -> dict[str, Any] | None: ...

    def load_machine_lifecycle(self, intake_id: str) -> dict[str, Any] | None: ...

    def get_specter_mcp_quota_gate(self, **request: Any) -> dict[str, Any] | None: ...

    def trip_specter_mcp_quota_gate(self, **request: Any) -> dict[str, Any] | None: ...

    def acquire_specter_mcp_quota_probe(self, **request: Any) -> dict[str, Any] | None: ...

    def finish_specter_mcp_quota_probe(self, **request: Any) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class MachineStartAccepted:
    """A typed acknowledgement that the existing runtime accepted the job."""

    job_id: str
    status: Literal["accepted", "pending", "queued", "running"]


@dataclass(frozen=True)
class MachineStartDefiniteRejection:
    """A bounded outcome proving the existing runtime did not accept the job."""

    status_code: Literal[400, 429, 503]
    error_code: str
    message: str
    blocked_until: str | None = None
    retry_after_seconds: int | None = None


MachineStartAdapter = Callable[
    [str, list[dict[str, str] | str], dict[str, Any], dict[str, str | None]],
    Awaitable[object],
]
MachineAvailabilityAdapter = Callable[[], Awaitable[dict[str, Any]]]


class MachineLifecycleDependencies:
    """Runtime dependencies injected into the machine lifecycle router."""

    def __init__(
        self,
        *,
        store: MachineLifecycleStore | None,
        start_adapter: MachineStartAdapter,
        availability_adapter: MachineAvailabilityAdapter | None = None,
    ) -> None:
        """Capture the persistence and existing RDI start seams."""
        self.store = store
        self.start_adapter = start_adapter
        self.availability_adapter = availability_adapter


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _stable_reference(prefix: str, material: dict[str, Any]) -> str:
    return f"{prefix}-{_sha256(material)[:32]}"


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
    return HTTPException(
        status_code=status_code,
        detail=detail,
        headers=headers,
    )


def _require_service_key(provided: str | None) -> None:
    expected = os.getenv(SERVICE_KEY_ENV)
    if expected is None or not expected:
        raise _problem(503, "machine_auth_not_configured", "Machine authentication is not configured.")
    if provided is None or not hmac.compare_digest(provided, expected):
        raise _problem(401, "machine_auth_invalid", "Machine authentication failed.")


class MachineAuthenticatedRoute(APIRoute):
    """Authenticate raw machine requests before FastAPI parses their bodies."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_handler = super().get_route_handler()

        async def authenticated_handler(request: Request) -> Response:
            _require_service_key(request.headers.get(SERVICE_KEY_HEADER))
            try:
                return await original_handler(request)
            except RequestValidationError:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "code": "machine_request_invalid",
                            "message": "Machine request is invalid.",
                            "contract_version": CONTRACT_VERSION,
                        }
                    },
                )

        return authenticated_handler


def _require_start_enabled() -> None:
    raw = os.getenv(START_ENABLED_ENV)
    if raw is None or raw == "" or raw.lower() in {"0", "false"}:
        raise _problem(503, "autonomous_start_disabled", "Autonomous machine start is disabled.")
    if raw != "true":
        raise _problem(503, "invalid_start_configuration", "Autonomous start configuration is invalid.")


def _daily_start_limit() -> int:
    preferred = os.getenv(DAILY_START_LIMIT_ENV)
    legacy = os.getenv(LEGACY_GLOBAL_START_LIMIT_ENV)
    if preferred is not None:
        if legacy not in {None, "", preferred}:
            raise _problem(
                503,
                "invalid_start_configuration",
                "Daily start limit configuration is ambiguous.",
            )
        raw = preferred
        if legacy is not None:
            _LOGGER.warning(
                "%s is deprecated; use %s",
                LEGACY_GLOBAL_START_LIMIT_ENV,
                DAILY_START_LIMIT_ENV,
            )
    elif legacy is not None:
        raw = legacy
        _LOGGER.warning(
            "%s is deprecated; use %s",
            LEGACY_GLOBAL_START_LIMIT_ENV,
            DAILY_START_LIMIT_ENV,
        )
    else:
        raw = "20"
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise _problem(503, "invalid_start_configuration", "Daily start limit configuration is invalid.") from exc
    if value < 1 or value > 20:
        raise _problem(503, "invalid_start_configuration", "Daily start limit configuration is invalid.")
    return value


def _prague_business_date(at: datetime | None = None) -> date:
    observed = at or datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("business-date clock must be timezone-aware")
    return observed.astimezone(_PRAGUE).date()


def _require_current_business_date(value: str) -> None:
    if date.fromisoformat(value) != _prague_business_date():
        raise _problem(
            409,
            "machine_business_date_closed",
            "Machine intake business date is not the current Europe/Prague date.",
        )


def _scoring_version() -> str:
    value = os.getenv(SCORING_VERSION_ENV)
    if value is None or not _IDENTIFIER_RE.fullmatch(value):
        raise _problem(503, "invalid_start_configuration", "RDI scoring version is not configured.")
    return value


def _target_environment(requested: str) -> Literal["staging", "production"]:
    configured = os.getenv(TARGET_ENVIRONMENT_ENV)
    if configured not in {"staging", "production"}:
        raise _problem(
            503,
            "machine_target_environment_not_configured",
            "Machine target environment is not configured.",
        )
    if requested != configured:
        raise _problem(
            409,
            "machine_target_environment_mismatch",
            "Machine request does not match the server target environment.",
        )
    return cast(Literal["staging", "production"], configured)


def _require_store(dependencies: MachineLifecycleDependencies) -> MachineLifecycleStore:
    store = dependencies.store
    if store is None or not store.is_configured():
        raise _problem(503, "machine_storage_not_configured", "Machine lifecycle storage is not configured.")
    return store


def _intake_record(
    request: MachineIntakeRequest,
    *,
    target_environment: Literal["staging", "production"],
) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    payload["target_environment"] = target_environment
    identity = {
        key: payload[key]
        for key in (
            "external_company_id",
            "campaign_id",
            "iteration_id",
            "source_run_id",
            "batch_id",
            "idempotency_key",
            "business_date",
            "business_timezone",
            "target_environment",
        )
    }
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "contract_version": CONTRACT_VERSION,
        "intake_id": _stable_reference("rdi-intake", identity),
        "idempotency_identity": _sha256(identity),
        "payload_hash": _sha256(payload),
        "rdi_company_id": None,
        "rdi_correlation_id": _stable_reference("rdi-correlation", identity),
        **payload,
        "intake_status": "accepted",
        "lifecycle_state": "accepted",
        "approval_required": False,
        "rejection_code": None,
        "job_id": None,
        "created_at": created_at,
        "updated_at": created_at,
        "started_at": None,
        "completed_at": None,
    }


def _intake_response(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "intake_id": record["intake_id"],
        "batch_id": record["batch_id"],
        "external_company_id": record["external_company_id"],
        "rdi_company_id": record["rdi_company_id"],
        "rdi_correlation_id": record["rdi_correlation_id"],
        "intake_status": record.get("intake_status", "accepted"),
        "rejection_code": record.get("rejection_code"),
        "approval_required": bool(record.get("approval_required", False)),
        "job_id": record.get("job_id"),
    }


def _machine_actor() -> dict[str, str | None]:
    return {
        "started_by_user_id": SERVICE_ACTOR,
        "started_by_display_name": SERVICE_ACTOR,
        "started_by_label": SERVICE_ACTOR,
    }


def _job_id(intake_id: str) -> str:
    digest = hashlib.sha256(intake_id.encode("utf-8")).hexdigest()[:32]
    return f"rdi-job-{digest}"


def _start_response(record: dict[str, Any]) -> dict[str, Any]:
    state = str(record.get("lifecycle_state") or "start_fenced")
    return {
        "contract_version": CONTRACT_VERSION,
        "intake_id": record["intake_id"],
        "external_company_id": record["external_company_id"],
        "rdi_company_id": record["rdi_company_id"],
        "rdi_correlation_id": record["rdi_correlation_id"],
        "job_id": record["job_id"],
        "lifecycle_state": state,
        "actor": SERVICE_ACTOR,
        "uncertain": state in {"start_fenced", "uncertain"},
        "business_date": str(record["business_date"]),
        "business_timezone": record["business_timezone"],
        "daily_start_limit": int(record["daily_start_limit"]),
        "daily_started_count": int(record["daily_started_count"]),
        "daily_remaining_capacity": int(record["daily_remaining_capacity"]),
    }


def _provider_block_problem(record: dict[str, Any]) -> HTTPException:
    retry_after = record.get("retry_after_seconds")
    try:
        retry_after_seconds = max(1, int(retry_after))
    except (TypeError, ValueError):
        next_probe_at = record.get("next_probe_at") or record.get("blocked_until")
        retry_after_seconds = 1
        if next_probe_at:
            try:
                parsed = datetime.fromisoformat(str(next_probe_at).replace("Z", "+00:00"))
                retry_after_seconds = max(
                    1,
                    int((parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()),
                )
            except (TypeError, ValueError):
                pass
    return _problem(
        429,
        "machine_specter_mcp_quota_exhausted",
        "Specter MCP quota is exhausted; no new machine analysis was accepted.",
        extra_detail={
            "provider": "specter_mcp",
            "blocked_until": record.get("blocked_until"),
            "retry_after_seconds": retry_after_seconds,
            "quota_remaining": "unknown",
        },
        headers={"Retry-After": str(retry_after_seconds)},
    )


def _reservation_problem(record: dict[str, Any]) -> HTTPException:
    action = record.get("action")
    if action == "provider_blocked":
        return _provider_block_problem(record)
    if action == "unknown":
        return _problem(404, "machine_intake_not_found", "Machine intake was not found.")
    if action == "environment_mismatch":
        return _problem(409, "machine_intake_environment_mismatch", "Machine intake environment does not match.")
    if action == "scope_mismatch":
        return _problem(409, "machine_intake_scope_mismatch", "Machine intake daily scope does not match.")
    if action == "business_date_closed":
        return _problem(409, "machine_business_date_closed", "Machine intake business date is closed.")
    if action == "rate_limited":
        return _problem(429, "machine_start_rate_limited", "The machine-start limit is exhausted.")
    if action == "terminal_invalid":
        return _problem(409, "machine_intake_terminal_invalid", "Machine intake cannot be started from its terminal state.")
    return _problem(409, "machine_start_rejected", "Machine start reservation was rejected.")


async def _machine_availability(
    dependencies: MachineLifecycleDependencies,
) -> dict[str, Any]:
    store = _require_store(dependencies)
    try:
        if dependencies.availability_adapter is not None:
            availability = await dependencies.availability_adapter()
        else:
            availability = get_specter_quota_availability(store)
    except SpecterQuotaGateUnavailable as exc:
        raise _problem(
            503,
            "machine_specter_mcp_gate_unavailable",
            "Specter MCP availability cannot be verified safely.",
        ) from exc
    return public_specter_quota_availability(availability)


def _finalize_or_fence(
    store: MachineLifecycleStore,
    *,
    reservation: dict[str, Any],
    lifecycle_state: str,
    safe_error_code: str | None = None,
    safe_error_class: str | None = None,
    safe_error_message: str | None = None,
) -> dict[str, Any]:
    finalized = store.finalize_machine_start(
        intake_id=reservation["intake_id"],
        job_id=reservation["job_id"],
        lifecycle_state=lifecycle_state,
        safe_error_code=safe_error_code,
        safe_error_class=safe_error_class,
        safe_error_message=safe_error_message,
        actor=SERVICE_ACTOR,
    )
    if finalized is not None:
        return {
            **finalized,
            **{
                key: reservation[key]
                for key in (
                    "daily_start_limit",
                    "daily_started_count",
                    "daily_remaining_capacity",
                )
            },
        }
    return {**reservation, "lifecycle_state": "start_fenced"}


def _load_lifecycle(
    dependencies: MachineLifecycleDependencies,
    intake_id: str,
) -> dict[str, Any]:
    store = _require_store(dependencies)
    record = store.load_machine_lifecycle(intake_id)
    if record is None:
        raise _problem(404, "machine_intake_not_found", "Machine intake was not found.")
    return record


def _timestamp(value: Any, *, field: str) -> str:
    text = str(value or "")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is invalid.") from exc
    if parsed.tzinfo is None:
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} lacks a timezone.")
    normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return normalized


def _optional_timestamp(value: Any, *, field: str) -> str | None:
    return _timestamp(value, field=field) if value not in (None, "") else None


def _latest_timestamp(*values: str | None) -> str:
    """Return the latest already-normalized UTC timestamp."""
    present = [value for value in values if value is not None]
    return max(
        present,
        key=lambda value: datetime.fromisoformat(value[:-1] + "+00:00"),
    )


def _exact_decimal(value: Any, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is invalid.")
    text = value if isinstance(value, str) else str(value)
    if len(text) > 64 or not _DECIMAL_RE.fullmatch(text):
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is invalid.")
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as exc:
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is invalid.") from exc
    if not decimal_value.is_finite() or decimal_value < 0 or decimal_value > 100:
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is outside the score range.")
    return text


def _bounded_text(value: Any, *, field: str, maximum: int = 128) -> str:
    text = str(value or "")
    if not text or len(text) > maximum or text != text.strip():
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is invalid.")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise _problem(503, "machine_terminal_contract_invalid", f"Persisted {field} is invalid.")
    return text


def _authoritative_company_id(value: Any) -> str:
    if not isinstance(value, str):
        raise _problem(503, "machine_terminal_contract_invalid", "Persisted rdi_company_id is invalid.")
    try:
        normalized = str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise _problem(503, "machine_terminal_contract_invalid", "Persisted rdi_company_id is invalid.") from exc
    if value != normalized:
        raise _problem(503, "machine_terminal_contract_invalid", "Persisted rdi_company_id is invalid.")
    return normalized


def _safe_error_token(value: Any, fallback: str) -> str:
    text = str(value or "")
    return text if _SAFE_TOKEN_RE.fullmatch(text) else fallback


def _safe_error_message(value: Any) -> str:
    text = str(value or "")
    lowered = text.lower()
    forbidden = ("bearer", "token", "secret", "credential", "traceback", "stack")
    if (
        not text
        or len(text) > 240
        or text != text.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in text)
        or any(marker in lowered for marker in forbidden)
    ):
        return "The RDI analysis ended without a publishable result."
    return text


def _status_payload(record: dict[str, Any]) -> dict[str, Any]:
    state = str(record.get("lifecycle_state") or "")
    terminal = state in {"succeeded", "failed", "cancelled", "rejected"}
    created_at = _timestamp(record["created_at"], field="created_at")
    updated_at = _timestamp(record.get("updated_at") or record["created_at"], field="updated_at")
    started_at = _optional_timestamp(record.get("started_at"), field="started_at")
    completed_at = _optional_timestamp(record.get("completed_at"), field="completed_at")
    return {
        "contract_version": CONTRACT_VERSION,
        "intake_id": record["intake_id"],
        "external_company_id": record["external_company_id"],
        "rdi_company_id": record["rdi_company_id"],
        "rdi_correlation_id": record["rdi_correlation_id"],
        "job_id": record.get("job_id"),
        "lifecycle_state": state,
        "terminal": terminal,
        "created_at": created_at,
        "updated_at": _latest_timestamp(updated_at, started_at, completed_at),
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _result_payload(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("lifecycle_state") != "succeeded":
        if record.get("lifecycle_state") in {"failed", "cancelled", "rejected"}:
            raise _problem(409, "machine_result_unavailable", "This terminal lifecycle has no successful result.")
        raise _problem(409, "machine_result_not_terminal", "Machine result is not terminally available.")
    result = record.get("terminal_result")
    if not isinstance(result, dict):
        raise _problem(503, "machine_terminal_contract_invalid", "Persisted terminal result is unavailable.")
    authoritative_company_id = _authoritative_company_id(result.get("rdi_company_id"))
    persisted_company_id = record.get("rdi_company_id")
    if persisted_company_id is not None and persisted_company_id != authoritative_company_id:
        raise _problem(409, "machine_result_identity_mismatch", "Persisted result company identity does not match intake.")
    result_external_id = result.get("external_company_id")
    if result_external_id is not None and result_external_id != record.get("external_company_id"):
        raise _problem(409, "machine_result_identity_mismatch", "Persisted result company identity does not match intake.")
    completed_at = _timestamp(result.get("completed_at"), field="completed_at")
    payload = {
        "contract_version": CONTRACT_VERSION,
        "result_id": _stable_reference(
            "rdi-result",
            {
                "intake_id": record["intake_id"],
                "job_id": record.get("job_id"),
                "rdi_company_id": authoritative_company_id,
                "completed_at": completed_at,
            },
        ),
        "intake_id": record["intake_id"],
        "external_company_id": record["external_company_id"],
        "rdi_company_id": authoritative_company_id,
        "rdi_correlation_id": record["rdi_correlation_id"],
        "job_id": _bounded_text(record.get("job_id"), field="job_id"),
        "final_status": "succeeded",
        "composite_score": _exact_decimal(result.get("composite_score"), field="composite_score"),
        "strategy_fit_score": _exact_decimal(result.get("strategy_fit_score"), field="strategy_fit_score"),
        "team_score": _exact_decimal(result.get("team_score"), field="team_score"),
        "upside_score": _exact_decimal(result.get("upside_score"), field="upside_score"),
        "rdi_bucket": _bounded_text(result.get("rdi_bucket"), field="rdi_bucket"),
        "completed_at": completed_at,
        "pipeline_version": _bounded_text(result.get("pipeline_version"), field="pipeline_version"),
        "scoring_version": _bounded_text(result.get("scoring_version"), field="scoring_version"),
    }
    payload["checksum"] = _sha256(payload)
    return payload


def _error_payload(record: dict[str, Any]) -> dict[str, Any]:
    state = str(record.get("lifecycle_state") or "")
    if state not in {"failed", "cancelled", "rejected"}:
        if state == "succeeded":
            raise _problem(409, "machine_error_unavailable", "This terminal lifecycle has no terminal error.")
        raise _problem(409, "machine_error_not_terminal", "Machine error is not terminally available.")
    return {
        "contract_version": CONTRACT_VERSION,
        "intake_id": record["intake_id"],
        "external_company_id": record["external_company_id"],
        "rdi_company_id": record["rdi_company_id"],
        "rdi_correlation_id": record["rdi_correlation_id"],
        "job_id": _bounded_text(record.get("job_id"), field="job_id"),
        "final_status": state,
        "error_code": _safe_error_token(record.get("safe_error_code"), "analysis_failed"),
        "error_class": _safe_error_token(record.get("safe_error_class"), "terminal_analysis_failure"),
        "message": _safe_error_message(record.get("safe_error_message")),
        "terminal_at": _timestamp(record.get("completed_at"), field="terminal_at"),
    }


def build_leadgen_machine_router(
    dependencies: MachineLifecycleDependencies,
) -> APIRouter:
    """Build the additive version-one machine lifecycle route family."""
    router = APIRouter(route_class=MachineAuthenticatedRoute)

    @router.get("/api/machine/leadgen/v1/availability")
    async def availability() -> dict[str, Any]:
        return await _machine_availability(dependencies)

    @router.post(
        "/api/machine/leadgen/v1/intakes",
        status_code=202,
        response_model=MachineIntakeResponse,
    )
    async def create_intake(
        request: MachineIntakeRequest,
    ) -> dict[str, Any]:
        target_environment = _target_environment(request.target_environment)
        _require_current_business_date(request.business_date)
        store = _require_store(dependencies)
        result = store.create_machine_intake(
            **_intake_record(request, target_environment=target_environment)
        )
        if result is None:
            raise _problem(503, "machine_intake_persistence_failed", "Machine intake could not be persisted.")
        if result.get("action") == "provider_blocked":
            raise _provider_block_problem(result)
        if result.get("action") == "conflict":
            raise _problem(409, "machine_intake_payload_conflict", "The idempotency identity already has different material payload.")
        return _intake_response(result)

    @router.post(
        "/api/machine/leadgen/v1/intakes/{intake_id}/start",
        status_code=202,
        response_model=MachineStartResponse,
    )
    async def start_intake(
        intake_id: str,
        request: MachineStartRequest,
    ) -> dict[str, Any]:
        _require_start_enabled()
        target_environment = _target_environment(request.target_environment)
        scoring_version = _scoring_version()
        intake_id = _validated_intake_reference(intake_id)
        store = _require_store(dependencies)
        reservation = store.reserve_machine_start(
            intake_id=intake_id,
            target_environment=target_environment,
            business_date=request.business_date,
            business_timezone=request.business_timezone,
            job_id=_job_id(intake_id),
            actor=SERVICE_ACTOR,
            daily_start_limit=_daily_start_limit(),
        )
        if reservation is None:
            raise _problem(503, "machine_start_reservation_failed", "Machine start reservation could not be persisted.")
        action = reservation.get("action")
        if action == "existing":
            return _start_response(reservation)
        if action != "reserved":
            raise _reservation_problem(reservation)

        actor = _machine_actor()
        context = {
            "source": "leadgen_machine",
            "target_environment": target_environment,
            "rdi_scoring_version": scoring_version,
            "leadgen_machine": {
                "contract_version": CONTRACT_VERSION,
                "intake_id": reservation["intake_id"],
                "external_company_id": reservation["external_company_id"],
                "canonical_domain": reservation["canonical_domain"],
                "campaign_id": reservation["campaign_id"],
                "iteration_id": reservation["iteration_id"],
                "source_run_id": reservation["source_run_id"],
                "batch_id": reservation["batch_id"],
                "rdi_correlation_id": reservation["rdi_correlation_id"],
                "target_environment": target_environment,
                "business_date": reservation["business_date"],
                "business_timezone": reservation["business_timezone"],
                "daily_start_limit": reservation["daily_start_limit"],
                "daily_started_count": reservation["daily_started_count"],
                "daily_remaining_capacity": reservation["daily_remaining_capacity"],
                "actor": SERVICE_ACTOR,
            },
        }
        try:
            remote_result = await dependencies.start_adapter(
                reservation["job_id"],
                [f"https://{reservation['canonical_domain']}"],
                context,
                actor,
            )
        except Exception:
            uncertain = _finalize_or_fence(
                store,
                reservation=reservation,
                lifecycle_state="uncertain",
                safe_error_code="remote_start_outcome_uncertain",
                safe_error_class="uncertain_remote_acceptance",
                safe_error_message="Remote start outcome is uncertain; replay will not start again.",
            )
            return _start_response(uncertain)

        if isinstance(remote_result, MachineStartDefiniteRejection):
            released = store.release_machine_start(
                intake_id=reservation["intake_id"],
                job_id=reservation["job_id"],
                actor=SERVICE_ACTOR,
            )
            if released is None:
                raise _problem(
                    503,
                    "machine_start_release_failed",
                    "The rejected start fence could not be safely released.",
                )
            raise _problem(
                remote_result.status_code,
                remote_result.error_code,
                remote_result.message,
                extra_detail={
                    "provider": "specter_mcp",
                    "blocked_until": remote_result.blocked_until,
                    "retry_after_seconds": remote_result.retry_after_seconds,
                    "quota_remaining": "unknown",
                }
                if remote_result.error_code == "machine_specter_mcp_quota_exhausted"
                else None,
                headers={"Retry-After": str(max(1, remote_result.retry_after_seconds or 1))}
                if remote_result.error_code == "machine_specter_mcp_quota_exhausted"
                else None,
            )

        if not isinstance(remote_result, MachineStartAccepted):
            uncertain = _finalize_or_fence(
                store,
                reservation=reservation,
                lifecycle_state="uncertain",
                safe_error_code="remote_start_outcome_uncertain",
                safe_error_class="uncertain_remote_acceptance",
                safe_error_message="Remote start outcome is uncertain; replay will not start again.",
            )
            return _start_response(uncertain)
        if remote_result.job_id != reservation["job_id"]:
            uncertain = _finalize_or_fence(
                store,
                reservation=reservation,
                lifecycle_state="uncertain",
                safe_error_code="remote_job_identity_mismatch",
                safe_error_class="uncertain_remote_acceptance",
                safe_error_message="Remote start returned an unexpected identity; replay will not start again.",
            )
            return _start_response(uncertain)

        queued = _finalize_or_fence(
            store,
            reservation=reservation,
            lifecycle_state="queued",
        )
        return _start_response(queued)

    @router.get(
        "/api/machine/leadgen/v1/intakes/{intake_id}/status",
        response_model=MachineStatusResponse,
    )
    async def get_status(
        intake_id: str,
    ) -> MachineStatusResponse | JSONResponse:
        intake_id = _validated_intake_reference(intake_id)
        payload = _status_payload(_load_lifecycle(dependencies, intake_id))
        model = MachineStatusResponse.model_validate(payload)
        if not model.terminal:
            return JSONResponse(status_code=202, content=model.model_dump(mode="json"))
        return model

    @router.get(
        "/api/machine/leadgen/v1/intakes/{intake_id}/result",
        response_model=MachineResultResponse,
    )
    async def get_result(
        intake_id: str,
    ) -> MachineResultResponse:
        intake_id = _validated_intake_reference(intake_id)
        return MachineResultResponse.model_validate(
            _result_payload(_load_lifecycle(dependencies, intake_id))
        )

    @router.get(
        "/api/machine/leadgen/v1/intakes/{intake_id}/error",
        response_model=MachineErrorResponse,
    )
    async def get_error(
        intake_id: str,
    ) -> MachineErrorResponse:
        intake_id = _validated_intake_reference(intake_id)
        return MachineErrorResponse.model_validate(
            _error_payload(_load_lifecycle(dependencies, intake_id))
        )

    return router


__all__ = [
    "CONTRACT_VERSION",
    "SERVICE_ACTOR",
    "MachineIntakeRequest",
    "MachineIntakeResponse",
    "MachineStartResponse",
    "MachineStatusResponse",
    "MachineResultResponse",
    "MachineErrorResponse",
    "MachineLifecycleDependencies",
    "build_leadgen_machine_router",
]
