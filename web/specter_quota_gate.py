"""Shared, fail-closed runtime gate for Specter MCP daily quota exhaustion."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, Protocol

from agent.ingest.specter_mcp_client import (
    SPECTER_MCP_QUOTA_ERROR_CODE,
    SpecterCompanyNotFoundError,
    SpecterQuotaLimitError,
    specter_quota_reset_hint,
)

logger = logging.getLogger(__name__)

GATE_MODE_ENV = "SPECTER_MCP_QUOTA_GATE_MODE"
GATE_MODE_OBSERVE = "observe"
GATE_MODE_ENFORCE = "enforce"
RECOVERY_PROBE_LEASE_SECONDS = 60


class SpecterQuotaGateUnavailable(RuntimeError):
    """The durable gate could not be read or changed safely."""


def requires_specter_mcp(
    *,
    input_mode: str,
    use_specter_mcp: bool = False,
    specter_urls: Any = None,
) -> bool:
    """Return whether one analysis shape will call Specter's MCP transport."""
    normalized_mode = str(input_mode or "").strip().lower()
    if normalized_mode == "pitchdeck":
        return bool(use_specter_mcp)
    if normalized_mode == "specter":
        return bool(specter_urls)
    return False


class SpecterQuotaGateStore(Protocol):
    def is_configured(self) -> bool: ...

    def get_specter_mcp_quota_gate(self, **request: Any) -> dict[str, Any] | None: ...

    def trip_specter_mcp_quota_gate(self, **request: Any) -> dict[str, Any] | None: ...

    def acquire_specter_mcp_quota_probe(self, **request: Any) -> dict[str, Any] | None: ...

    def finish_specter_mcp_quota_probe(self, **request: Any) -> dict[str, Any] | None: ...


def specter_quota_gate_mode() -> str:
    value = (os.getenv(GATE_MODE_ENV) or GATE_MODE_OBSERVE).strip().lower()
    if value not in {GATE_MODE_OBSERVE, GATE_MODE_ENFORCE}:
        raise SpecterQuotaGateUnavailable(
            f"{GATE_MODE_ENV} must be 'observe' or 'enforce'."
        )
    return value


def specter_quota_gate_enforced() -> bool:
    return specter_quota_gate_mode() == GATE_MODE_ENFORCE


def specter_quota_target_environment() -> str:
    configured = (os.getenv("RDI_LEADGEN_TARGET_ENVIRONMENT") or "").strip().lower()
    if configured in {"staging", "production"}:
        return configured
    app_env = (os.getenv("APP_ENV") or "development").strip().lower()
    return "production" if app_env in {"prod", "production"} else "staging"


def _open_fallback(*, storage_available: bool) -> dict[str, Any]:
    return {
        "provider": "specter_mcp",
        "target_environment": specter_quota_target_environment(),
        "state": "open",
        "enforcement_enabled": False,
        "accepting_new_analyses": True,
        "quota_remaining": "unknown",
        "blocked_until": None,
        "next_probe_at": None,
        "retry_after_seconds": 0,
        "reason_code": None,
        "observed_at": None,
        "gate_storage_available": storage_available,
    }


def _require_payload(payload: Any, operation: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SpecterQuotaGateUnavailable(
            f"Specter MCP quota gate {operation} did not return a record."
        )
    state = str(payload.get("state") or "")
    if state not in {"open", "blocked", "probing"}:
        raise SpecterQuotaGateUnavailable(
            f"Specter MCP quota gate {operation} returned an invalid state."
        )
    normalized = dict(payload)
    normalized["gate_storage_available"] = True
    return normalized


def get_specter_quota_availability(
    store: SpecterQuotaGateStore | None,
) -> dict[str, Any]:
    enforced = specter_quota_gate_enforced()
    gate_read = getattr(store, "get_specter_mcp_quota_gate", None)
    if store is None or not store.is_configured() or not callable(gate_read):
        if enforced:
            raise SpecterQuotaGateUnavailable(
                "Specter MCP quota gate storage is not configured."
            )
        logger.warning("Specter MCP quota gate is observing without durable storage")
        return _open_fallback(storage_available=False)

    payload = gate_read(
        target_environment=specter_quota_target_environment(),
        enforcement_enabled=enforced,
    )
    if payload is None and not enforced:
        logger.warning("Specter MCP quota gate read failed in observe mode")
        return _open_fallback(storage_available=False)
    return _require_payload(payload, "read")


def trip_specter_quota_gate(
    store: SpecterQuotaGateStore | None,
    *,
    error: BaseException,
    source_component: str,
    source_job_id: str | None = None,
    retry_after_seconds: int | None = None,
    reason_code: str = SPECTER_MCP_QUOTA_ERROR_CODE,
) -> dict[str, Any]:
    enforced = specter_quota_gate_enforced()
    gate_trip = getattr(store, "trip_specter_mcp_quota_gate", None)
    if store is None or not store.is_configured() or not callable(gate_trip):
        if enforced:
            raise SpecterQuotaGateUnavailable(
                "Specter MCP quota gate could not persist quota exhaustion."
            )
        logger.warning(
            "Specter MCP provider block could not be persisted: reason=%s source=%s job=%s",
            reason_code,
            source_component,
            source_job_id or "-",
        )
        return _open_fallback(storage_available=False)

    reset_hint = getattr(error, "reset_hint", None) or specter_quota_reset_hint(
        str(error)
    )
    payload = gate_trip(
        target_environment=specter_quota_target_environment(),
        enforcement_enabled=enforced,
        reason_code=reason_code,
        reset_hint=reset_hint,
        source_component=source_component,
        source_job_id=source_job_id,
        retry_after_seconds=retry_after_seconds,
    )
    return _require_payload(payload, "trip")


def maybe_recover_specter_quota_gate(
    store: SpecterQuotaGateStore | None,
    *,
    probe: Callable[[], Any],
) -> dict[str, Any]:
    availability = get_specter_quota_availability(store)
    if availability.get("accepting_new_analyses"):
        return availability
    acquire_probe = getattr(store, "acquire_specter_mcp_quota_probe", None)
    finish_probe = getattr(store, "finish_specter_mcp_quota_probe", None)
    if store is None or not callable(acquire_probe) or not callable(finish_probe):
        raise SpecterQuotaGateUnavailable("Specter MCP quota gate store is missing.")

    token = str(uuid.uuid4())
    acquired = acquire_probe(
        target_environment=specter_quota_target_environment(),
        enforcement_enabled=True,
        probe_lease_token=token,
        lease_seconds=RECOVERY_PROBE_LEASE_SECONDS,
    )
    acquired = _require_payload(acquired, "probe acquisition")
    if acquired.get("action") != "acquired":
        return acquired

    succeeded = False
    reason_code: str | None = None
    try:
        probe()
        succeeded = True
    except SpecterCompanyNotFoundError:
        # A definitive not-found result proves the MCP accepted the call.
        succeeded = True
    except SpecterQuotaLimitError:
        reason_code = SPECTER_MCP_QUOTA_ERROR_CODE
    except Exception:
        logger.warning("Specter MCP recovery probe failed", exc_info=True)
        reason_code = "specter_mcp_recovery_probe_failed"

    finished = finish_probe(
        target_environment=specter_quota_target_environment(),
        enforcement_enabled=True,
        probe_lease_token=token,
        succeeded=succeeded,
        reason_code=reason_code,
    )
    return _require_payload(finished, "probe completion")


_PUBLIC_FIELDS = (
    "provider",
    "target_environment",
    "state",
    "enforcement_enabled",
    "accepting_new_analyses",
    "quota_remaining",
    "blocked_until",
    "next_probe_at",
    "retry_after_seconds",
    "reason_code",
    "reset_hint",
    "observed_at",
    "updated_at",
    "gate_storage_available",
)


def public_specter_quota_availability(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the stable, non-sensitive availability contract."""
    return {key: payload.get(key) for key in _PUBLIC_FIELDS if key in payload}
