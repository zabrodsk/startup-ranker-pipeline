"""Atomic Specter quota broker helpers and stable public contracts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from zoneinfo import ZoneInfo

BROKER_MODE_ENV = "SPECTER_MCP_QUOTA_BROKER_MODE"
LEGACY_MODE_ENV = "SPECTER_MCP_QUOTA_GATE_MODE"
BROKER_MODE_OBSERVE = "observe"
BROKER_MODE_ENFORCE = "enforce"
BUSINESS_TIMEZONE = "Europe/Prague"
_PRAGUE = ZoneInfo(BUSINESS_TIMEZONE)

_CIRCUIT_TO_LEGACY_STATE = {
    "closed": "open",
    "open": "blocked",
    "probing": "probing",
}
_LEGACY_TO_CIRCUIT_STATE = {
    "open": "closed",
    "blocked": "open",
    "probing": "probing",
}


@dataclass(frozen=True)
class SpecterQuotaPolicy:
    observed_limit: int = 250
    safety_reserve: int = 25
    company_cap: int = 8
    founder_profile_cap: int = 3
    scheduled_import_allowance: int = 40
    recovery_allowance: int = 5
    campaign_reserve_cap: int = 160


@dataclass(frozen=True)
class SpecterQuotaUsage:
    total: int = 0
    company: int = 0
    founder_profiles: int = 0
    scheduled_import: int = 0
    recovery_probe: int = 0
    autonomous_campaign: int = 0


def specter_quota_decision(
    *,
    policy: SpecterQuotaPolicy,
    usage: SpecterQuotaUsage,
    quota_class: str,
    operation: str,
    remaining_rdi_slots: int | None,
    circuit_state: str = "closed",
) -> tuple[bool, str | None]:
    """Mirror the SQL policy for deterministic preflight and acceptance simulation."""
    if circuit_state != "closed":
        return False, "specter_mcp_quota_exhausted"
    if usage.company >= policy.company_cap:
        return False, "company_cap_exhausted"
    if operation == "get_person_profile" and usage.founder_profiles >= policy.founder_profile_cap:
        return False, "founder_profile_cap_exhausted"
    if (
        quota_class == "scheduled_import"
        and usage.scheduled_import >= policy.scheduled_import_allowance
    ):
        return False, "scheduled_import_allowance_exhausted"
    if quota_class == "recovery_probe" and usage.recovery_probe >= policy.recovery_allowance:
        return False, "recovery_allowance_exhausted"

    slots = 20 if remaining_rdi_slots is None else max(remaining_rdi_slots, 0)
    campaign_target = min(policy.campaign_reserve_cap, slots * policy.company_cap)
    remaining_import = max(policy.scheduled_import_allowance - usage.scheduled_import, 0)
    remaining_recovery = max(policy.recovery_allowance - usage.recovery_probe, 0)
    remaining_campaign = max(campaign_target - usage.autonomous_campaign, 0)
    if quota_class == "autonomous_campaign":
        floor = policy.safety_reserve + remaining_import + remaining_recovery
    elif quota_class == "scheduled_import":
        floor = policy.safety_reserve + remaining_recovery + remaining_campaign
    elif quota_class == "recovery_probe":
        floor = policy.safety_reserve + remaining_import + remaining_campaign
    else:
        floor = (
            policy.safety_reserve
            + remaining_import
            + remaining_recovery
            + remaining_campaign
        )
    if usage.total >= policy.observed_limit - floor:
        return False, "quota_estimate_exhausted"
    return True, None


@dataclass(frozen=True)
class SpecterQuotaBrokerRequest:
    target_environment: str
    business_date: str
    business_timezone: str
    consumer: str
    operation: str | None
    quota_class: str
    company_ref: str | None
    remaining_rdi_slots: int | None
    actor: str
    metadata: dict[str, Any]
    intake_id: str | None
    enforcement_enabled: bool


_broker_request_var: ContextVar[SpecterQuotaBrokerRequest | None] = ContextVar(
    "specter_quota_broker_request",
    default=None,
)


def specter_quota_broker_mode() -> str:
    raw = (os.getenv(BROKER_MODE_ENV) or "").strip().lower()
    if raw not in {BROKER_MODE_OBSERVE, BROKER_MODE_ENFORCE}:
        legacy = (os.getenv(LEGACY_MODE_ENV) or "").strip().lower()
        if legacy in {BROKER_MODE_OBSERVE, BROKER_MODE_ENFORCE}:
            return legacy
        return BROKER_MODE_OBSERVE
    return raw


def specter_quota_broker_configured() -> bool:
    raw = (os.getenv(BROKER_MODE_ENV) or "").strip().lower()
    legacy = (os.getenv(LEGACY_MODE_ENV) or "").strip().lower()
    return raw in {BROKER_MODE_OBSERVE, BROKER_MODE_ENFORCE} or legacy in {
        BROKER_MODE_OBSERVE,
        BROKER_MODE_ENFORCE,
    }


def specter_quota_broker_enforced() -> bool:
    return specter_quota_broker_mode() == BROKER_MODE_ENFORCE


def specter_quota_business_date(at: datetime | None = None) -> str:
    observed = at or datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("broker business date requires timezone-aware datetime")
    return observed.astimezone(_PRAGUE).date().isoformat()


def normalize_specter_quota_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    circuit_state = str(normalized.get("circuit_state") or "").strip().lower()
    legacy_state = str(normalized.get("state") or "").strip().lower()
    if circuit_state not in _CIRCUIT_TO_LEGACY_STATE:
        circuit_state = _LEGACY_TO_CIRCUIT_STATE.get(legacy_state, "closed")
    normalized["circuit_state"] = circuit_state
    normalized["state"] = _CIRCUIT_TO_LEGACY_STATE[circuit_state]
    if "reason" not in normalized and normalized.get("reason_code") is not None:
        normalized["reason"] = normalized.get("reason_code")
    if "reason_code" not in normalized and normalized.get("reason") is not None:
        normalized["reason_code"] = normalized.get("reason")
    if "retry_at" not in normalized:
        normalized["retry_at"] = (
            normalized.get("next_probe_at") or normalized.get("blocked_until")
        )
    if "business_date" not in normalized:
        normalized["business_date"] = specter_quota_business_date()
    if "estimated_remaining" not in normalized:
        normalized["estimated_remaining"] = normalized.get("quota_remaining", "unknown")
    if "quota_remaining" not in normalized:
        normalized["quota_remaining"] = normalized.get("estimated_remaining", "unknown")
    if "accepting_new_analyses" not in normalized:
        normalized["accepting_new_analyses"] = circuit_state == "closed"
    raw_status = str(normalized.get("status") or "").strip().lower()
    normalized["status_internal"] = raw_status
    normalized["status"] = external_specter_quota_status(raw_status)
    return normalized


_PUBLIC_AUTH_FIELDS = (
    "authorization_id",
    "status",
    "circuit_state",
    "state",
    "business_date",
    "retry_at",
    "reason",
    "reason_code",
    "estimated_remaining",
    "quota_remaining",
    "accepting_new_analyses",
    "target_environment",
    "provider",
    "status_internal",
)


def public_specter_quota_authorization(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_specter_quota_payload(payload)
    return {
        key: normalized.get(key)
        for key in _PUBLIC_AUTH_FIELDS
        if key in normalized
    }

def external_specter_quota_status(status: str | None) -> str | None:
    normalized = str(status or "").strip().lower()
    if normalized == "reserved":
        return "authorized"
    if normalized == "denied":
        return "deferred"
    return normalized or None


def specter_quota_target_environment() -> str:
    configured = (os.getenv("RDI_LEADGEN_TARGET_ENVIRONMENT") or "").strip().lower()
    if configured in {"staging", "production"}:
        return configured
    app_env = (os.getenv("APP_ENV") or "development").strip().lower()
    return "production" if app_env in {"prod", "production"} else "staging"


def current_specter_quota_broker_request() -> SpecterQuotaBrokerRequest | None:
    return _broker_request_var.get()


def build_specter_quota_broker_request(
    *,
    consumer: str,
    operation: str | None,
    quota_class: str,
    company_ref: str | None = None,
    remaining_rdi_slots: int | None = None,
    intake_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpecterQuotaBrokerRequest:
    bounded_remaining_slots: int | None
    if remaining_rdi_slots is None:
        bounded_remaining_slots = None
    else:
        bounded_remaining_slots = max(int(remaining_rdi_slots), 0)
    return SpecterQuotaBrokerRequest(
        target_environment=specter_quota_target_environment(),
        business_date=specter_quota_business_date(),
        business_timezone=BUSINESS_TIMEZONE,
        consumer=consumer,
        operation=operation,
        quota_class=quota_class,
        company_ref=company_ref,
        remaining_rdi_slots=bounded_remaining_slots,
        actor="service:rockaway-leadgen",
        metadata=dict(metadata or {}),
        intake_id=intake_id,
        enforcement_enabled=specter_quota_broker_enforced(),
    )


@contextmanager
def use_specter_quota_broker(
    request: SpecterQuotaBrokerRequest | None,
) -> Iterator[None]:
    token: Token[SpecterQuotaBrokerRequest | None] | None = None
    if request is not None:
        token = _broker_request_var.set(request)
    try:
        yield
    finally:
        if token is not None:
            _broker_request_var.reset(token)
