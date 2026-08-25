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
from starlette.responses import Response

from web.leadgen_domain import company_source_host, normalize_company_domain
from web.leadgen_machine import (
    MachineStartAccepted,
    MachineStartDefiniteRejection,
    _daily_start_limit,
    _machine_actor,
    _require_service_key,
    _require_start_enabled,
    _scoring_version,
    _stable_reference,
    _target_environment,
    _timestamp,
)
from web.leadgen_machine import (
    _error_payload as _v1_error_payload,
)
from web.leadgen_machine import (
    _result_payload as _v1_result_payload,
)
from web.specter_quota_broker import (
    BUSINESS_TIMEZONE as BROKER_TIMEZONE,
)
from web.specter_quota_broker import (
    public_specter_quota_authorization,
    specter_quota_broker_enforced,
    specter_quota_business_date,
)

CONTRACT_VERSION = "rdi.leadgen-machine.v2"
SERVICE_ACTOR = "service:rockaway-leadgen"
BUNDLE_SCHEMA_VERSION_V1 = "frozen-leadgen-evidence-bundle-v1"
BUNDLE_SCHEMA_VERSION_V2 = "frozen-leadgen-evidence-bundle-v2"
BUNDLE_SCHEMA_VERSION = BUNDLE_SCHEMA_VERSION_V2
BUSINESS_TIMEZONE = "Europe/Prague"
V2_ENABLED_ENV = "RDI_LEADGEN_MACHINE_V2_ENABLED"
RESEARCH_PACKET_SCHEMA_VERSION = "research-evidence-packet-v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPECTER_ID_RE = re.compile(r"^[0-9a-f]{24}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INTAKE_RE = re.compile(r"^rdi-v2-intake-[0-9a-f]{32}$")
_LEGACY_SCORING_VERSION_BY_PIPELINE = {"v1": "ranking-v1"}
_RESEARCH_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "company_ref",
        "identity",
        "claims",
        "contradiction_checked",
        "objective_coverage",
        "stale_objectives",
        "analysis_ready",
        "specter_refresh_required",
        "specter_evidence_state",
        "quota_authorization_id",
        "assessment",
        "packet_sha256",
    }
)
_RESEARCH_PACKET_IDENTITY_FIELDS = frozenset({"domain", "website_url"})
_RESEARCH_PACKET_STATES = frozenset(
    {
        "not_required",
        "cached_complete",
        "cached_partial",
        "fresh_authorized",
        "fresh_deferred_quota",
        "unavailable",
    }
)
_RESEARCH_OBJECTIVES = (
    "european_connection",
    "stage_and_funding",
    "founder_prior_execution",
    "founder_market_fit",
    "product_software_usp",
    "customer_or_deployment",
    "commercial_traction",
    "moat_or_defensibility",
    "market_problem_and_buyer",
    "momentum",
)
_RESEARCH_OBJECTIVE_CATEGORIES = {
    "european_connection": "geography",
    "stage_and_funding": "funding",
    "founder_prior_execution": "team",
    "founder_market_fit": "team",
    "product_software_usp": "product",
    "customer_or_deployment": "traction",
    "commercial_traction": "traction",
    "moat_or_defensibility": "product",
    "market_problem_and_buyer": "market",
    "momentum": "talent",
}
_RESEARCH_CORE_OBJECTIVES = (
    "european_connection",
    "stage_and_funding",
    "product_software_usp",
    "market_problem_and_buyer",
)
_RESEARCH_FOUNDER_OBJECTIVES = (
    "founder_prior_execution",
    "founder_market_fit",
)
# These shared profile/publishing hosts can carry evidence, but they do not
# establish independent corroboration. Keep this aligned with LeadGen's
# SHARED_COMPANY_IDENTITY_HOSTS contract. Editorial publications such as
# eu-startups.com remain valid independent evidence even though RDI also blocks
# them from being mistaken for a company's canonical domain.
_NON_INDEPENDENT_RESEARCH_HOSTS = frozenset(
    {
        "angel.co",
        "azurewebsites.net",
        "bitbucket.org",
        "blogspot.com",
        "bsky.app",
        "crunchbase.com",
        "discord.com",
        "discord.gg",
        "facebook.com",
        "github.com",
        "github.io",
        "gitlab.com",
        "gitlab.io",
        "herokuapp.com",
        "instagram.com",
        "linkedin.com",
        "medium.com",
        "notion.site",
        "notion.so",
        "pages.dev",
        "substack.com",
        "threads.net",
        "tiktok.com",
        "t.me",
        "twitter.com",
        "vercel.app",
        "webflow.io",
        "wixsite.com",
        "x.com",
        "youtu.be",
        "youtube.com",
    }
)
_RESEARCH_CLAIM_METADATA_FIELDS = frozenset(
    {
        "source_kind",
        "packet_sha256",
        "schema_version",
        "objective",
        "evidence_id",
        "category",
        "status",
        "publisher_domain",
        "observed_at",
        "published_at",
        "retrieved_at",
        "confidence",
        "confidence_reason_codes",
        "content_sha256",
    }
)


def _non_independent_research_host(domain: str) -> bool:
    return any(
        domain == shared or domain.endswith(f".{shared}")
        for shared in _NON_INDEPENDENT_RESEARCH_HOSTS
    )


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


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected_fields: frozenset[str],
    field_name: str,
) -> None:
    if frozenset(payload) != expected_fields:
        raise ValueError(f"{field_name} fields do not match the canonical schema")


def _normalize_packet_host(value: Any, *, field_name: str) -> str:
    normalized = normalize_company_domain(str(value or ""))
    if not normalized:
        raise ValueError(f"{field_name} must be a canonical hostname")
    return normalized


def _derived_research_packet_fields(
    payload: Mapping[str, Any],
    *,
    canonical_domain: str,
) -> dict[str, Any]:
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("research packet claims must be an array")
    stale_objectives = payload.get("stale_objectives")
    if not isinstance(stale_objectives, list):
        raise ValueError("research packet stale_objectives must be an array")
    stale_objective_set = {str(item) for item in stale_objectives}
    supported: set[str] = set()
    contradicted: set[str] = set()
    contradiction_refs: list[str] = []
    supporting_signals: list[dict[str, Any]] = []
    primary_signals: list[dict[str, Any]] = []
    for claim in raw_claims:
        if not isinstance(claim, Mapping) or frozenset(claim) != {"objective", "evidence"}:
            raise ValueError("research packet claims must contain canonical objective/evidence pairs")
        objective = str(claim.get("objective") or "")
        if objective not in _RESEARCH_OBJECTIVES:
            raise ValueError("research packet claim objective is invalid")
        evidence = claim.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("research packet evidence must be an object")
        expected_category = _RESEARCH_OBJECTIVE_CATEGORIES[objective]
        if str(evidence.get("category") or "") != expected_category:
            raise ValueError("research packet evidence category does not match its objective")
        if str(evidence.get("subject_company_ref") or "") != canonical_domain:
            raise ValueError("research packet evidence subject does not match bundle domain")
        publisher_domain = _normalize_packet_host(
            evidence.get("publisher_domain"),
            field_name="research packet evidence publisher_domain",
        )
        source_domain = _normalize_packet_host(
            evidence.get("source_url"),
            field_name="research packet evidence source_url",
        )
        if publisher_domain != source_domain:
            raise ValueError("research packet evidence publisher_domain does not match source_url")
        status = str(evidence.get("status") or "")
        if status == "supports":
            supported.add(objective)
            supporting_signals.append(
                {
                    "evidence_id": str(evidence.get("evidence_id") or ""),
                    "producer_origin": str(evidence.get("producer_origin") or ""),
                    "publisher_domain": publisher_domain,
                    "source_url": str(evidence.get("source_url") or ""),
                    "category": expected_category,
                    "is_primary": bool(evidence.get("is_primary")),
                    "is_company_owned": bool(evidence.get("is_company_owned")),
                }
            )
            if bool(evidence.get("is_primary")) and bool(evidence.get("is_company_owned")):
                primary_signals.append(supporting_signals[-1])
        elif status == "contradicts":
            contradicted.add(objective)
            contradiction_refs.append(str(evidence.get("evidence_id") or ""))
    objective_coverage = {
        objective: (
            "contradicted"
            if objective in contradicted
            else "stale"
            if objective in stale_objective_set
            else "supported"
            if objective in supported
            else "missing"
        )
        for objective in _RESEARCH_OBJECTIVES
    }
    missing_objectives = sorted(
        objective
        for objective in _RESEARCH_OBJECTIVES
        if objective not in supported and objective not in contradicted
    )
    independent = {
        candidate["evidence_id"]
        for anchor in primary_signals
        for candidate in supporting_signals
        if candidate["evidence_id"] != anchor["evidence_id"]
        and candidate["source_url"] != anchor["source_url"]
        and candidate["publisher_domain"] != anchor["publisher_domain"]
        and candidate["producer_origin"] != anchor["producer_origin"]
        and not _non_independent_research_host(candidate["publisher_domain"])
    }
    compound_evidence = bool(
        primary_signals
        and independent
        and len({candidate["category"] for candidate in supporting_signals}) >= 2
        and not contradiction_refs
    )
    contradiction_checked = bool(payload.get("contradiction_checked"))
    assessment = {
        "missing_objectives": missing_objectives,
        "contradicted_objectives": sorted(contradicted),
        "contradiction_refs": contradiction_refs,
        "contradiction_checked": contradiction_checked,
        "compound_evidence": compound_evidence,
        "preferred_research_ready": bool(
            contradiction_checked
            and not missing_objectives
            and compound_evidence
            and not contradiction_refs
        ),
    }
    has_verified_company_owned_identity = any(
        candidate["is_primary"]
        and candidate["is_company_owned"]
        and (
            candidate["publisher_domain"] == canonical_domain
            or candidate["publisher_domain"].endswith(f".{canonical_domain}")
        )
        for candidate in supporting_signals
    )
    analysis_ready = bool(
        contradiction_checked
        and all(objective_coverage[objective] == "supported" for objective in _RESEARCH_CORE_OBJECTIVES)
        and any(
            objective_coverage[objective] == "supported"
            for objective in _RESEARCH_FOUNDER_OBJECTIVES
        )
        and has_verified_company_owned_identity
        and not any(
            str((claim.get("evidence") or {}).get("category") or "") == "identity"
            and str((claim.get("evidence") or {}).get("status") or "") == "contradicts"
            for claim in raw_claims
            if isinstance(claim, Mapping)
        )
    )
    specter_evidence_state = str(payload.get("specter_evidence_state") or "")
    specter_refresh_required = specter_evidence_state in {
        "cached_partial",
        "fresh_authorized",
        "fresh_deferred_quota",
        "unavailable",
    }
    return {
        "objective_coverage": objective_coverage,
        "assessment": assessment,
        "analysis_ready": analysis_ready,
        "specter_refresh_required": specter_refresh_required,
    }


def _validate_research_packet_payload(
    payload: Mapping[str, Any],
    *,
    canonical_domain: str,
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        expected_fields=_RESEARCH_PACKET_FIELDS,
        field_name="research packet",
    )
    if payload.get("schema_version") != RESEARCH_PACKET_SCHEMA_VERSION:
        raise ValueError("research packet schema_version is invalid")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("research packet identity must be an object")
    _require_exact_fields(
        identity,
        expected_fields=_RESEARCH_PACKET_IDENTITY_FIELDS,
        field_name="research packet identity",
    )
    company_ref = _domain(payload.get("company_ref"))
    if company_ref != canonical_domain:
        raise ValueError("research packet company_ref does not match bundle")
    packet_domain = _domain(identity.get("domain") or identity.get("website_url"))
    if packet_domain != canonical_domain:
        raise ValueError("research packet identity does not match bundle")
    if not isinstance(payload.get("claims"), list):
        raise ValueError("research packet claims must be an array")
    if not isinstance(payload.get("contradiction_checked"), bool):
        raise ValueError("research packet contradiction_checked must be bool")
    if not isinstance(payload.get("objective_coverage"), Mapping):
        raise ValueError("research packet objective_coverage must be an object")
    if not isinstance(payload.get("stale_objectives"), list):
        raise ValueError("research packet stale_objectives must be an array")
    if not isinstance(payload.get("analysis_ready"), bool):
        raise ValueError("research packet analysis_ready must be bool")
    if not isinstance(payload.get("specter_refresh_required"), bool):
        raise ValueError("research packet specter_refresh_required must be bool")
    specter_evidence_state = payload.get("specter_evidence_state")
    if (
        not isinstance(specter_evidence_state, str)
        or specter_evidence_state not in _RESEARCH_PACKET_STATES
    ):
        raise ValueError("research packet specter_evidence_state is invalid")
    quota_authorization_id = payload.get("quota_authorization_id")
    if quota_authorization_id not in {None, ""}:
        _identifier(str(quota_authorization_id))
    if not isinstance(payload.get("assessment"), Mapping):
        raise ValueError("research packet assessment must be an object")
    packet_sha256 = _require_sha256(
        payload.get("packet_sha256"),
        field_name="research packet sha256",
    )
    content = dict(payload)
    content.pop("packet_sha256", None)
    if _sha256(content) != packet_sha256:
        raise ValueError("research packet hash does not match canonical content")
    derived = _derived_research_packet_fields(
        payload,
        canonical_domain=canonical_domain,
    )
    if dict(payload.get("objective_coverage") or {}) != derived["objective_coverage"]:
        raise ValueError("research packet objective_coverage does not match canonical content")
    if dict(payload.get("assessment") or {}) != derived["assessment"]:
        raise ValueError("research packet assessment does not match canonical content")
    if payload.get("analysis_ready") != derived["analysis_ready"]:
        raise ValueError("research packet analysis_ready does not match canonical content")
    if payload.get("specter_refresh_required") != derived["specter_refresh_required"]:
        raise ValueError(
            "research packet specter_refresh_required does not match canonical content"
        )
    return {
        "analysis_ready": payload.get("analysis_ready"),
        "specter_evidence_state": specter_evidence_state,
        "quota_authorization_id": (
            None if quota_authorization_id in {None, ""} else str(quota_authorization_id)
        ),
        "packet_sha256": packet_sha256,
    }


def _validate_research_chunk_metadata(
    chunks: list[BundleChunk],
    *,
    packet_sha256: str | None,
) -> None:
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if len(metadata) > 16:
            raise ValueError("chunk metadata exceeds bounded limits")
        source_kind = metadata.get("source_kind")
        if source_kind != "research_evidence_claim":
            continue
        if frozenset(metadata) - _RESEARCH_CLAIM_METADATA_FIELDS:
            raise ValueError("research packet chunk metadata is invalid")
        if packet_sha256 is None:
            raise ValueError("research packet chunks require a research packet payload")
        if metadata.get("packet_sha256") != packet_sha256:
            raise ValueError("research packet chunk metadata does not match bundle packet")
        if metadata.get("schema_version") != RESEARCH_PACKET_SCHEMA_VERSION:
            raise ValueError("research packet chunk schema_version is invalid")
        if not isinstance(metadata.get("objective"), str) or not metadata.get("objective"):
            raise ValueError("research packet chunk objective is required")
        if not isinstance(metadata.get("evidence_id"), str) or not metadata.get("evidence_id"):
            raise ValueError("research packet chunk evidence_id is required")
        for required_text_field in (
            "category",
            "status",
            "publisher_domain",
            "observed_at",
            "retrieved_at",
            "confidence",
        ):
            if (
                not isinstance(metadata.get(required_text_field), str)
                or not metadata.get(required_text_field)
            ):
                raise ValueError(f"research packet chunk {required_text_field} is required")
        confidence_reason_codes = metadata.get("confidence_reason_codes")
        if not isinstance(confidence_reason_codes, list):
            raise ValueError("research packet chunk confidence_reason_codes must be an array")
        if metadata.get("content_sha256") not in {None, ""}:
            _require_sha256(
                metadata.get("content_sha256"),
                field_name="research packet chunk content_sha256",
            )


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
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id")
    @classmethod
    def _chunk_identifier(cls, value: str) -> str:
        return _identifier(value)


class LeadGenAuthorizationManifest(BaseModel):
    """Immutable LeadGen authorization fields embedded in a frozen bundle."""

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
    """Validated frozen LeadGen evidence bundle accepted by the v2 intake API."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["frozen-leadgen-evidence-bundle-v1", "frozen-leadgen-evidence-bundle-v2"]
    external_company_id: str
    canonical_domain: str
    specter_company_id: str | None
    requires_specter_mcp: bool
    parent_bundle_sha256: str | None
    created_at: datetime
    analysis_ready: bool | None = None
    specter_evidence_state: str | None = None
    quota_authorization_id: str | None = None
    company: dict[str, Any]
    evidence_chunks: list[BundleChunk] = Field(max_length=10_000)
    components: list[BundleComponent] = Field(max_length=256)
    component_payloads: dict[str, Any]
    research_evidence_packet: dict[str, Any] | None = None
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

    @field_validator("quota_authorization_id")
    @classmethod
    def _quota_authorization_id(cls, value: str | None) -> str | None:
        if value in {None, ""}:
            return None
        return _identifier(value)

    @model_validator(mode="after")
    def _lineage_matches(self) -> FrozenLeadGenEvidenceBundleV1:
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
        packet_details: dict[str, Any] | None = None
        if self.schema_version == BUNDLE_SCHEMA_VERSION_V2:
            required_fields = {
                "analysis_ready",
                "specter_evidence_state",
                "quota_authorization_id",
                "research_evidence_packet",
            }
            if not required_fields.issubset(self.model_fields_set):
                raise ValueError("bundle v2 requires packet mirror fields")
            if not self.research_evidence_packet:
                raise ValueError("bundle v2 requires a research packet")
            packet_details = _validate_research_packet_payload(
                self.research_evidence_packet,
                canonical_domain=self.canonical_domain,
            )
            if (
                self.analysis_ready is not None
                and self.analysis_ready != packet_details["analysis_ready"]
            ):
                raise ValueError("bundle analysis_ready does not match research packet")
            if (
                self.specter_evidence_state is not None
                and self.specter_evidence_state
                != packet_details["specter_evidence_state"]
            ):
                raise ValueError("bundle specter evidence state does not match research packet")
            if (
                self.quota_authorization_id is not None
                and self.quota_authorization_id
                != packet_details["quota_authorization_id"]
            ):
                raise ValueError("bundle quota authorization does not match research packet")
            if self.analysis_ready and self.requires_specter_mcp:
                raise ValueError("analysis-ready bundle-complete packets must not require Specter")
        _validate_research_chunk_metadata(
            self.evidence_chunks,
            packet_sha256=(
                None if packet_details is None else packet_details["packet_sha256"]
            ),
        )
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Return the persisted canonical payload without introducing hash drift."""
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


class MachineV2SpecterReservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_environment: Literal["staging", "production"]
    business_timezone: Literal["Europe/Prague"]
    consumer: str
    company_ref: str | None = None
    operation: str
    quota_class: Literal[
        "flex",
        "flexible_pool",
        "manual_batch",
        "promoted_candidate_refresh",
        "scheduled_import",
        "recovery_probe",
        "autonomous_campaign",
    ]
    idempotency_key: str
    remaining_rdi_slots: int | None = Field(default=None, ge=0, le=20)
    intake_id: str | None = None

    @field_validator("consumer", "operation", "idempotency_key")
    @classmethod
    def _safe_identifier(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("company_ref")
    @classmethod
    def _safe_company_ref(cls, value: str | None) -> str | None:
        if value in {None, ""}:
            return None
        if (
            not isinstance(value, str)
            or len(value) > 255
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("must be bounded clean text")
        return value

    @field_validator("intake_id")
    @classmethod
    def _safe_intake_id(cls, value: str | None) -> str | None:
        if value in {None, ""}:
            return None
        if not _INTAKE_RE.fullmatch(value):
            raise ValueError("must be a machine v2 intake id")
        return value

    @model_validator(mode="after")
    def _require_scope_anchor(self) -> MachineV2SpecterReservationRequest:
        if self.intake_id is None and self.company_ref is None:
            raise ValueError("company_ref or intake_id is required")
        return self


class MachineV2SpecterFinalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_environment: Literal["staging", "production"]
    operation: str
    outcome: Literal["succeeded", "failed", "released"]
    provider_quota_error: bool = False
    reason_code: str | None = None
    intake_id: str | None = None

    @field_validator("operation")
    @classmethod
    def _operation(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("reason_code")
    @classmethod
    def _reason_code(cls, value: str | None) -> str | None:
        if value in {None, ""}:
            return None
        return _identifier(value)

    @field_validator("intake_id")
    @classmethod
    def _finalize_intake_id(cls, value: str | None) -> str | None:
        if value in {None, ""}:
            return None
        if not _INTAKE_RE.fullmatch(value):
            raise ValueError("must be a machine v2 intake id")
        return value


class MachineV2Store(Protocol):
    def is_configured(self) -> bool: ...

    def put_machine_v2_evidence_bundle(self, **record: Any) -> dict[str, Any] | None: ...

    def create_machine_v2_intake(self, **record: Any) -> dict[str, Any] | None: ...

    def reserve_machine_v2_start(self, **record: Any) -> dict[str, Any] | None: ...

    def finalize_machine_v2_start(self, **record: Any) -> dict[str, Any] | None: ...

    def release_machine_v2_start(self, **record: Any) -> dict[str, Any] | None: ...

    def load_machine_v2_lifecycle(self, intake_id: str) -> dict[str, Any] | None: ...

    def reserve_specter_quota_authorization(self, **record: Any) -> dict[str, Any] | None: ...

    def commit_specter_quota_authorization(self, **record: Any) -> dict[str, Any] | None: ...

    def release_specter_quota_authorization(self, **record: Any) -> dict[str, Any] | None: ...


StartAdapter = Callable[
    [str, list[dict[str, str] | str], dict[str, Any], dict[str, str | None]],
    Awaitable[MachineStartAccepted | MachineStartDefiniteRejection | Any],
]


class MachineV2Dependencies:
    """Runtime collaborators required by the LeadGen machine v2 router."""

    def __init__(self, *, store: MachineV2Store | None, start_adapter: StartAdapter) -> None:
        """Store the persistence adapter and delegated start implementation."""
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
    """Build the authenticated LeadGen machine v2 API surface."""
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
            schema_version=request.schema_version,
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

    @router.post("/api/machine/leadgen/v2/specter/reservations", status_code=201)
    async def reserve_specter_quota(
        request: MachineV2SpecterReservationRequest,
    ) -> dict[str, Any]:
        _require_v2_enabled()
        target_environment = _target_environment(request.target_environment)
        if request.business_timezone != BROKER_TIMEZONE:
            raise _problem(409, "machine_v2_broker_timezone_invalid", "Machine v2 broker timezone is invalid.")
        store = _require_store(dependencies)
        company_ref = request.company_ref
        metadata: dict[str, Any] = {}
        if request.intake_id is not None:
            lifecycle = store.load_machine_v2_lifecycle(request.intake_id)
            if lifecycle is None:
                raise _problem(404, "machine_v2_intake_not_found", "Machine v2 intake was not found.")
            if lifecycle.get("target_environment") != target_environment:
                raise _problem(409, "machine_v2_intake_environment_mismatch", "Machine v2 intake environment does not match.")
            metadata["intake_id"] = request.intake_id
            if company_ref is None:
                company_ref = f"domain:{lifecycle['canonical_domain']}"
        result = store.reserve_specter_quota_authorization(
            target_environment=target_environment,
            business_date=specter_quota_business_date(),
            business_timezone=request.business_timezone,
            consumer=request.consumer,
            company_ref=company_ref,
            operation=request.operation,
            quota_class=request.quota_class,
            idempotency_key=request.idempotency_key,
            enforcement_enabled=specter_quota_broker_enforced(),
            remaining_rdi_slots=request.remaining_rdi_slots,
            actor=SERVICE_ACTOR,
            metadata=metadata,
        )
        if result is None:
            raise _problem(503, "machine_v2_broker_reservation_failed", "Specter quota reservation could not be persisted.")
        return {
            "contract_version": CONTRACT_VERSION,
            **public_specter_quota_authorization(result),
        }

    @router.post("/api/machine/leadgen/v2/specter/reservations/{authorization_id}/commit")
    async def commit_specter_quota(
        authorization_id: str,
        request: MachineV2SpecterFinalizeRequest,
    ) -> dict[str, Any]:
        _require_v2_enabled()
        if request.outcome == "released":
            raise _problem(409, "machine_v2_broker_finalize_invalid", "Specter quota commit outcome is invalid.")
        target_environment = _target_environment(request.target_environment)
        result = _require_store(dependencies).commit_specter_quota_authorization(
            authorization_id=authorization_id,
            target_environment=target_environment,
            operation=request.operation,
            outcome=request.outcome,
            provider_quota_error=bool(request.provider_quota_error),
            reason_code=request.reason_code,
            intake_id=request.intake_id,
            actor=SERVICE_ACTOR,
        )
        if result is None:
            raise _problem(503, "machine_v2_broker_commit_failed", "Specter quota commit could not be persisted.")
        return {
            "contract_version": CONTRACT_VERSION,
            **public_specter_quota_authorization(result),
        }

    @router.post("/api/machine/leadgen/v2/specter/reservations/{authorization_id}/release")
    async def release_specter_quota(
        authorization_id: str,
        request: MachineV2SpecterFinalizeRequest,
    ) -> dict[str, Any]:
        _require_v2_enabled()
        if request.outcome != "released":
            raise _problem(409, "machine_v2_broker_finalize_invalid", "Specter quota release outcome is invalid.")
        target_environment = _target_environment(request.target_environment)
        result = _require_store(dependencies).release_specter_quota_authorization(
            authorization_id=authorization_id,
            target_environment=target_environment,
            operation=request.operation,
            intake_id=request.intake_id,
            reason_code=request.reason_code,
            actor=SERVICE_ACTOR,
        )
        if result is None:
            raise _problem(503, "machine_v2_broker_release_failed", "Specter quota release could not be persisted.")
        return {
            "contract_version": CONTRACT_VERSION,
            **public_specter_quota_authorization(result),
        }

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
