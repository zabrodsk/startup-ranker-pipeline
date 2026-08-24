"""Child-process worker for a single Specter company.

The parent process stays lightweight and delegates one company at a time to a
fresh subprocess so the OS can reclaim analysis memory after every company.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from hashlib import sha256
from pathlib import Path
from typing import Any

import web.db as db
from agent.batch import evaluate_from_specter
from agent.dataclasses.company import Company
from agent.ingest.specter_ingest import _company_slug, ingest_specter_company
from agent.ingest.specter_mcp_client import (
    SPECTER_MCP_QUOTA_ERROR_CODE,
    SPECTER_MCP_UNAVAILABLE_ERROR_CODE,
    SpecterCompanyNotFoundError,
    SpecterDisambiguationError,
    SpecterMCPError,
    SpecterQuotaLimitError,
    fetch_specter_company,
    specter_quota_reset_hint,
)
from agent.ingest.store import Chunk, EvidenceStore
from agent.llm_catalog import serialize_selection
from agent.llm_policy import (
    build_phase_model_policy,
    build_pipeline_policy,
    normalize_phase_models,
    normalize_premium_phase_models,
    normalize_quality_tier,
)
from agent.run_context import RunTelemetryCollector, use_run_context
from web import app as web_app
from web.leadgen_domain import normalize_company_domain
from web.leadgen_machine_v2 import (
    FrozenLeadGenEvidenceBundleV1,
    canonical_bundle_sha256,
)
from web.specter_quota_broker import (
    SpecterQuotaBrokerRequest,
    build_specter_quota_broker_request,
    use_specter_quota_broker,
)
from web.specter_quota_gate import SpecterQuotaGateUnavailable, trip_specter_quota_gate

EVENT_PREFIX = "__SPECTER_COMPANY_EVENT__"


def _emit_event(payload: dict[str, Any]) -> None:
    print(f"{EVENT_PREFIX}{json.dumps(payload, ensure_ascii=True)}", flush=True)


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_frozen_leadgen_bundle(
    run_config: dict[str, Any],
    *,
    expected_domain: str | None,
) -> tuple[Company, EvidenceStore] | None:
    """Reconstruct the normal RDI inputs without a new Specter request."""
    context = run_config.get("leadgen_machine_v2")
    if not isinstance(context, dict) or context.get("requires_specter_mcp") is not False:
        return None
    bundle_sha256 = str(context.get("evidence_bundle_sha256") or "")
    record = db.load_machine_v2_evidence_bundle(bundle_sha256)
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict):
        raise ValueError("Frozen LeadGen evidence bundle is unavailable")
    if canonical_bundle_sha256(payload) != bundle_sha256:
        raise ValueError("Frozen LeadGen evidence bundle hash mismatch")
    bundle = FrozenLeadGenEvidenceBundleV1.model_validate(payload)
    requested_domain = normalize_company_domain(expected_domain or "")
    if requested_domain and requested_domain != bundle.canonical_domain:
        raise ValueError("Frozen LeadGen evidence bundle domain mismatch")
    company = Company.model_validate(bundle.company)
    bundle_chunks = [Chunk(**chunk.model_dump()) for chunk in bundle.evidence_chunks]
    packet_chunks = _packet_claim_chunks(bundle)
    packet_lineage_keys = {
        key
        for chunk in packet_chunks
        for key in _chunk_lineage_keys(chunk)
    }
    store = EvidenceStore(
        startup_slug=_company_slug(company.name),
        chunks=[
            *packet_chunks,
            *[
                chunk
                for chunk in bundle_chunks
                if not packet_lineage_keys.intersection(_chunk_lineage_keys(chunk))
            ],
        ],
    )
    return company, store


def _packet_claim_chunks(bundle: FrozenLeadGenEvidenceBundleV1) -> list[Chunk]:
    """Materialize canonical research-packet claims as additive evidence chunks."""
    packet = bundle.research_evidence_packet
    if not isinstance(packet, dict):
        return []

    claims = packet.get("claims")
    if not isinstance(claims, list):
        return []

    stale_objectives = {
        str(item)
        for item in (packet.get("stale_objectives") or [])
        if isinstance(item, str)
    }
    packet_sha256 = str(packet.get("packet_sha256") or "")
    chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        objective = str(claim.get("objective") or "").strip()
        evidence = claim.get("evidence")
        if not objective or not isinstance(evidence, dict):
            continue
        text = str(evidence.get("claim") or "").strip()
        if not text:
            continue
        raw_evidence_id = str(evidence.get("evidence_id") or "").strip()
        chunk_id = _packet_claim_chunk_id(raw_evidence_id, objective, index)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        metadata = {
            "lineage_source": "leadgen_research_packet",
            "schema_version": packet.get("schema_version"),
            "packet_sha256": packet_sha256 or None,
            "objective": objective,
            "evidence_id": raw_evidence_id or None,
            "category": evidence.get("category"),
            "status": evidence.get("status"),
            "source_url": evidence.get("source_url"),
            "publisher_domain": evidence.get("publisher_domain"),
            "producer_origin": evidence.get("producer_origin"),
            "source_family": evidence.get("source_family"),
            "observed_at": evidence.get("observed_at"),
            "published_at": evidence.get("published_at"),
            "retrieved_at": evidence.get("retrieved_at"),
            "confidence": evidence.get("confidence"),
            "confidence_reason_codes": evidence.get("confidence_reason_codes") or [],
            "content_sha256": evidence.get("content_sha256"),
            "provenance_ref": evidence.get("provenance_ref"),
            "subject_company_ref": evidence.get("subject_company_ref"),
            "is_primary": bool(evidence.get("is_primary")),
            "is_company_owned": bool(evidence.get("is_company_owned")),
            "stale_objective": objective in stale_objectives,
        }
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                source_file=str(
                    evidence.get("source_url") or "leadgen:research_packet"
                ),
                page_or_slide=objective,
                metadata={
                    key: value for key, value in metadata.items() if value is not None
                },
            )
        )
    return chunks


def _chunk_lineage_keys(chunk: Chunk) -> set[tuple[str, ...]]:
    metadata = dict(chunk.metadata or {})
    keys: set[tuple[str, ...]] = set()
    evidence_id = str(metadata.get("evidence_id") or "").strip()
    content_sha256 = str(metadata.get("content_sha256") or "").strip()
    if evidence_id:
        keys.add(("evidence_id", evidence_id))
    if content_sha256:
        keys.add(("content_sha256", content_sha256))
    if not keys:
        keys.add(("claim", str(chunk.source_file).strip(), str(chunk.text).strip()))
    return keys


def _packet_claim_chunk_id(
    evidence_id: str,
    objective: str,
    index: int,
) -> str:
    token = evidence_id
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_", ".", ":"} else "-"
        for character in token.lower()
    ).strip("-")
    if normalized:
        return f"leadgen-packet:{normalized[:90]}"
    digest = sha256(f"{objective}:{index}".encode()).hexdigest()[:16]
    return f"leadgen-packet:{digest}"


def _pipeline_policy_from_run_config(run_config: dict[str, Any]) -> Any | None:
    phase_models = normalize_phase_models(run_config.get("phase_models"))
    if run_config.get("phase_models"):
        return build_phase_model_policy(phase_models)

    quality_tier = normalize_quality_tier(run_config.get("quality_tier"))
    if quality_tier:
        premium_phase_models = normalize_premium_phase_models(
            run_config.get("premium_phase_models")
        )
        return build_pipeline_policy(
            quality_tier,
            premium_phase_models if quality_tier == "premium" else None,
        )
    return None


def _llm_selection_from_run_config(run_config: dict[str, Any]) -> dict[str, str]:
    return serialize_selection(
        run_config.get("llm_provider"),
        run_config.get("llm_model"),
    )


def _specter_broker_request_from_run_config(
    run_config: dict[str, Any],
) -> SpecterQuotaBrokerRequest | None:
    context = run_config.get("leadgen_machine_v2")
    if not isinstance(context, dict) or context.get("requires_specter_mcp") is not True:
        return None
    canonical_domain = normalize_company_domain(str(context.get("canonical_domain") or ""))
    if not canonical_domain:
        return None
    remaining_slots = context.get("daily_remaining_capacity")
    try:
        normalized_remaining_slots = max(int(remaining_slots), 0)
    except (TypeError, ValueError):
        normalized_remaining_slots = None
    return build_specter_quota_broker_request(
        consumer="rdi",
        operation=None,
        quota_class="autonomous_campaign",
        company_ref=f"domain:{canonical_domain}",
        remaining_rdi_slots=normalized_remaining_slots,
        intake_id=str(context.get("intake_id") or "") or None,
        metadata={"source": "leadgen_machine_v2"},
    )


def _init_worker_cache(job_id: str, run_config: dict[str, Any], versions: dict[str, Any]) -> None:
    llm_selection = _llm_selection_from_run_config(run_config)
    web_app._results_cache[job_id] = {
        "input_mode": run_config.get("input_mode", "specter"),
        "vc_investment_strategy": run_config.get("vc_investment_strategy"),
        "instructions": run_config.get("instructions"),
        "use_web_search": run_config.get("use_web_search", False),
        "llm_selection": llm_selection,
        "phase_models": run_config.get("phase_models"),
        "quality_tier": run_config.get("quality_tier"),
        "premium_phase_models": run_config.get("premium_phase_models"),
        "effective_phase_models": run_config.get("effective_phase_models"),
        "run_config": dict(run_config),
        "model_executions": [],
        "versions": dict(versions or {}),
        "run_costs_aggregate": web_app._empty_run_costs_summary(),
        "files": [],
    }


def _persist_company_telemetry(
    job_id: str,
    collector: RunTelemetryCollector,
    run_config: dict[str, Any],
    versions: dict[str, Any],
) -> None:
    rows = collector.snapshot_model_executions()
    if rows:
        db.persist_model_executions(
            job_id,
            rows,
            run_config=run_config,
            versions=versions,
        )


def _handle_fetch_failure(
    args: argparse.Namespace,
    job_id: str,
    run_config: dict[str, Any],
    versions: dict[str, Any],
    exc: Exception,
) -> int:
    """Persist + emit a structured per-company failure for fetch/ingest crashes.

    Company identity is synthesized from the task args since no Company or
    EvidenceStore exists yet. Emits the same company_complete{status:"error"}
    event the evaluation except-path uses, so the parent counts the failure
    and the UI shows the specific message instead of a generic exit code.
    """
    quota_exhausted = isinstance(exc, SpecterQuotaLimitError)
    provider_unavailable = isinstance(exc, SpecterMCPError) and not isinstance(
        exc,
        (SpecterCompanyNotFoundError, SpecterDisambiguationError),
    )
    provider_blocked = quota_exhausted or provider_unavailable
    provider_error_code = (
        SPECTER_MCP_QUOTA_ERROR_CODE
        if quota_exhausted
        else SPECTER_MCP_UNAVAILABLE_ERROR_CODE
    )
    error_message = (
        "Specter MCP quota exhausted."
        if quota_exhausted
        else "Specter MCP is temporarily unavailable."
        if provider_unavailable
        else f"{type(exc).__name__}: {exc}"[:1000]
    )
    name = (
        args.expected_name
        or args.specter_url
        or (f"company #{args.company_index}" if args.company_index is not None else "Unknown")
    ).strip()
    slug = _company_slug(name) or f"company-{args.absolute_index}"
    company = Company(name=name)
    store = EvidenceStore(startup_slug=slug, chunks=[])
    status = "blocked" if provider_blocked else "error"
    reset_hint = getattr(exc, "reset_hint", None) or specter_quota_reset_hint(error_message)
    try:
        if provider_blocked:
            trip_specter_quota_gate(
                db,
                error=exc,
                source_component="specter_company_worker",
                source_job_id=job_id,
                retry_after_seconds=None if quota_exhausted else 300,
                reason_code=provider_error_code,
            )
            db.insert_analysis_event(
                job_id,
                message=(
                    "Specter MCP quota exhausted; analysis is waiting for provider reset."
                    if quota_exhausted
                    else "Specter MCP is temporarily unavailable; analysis is waiting for provider recovery."
                ),
                event_type=(
                    "specter_mcp_quota_blocked"
                    if quota_exhausted
                    else "specter_mcp_provider_blocked"
                ),
                stage="specter_company_worker.fetch",
                payload={
                    "error_code": provider_error_code,
                    "quota_remaining": "unknown",
                    "reset_hint": reset_hint,
                },
            )
        else:
            db.insert_analysis_error(
                job_id,
                message=error_message,
                stage="specter_company_worker.fetch",
                error_type=type(exc).__name__,
                company_slug=slug,
            )
            failure_payload = web_app._failure_result_payload(
                job_id,
                company=company,
                store=store,
                slug=slug,
                status=status,
                error_message=error_message,
            )
            result_row = {
                "slug": slug,
                "company": company,
                "company_name": company.name,
                "evidence_store": store,
                "final_state": {
                    "final_arguments": [],
                    "final_decision": status,
                    "ranking_result": None,
                    "all_qa_pairs": [],
                },
                "analysis_status": status,
                "error": error_message,
                "skipped": False,
            }
            db.persist_company_failure_result(
                job_id_legacy=job_id,
                result_row=result_row,
                company_payload=failure_payload,
                run_config=run_config,
                versions=versions,
            )
    except SpecterQuotaGateUnavailable:
        print("Specter MCP quota gate persistence unavailable", file=sys.stderr)
    except Exception:
        # Persistence is best-effort; the structured event below still reaches
        # the parent so the failure is counted and surfaced.
        traceback.print_exc()
    event = {
        "type": "company_complete",
        "company_name": name,
        "absolute_index": args.absolute_index,
        "status": status,
        "error": error_message[:500],
        "error_type": type(exc).__name__,
    }
    if provider_blocked:
        event["provider_blocked"] = True
        event["error_code"] = provider_error_code
        event["quota_remaining"] = "unknown"
        if quota_exhausted:
            event["quota_exhausted"] = True
        if reset_hint:
            event["reset_hint"] = reset_hint
    _emit_event(event)
    return 0


async def _process_company(args: argparse.Namespace) -> int:
    payload = _read_json(args.config_path)
    run_config = payload.get("run_config") or {}
    versions = payload.get("versions") or {}
    job_id = args.job_id
    use_web_search = bool(args.use_web_search)
    # Per-run web-search intensity (Off/Targeted/Full) rides in run_config; carry
    # it on the run context so the answering hook can gate Targeted vs Full.
    web_search_mode = run_config.get("web_search_mode")
    vc_investment_strategy = args.vc_investment_strategy or run_config.get("vc_investment_strategy")
    llm_selection = _llm_selection_from_run_config(run_config)
    pipeline_policy = _pipeline_policy_from_run_config(run_config)

    _init_worker_cache(job_id, run_config, versions)

    # fetch/ingest run before any Company/EvidenceStore exists. Without this
    # guard a failure here (e.g. Specter auth outage) kills the child with a
    # bare non-zero exit: no structured event, no analysis_errors row, and the
    # UI only shows a generic "exited with code 1" (the 2026-06-11 outage UX).
    try:
        broker_request = _specter_broker_request_from_run_config(run_config)
        frozen_inputs = _load_frozen_leadgen_bundle(
            run_config,
            expected_domain=args.specter_url,
        )
        if frozen_inputs is not None:
            company, store = frozen_inputs
        elif args.specter_url:
            with use_specter_quota_broker(broker_request):
                company, store = fetch_specter_company(
                    args.specter_url,
                    expected_name=args.expected_name or None,
                    fetch_full_team=bool(args.fetch_full_team),
                    known_company_id=args.specter_company_id or None,
                )
        else:
            if not args.specter_companies or args.company_index is None:
                raise ValueError(
                    "specter_company_worker requires either --specter-url or "
                    "--specter-companies + --company-index"
                )
            company, store = ingest_specter_company(
                args.specter_companies,
                args.specter_people,
                company_index=args.company_index,
            )
    except Exception as exc:
        if isinstance(exc, SpecterQuotaLimitError):
            print("Specter MCP quota exhausted during company fetch", file=sys.stderr)
        else:
            traceback.print_exc()
        return _handle_fetch_failure(args, job_id, run_config, versions, exc)
    collector = RunTelemetryCollector(selected_llm=llm_selection)
    web_app._results_cache[job_id]["telemetry_collector"] = collector

    def _on_progress(message: str) -> None:
        _emit_event(
            {
                "type": "progress",
                "company_name": company.name,
                "absolute_index": args.absolute_index,
                "message": message,
            }
        )

    try:
        with use_run_context(
            llm_selection=llm_selection,
            telemetry_collector=collector,
            pipeline_policy=pipeline_policy,
            web_search_mode=web_search_mode,
        ):
            result = await evaluate_from_specter(
                company,
                store,
                k=8,
                use_web_search=use_web_search,
                on_progress=_on_progress,
                vc_investment_strategy=vc_investment_strategy,
            )

        # Duplicate-run gate (Sprint 3): persist the evidence fingerprint the
        # parent computed at dispatch time so future runs can match this row.
        if getattr(args, "evidence_fingerprint", None):
            result["evidence_fingerprint"] = args.evidence_fingerprint

        company_payload = web_app._single_result_payload(job_id, result)
        persisted = db.persist_company_result(
            job_id_legacy=job_id,
            result_row=result,
            company_payload=company_payload,
            run_config=run_config,
            versions=versions,
        )
        _persist_company_telemetry(job_id, collector, run_config, versions)
        if not persisted:
            raise RuntimeError(f"Failed to persist company result for {company.name}")

        _emit_event(
            {
                "type": "company_complete",
                "company_name": company.name,
                "absolute_index": args.absolute_index,
                "status": "done",
            }
        )
    except Exception as exc:
        traceback.print_exc()
        error_message = str(exc)[:1000]
        status = "timeout" if isinstance(exc, TimeoutError) else "error"
        db.insert_analysis_error(
            job_id,
            message=error_message,
            stage="specter_company_worker",
            error_type=type(exc).__name__,
            company_slug=store.startup_slug,
        )
        failure_payload = web_app._failure_result_payload(
            job_id,
            company=company,
            store=store,
            slug=store.startup_slug,
            status=status,
            error_message=error_message,
        )
        db.persist_company_failure_result(
            job_id_legacy=job_id,
            result_row={
                "slug": store.startup_slug,
                "company": company,
                "company_name": company.name,
                "evidence_store": store,
                "final_state": {
                    "final_arguments": [],
                    "final_decision": status,
                    "ranking_result": None,
                    "all_qa_pairs": [],
                },
                "analysis_status": status,
                "error": error_message,
                "skipped": False,
            },
            company_payload=failure_payload,
            run_config=run_config,
            versions=versions,
        )
        _persist_company_telemetry(job_id, collector, run_config, versions)
        _emit_event(
            {
                "type": "company_complete",
                "company_name": company.name,
                "absolute_index": args.absolute_index,
                "status": status,
                "error": error_message[:500],
            }
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--specter-companies")
    parser.add_argument("--specter-people")
    parser.add_argument("--company-index", type=int)
    parser.add_argument("--specter-url", help="URL or domain for MCP-based intake")
    parser.add_argument(
        "--expected-name",
        help="Expected company name; used to verify Specter MCP disambiguation",
    )
    parser.add_argument(
        "--specter-company-id",
        help=(
            "Specter company id resolved by the web preflight (Sprint 3 W7). "
            "Skips find_company + match verification; stale ids fall back to "
            "full resolution inside fetch_specter_company."
        ),
    )
    parser.add_argument(
        "--evidence-fingerprint",
        help=(
            "Evidence fingerprint computed by the parent at dispatch time "
            "(Sprint 3 dup-run gate); persisted on the company_runs row."
        ),
    )
    parser.add_argument("--absolute-index", type=int, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--vc-investment-strategy")
    parser.add_argument("--use-web-search", action="store_true")
    parser.add_argument(
        "--fetch-full-team",
        action="store_true",
        help=(
            "When set, fan out to get_person_profile per founder/key person "
            "for full LinkedIn-grade career history. Adds ~60%% more MCP "
            "calls per company. Only affects URL-based intake."
        ),
    )
    args = parser.parse_args()

    return asyncio.run(_process_company(args))


if __name__ == "__main__":
    sys.exit(main())
