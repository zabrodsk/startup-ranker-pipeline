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
from pathlib import Path
from typing import Any

import web.db as db
from agent.batch import evaluate_from_specter
from agent.dataclasses.company import Company
from agent.ingest.specter_ingest import _company_slug, ingest_specter_company
from agent.ingest.specter_mcp_client import (
    SPECTER_MCP_QUOTA_ERROR_CODE,
    SpecterQuotaLimitError,
    fetch_specter_company,
    specter_quota_reset_hint,
)
from agent.ingest.store import EvidenceStore
from agent.ingest.web_fallback import (
    fetch_company_homepage,
    quota_web_fallback_allowed,
)
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

EVENT_PREFIX = "__SPECTER_COMPANY_EVENT__"


def _emit_event(payload: dict[str, Any]) -> None:
    print(f"{EVENT_PREFIX}{json.dumps(payload, ensure_ascii=True)}", flush=True)


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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
    error_message = f"{type(exc).__name__}: {exc}"[:1000]
    name = (
        args.expected_name
        or args.specter_url
        or (f"company #{args.company_index}" if args.company_index is not None else "Unknown")
    ).strip()
    slug = _company_slug(name) or f"company-{args.absolute_index}"
    company = Company(name=name)
    store = EvidenceStore(startup_slug=slug, chunks=[])
    status = "error"
    quota_exhausted = isinstance(exc, SpecterQuotaLimitError)
    reset_hint = getattr(exc, "reset_hint", None) or specter_quota_reset_hint(error_message)
    try:
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
        if quota_exhausted:
            failure_payload["error_code"] = SPECTER_MCP_QUOTA_ERROR_CODE
            failure_payload["quota_remaining"] = "unknown"
            result_row["error_code"] = SPECTER_MCP_QUOTA_ERROR_CODE
            result_row["quota_remaining"] = "unknown"
            if reset_hint:
                failure_payload["reset_hint"] = reset_hint
                result_row["reset_hint"] = reset_hint
        db.persist_company_failure_result(
            job_id_legacy=job_id,
            result_row=result_row,
            company_payload=failure_payload,
            run_config=run_config,
            versions=versions,
        )
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
    if quota_exhausted:
        event["quota_exhausted"] = True
        event["error_code"] = SPECTER_MCP_QUOTA_ERROR_CODE
        event["quota_remaining"] = "unknown"
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
    used_quota_web_fallback = False
    try:
        if args.specter_url:
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
    except SpecterQuotaLimitError as exc:
        if not quota_web_fallback_allowed(
            run_config,
            use_web_search=use_web_search,
            specter_url=args.specter_url,
            expected_name=args.expected_name,
        ):
            traceback.print_exc()
            return _handle_fetch_failure(args, job_id, run_config, versions, exc)
        try:
            company, store = fetch_company_homepage(
                args.specter_url,
                expected_name=args.expected_name,
            )
            used_quota_web_fallback = True
        except Exception:
            traceback.print_exc()
            return _handle_fetch_failure(args, job_id, run_config, versions, exc)
    except Exception as exc:
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

    if used_quota_web_fallback:
        _on_progress(
            "Specter daily quota is exhausted; continuing with identity-checked "
            "company-homepage bootstrap and web evidence search."
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
