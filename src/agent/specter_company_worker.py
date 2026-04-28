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

from agent.batch import evaluate_from_specter
from agent.ingest.specter_ingest import ingest_specter_company
from agent.ingest.specter_mcp_client import (
    SpecterDisambiguationError,
    SpecterMCPError,
    fetch_specter_company,
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
import web.db as db
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


async def _process_company(args: argparse.Namespace) -> int:
    payload = _read_json(args.config_path)
    run_config = payload.get("run_config") or {}
    versions = payload.get("versions") or {}
    job_id = args.job_id
    use_web_search = bool(args.use_web_search)
    vc_investment_strategy = args.vc_investment_strategy or run_config.get("vc_investment_strategy")
    llm_selection = _llm_selection_from_run_config(run_config)
    pipeline_policy = _pipeline_policy_from_run_config(run_config)

    _init_worker_cache(job_id, run_config, versions)

    if args.specter_url:
        company, store = fetch_specter_company(
            args.specter_url,
            expected_name=args.expected_name or None,
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
        ):
            result = await evaluate_from_specter(
                company,
                store,
                k=8,
                use_web_search=use_web_search,
                on_progress=_on_progress,
                vc_investment_strategy=vc_investment_strategy,
            )

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
    parser.add_argument("--absolute-index", type=int, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--vc-investment-strategy")
    parser.add_argument("--use-web-search", action="store_true")
    args = parser.parse_args()

    return asyncio.run(_process_company(args))


if __name__ == "__main__":
    sys.exit(main())
