"""Supabase persistence for Rockaway Deal Intelligence analyses and job telemetry."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_client: Client | None = None
_client_config: tuple[str, str] | None = None
BUCKET_EXCEL = "analysis-exports"


def _normalize_company_key(name: str | None, domain: str | None = None, slug: str | None = None) -> str:
    base = (domain or "").strip().lower()
    if base:
        base = re.sub(r"^https?://", "", base)
        base = re.sub(r"^www\.", "", base)
        base = base.strip("/")
        if base:
            return f"domain:{base}"

    slug_base = (slug or "").strip().lower()
    if slug_base:
        return f"slug:{slug_base}"

    text = (name or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return f"name:{text or 'unknown'}"


def _normalize_text_token(value: str | None) -> str:
    text = (value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _strip_legacy_company_key_suffix(company_key: str | None) -> str:
    return re.sub(r"--legacy-\d+$", "", (company_key or "").strip().lower())


def _company_history_group_key(row: dict[str, Any]) -> str:
    result_payload = _serialize(row.get("result_payload") or {})
    summary_row = ((result_payload.get("summary_rows") or [{}]) or [{}])[0]
    company_name = (
        row.get("company_name")
        or result_payload.get("company_name")
        or summary_row.get("company_name")
    )
    if company_name:
        return f"name:{_normalize_text_token(company_name)}"

    startup_slug = (
        row.get("startup_slug")
        or result_payload.get("startup_slug")
        or summary_row.get("startup_slug")
    )
    if startup_slug:
        return f"slug:{_normalize_text_token(startup_slug)}"

    company_key = _strip_legacy_company_key_suffix(row.get("company_key"))
    return company_key or "name:unknown"


def _company_payload_from_result(
    full_payload: dict[str, Any],
    result_row: dict[str, Any],
) -> dict[str, Any]:
    payload = _serialize(full_payload or {})
    if payload.get("mode") == "single":
        return payload

    slug = result_row.get("slug") or result_row.get("startup_slug") or ""
    company = result_row.get("company")
    company_name = getattr(company, "name", None) or result_row.get("company_name") or slug

    summary_rows = payload.get("summary_rows") or []
    summary_row = next(
        (
            row for row in summary_rows
            if (row.get("startup_slug") or "") == slug
            or str(row.get("company_name") or "").strip().lower() == str(company_name).strip().lower()
        ),
        {},
    )

    argument_rows = [
        row for row in (payload.get("argument_rows") or [])
        if (row.get("startup_slug") or "") == slug
        or str(row.get("company_name") or "").strip().lower() == str(company_name).strip().lower()
    ]
    qa_rows = [
        row for row in (payload.get("qa_provenance_rows") or [])
        if (row.get("startup_slug") or "") == slug
        or str(row.get("company_name") or "").strip().lower() == str(company_name).strip().lower()
    ]
    failed_rows = [
        row for row in (payload.get("failed_rows") or [])
        if (row.get("startup_slug") or "") == slug
        or str(row.get("company_name") or "").strip().lower() == str(company_name).strip().lower()
    ]

    founders = (payload.get("founders_by_slug") or {}).get(slug) or summary_row.get("founders") or []
    team_members = (payload.get("team_members_by_slug") or {}).get(slug) or summary_row.get("team_members") or founders

    return {
        "mode": "single",
        "source_mode": payload.get("mode") or "batch",
        "startup_slug": slug,
        "company_name": company_name,
        "industry": getattr(company, "industry", None) or summary_row.get("industry") or "",
        "tagline": getattr(company, "tagline", None) or "",
        "about": getattr(company, "about", None) or "",
        "decision": summary_row.get("decision") or result_row.get("decision") or "unknown",
        "total_score": summary_row.get("total_score"),
        "avg_pro": summary_row.get("avg_pro"),
        "avg_contra": summary_row.get("avg_contra"),
        "summary_rows": [summary_row] if summary_row else [],
        "argument_rows": argument_rows,
        "qa_provenance_rows": qa_rows,
        "failed_rows": failed_rows,
        "founders": founders,
        "team_members": team_members,
        "ranking_result": {
            "rank": summary_row.get("rank"),
            "percentile": summary_row.get("percentile"),
            "composite_score": summary_row.get("composite_score"),
            "strategy_fit_score": summary_row.get("strategy_fit_score"),
            "team_score": summary_row.get("team_score"),
            "upside_score": summary_row.get("upside_score"),
            "bucket": summary_row.get("bucket"),
            "strategy_fit_summary": summary_row.get("strategy_fit_summary"),
            "team_summary": summary_row.get("team_summary"),
            "potential_summary": summary_row.get("potential_summary"),
            "key_points": summary_row.get("key_points"),
            "red_flags": summary_row.get("red_flags"),
        } if summary_row else None,
    }


def _upsert_company(
    client: Client,
    *,
    name: str,
    industry: str | None,
    tagline: str | None,
    about: str | None,
    team: list[Any] | None,
    domain: str | None,
    company_key: str,
) -> str | None:
    payload = {
        "name": name,
        "industry": industry,
        "tagline": tagline,
        "about": about,
        "team": _serialize(team or []),
        "domain": domain,
        "company_key": company_key,
    }
    try:
        row = client.table("companies").upsert(payload, on_conflict="company_key").execute()
        if row.data:
            return row.data[0].get("id")
    except Exception:
        pass

    try:
        row = (
            client.table("companies")
            .select("id")
            .eq("company_key", company_key)
            .limit(1)
            .execute()
        )
        if row.data:
            return row.data[0].get("id")
    except Exception:
        pass
    return None


def _get_supabase_config() -> tuple[str, str]:
    return (
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


def _get_client() -> Client | None:
    global _client, _client_config
    config = _get_supabase_config()
    if not all(config):
        _client = None
        _client_config = None
        return None

    if _client is None or _client_config != config:
        _client = create_client(*config)
        _client_config = config

    return _client


def is_configured() -> bool:
    return all(_get_supabase_config())


def _serialize(obj: Any) -> Any:
    """Convert Pydantic/objects to JSON-serializable dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _get_job_uuid(client: Client, job_id_legacy: str) -> str | None:
    try:
        row = (
            client.table("jobs")
            .select("id")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        )
        if row.data:
            return row.data[0].get("id")
    except Exception:
        pass
    return None


def upsert_job(
    job_id_legacy: str,
    *,
    run_config: dict[str, Any] | None = None,
    versions: dict[str, Any] | None = None,
) -> str | None:
    """Insert/update a job row by legacy id and return UUID if available."""
    client = _get_client()
    if not client:
        return None

    rc = run_config or {}
    vv = versions or {}
    payload = {
        "job_id_legacy": job_id_legacy,
        "input_mode": rc.get("input_mode", "pitchdeck"),
        "vc_investment_strategy": rc.get("vc_investment_strategy"),
        "instructions": rc.get("instructions"),
        "use_web_search": rc.get("use_web_search", False),
        "run_config": _serialize(rc),
        "app_version": vv.get("app_version"),
        "prompt_version": vv.get("prompt_version"),
        "pipeline_version": vv.get("pipeline_version"),
        "schema_version": vv.get("schema_version"),
    }

    try:
        result = client.table("jobs").upsert(payload, on_conflict="job_id_legacy").execute()
        if result.data:
            return result.data[0].get("id")
    except Exception:
        return _get_job_uuid(client, job_id_legacy)

    return _get_job_uuid(client, job_id_legacy)


def insert_analysis_event(
    job_id_legacy: str,
    *,
    message: str,
    event_type: str = "progress",
    stage: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    client = _get_client()
    if not client:
        return
    try:
        job_uuid = _get_job_uuid(client, job_id_legacy)
        client.table("analysis_events").insert(
            {
                "job_id": job_uuid,
                "job_id_legacy": job_id_legacy,
                "event_type": event_type,
                "stage": stage,
                "message": message,
                "payload": _serialize(payload or {}),
            }
        ).execute()
    except Exception:
        pass


def upsert_job_control(
    job_id_legacy: str,
    *,
    pause_requested: bool,
    stop_requested: bool,
    last_action: str | None,
) -> None:
    client = _get_client()
    if not client:
        return
    try:
        job_uuid = _get_job_uuid(client, job_id_legacy)
        client.table("job_controls").upsert(
            {
                "job_id": job_uuid,
                "job_id_legacy": job_id_legacy,
                "pause_requested": pause_requested,
                "stop_requested": stop_requested,
                "last_action": last_action,
            },
            on_conflict="job_id_legacy",
        ).execute()
    except Exception:
        pass


def insert_job_status_history(
    job_id_legacy: str,
    *,
    status: str,
    progress: str | None = None,
    source: str = "app",
) -> None:
    client = _get_client()
    if not client:
        return
    try:
        job_uuid = _get_job_uuid(client, job_id_legacy)
        client.table("job_status_history").insert(
            {
                "job_id": job_uuid,
                "job_id_legacy": job_id_legacy,
                "status": status,
                "progress": progress,
                "source": source,
            }
        ).execute()
    except Exception:
        pass


def insert_analysis_error(
    job_id_legacy: str,
    *,
    message: str,
    stage: str | None = None,
    error_type: str | None = None,
    details: dict[str, Any] | None = None,
    company_slug: str | None = None,
) -> None:
    client = _get_client()
    if not client:
        return
    try:
        job_uuid = _get_job_uuid(client, job_id_legacy)
        client.table("analysis_errors").insert(
            {
                "job_id": job_uuid,
                "job_id_legacy": job_id_legacy,
                "company_slug": company_slug,
                "stage": stage,
                "error_type": error_type,
                "message": message,
                "details": _serialize(details or {}),
            }
        ).execute()
    except Exception:
        pass


def upsert_person_profile_job(
    person_job_id: str,
    *,
    status: str,
    progress: str | None = None,
    request_payload: dict[str, Any] | None = None,
    result_payload: dict[str, Any] | None = None,
    error: str | None = None,
    company_slug: str | None = None,
    person_key: str | None = None,
) -> None:
    client = _get_client()
    if not client:
        return

    payload = {
        "person_job_id": person_job_id,
        "company_slug": company_slug,
        "person_key": person_key,
        "status": status,
        "progress": progress,
        "request_payload": _serialize(request_payload or {}),
        "result_payload": _serialize(result_payload) if result_payload is not None else None,
        "error": error,
    }
    try:
        client.table("person_profile_jobs").upsert(payload, on_conflict="person_job_id").execute()
    except Exception:
        pass


def load_person_profile_job(person_job_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None
    try:
        row = (
            client.table("person_profile_jobs")
            .select("person_job_id, status, progress, result_payload, error")
            .eq("person_job_id", person_job_id)
            .limit(1)
            .execute()
        )
        if not row.data:
            return None
        return row.data[0]
    except Exception:
        return None


def persist_analysis(
    job_id_legacy: str,
    results_list: list[dict[str, Any]],
    results_payload: dict[str, Any],
    excel_path: str,
    run_config: dict[str, Any],
    *,
    versions: dict[str, Any] | None = None,
    source_files: list[dict[str, Any]] | None = None,
    model_executions: list[dict[str, Any]] | None = None,
) -> bool:
    """Persist analysis results to Supabase.

    Creates/updates job, companies, pitch_decks, chunks, analyses. Uploads Excel to Storage.
    Stores results_payload for fast retrieval without reconstruction.
    Also stores source-file metadata and model execution telemetry.
    """
    client = _get_client()
    if not client:
        return False

    evaluated = [r for r in results_list if not r.get("skipped")]
    if not evaluated:
        return False

    try:
        ensure_excel_bucket()

        job_uuid = upsert_job(job_id_legacy, run_config=run_config, versions=versions)

        excel_storage_path: str | None = None
        excel_file = Path(excel_path)
        if excel_file.exists():
            try:
                with open(excel_file, "rb") as f:
                    client.storage.from_(BUCKET_EXCEL).upload(
                        path=f"{job_id_legacy}/results.xlsx",
                        file=f,
                        file_options={
                            "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "upsert": "true",
                        },
                    )
                excel_storage_path = f"{job_id_legacy}/results.xlsx"
            except Exception:
                pass

        for file_info in source_files or []:
            try:
                client.table("source_files").insert(
                    {
                        "job_id": job_uuid,
                        "job_id_legacy": job_id_legacy,
                        "file_name": file_info.get("name"),
                        "file_size_bytes": file_info.get("size"),
                        "mime_type": file_info.get("mime_type"),
                        "sha256": file_info.get("sha256"),
                        "local_path": file_info.get("local_path"),
                        "storage_path": file_info.get("storage_path"),
                    }
                ).execute()
            except Exception:
                pass

        for exec_row in model_executions or []:
            try:
                client.table("model_executions").insert(
                    {
                        "job_id": job_uuid,
                        "job_id_legacy": job_id_legacy,
                        "company_slug": exec_row.get("company_slug"),
                        "stage": exec_row.get("stage", "scoring"),
                        "provider": exec_row.get("provider"),
                        "model": exec_row.get("model"),
                        "request_timeout_seconds": exec_row.get("request_timeout_seconds"),
                        "max_retries": exec_row.get("max_retries"),
                        "latency_ms": exec_row.get("latency_ms"),
                        "prompt_tokens": exec_row.get("prompt_tokens"),
                        "completion_tokens": exec_row.get("completion_tokens"),
                        "total_tokens": exec_row.get("total_tokens"),
                        "status": exec_row.get("status", "done"),
                        "error_message": exec_row.get("error_message"),
                        "metadata": _serialize(exec_row.get("metadata") or {}),
                    }
                ).execute()
            except Exception:
                pass

        payload_serialized = _serialize(results_payload)

        for r in evaluated:
            company = r.get("company")
            slug = r.get("slug", "unknown")
            final_state = r.get("final_state", {})
            store = r.get("evidence_store")
            company_name = getattr(company, "name", None) or r.get("company_name") or slug
            company_domain = getattr(company, "domain", None)
            company_key = _normalize_company_key(company_name, company_domain, slug)
            company_payload = _company_payload_from_result(payload_serialized, r)
            summary_row = ((company_payload.get("summary_rows") or [{}]) or [{}])[0]
            ranking_result = company_payload.get("ranking_result") or {}

            company_id = None
            company_id = _upsert_company(
                client,
                name=company_name,
                industry=getattr(company, "industry", None),
                tagline=getattr(company, "tagline", None),
                about=getattr(company, "about", None),
                team=getattr(company, "team", None),
                domain=company_domain,
                company_key=company_key,
            )

            pitch_deck_id = None
            if store and company_id:
                pd_row = client.table("pitch_decks").insert(
                    {
                        "company_id": company_id,
                        "storage_path": f"jobs/{job_id_legacy}/{slug}",
                        "original_filename": f"{slug}.pdf",
                    }
                ).execute()
                if pd_row.data:
                    pitch_deck_id = pd_row.data[0]["id"]

                for idx, chunk in enumerate(store.chunks):
                    client.table("chunks").insert(
                        {
                            "pitch_deck_id": pitch_deck_id,
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "source_file": chunk.source_file,
                            "page_or_slide": str(chunk.page_or_slide) if chunk.page_or_slide is not None else None,
                            "sort_order": idx,
                        }
                    ).execute()

            state = _serialize(final_state)
            client.table("analyses").insert(
                {
                    "pitch_deck_id": pitch_deck_id,
                    "company_id": company_id,
                    "job_id": job_uuid,
                    "job_id_legacy": job_id_legacy,
                    "state": state,
                    "results_payload": payload_serialized,
                    "status": "done",
                    "run_config": _serialize(run_config),
                    "excel_storage_path": excel_storage_path,
                }
            ).execute()

            try:
                client.table("company_runs").upsert(
                    {
                        "company_id": company_id,
                        "job_id": job_uuid,
                        "job_id_legacy": job_id_legacy,
                        "company_key": company_key,
                        "company_name": company_name,
                        "startup_slug": slug,
                        "input_order": summary_row.get("specter_input_order"),
                        "decision": company_payload.get("decision"),
                        "total_score": company_payload.get("total_score"),
                        "composite_score": ranking_result.get("composite_score"),
                        "bucket": ranking_result.get("bucket"),
                        "mode": run_config.get("input_mode", "pitchdeck"),
                        "run_created_at": datetime.now(timezone.utc).isoformat(),
                        "result_payload": company_payload,
                    },
                    on_conflict="job_id_legacy,company_key",
                ).execute()
            except Exception:
                pass

        insert_job_status_history(
            job_id_legacy,
            status="done",
            progress="Analysis complete",
            source="persist_analysis",
        )
        upsert_job_control(
            job_id_legacy,
            pause_requested=False,
            stop_requested=False,
            last_action="done",
        )

        return True
    except Exception:
        return False


def load_job_results(job_id_legacy: str) -> dict[str, Any] | None:
    """Load results for a job from Supabase. Returns dict with results and excel_storage_path or None."""
    client = _get_client()
    if not client:
        return None

    try:
        analyses = (
            client.table("analyses")
            .select("results_payload, excel_storage_path")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        )
        if not analyses.data:
            return None

        first = analyses.data[0]
        results = first.get("results_payload")
        if results is None:
            return None

        out: dict[str, Any] = {"results": results}
        if first.get("excel_storage_path"):
            out["excel_storage_path"] = first["excel_storage_path"]
        return out
    except Exception:
        return None


def load_all_completed_jobs() -> dict[str, dict[str, Any]]:
    """Load all completed analyses from Supabase, keyed by job_id_legacy."""
    client = _get_client()
    if not client:
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        rows = (
            client.table("analyses")
            .select("job_id_legacy, results_payload, excel_storage_path, created_at")
            .eq("status", "done")
            .order("created_at", desc=True)
            .execute()
        )
        if not rows.data:
            return {}

        seen: set[str] = set()
        for r in rows.data:
            jid = r.get("job_id_legacy")
            if not jid or jid in seen:
                continue
            payload = r.get("results_payload")
            if payload is None:
                continue
            seen.add(jid)
            out[jid] = {
                "results": payload,
                "excel_storage_path": r.get("excel_storage_path"),
            }
    except Exception:
        pass
    return out


def list_saved_jobs(limit: int = 200) -> list[dict[str, Any]]:
    """Return saved jobs with latest known status and optional results payload."""
    client = _get_client()
    if not client:
        return []

    try:
        jobs_resp = (
            client.table("jobs")
            .select("job_id_legacy, input_mode, use_web_search, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        jobs = jobs_resp.data or []
        if not jobs:
            return []

        job_ids = [row.get("job_id_legacy") for row in jobs if row.get("job_id_legacy")]
        if not job_ids:
            return []

        status_rows = (
            client.table("job_status_history")
            .select("job_id_legacy, status, progress, created_at")
            .in_("job_id_legacy", job_ids)
            .order("created_at", desc=True)
            .limit(max(limit * 8, 200))
            .execute()
        )
        latest_status_by_job: dict[str, dict[str, Any]] = {}
        for row in status_rows.data or []:
            jid = row.get("job_id_legacy")
            if jid and jid not in latest_status_by_job:
                latest_status_by_job[jid] = row

        analysis_rows = (
            client.table("analyses")
            .select("job_id_legacy, status, results_payload, excel_storage_path, created_at")
            .in_("job_id_legacy", job_ids)
            .order("created_at", desc=True)
            .limit(max(limit * 4, 100))
            .execute()
        )
        latest_analysis_by_job: dict[str, dict[str, Any]] = {}
        for row in analysis_rows.data or []:
            jid = row.get("job_id_legacy")
            if jid and jid not in latest_analysis_by_job:
                latest_analysis_by_job[jid] = row

        out: list[dict[str, Any]] = []
        for row in jobs:
            jid = row.get("job_id_legacy")
            if not jid:
                continue

            latest_status = latest_status_by_job.get(jid, {})
            latest_analysis = latest_analysis_by_job.get(jid, {})
            status = latest_status.get("status") or latest_analysis.get("status") or "pending"
            progress = latest_status.get("progress")
            created_at = latest_analysis.get("created_at") or row.get("created_at")

            out.append(
                {
                    "job_id": jid,
                    "status": status,
                    "progress": progress,
                    "created_at": created_at,
                    "input_mode": row.get("input_mode"),
                    "use_web_search": row.get("use_web_search"),
                    "results": latest_analysis.get("results_payload"),
                    "excel_storage_path": latest_analysis.get("excel_storage_path"),
                }
            )

        return out
    except Exception:
        return []


def _fetch_company_run_rows(client: Client, limit_runs: int) -> list[dict[str, Any]]:
    rows = (
        client.table("company_runs")
        .select(
            "company_key, company_name, startup_slug, job_id_legacy, decision, total_score, "
            "composite_score, bucket, mode, input_order, run_created_at, created_at, result_payload"
        )
        .order("run_created_at", desc=True)
        .limit(limit_runs)
        .execute()
    )
    return rows.data or []


def _reconcile_missing_company_runs(
    client: Client,
    existing_rows: list[dict[str, Any]],
    *,
    limit_jobs: int,
) -> int:
    existing_job_ids = {
        row.get("job_id_legacy")
        for row in existing_rows
        if row.get("job_id_legacy")
    }
    if not existing_job_ids:
        return 0

    inserted = 0
    try:
        analyses = (
            client.table("analyses")
            .select("job_id_legacy, results_payload, created_at")
            .order("created_at", desc=True)
            .limit(limit_jobs)
            .execute()
        )
        missing_analyses = [
            row for row in (analyses.data or [])
            if row.get("job_id_legacy") and row.get("job_id_legacy") not in existing_job_ids
        ]
        if not missing_analyses:
            return 0

        missing_job_ids = [row.get("job_id_legacy") for row in missing_analyses if row.get("job_id_legacy")]
        jobs = (
            client.table("jobs")
            .select("job_id_legacy, input_mode")
            .in_("job_id_legacy", missing_job_ids)
            .limit(len(missing_job_ids))
            .execute()
        )
        mode_by_job = {row.get("job_id_legacy"): row.get("input_mode") for row in (jobs.data or [])}

        for row in missing_analyses:
            job_id_legacy = row.get("job_id_legacy")
            payload = row.get("results_payload")
            if not job_id_legacy or not payload:
                continue
            for run in _extract_company_runs_from_payload(
                job_id_legacy,
                payload,
                created_at=row.get("created_at"),
                mode=mode_by_job.get(job_id_legacy),
            ):
                try:
                    client.table("company_runs").upsert(
                        run,
                        on_conflict="job_id_legacy,company_key",
                    ).execute()
                    inserted += 1
                except Exception:
                    pass
    except Exception:
        return inserted

    return inserted


def list_company_histories(limit_runs: int = 1000) -> list[dict[str, Any]]:
    """Return grouped company histories with per-run records."""
    client = _get_client()
    if not client:
        return []

    try:
        rows_data = _fetch_company_run_rows(client, limit_runs)
        if not rows_data:
            backfill_company_runs_from_analyses()
            rows_data = _fetch_company_run_rows(client, limit_runs)
        else:
            inserted = _reconcile_missing_company_runs(
                client,
                rows_data,
                limit_jobs=max(limit_runs, 200),
            )
            if inserted:
                rows_data = _fetch_company_run_rows(client, limit_runs)

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows_data:
            company_key = row.get("company_key")
            if not company_key:
                continue

            group_key = _company_history_group_key(row)
            run = {
                "job_id": row.get("job_id_legacy"),
                "startup_slug": row.get("startup_slug"),
                "decision": row.get("decision"),
                "total_score": row.get("total_score"),
                "composite_score": row.get("composite_score"),
                "bucket": row.get("bucket"),
                "mode": row.get("mode"),
                "input_order": row.get("input_order"),
                "created_at": row.get("run_created_at") or row.get("created_at"),
                "results": row.get("result_payload"),
            }
            entry = grouped.setdefault(
                group_key,
                {
                    "company_key": company_key,
                    "company_name": row.get("company_name") or row.get("startup_slug") or company_key,
                    "latest_score": row.get("composite_score"),
                    "latest_total_score": row.get("total_score"),
                    "latest_input_order": row.get("input_order"),
                    "latest_run_at": row.get("run_created_at") or row.get("created_at"),
                    "runs": [],
                },
            )
            existing_run = next(
                (
                    existing
                    for existing in entry["runs"]
                    if existing.get("job_id") and existing.get("job_id") == run.get("job_id")
                ),
                None,
            )
            if existing_run:
                existing_ts = str(existing_run.get("created_at") or "")
                current_ts = str(run.get("created_at") or "")
                if current_ts > existing_ts:
                    entry["runs"].remove(existing_run)
                    entry["runs"].append(run)
                continue

            entry["runs"].append(run)

        for entry in grouped.values():
            entry["runs"].sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)
            if entry["runs"]:
                latest = entry["runs"][0]
                latest_result = latest.get("results") or {}
                latest_summary = ((latest_result.get("summary_rows") or [{}]) or [{}])[0]
                entry["company_key"] = (
                    latest.get("startup_slug")
                    or latest_result.get("startup_slug")
                    or latest_summary.get("startup_slug")
                    or entry.get("company_key")
                )
                entry["company_name"] = (
                    latest_result.get("company_name")
                    or latest_summary.get("company_name")
                    or entry.get("company_name")
                )
                entry["latest_score"] = latest.get("composite_score")
                entry["latest_total_score"] = latest.get("total_score")
                entry["latest_input_order"] = latest.get("input_order")
                entry["latest_run_at"] = latest.get("created_at")

        return list(grouped.values())
    except Exception:
        return []


def _extract_company_runs_from_payload(
    job_id_legacy: str,
    payload: dict[str, Any],
    *,
    created_at: str | None,
    mode: str | None,
) -> list[dict[str, Any]]:
    serialized = _serialize(payload or {})
    payload_mode = serialized.get("mode")
    if payload_mode == "single":
        company_name = serialized.get("company_name") or serialized.get("startup_slug") or job_id_legacy
        company_key = _normalize_company_key(company_name, None, serialized.get("startup_slug"))
        ranking = serialized.get("ranking_result") or {}
        return [{
            "job_id_legacy": job_id_legacy,
            "company_key": company_key,
            "company_name": company_name,
            "startup_slug": serialized.get("startup_slug"),
            "input_order": None,
            "decision": serialized.get("decision"),
            "total_score": serialized.get("total_score"),
            "composite_score": ranking.get("composite_score"),
            "bucket": ranking.get("bucket"),
            "mode": mode or payload_mode,
            "run_created_at": created_at,
            "result_payload": serialized,
        }]

    rows = []
    for summary_row in serialized.get("summary_rows") or []:
        company_name = summary_row.get("company_name") or summary_row.get("startup_slug") or job_id_legacy
        startup_slug = summary_row.get("startup_slug")
        company_key = _normalize_company_key(company_name, None, startup_slug)
        row_payload = {
            "mode": "single",
            "source_mode": payload_mode or "batch",
            "startup_slug": startup_slug,
            "company_name": company_name,
            "decision": summary_row.get("decision"),
            "total_score": summary_row.get("total_score"),
            "avg_pro": summary_row.get("avg_pro"),
            "avg_contra": summary_row.get("avg_contra"),
            "summary_rows": [summary_row],
            "argument_rows": [
                row for row in (serialized.get("argument_rows") or [])
                if (row.get("startup_slug") or "") == startup_slug
            ],
            "qa_provenance_rows": [
                row for row in (serialized.get("qa_provenance_rows") or [])
                if (row.get("startup_slug") or "") == startup_slug
            ],
            "founders": summary_row.get("founders") or [],
            "team_members": summary_row.get("team_members") or [],
            "ranking_result": {
                "rank": summary_row.get("rank"),
                "percentile": summary_row.get("percentile"),
                "composite_score": summary_row.get("composite_score"),
                "strategy_fit_score": summary_row.get("strategy_fit_score"),
                "team_score": summary_row.get("team_score"),
                "upside_score": summary_row.get("upside_score"),
                "bucket": summary_row.get("bucket"),
                "strategy_fit_summary": summary_row.get("strategy_fit_summary"),
                "team_summary": summary_row.get("team_summary"),
                "potential_summary": summary_row.get("potential_summary"),
                "key_points": summary_row.get("key_points"),
                "red_flags": summary_row.get("red_flags"),
            },
        }
        rows.append({
            "job_id_legacy": job_id_legacy,
            "company_key": company_key,
            "company_name": company_name,
            "startup_slug": startup_slug,
            "input_order": summary_row.get("specter_input_order"),
            "decision": summary_row.get("decision"),
            "total_score": summary_row.get("total_score"),
            "composite_score": summary_row.get("composite_score"),
            "bucket": summary_row.get("bucket"),
            "mode": mode or payload_mode,
            "run_created_at": created_at,
            "result_payload": row_payload,
        })
    return rows


def backfill_company_runs_from_analyses(limit_jobs: int = 500) -> int:
    """Populate company_runs from historical analyses rows when the new table is empty."""
    client = _get_client()
    if not client:
        return 0

    inserted = 0
    try:
        existing = (
            client.table("company_runs")
            .select("id", count="exact", head=True)
            .execute()
        )
        if (existing.count or 0) > 0:
            return 0
    except Exception:
        return 0

    try:
        analyses = (
            client.table("analyses")
            .select("job_id_legacy, results_payload, created_at")
            .order("created_at", desc=True)
            .limit(limit_jobs)
            .execute()
        )
        jobs = (
            client.table("jobs")
            .select("job_id_legacy, input_mode")
            .limit(limit_jobs)
            .execute()
        )
        mode_by_job = {row.get("job_id_legacy"): row.get("input_mode") for row in (jobs.data or [])}
        seen_jobs: set[str] = set()
        for row in analyses.data or []:
            job_id_legacy = row.get("job_id_legacy")
            payload = row.get("results_payload")
            if not job_id_legacy or not payload or job_id_legacy in seen_jobs:
                continue
            seen_jobs.add(job_id_legacy)
            for run in _extract_company_runs_from_payload(
                job_id_legacy,
                payload,
                created_at=row.get("created_at"),
                mode=mode_by_job.get(job_id_legacy),
            ):
                try:
                    client.table("company_runs").upsert(
                        run,
                        on_conflict="job_id_legacy,company_key",
                    ).execute()
                    inserted += 1
                except Exception:
                    pass
    except Exception:
        return inserted

    return inserted


def backfill_all_company_runs_from_analyses(limit_jobs: int = 500) -> int:
    """Populate company_runs from all analyses (including when table already has rows)."""
    client = _get_client()
    if not client:
        return 0

    inserted = 0
    try:
        analyses = (
            client.table("analyses")
            .select("job_id_legacy, results_payload, created_at")
            .order("created_at", desc=True)
            .limit(limit_jobs)
            .execute()
        )
        jobs = (
            client.table("jobs")
            .select("job_id_legacy, input_mode")
            .limit(limit_jobs)
            .execute()
        )
        mode_by_job = {row.get("job_id_legacy"): row.get("input_mode") for row in (jobs.data or [])}
        seen_jobs: set[str] = set()
        for row in analyses.data or []:
            job_id_legacy = row.get("job_id_legacy")
            payload = row.get("results_payload")
            if not job_id_legacy or not payload or job_id_legacy in seen_jobs:
                continue
            seen_jobs.add(job_id_legacy)
            for run in _extract_company_runs_from_payload(
                job_id_legacy,
                payload,
                created_at=row.get("created_at"),
                mode=mode_by_job.get(job_id_legacy),
            ):
                try:
                    client.table("company_runs").upsert(
                        run,
                        on_conflict="job_id_legacy,company_key",
                    ).execute()
                    inserted += 1
                except Exception:
                    pass
    except Exception:
        return inserted

    return inserted


def download_excel_bytes(job_id_legacy: str) -> bytes | None:
    """Download Excel file from Supabase Storage. Returns None if not found."""
    client = _get_client()
    if not client:
        return None

    try:
        path = f"{job_id_legacy}/results.xlsx"
        data = client.storage.from_(BUCKET_EXCEL).download(path)
        return data
    except Exception:
        return None


def load_analyses_by_company(company_name: str) -> list[dict[str, Any]]:
    """Load analyses for a company by name. Returns list of {job_id_legacy, results_payload, created_at}."""
    client = _get_client()
    if not client:
        return []

    out: list[dict[str, Any]] = []
    try:
        company_runs = (
            client.table("company_runs")
            .select("job_id_legacy, result_payload, run_created_at, company_name")
            .filter("company_name", "ilike", f"%{company_name}%")
            .order("run_created_at", desc=True)
            .limit(50)
            .execute()
        )
        if company_runs.data:
            for r in company_runs.data:
                payload = r.get("result_payload")
                if payload:
                    out.append({
                        "job_id": r.get("job_id_legacy"),
                        "results": payload,
                        "created_at": r.get("run_created_at"),
                    })
            return out
    except Exception:
        pass

    try:
        companies = (
            client.table("companies")
            .select("id")
            .filter("name", "ilike", f"%{company_name}%")
            .execute()
        )
        if not companies.data:
            return []
        company_ids = [c["id"] for c in companies.data]

        analyses = (
            client.table("analyses")
            .select("job_id_legacy, results_payload, created_at")
            .in_("company_id", company_ids)
            .eq("status", "done")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        for r in analyses.data or []:
            payload = r.get("results_payload")
            if payload:
                out.append({
                    "job_id": r.get("job_id_legacy"),
                    "results": payload,
                    "created_at": r.get("created_at"),
                })
    except Exception:
        pass
    return out


def ensure_excel_bucket() -> None:
    """Create the Excel storage bucket if it does not exist."""
    client = _get_client()
    if not client:
        return
    try:
        client.storage.get_bucket(BUCKET_EXCEL)
    except Exception:
        try:
            client.storage.create_bucket(BUCKET_EXCEL, options={"public": False})
        except Exception:
            pass
