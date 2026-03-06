"""Supabase persistence for Startup Ranker analyses and job telemetry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from supabase import Client, create_client

_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_client: Client | None = None
BUCKET_EXCEL = "analysis-exports"


def _get_client() -> Client | None:
    global _client
    if not _SUPABASE_URL or not _SUPABASE_SERVICE_KEY:
        return None
    if _client is None:
        _client = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_KEY)
    return _client


def is_configured() -> bool:
    return bool(_SUPABASE_URL and _SUPABASE_SERVICE_KEY)


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

            company_id = None
            if company:
                company_data = {
                    "name": getattr(company, "name", slug),
                    "industry": getattr(company, "industry"),
                    "tagline": getattr(company, "tagline"),
                    "about": getattr(company, "about"),
                    "team": _serialize(getattr(company, "team", []) or []),
                    "domain": getattr(company, "domain", None),
                }
                company_row = client.table("companies").insert(company_data).execute()
                if company_row.data:
                    company_id = company_row.data[0]["id"]

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
