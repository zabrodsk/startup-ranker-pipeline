"""Supabase persistence for Rockaway Deal Intelligence analyses and job telemetry."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client
from agent.run_context import build_run_costs_from_model_executions

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

_client: Client | None = None
_client_config: tuple[str, str] | None = None
logger = logging.getLogger(__name__)
SOURCE_FILES_BUCKET = os.getenv("SUPABASE_SOURCE_FILES_BUCKET", "analysis-inputs")
WORKER_ACTIVE_STATUSES = {"queued", "claimed", "running", "finalizing"}
WORKER_TERMINAL_STATUSES = {"done", "error", "interrupted", "stopped"}
WORKER_EXECUTION_STALE_SECONDS = int(os.getenv("SPECTER_WORKER_STALE_SECONDS", "120"))
WORKER_QUEUE_STALE_SECONDS = int(os.getenv("SPECTER_WORKER_QUEUE_STALE_SECONDS", "900"))


def _log_supabase_error(
    operation: str,
    table: str,
    exc: Exception,
    *,
    job_id_legacy: str | None = None,
    include_traceback: bool = False,
) -> None:
    log_fn = logger.exception if include_traceback else logger.warning
    log_fn(
        "Supabase operation failed: operation=%s table=%s job_id_legacy=%s error=%s",
        operation,
        table,
        job_id_legacy or "-",
        exc,
    )


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


def _extract_started_by_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    started_by = {
        "started_by_user_id": source.get("started_by_user_id"),
        "started_by_email": source.get("started_by_email"),
        "started_by_display_name": source.get("started_by_display_name"),
        "started_by_label": source.get("started_by_label"),
    }
    return {key: value for key, value in started_by.items() if value not in (None, "")}


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


def _ranking_sort_key_from_payload(payload: dict[str, Any]) -> tuple[float, float, float, int]:
    ranking = _serialize(payload.get("ranking_result") or {})
    dimension_scores = ranking.get("dimension_scores") or []
    adjusted_scores = [
        float(item.get("adjusted_score"))
        for item in dimension_scores
        if isinstance(item, dict) and isinstance(item.get("adjusted_score"), (int, float))
    ]
    confidences = [
        float(item.get("confidence"))
        for item in dimension_scores
        if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))
    ]
    critical_gaps = sum(
        len(item.get("critical_gaps") or [])
        for item in dimension_scores
        if isinstance(item, dict)
    )
    composite_score = ranking.get("composite_score")
    total_score = payload.get("total_score")
    primary_score = (
        float(composite_score)
        if isinstance(composite_score, (int, float))
        else float(total_score) if isinstance(total_score, (int, float)) else 0.0
    )
    min_dimension_score = min(adjusted_scores) if adjusted_scores else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return (
        -primary_score,
        -min_dimension_score,
        -avg_confidence,
        critical_gaps,
    )


def _job_result_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    data = _serialize(payload or {})
    if not data:
        return None

    summary_rows = data.get("summary_rows") or []
    first_summary = summary_rows[0] if summary_rows else {}
    ranking = _serialize(data.get("ranking_result") or {})
    return {
        "mode": data.get("mode"),
        "company_name": data.get("company_name") or first_summary.get("company_name"),
        "startup_slug": data.get("startup_slug") or first_summary.get("startup_slug"),
        "summary_count": len(summary_rows),
        "decision": data.get("decision") or first_summary.get("decision"),
        "total_score": data.get("total_score") if data.get("total_score") is not None else first_summary.get("total_score"),
        "composite_score": ranking.get("composite_score") if ranking.get("composite_score") is not None else first_summary.get("composite_score"),
        "bucket": ranking.get("bucket") or first_summary.get("bucket"),
    }


def _sorted_completed_company_payloads(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        payload = _serialize(row.get("result_payload") or {})
        if not payload:
            continue
        prepared.append((row, payload))

    def _order_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, str, str]:
        row, payload = item
        input_order = row.get("input_order")
        summary_row = ((payload.get("summary_rows") or [{}]) or [{}])[0]
        fallback_order = summary_row.get("specter_input_order")
        order_value = input_order if isinstance(input_order, int) else fallback_order
        if not isinstance(order_value, int):
            order_value = 10**9
        created_at = str(row.get("run_created_at") or row.get("created_at") or "")
        slug = str(row.get("startup_slug") or payload.get("startup_slug") or "")
        return (order_value, created_at, slug)

    prepared.sort(key=_order_key)
    return prepared


def _compose_results_payload_from_company_runs(
    rows: list[dict[str, Any]],
    *,
    preferred_mode: str | None = None,
    snapshot_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prepared = _sorted_completed_company_payloads(rows)
    if not prepared:
        snapshot = _serialize(snapshot_payload or {})
        return snapshot or None

    snapshot = _serialize(snapshot_payload or {})
    resolved_mode = preferred_mode or snapshot.get("mode")
    if not resolved_mode:
        only_payload = prepared[0][1]
        resolved_mode = "single" if len(prepared) == 1 and only_payload.get("mode") == "single" else "batch"

    if resolved_mode == "single" and len(prepared) == 1:
        payload = _serialize(prepared[0][1])
        for key in ("llm", "llm_selection", "run_costs", "batch_chunking", "job_status", "job_message"):
            if key in snapshot:
                payload[key] = _serialize(snapshot.get(key))
        return payload

    ranked_payloads = [payload for _, payload in prepared]
    ranked_payloads.sort(key=_ranking_sort_key_from_payload)
    company_count = len(ranked_payloads)

    summary_rows: list[dict[str, Any]] = []
    argument_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    founders_by_slug: dict[str, list[dict[str, Any]]] = {}
    team_members_by_slug: dict[str, list[dict[str, Any]]] = {}

    for index, payload in enumerate(ranked_payloads, start=1):
        summary_row = _serialize(((payload.get("summary_rows") or [{}]) or [{}])[0])
        if not summary_row:
            summary_row = {
                "startup_slug": payload.get("startup_slug"),
                "company_name": payload.get("company_name"),
                "decision": payload.get("decision"),
                "total_score": payload.get("total_score"),
            }

        ranking = _serialize(payload.get("ranking_result") or {})
        summary_row["rank"] = index
        summary_row["percentile"] = round(100.0 * (company_count - index + 1) / company_count, 1)
        for key in (
            "composite_score",
            "strategy_fit_score",
            "team_score",
            "upside_score",
            "bucket",
            "strategy_fit_summary",
            "team_summary",
            "potential_summary",
        ):
            if key in ranking:
                summary_row[key] = _serialize(ranking.get(key))
        for key in ("key_points", "red_flags"):
            if key not in ranking:
                continue
            value = _serialize(ranking.get(key))
            summary_row[key] = "\n".join(value) if isinstance(value, list) else value

        slug = (
            summary_row.get("startup_slug")
            or payload.get("startup_slug")
            or _normalize_text_token(summary_row.get("company_name") or payload.get("company_name"))
        )
        founders = _serialize(payload.get("founders") or summary_row.get("founders") or [])
        team_members = _serialize(payload.get("team_members") or summary_row.get("team_members") or founders)
        summary_row["founders"] = founders
        summary_row["team_members"] = team_members
        founders_by_slug[slug] = founders
        team_members_by_slug[slug] = team_members
        summary_rows.append(summary_row)
        argument_rows.extend(_serialize(payload.get("argument_rows") or []))
        qa_rows.extend(_serialize(payload.get("qa_provenance_rows") or []))

    results_payload = {
        "mode": "batch",
        "num_companies": len(summary_rows),
        "num_skipped": 0,
        "summary_rows": summary_rows,
        "argument_rows": argument_rows,
        "qa_provenance_rows": qa_rows,
        "failed_rows": [],
        "founders_by_slug": founders_by_slug,
        "team_members_by_slug": team_members_by_slug,
    }
    for key in ("llm", "llm_selection", "run_costs", "batch_chunking", "job_status", "job_message"):
        if key in snapshot:
            results_payload[key] = _serialize(snapshot.get(key))
    return results_payload


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
    except Exception as exc:
        _log_supabase_error("upsert_company", "companies", exc)

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
    except Exception as exc:
        _log_supabase_error("select_company_after_upsert", "companies", exc)
    return None


def _select_existing_company_analyses(
    client: Client,
    *,
    job_id_legacy: str,
    company_id: str | None,
) -> list[dict[str, Any]]:
    if not company_id:
        return []
    try:
        rows = (
            client.table("analyses")
            .select("id, pitch_deck_id")
            .eq("job_id_legacy", job_id_legacy)
            .eq("company_id", company_id)
            .execute()
        )
        return list(rows.data or [])
    except Exception as exc:
        _log_supabase_error(
            "select_existing_company_analyses",
            "analyses",
            exc,
            job_id_legacy=job_id_legacy,
        )
        return []


def _replace_company_analysis_records(
    client: Client,
    *,
    job_id_legacy: str,
    company_id: str | None,
    replace_documents: bool,
) -> str | None:
    existing_rows = _select_existing_company_analyses(
        client,
        job_id_legacy=job_id_legacy,
        company_id=company_id,
    )
    pitch_deck_ids = [row.get("pitch_deck_id") for row in existing_rows if row.get("pitch_deck_id")]
    preserved_pitch_deck_id = pitch_deck_ids[0] if pitch_deck_ids and not replace_documents else None

    if company_id:
        try:
            client.table("analyses").delete().eq("job_id_legacy", job_id_legacy).eq("company_id", company_id).execute()
        except Exception as exc:
            _log_supabase_error(
                "delete_existing_company_analyses",
                "analyses",
                exc,
                job_id_legacy=job_id_legacy,
            )

    if replace_documents:
        for pitch_deck_id in pitch_deck_ids:
            try:
                client.table("chunks").delete().eq("pitch_deck_id", pitch_deck_id).execute()
            except Exception as exc:
                _log_supabase_error(
                    "delete_existing_chunks",
                    "chunks",
                    exc,
                    job_id_legacy=job_id_legacy,
                )
            try:
                client.table("pitch_decks").delete().eq("id", pitch_deck_id).execute()
            except Exception as exc:
                _log_supabase_error(
                    "delete_existing_pitch_decks",
                    "pitch_decks",
                    exc,
                    job_id_legacy=job_id_legacy,
                )

    return preserved_pitch_deck_id


def _persist_company_analysis_row(
    client: Client,
    *,
    job_uuid: str,
    job_id_legacy: str,
    result_row: dict[str, Any],
    company_payload: dict[str, Any],
    run_config: dict[str, Any],
    excel_storage_path: str | None,
    replace_documents: bool,
) -> bool:
    company = result_row.get("company")
    slug = result_row.get("slug", "unknown")
    final_state = result_row.get("final_state", {})
    store = result_row.get("evidence_store")
    analysis_status = str(result_row.get("analysis_status") or "done")
    company_name = getattr(company, "name", None) or result_row.get("company_name") or slug
    company_domain = getattr(company, "domain", None)
    company_key = _normalize_company_key(company_name, company_domain, slug)
    ranking_result = company_payload.get("ranking_result") or {}
    summary_row = ((company_payload.get("summary_rows") or [{}]) or [{}])[0]

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

    pitch_deck_id = _replace_company_analysis_records(
        client,
        job_id_legacy=job_id_legacy,
        company_id=company_id,
        replace_documents=replace_documents,
    )

    if store and company_id and (replace_documents or not pitch_deck_id):
        pd_row = client.table("pitch_decks").insert(
            {
                "company_id": company_id,
                "storage_path": f"jobs/{job_id_legacy}/{slug}",
                "original_filename": f"{slug}.pdf",
            }
        ).execute()
        if pd_row.data:
            pitch_deck_id = pd_row.data[0]["id"]

        if pitch_deck_id:
            for idx, chunk in enumerate(getattr(store, "chunks", []) or []):
                client.table("chunks").insert(
                    {
                        "pitch_deck_id": pitch_deck_id,
                        "chunk_id": getattr(chunk, "chunk_id", None),
                        "text": getattr(chunk, "text", ""),
                        "source_file": getattr(chunk, "source_file", ""),
                        "page_or_slide": (
                            str(getattr(chunk, "page_or_slide", None))
                            if getattr(chunk, "page_or_slide", None) is not None
                            else None
                        ),
                        "sort_order": idx,
                    }
                ).execute()

    client.table("analyses").insert(
        {
            "pitch_deck_id": pitch_deck_id,
            "company_id": company_id,
            "job_id": job_uuid,
            "job_id_legacy": job_id_legacy,
            "state": _serialize(final_state),
            "results_payload": _serialize(company_payload),
            "status": analysis_status,
            "run_config": _serialize(run_config),
            "excel_storage_path": excel_storage_path,
        }
    ).execute()

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
            "result_payload": _serialize(company_payload),
            **_extract_started_by_fields(run_config),
        },
        on_conflict="job_id_legacy,company_key",
    ).execute()
    return True


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


def _execution_metadata(exec_row: dict[str, Any]) -> dict[str, Any]:
    metadata = _serialize(exec_row.get("metadata") or {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("service", exec_row.get("service", "llm"))
    if exec_row.get("status") is not None:
        metadata.setdefault("status", exec_row.get("status"))
    if exec_row.get("estimated_cost_usd") is not None:
        metadata["estimated_cost_usd"] = exec_row.get("estimated_cost_usd")
    if exec_row.get("request_count") is not None:
        metadata["request_count"] = exec_row.get("request_count")
    return metadata


def _hydrate_execution_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    hydrated = dict(row)
    hydrated["service"] = row.get("service") or metadata.get("service") or "llm"
    hydrated["status"] = row.get("status") or metadata.get("status") or "done"
    hydrated["request_count"] = row.get("request_count")
    if hydrated["request_count"] is None:
        hydrated["request_count"] = metadata.get("request_count")
    hydrated["estimated_cost_usd"] = row.get("estimated_cost_usd")
    if hydrated["estimated_cost_usd"] is None:
        hydrated["estimated_cost_usd"] = metadata.get("estimated_cost_usd")
    return hydrated


def _extract_worker_state(run_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(run_config, dict):
        return {}
    worker_state = run_config.get("worker_state")
    return _serialize(worker_state) if isinstance(worker_state, dict) else {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _worker_state_timeout_seconds(worker_state: dict[str, Any]) -> int | None:
    status = str(worker_state.get("status") or "").strip().lower()
    if status not in WORKER_ACTIVE_STATUSES:
        return None
    if status == "queued":
        return max(WORKER_QUEUE_STALE_SECONDS, 1)
    return max(WORKER_EXECUTION_STALE_SECONDS, 1)


def _worker_state_is_stale(worker_state: dict[str, Any]) -> bool:
    timeout_seconds = _worker_state_timeout_seconds(worker_state)
    if timeout_seconds is None:
        return False
    heartbeat = _parse_timestamp(worker_state.get("last_heartbeat_at"))
    if heartbeat is None:
        return False
    return _utcnow() - heartbeat > timedelta(seconds=timeout_seconds)


def _resolved_worker_state(worker_state: dict[str, Any]) -> tuple[str, str | None, bool]:
    status = str(worker_state.get("status") or "").strip().lower()
    progress = _worker_progress_from_state(worker_state)
    if status not in WORKER_ACTIVE_STATUSES:
        return status, progress, False
    if _worker_state_is_stale(worker_state):
        if status == "queued":
            return "interrupted", "Worker queue stalled before processing started.", False
        return "interrupted", "Worker interrupted before completion.", False
    return status, progress, True


def _worker_state_terminal_timestamp(worker_state: dict[str, Any]) -> datetime | None:
    return _parse_timestamp(worker_state.get("run_finished_at")) or _parse_timestamp(
        worker_state.get("last_heartbeat_at")
    )


def _merge_worker_state(
    run_config: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(run_config or {})
    worker_state = dict(_extract_worker_state(merged))
    for key, value in (updates or {}).items():
        if value is None:
            worker_state.pop(key, None)
        else:
            worker_state[key] = _serialize(value)
    if worker_state:
        merged["worker_state"] = worker_state
    else:
        merged.pop("worker_state", None)
    return merged


def _worker_progress_from_state(worker_state: dict[str, Any]) -> str | None:
    progress = worker_state.get("progress")
    if isinstance(progress, str) and progress.strip():
        return progress
    status = str(worker_state.get("status") or "").strip().lower()
    active_company = str(worker_state.get("active_company_slug") or "").strip()
    completed = worker_state.get("completed_companies")
    total = worker_state.get("total_companies")
    if status == "queued":
        return "Queued for worker..."
    if status == "claimed":
        return "Worker claimed job..."
    if status == "finalizing":
        return "Finalizing batch results..."
    if status == "running" and active_company:
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            next_index = min(completed + 1, total)
            return f"Worker running — {active_company} ({next_index}/{total})"
        return f"Worker running — {active_company}"
    if status in WORKER_TERMINAL_STATUSES:
        return "Worker finished." if status == "done" else "Worker interrupted."
    return None


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
    except Exception as exc:
        _log_supabase_error("get_job_uuid", "jobs", exc, job_id_legacy=job_id_legacy)
    return None


def upsert_job(
    job_id_legacy: str,
    *,
    run_config: dict[str, Any] | None = None,
    versions: dict[str, Any] | None = None,
    worker_state: dict[str, Any] | None = None,
) -> str | None:
    """Insert/update a job row by legacy id and return UUID if available."""
    client = _get_client()
    if not client:
        return None

    rc = _merge_worker_state(run_config or {}, worker_state)
    vv = versions or {}
    started_by = _extract_started_by_fields(rc)
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
        **started_by,
    }

    try:
        result = client.table("jobs").upsert(payload, on_conflict="job_id_legacy").execute()
        if result.data:
            return result.data[0].get("id")
    except Exception as exc:
        _log_supabase_error("upsert_job", "jobs", exc, job_id_legacy=job_id_legacy)
        return _get_job_uuid(client, job_id_legacy)

    return _get_job_uuid(client, job_id_legacy)


def ensure_source_files_bucket() -> None:
    client = _get_client()
    if not client:
        return
    try:
        existing = client.storage.list_buckets()
        if any(
            (
                getattr(bucket, "name", None)
                or (bucket.get("name") if isinstance(bucket, dict) else None)
            )
            == SOURCE_FILES_BUCKET
            for bucket in existing or []
        ):
            return
    except Exception:
        pass
    try:
        client.storage.create_bucket(
            SOURCE_FILES_BUCKET,
            options={"public": False},
        )
    except Exception as exc:
        _log_supabase_error("ensure_source_files_bucket", "storage", exc)


def get_authenticated_supabase_user(jwt: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client or not jwt:
        return None
    try:
        response = client.auth.get_user(jwt)
        user = getattr(response, "user", None)
        if user is None:
            return None
        if hasattr(user, "model_dump"):
            return user.model_dump()
        if isinstance(user, dict):
            return user
    except Exception as exc:
        _log_supabase_error("get_authenticated_supabase_user", "auth", exc)
    return None


def upload_source_file(
    job_id_legacy: str,
    local_path: str | Path,
    *,
    file_name: str | None = None,
    mime_type: str | None = None,
) -> str | None:
    client = _get_client()
    if not client:
        return None

    source_path = Path(local_path)
    if not source_path.exists():
        return None

    ensure_source_files_bucket()
    file_token = re.sub(r"[^a-zA-Z0-9._-]+", "-", file_name or source_path.name).strip("-") or source_path.name
    storage_path = f"jobs/{job_id_legacy}/inputs/{file_token}"
    try:
        payload = source_path.read_bytes()
        options = {"upsert": "true"}
        if mime_type:
            options["content-type"] = mime_type
        client.storage.from_(SOURCE_FILES_BUCKET).upload(storage_path, payload, options)
        return storage_path
    except Exception as exc:
        _log_supabase_error("upload_source_file", "storage", exc, job_id_legacy=job_id_legacy)
        return None


def download_source_file_to_path(
    storage_path: str,
    destination_path: str | Path,
) -> bool:
    client = _get_client()
    if not client:
        return False
    try:
        data = client.storage.from_(SOURCE_FILES_BUCKET).download(storage_path)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return True
    except Exception as exc:
        _log_supabase_error("download_source_file_to_path", "storage", exc)
        return False


def persist_source_files(
    job_id_legacy: str,
    source_files: list[dict[str, Any]],
    *,
    run_config: dict[str, Any] | None = None,
    versions: dict[str, Any] | None = None,
    worker_state: dict[str, Any] | None = None,
) -> bool:
    client = _get_client()
    if not client:
        return False

    job_uuid = upsert_job(
        job_id_legacy,
        run_config=run_config,
        versions=versions,
        worker_state=worker_state,
    )
    if not job_uuid:
        return False

    try:
        client.table("source_files").delete().eq("job_id_legacy", job_id_legacy).execute()
    except Exception as exc:
        _log_supabase_error("persist_source_files.delete_existing", "source_files", exc, job_id_legacy=job_id_legacy)

    success = True
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
        except Exception as exc:
            success = False
            _log_supabase_error("persist_source_files.insert", "source_files", exc, job_id_legacy=job_id_legacy)
    return success


def load_source_files(job_id_legacy: str) -> list[dict[str, Any]]:
    client = _get_client()
    if not client:
        return []
    try:
        rows = (
            client.table("source_files")
            .select("file_name, file_size_bytes, mime_type, sha256, local_path, storage_path")
            .eq("job_id_legacy", job_id_legacy)
            .order("created_at")
            .execute()
        )
        out: list[dict[str, Any]] = []
        for row in rows.data or []:
            out.append(
                {
                    "name": row.get("file_name"),
                    "size": row.get("file_size_bytes"),
                    "mime_type": row.get("mime_type"),
                    "sha256": row.get("sha256"),
                    "local_path": row.get("local_path"),
                    "storage_path": row.get("storage_path"),
                }
            )
        return out
    except Exception as exc:
        _log_supabase_error("load_source_files", "source_files", exc, job_id_legacy=job_id_legacy)
        return []


def queue_specter_worker_job(
    job_id_legacy: str,
    *,
    run_config: dict[str, Any],
    versions: dict[str, Any] | None = None,
    total_companies: int,
    progress: str | None = None,
) -> bool:
    worker_state = {
        "status": "queued",
        "progress": progress or "Queued for worker...",
        "total_companies": total_companies,
        "completed_companies": 0,
        "failed_companies": 0,
        "active_company_slug": None,
        "active_company_index": None,
        "last_heartbeat_at": _utcnow().isoformat(),
        "run_started_at": None,
        "run_finished_at": None,
        "worker_service_enabled": True,
    }
    job_uuid = upsert_job(
        job_id_legacy,
        run_config=run_config,
        versions=versions,
        worker_state=worker_state,
    )
    if not job_uuid:
        return False
    insert_job_status_history(
        job_id_legacy,
        status="running",
        progress=worker_state["progress"],
        source="worker_queue",
    )
    insert_analysis_event(
        job_id_legacy,
        message=worker_state["progress"],
        event_type="worker_queue",
        stage="queue",
        payload={"worker_state": worker_state},
    )
    return True


def list_claimable_specter_worker_jobs(limit: int = 10) -> list[dict[str, Any]]:
    client = _get_client()
    if not client:
        return []
    try:
        rows = (
            client.table("jobs")
            .select("job_id_legacy, input_mode, use_web_search, created_at, run_config")
            .eq("input_mode", "specter")
            .order("created_at", desc=True)
            .limit(max(limit * 10, 50))
            .execute()
        )
    except Exception as exc:
        _log_supabase_error("list_claimable_specter_worker_jobs", "jobs", exc)
        return []

    claimable: list[dict[str, Any]] = []
    for row in rows.data or []:
        worker_state = _extract_worker_state(row.get("run_config") or {})
        status = str(worker_state.get("status") or "").strip().lower()
        if status in {"queued", "claimed", "running", "finalizing"}:
            claimable.append(
                {
                    "job_id": row.get("job_id_legacy"),
                    "input_mode": row.get("input_mode"),
                    "use_web_search": row.get("use_web_search"),
                    "created_at": row.get("created_at"),
                    "run_config": _serialize(row.get("run_config") or {}),
                    "worker_state": worker_state,
                }
            )
        if len(claimable) >= limit:
            break
    return claimable


def claim_specter_worker_job(
    job_id_legacy: str,
    *,
    worker_id: str,
) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None
    try:
        rows = (
            client.table("jobs")
            .select("job_id_legacy, input_mode, use_web_search, created_at, run_config")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        ).data or []
        if not rows:
            return None
        row = rows[0]
        worker_state = _extract_worker_state(row.get("run_config") or {})
        status = str(worker_state.get("status") or "").strip().lower()
        if status not in {"queued", "claimed", "running"}:
            return None
        now = _utcnow().isoformat()
        updated_state = {
            "status": "running",
            "progress": worker_state.get("progress") or "Worker claimed job...",
            "worker_id": worker_id,
            "last_heartbeat_at": now,
            "run_started_at": worker_state.get("run_started_at") or now,
            "worker_service_enabled": True,
        }
        run_config = _merge_worker_state(row.get("run_config") or {}, updated_state)
        upsert_job(job_id_legacy, run_config=run_config)
        return {
            "job_id": row.get("job_id_legacy"),
            "input_mode": row.get("input_mode"),
            "use_web_search": row.get("use_web_search"),
            "created_at": row.get("created_at"),
            "run_config": run_config,
            "worker_state": _extract_worker_state(run_config),
        }
    except Exception as exc:
        _log_supabase_error("claim_specter_worker_job", "jobs", exc, job_id_legacy=job_id_legacy)
        return None


def heartbeat_specter_worker_job(
    job_id_legacy: str,
    *,
    progress: str | None = None,
    status: str | None = None,
    active_company_slug: str | None = None,
    active_company_index: int | None = None,
    completed_companies: int | None = None,
    failed_companies: int | None = None,
    total_companies: int | None = None,
    worker_id: str | None = None,
) -> bool:
    client = _get_client()
    if not client:
        return False
    try:
        rows = (
            client.table("jobs")
            .select("run_config")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        ).data or []
        if not rows:
            return False
        run_config = rows[0].get("run_config") or {}
        updates = {
            "status": status,
            "progress": progress,
            "active_company_slug": active_company_slug,
            "active_company_index": active_company_index,
            "completed_companies": completed_companies,
            "failed_companies": failed_companies,
            "total_companies": total_companies,
            "worker_id": worker_id,
            "last_heartbeat_at": _utcnow().isoformat(),
            "worker_service_enabled": True,
        }
        upsert_job(job_id_legacy, run_config=run_config, worker_state=updates)
        return True
    except Exception as exc:
        _log_supabase_error("heartbeat_specter_worker_job", "jobs", exc, job_id_legacy=job_id_legacy)
        return False


def finish_specter_worker_job(
    job_id_legacy: str,
    *,
    status: str,
    progress: str | None,
    completed_companies: int | None = None,
    failed_companies: int | None = None,
    total_companies: int | None = None,
) -> bool:
    client = _get_client()
    if not client:
        return False
    try:
        rows = (
            client.table("jobs")
            .select("run_config")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        ).data or []
        if not rows:
            return False
        now = _utcnow().isoformat()
        run_config = rows[0].get("run_config") or {}
        upsert_job(
            job_id_legacy,
            run_config=run_config,
            worker_state={
                "status": status,
                "progress": progress,
                "completed_companies": completed_companies,
                "failed_companies": failed_companies,
                "total_companies": total_companies,
                "active_company_slug": None,
                "active_company_index": None,
                "last_heartbeat_at": now,
                "run_finished_at": now,
                "worker_service_enabled": True,
            },
        )
        insert_job_status_history(job_id_legacy, status=status, progress=progress, source="worker_finish")
        return True
    except Exception as exc:
        _log_supabase_error("finish_specter_worker_job", "jobs", exc, job_id_legacy=job_id_legacy)
        return False


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
    except Exception as exc:
        _log_supabase_error("insert_analysis_event", "analysis_events", exc, job_id_legacy=job_id_legacy)


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
    except Exception as exc:
        _log_supabase_error("upsert_job_control", "job_controls", exc, job_id_legacy=job_id_legacy)


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
    except Exception as exc:
        _log_supabase_error(
            "insert_job_status_history",
            "job_status_history",
            exc,
            job_id_legacy=job_id_legacy,
        )


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
    except Exception as exc:
        _log_supabase_error("insert_analysis_error", "analysis_errors", exc, job_id_legacy=job_id_legacy)


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
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.table("person_profile_jobs").upsert(payload, on_conflict="person_job_id").execute()
    except Exception as exc:
        _log_supabase_error("upsert_person_profile_job", "person_profile_jobs", exc)


def load_person_profile_job(person_job_id: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None
    try:
        row = (
            client.table("person_profile_jobs")
            .select("person_job_id, status, progress, request_payload, result_payload, error")
            .eq("person_job_id", person_job_id)
            .limit(1)
            .execute()
        )
        if not row.data:
            return None
        return row.data[0]
    except Exception as exc:
        _log_supabase_error("load_person_profile_job", "person_profile_jobs", exc)
        return None


def load_latest_person_profile_job(company_slug: str, person_key: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None

    normalized_slug = (company_slug or "").strip()
    normalized_person_key = (person_key or "").strip()
    if not normalized_slug or not normalized_person_key:
        return None

    try:
        row = (
            client.table("person_profile_jobs")
            .select(
                "person_job_id, company_slug, person_key, status, progress, request_payload, "
                "result_payload, error, created_at, updated_at"
            )
            .eq("company_slug", normalized_slug)
            .eq("person_key", normalized_person_key)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not row.data:
            return None
        return row.data[0]
    except Exception as exc:
        _log_supabase_error("load_latest_person_profile_job", "person_profile_jobs", exc)
        return None


def prune_person_profile_jobs(company_slug: str, person_key: str, keep_person_job_id: str) -> bool:
    client = _get_client()
    if not client:
        return False

    normalized_slug = (company_slug or "").strip()
    normalized_person_key = (person_key or "").strip()
    normalized_keep_id = (keep_person_job_id or "").strip()
    if not normalized_slug or not normalized_person_key or not normalized_keep_id:
        return False

    try:
        (
            client.table("person_profile_jobs")
            .delete()
            .eq("company_slug", normalized_slug)
            .eq("person_key", normalized_person_key)
            .neq("person_job_id", normalized_keep_id)
            .execute()
        )
        return True
    except Exception as exc:
        _log_supabase_error("prune_person_profile_jobs", "person_profile_jobs", exc)
        return False


def persist_analysis(
    job_id_legacy: str,
    results_list: list[dict[str, Any]],
    results_payload: dict[str, Any],
    run_config: dict[str, Any],
    *,
    versions: dict[str, Any] | None = None,
    source_files: list[dict[str, Any]] | None = None,
    model_executions: list[dict[str, Any]] | None = None,
) -> bool:
    """Persist analysis results to Supabase.

    Creates/updates job, companies, pitch_decks, chunks, analyses.
    Stores results_payload for fast retrieval without reconstruction.
    Also stores source-file metadata and model execution telemetry.
    """
    client = _get_client()
    if not client:
        return False

    snapshot_payload = _serialize(results_payload or {})
    if not snapshot_payload:
        return False

    try:
        import time as _time
        _t0 = _time.monotonic()
        print(f"[persist_analysis] start job={job_id_legacy}")
        job_uuid = upsert_job(job_id_legacy, run_config=run_config, versions=versions)
        if not job_uuid:
            print(f"[persist_analysis] upsert_job returned None, aborting")
            return False
        print(f"[persist_analysis] upsert_job done ({_time.monotonic()-_t0:.1f}s)")

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
            except Exception as exc:
                _log_supabase_error(
                    "persist_analysis.insert_source_file",
                    "source_files",
                    exc,
                    job_id_legacy=job_id_legacy,
                )

        _EXEC_BATCH_SIZE = 200
        exec_rows_to_insert = [
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
                "metadata": _execution_metadata(exec_row),
            }
            for exec_row in (model_executions or [])
        ]
        print(f"[persist_analysis] inserting {len(exec_rows_to_insert)} model_executions in batches ({_time.monotonic()-_t0:.1f}s)")
        for i in range(0, len(exec_rows_to_insert), _EXEC_BATCH_SIZE):
            try:
                client.table("model_executions").insert(
                    exec_rows_to_insert[i : i + _EXEC_BATCH_SIZE]
                ).execute()
            except Exception as exc:
                _log_supabase_error(
                    "persist_analysis.insert_model_executions_batch",
                    "model_executions",
                    exc,
                    job_id_legacy=job_id_legacy,
                )
        print(f"[persist_analysis] model_executions done ({_time.monotonic()-_t0:.1f}s)")

        try:
            client.table("analyses").delete().eq("job_id_legacy", job_id_legacy).is_("company_id", "null").execute()
        except Exception as exc:
            _log_supabase_error(
                "persist_analysis.delete_snapshot_rows",
                "analyses",
                exc,
                job_id_legacy=job_id_legacy,
            )

        print(f"[persist_analysis] inserting analyses row, payload size={len(str(snapshot_payload))} chars ({_time.monotonic()-_t0:.1f}s)")
        client.table("analyses").insert(
            {
                "pitch_deck_id": None,
                "company_id": None,
                "job_id": job_uuid,
                "job_id_legacy": job_id_legacy,
                "state": {},
                "results_payload": snapshot_payload,
                "status": snapshot_payload.get("job_status") or "done",
                "run_config": _serialize(run_config),
                "excel_storage_path": None,
            }
        ).execute()

        print(f"[persist_analysis] analyses row inserted ({_time.monotonic()-_t0:.1f}s)")
        insert_job_status_history(
            job_id_legacy,
            status=snapshot_payload.get("job_status") or "done",
            progress=snapshot_payload.get("job_message") or "Analysis complete",
            source="persist_analysis",
        )
        upsert_job_control(
            job_id_legacy,
            pause_requested=False,
            stop_requested=(snapshot_payload.get("job_status") == "stopped"),
            last_action=snapshot_payload.get("job_status") or "done",
        )
        print(f"[persist_analysis] done total={_time.monotonic()-_t0:.1f}s")

        return True
    except Exception as exc:
        _log_supabase_error(
            "persist_analysis",
            "jobs,source_files,model_executions,analyses,job_status_history,job_controls",
            exc,
            job_id_legacy=job_id_legacy,
            include_traceback=True,
        )
        return False


def persist_analysis_snapshot(
    job_id_legacy: str,
    *,
    results_payload: dict[str, Any],
    run_config: dict[str, Any],
    versions: dict[str, Any] | None = None,
    worker_state: dict[str, Any] | None = None,
) -> bool:
    client = _get_client()
    if not client:
        return False

    snapshot_payload = _serialize(results_payload or {})
    if not snapshot_payload:
        return False

    try:
        job_uuid = upsert_job(
            job_id_legacy,
            run_config=run_config,
            versions=versions,
            worker_state=worker_state,
        )
        if not job_uuid:
            return False
        try:
            client.table("analyses").delete().eq("job_id_legacy", job_id_legacy).is_("company_id", "null").execute()
        except Exception as exc:
            _log_supabase_error(
                "persist_analysis_snapshot.delete_existing",
                "analyses",
                exc,
                job_id_legacy=job_id_legacy,
            )

        client.table("analyses").insert(
            {
                "pitch_deck_id": None,
                "company_id": None,
                "job_id": job_uuid,
                "job_id_legacy": job_id_legacy,
                "state": {},
                "results_payload": snapshot_payload,
                "status": snapshot_payload.get("job_status") or "done",
                "run_config": _serialize(run_config),
                "excel_storage_path": None,
            }
        ).execute()
        insert_job_status_history(
            job_id_legacy,
            status=snapshot_payload.get("job_status") or "done",
            progress=snapshot_payload.get("job_message") or "Analysis complete",
            source="persist_analysis_snapshot",
        )
        upsert_job_control(
            job_id_legacy,
            pause_requested=False,
            stop_requested=(snapshot_payload.get("job_status") == "stopped"),
            last_action=snapshot_payload.get("job_status") or "done",
        )
        return True
    except Exception as exc:
        _log_supabase_error(
            "persist_analysis_snapshot",
            "jobs,analyses,job_status_history,job_controls",
            exc,
            job_id_legacy=job_id_legacy,
            include_traceback=True,
        )
        return False


def persist_model_executions(
    job_id_legacy: str,
    model_executions: list[dict[str, Any]],
    *,
    run_config: dict[str, Any] | None = None,
    versions: dict[str, Any] | None = None,
) -> bool:
    """Persist model execution telemetry incrementally for long-running chunked jobs."""
    client = _get_client()
    if not client or not model_executions:
        return False

    try:
        job_uuid = upsert_job(job_id_legacy, run_config=run_config, versions=versions)
        if not job_uuid:
            return False

        rows = [
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
                "metadata": _execution_metadata(exec_row),
            }
            for exec_row in model_executions
        ]

        batch_size = 200
        for i in range(0, len(rows), batch_size):
            client.table("model_executions").insert(rows[i : i + batch_size]).execute()
        return True
    except Exception as exc:
        _log_supabase_error(
            "persist_model_executions",
            "jobs,model_executions",
            exc,
            job_id_legacy=job_id_legacy,
            include_traceback=True,
        )
        return False


def persist_company_result(
    *,
    job_id_legacy: str,
    result_row: dict[str, Any],
    company_payload: dict[str, Any],
    run_config: dict[str, Any],
    versions: dict[str, Any] | None = None,
) -> bool:
    """Persist one completed company immediately without waiting for the full batch."""
    client = _get_client()
    if not client or result_row.get("skipped"):
        return False

    try:
        job_uuid = upsert_job(job_id_legacy, run_config=run_config, versions=versions)
        if not job_uuid:
            return False

        return _persist_company_analysis_row(
            client,
            job_uuid=job_uuid,
            job_id_legacy=job_id_legacy,
            result_row=result_row,
            company_payload=company_payload,
            run_config=run_config,
            excel_storage_path=None,
            replace_documents=True,
        )
    except Exception as exc:
        _log_supabase_error(
            "persist_company_result",
            "jobs,companies,pitch_decks,chunks,analyses,company_runs",
            exc,
            job_id_legacy=job_id_legacy,
            include_traceback=True,
        )
        return False


def persist_company_failure_result(
    *,
    job_id_legacy: str,
    result_row: dict[str, Any],
    company_payload: dict[str, Any],
    run_config: dict[str, Any],
    versions: dict[str, Any] | None = None,
) -> bool:
    """Persist a timed-out or failed company so batch progress is not lost."""
    client = _get_client()
    if not client:
        return False

    try:
        job_uuid = upsert_job(job_id_legacy, run_config=run_config, versions=versions)
        if not job_uuid:
            return False

        return _persist_company_analysis_row(
            client,
            job_uuid=job_uuid,
            job_id_legacy=job_id_legacy,
            result_row=result_row,
            company_payload=company_payload,
            run_config=run_config,
            excel_storage_path=None,
            replace_documents=True,
        )
    except Exception as exc:
        _log_supabase_error(
            "persist_company_failure_result",
            "jobs,companies,pitch_decks,chunks,analyses,company_runs",
            exc,
            job_id_legacy=job_id_legacy,
            include_traceback=True,
        )
        return False


def _load_latest_analysis_snapshot(
    client: Client,
    job_id_legacy: str,
) -> dict[str, Any] | None:
    try:
        analyses = (
            client.table("analyses")
            .select("results_payload, status, created_at")
            .eq("job_id_legacy", job_id_legacy)
            .is_("company_id", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if analyses.data:
            return analyses.data[0]
    except Exception as exc:
        _log_supabase_error(
            "load_latest_analysis_snapshot",
            "analyses",
            exc,
            job_id_legacy=job_id_legacy,
        )
        return None
    return None


def _load_company_run_rows_for_job(client: Client, job_id_legacy: str) -> list[dict[str, Any]]:
    try:
        rows = (
            client.table("company_runs")
            .select(
                "company_key, company_name, startup_slug, job_id_legacy, decision, total_score, "
                "composite_score, bucket, mode, input_order, run_created_at, created_at, result_payload"
            )
            .eq("job_id_legacy", job_id_legacy)
            .limit(500)
            .execute()
        )
        return rows.data or []
    except Exception as exc:
        _log_supabase_error("load_company_run_rows_for_job", "company_runs", exc, job_id_legacy=job_id_legacy)
        return []


def _load_company_progress_rows_for_job(client: Client, job_id_legacy: str) -> list[dict[str, Any]]:
    try:
        rows = (
            client.table("company_runs")
            .select("decision")
            .eq("job_id_legacy", job_id_legacy)
            .limit(500)
            .execute()
        )
        return rows.data or []
    except Exception as exc:
        _log_supabase_error("load_company_progress_rows_for_job", "company_runs", exc, job_id_legacy=job_id_legacy)
        return []


def _can_reconstruct_job_results(
    client: Client,
    *,
    job_id_legacy: str,
    preferred_mode: str | None = None,
    snapshot_payload: dict[str, Any] | None = None,
) -> bool:
    try:
        payload = _compose_results_payload_from_company_runs(
            _load_company_run_rows_for_job(client, job_id_legacy),
            preferred_mode=preferred_mode,
            snapshot_payload=snapshot_payload,
        )
        return isinstance(payload, dict) and bool(payload)
    except Exception as exc:
        _log_supabase_error(
            "can_reconstruct_job_results",
            "company_runs,analyses",
            exc,
            job_id_legacy=job_id_legacy,
        )
        return False


def load_analysis_events(job_id_legacy: str, limit: int = 200) -> list[str]:
    client = _get_client()
    if not client:
        return []
    try:
        rows = (
            client.table("analysis_events")
            .select("message, created_at")
            .eq("job_id_legacy", job_id_legacy)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        messages = [str(row.get("message") or "") for row in rows.data or [] if str(row.get("message") or "").strip()]
        messages.reverse()
        return messages
    except Exception as exc:
        _log_supabase_error("load_analysis_events", "analysis_events", exc, job_id_legacy=job_id_legacy)
        return []


def load_job_company_runs(job_id_legacy: str) -> list[dict[str, Any]]:
    """Return persisted company-run rows for a job."""
    client = _get_client()
    if not client:
        return []
    return _load_company_run_rows_for_job(client, job_id_legacy)


def load_job_progress_snapshot(
    job_id_legacy: str,
    *,
    preferred_mode: str | None = None,
) -> dict[str, Any] | None:
    """Return a lightweight progress snapshot without reconstructing the full report."""
    client = _get_client()
    if not client:
        return None

    try:
        snapshot_row = _load_latest_analysis_snapshot(client, job_id_legacy) or {}
        snapshot_payload = _serialize(snapshot_row.get("results_payload") or {})
        progress_rows = _load_company_progress_rows_for_job(client, job_id_legacy)
        total_rows = len(progress_rows)
        failed_rows = sum(
            1
            for row in progress_rows
            if str(row.get("decision") or "").strip().lower() in {"error", "timeout"}
        )
        completed_rows = max(0, total_rows - failed_rows)
        mode = preferred_mode or snapshot_payload.get("mode")
        if not mode:
            mode = "single" if total_rows == 1 else "batch"

        payload = {
            "mode": mode,
            "num_companies": completed_rows,
            "num_skipped": failed_rows,
            "summary_rows_count": completed_rows,
            "failed_rows_count": failed_rows,
            "job_status": snapshot_payload.get("job_status"),
            "job_message": snapshot_payload.get("job_message"),
        }
        if mode == "single":
            for key in (
                "company_name",
                "startup_slug",
                "decision",
                "total_score",
                "avg_pro",
                "avg_contra",
            ):
                if key in snapshot_payload:
                    payload[key] = snapshot_payload.get(key)
        return {"results": payload}
    except Exception as exc:
        _log_supabase_error("load_job_progress_snapshot", "analyses,company_runs", exc, job_id_legacy=job_id_legacy)
        return None


def load_run_costs(job_id_legacy: str) -> dict[str, Any] | None:
    client = _get_client()
    if not client:
        return None

    try:
        batch_size = 1000
        start = 0
        rows: list[dict[str, Any]] = []
        while True:
            resp = (
                client.table("model_executions")
                .select(
                    "provider, model, status, prompt_tokens, completion_tokens, "
                    "total_tokens, metadata"
                )
                .eq("job_id_legacy", job_id_legacy)
                .range(start, start + batch_size - 1)
                .execute()
            )
            batch = [_hydrate_execution_row(row) for row in list(resp.data or [])]
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < batch_size:
                break
            start += batch_size

        return build_run_costs_from_model_executions(rows)
    except Exception as exc:
        _log_supabase_error("load_run_costs", "model_executions", exc, job_id_legacy=job_id_legacy)
        return None


def load_job_results(
    job_id_legacy: str,
    *,
    preferred_mode: str | None = None,
) -> dict[str, Any] | None:
    """Load and reconstruct results for a job from Supabase."""
    client = _get_client()
    if not client:
        return None

    try:
        snapshot_row = _load_latest_analysis_snapshot(client, job_id_legacy) or {}
        company_rows = _load_company_run_rows_for_job(client, job_id_legacy)
        results = _compose_results_payload_from_company_runs(
            company_rows,
            preferred_mode=preferred_mode,
            snapshot_payload=snapshot_row.get("results_payload") if isinstance(snapshot_row, dict) else None,
        )
        if results is None:
            return None
        run_costs = load_run_costs(job_id_legacy)
        if isinstance(run_costs, dict) and (
            run_costs.get("status") != "unavailable" or "run_costs" not in results
        ):
            results["run_costs"] = run_costs
        return {"results": results}
    except Exception as exc:
        _log_supabase_error("load_job_results", "analyses,company_runs,model_executions", exc, job_id_legacy=job_id_legacy)
        return None


def load_job_status(job_id_legacy: str) -> dict[str, Any] | None:
    """Load the latest persisted job status, preferring terminal saved analyses."""
    client = _get_client()
    if not client:
        return None

    latest_status: dict[str, Any] = {}
    job_row: dict[str, Any] = {}
    worker_state: dict[str, Any] = {}
    try:
        job_resp = (
            client.table("jobs")
            .select("run_config")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        )
        job_row = (job_resp.data or [{}])[0] or {}
        worker_state = _extract_worker_state(job_row.get("run_config") or {})
    except Exception as exc:
        _log_supabase_error("load_job_status.jobs", "jobs", exc, job_id_legacy=job_id_legacy)
        job_row = {}
        worker_state = {}
    try:
        status_resp = (
            client.table("job_status_history")
            .select("status, progress, created_at")
            .eq("job_id_legacy", job_id_legacy)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        latest_status = (status_resp.data or [{}])[0] or {}
    except Exception as exc:
        _log_supabase_error("load_job_status", "job_status_history", exc, job_id_legacy=job_id_legacy)
        latest_status = {}

    latest_analysis = _load_latest_analysis_snapshot(client, job_id_legacy) or {}
    worker_status, worker_progress, worker_active = _resolved_worker_state(worker_state)
    latest_status_ts = _parse_timestamp(latest_status.get("created_at"))
    latest_analysis_ts = _parse_timestamp(latest_analysis.get("created_at"))
    worker_terminal_ts = _worker_state_terminal_timestamp(worker_state)
    if not latest_status and not latest_analysis and not worker_status:
        return None

    analysis_status = str(latest_analysis.get("status") or "").strip().lower()
    status = str(latest_status.get("status") or analysis_status or worker_status or "pending").strip().lower()
    progress = latest_status.get("progress") or worker_progress
    snapshot_payload = latest_analysis.get("results_payload")
    if isinstance(snapshot_payload, dict) and not progress:
        progress = snapshot_payload.get("job_message")

    if worker_active and analysis_status not in {"done", "error", "stopped"}:
        status = worker_status if worker_status != "claimed" else "running"
        progress = worker_progress or progress
    elif analysis_status in {"done", "error", "stopped"} and (
        latest_status_ts is None or (latest_analysis_ts is not None and latest_analysis_ts >= latest_status_ts)
    ):
        status = analysis_status
        if isinstance(snapshot_payload, dict):
            progress = snapshot_payload.get("job_message") or progress
    elif worker_status in WORKER_TERMINAL_STATUSES and (
        latest_status_ts is None or (worker_terminal_ts is not None and worker_terminal_ts >= latest_status_ts)
    ):
        status = worker_status
        progress = worker_progress or progress
    elif worker_status == "interrupted" and analysis_status not in {"done", "error", "stopped"}:
        status = "interrupted"
        progress = worker_progress or progress
    elif worker_status in WORKER_TERMINAL_STATUSES and status not in {"done", "error", "stopped"}:
        status = worker_status
        progress = worker_progress or progress
    elif analysis_status in {"done", "error", "stopped"} and status not in {"done", "error", "stopped"}:
        status = analysis_status
        if isinstance(snapshot_payload, dict):
            progress = snapshot_payload.get("job_message") or progress

    return {
        "status": status or "pending",
        "progress": progress,
        "worker_active": worker_active,
    }


def load_saved_job(job_id_legacy: str) -> dict[str, Any] | None:
    """Load a single saved job row merged with latest status/analysis metadata."""
    for row in list_saved_jobs(limit=500):
        if row.get("job_id") == job_id_legacy:
            return row
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
            .select("job_id_legacy, results_payload, created_at")
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
            rebuilt = load_job_results(jid)
            if not rebuilt or rebuilt.get("results") is None:
                continue
            seen.add(jid)
            out[jid] = {
                "results": rebuilt.get("results"),
            }
    except Exception as exc:
        _log_supabase_error("load_all_completed_jobs", "analyses", exc)
    return out


def list_saved_jobs(limit: int = 200) -> list[dict[str, Any]]:
    """Return saved jobs with latest known status and optional results payload."""
    client = _get_client()
    if not client:
        return []

    try:
        jobs_resp = (
            client.table("jobs")
            .select(
                "job_id_legacy, input_mode, use_web_search, created_at, run_config, "
                "started_by_user_id, started_by_email, started_by_display_name, started_by_label"
            )
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
            .limit(max(limit * 20, 1000))
            .execute()
        )
        latest_status_by_job: dict[str, dict[str, Any]] = {}
        for row in status_rows.data or []:
            jid = row.get("job_id_legacy")
            if jid and jid not in latest_status_by_job:
                latest_status_by_job[jid] = row

        analysis_rows = (
            client.table("analyses")
            .select("job_id_legacy, status, results_payload, created_at")
            .in_("job_id_legacy", job_ids)
            .is_("company_id", "null")
            .order("created_at", desc=True)
            .limit(max(limit * 4, 100))
            .execute()
        )
        latest_analysis_by_job: dict[str, dict[str, Any]] = {}
        for row in analysis_rows.data or []:
            jid = row.get("job_id_legacy")
            if jid and jid not in latest_analysis_by_job:
                latest_analysis_by_job[jid] = row

        reconstructable_job_ids: set[str] = set()
        reconstructable_probe_by_job: dict[str, bool] = {}
        terminal_without_snapshot_candidates: list[str] = []
        for row in jobs:
            jid = row.get("job_id_legacy")
            if not jid:
                continue
            latest_status = latest_status_by_job.get(jid, {})
            latest_analysis = latest_analysis_by_job.get(jid, {})
            analysis_status = str(latest_analysis.get("status") or "").strip().lower()
            status = str(latest_status.get("status") or analysis_status or "").strip().lower()
            snapshot_payload = latest_analysis.get("results_payload")
            if status in {"done", "error", "stopped"} and not (isinstance(snapshot_payload, dict) and snapshot_payload):
                terminal_without_snapshot_candidates.append(jid)

        if terminal_without_snapshot_candidates:
            try:
                company_rows = (
                    client.table("company_runs")
                    .select("job_id_legacy")
                    .in_("job_id_legacy", terminal_without_snapshot_candidates)
                    .limit(max(len(terminal_without_snapshot_candidates) * 8, 100))
                    .execute()
                )
                reconstructable_job_ids = {
                    row.get("job_id_legacy")
                    for row in (company_rows.data or [])
                    if row.get("job_id_legacy")
                }
            except Exception as exc:
                _log_supabase_error(
                    "list_saved_jobs.company_runs",
                    "company_runs",
                    exc,
                )
                reconstructable_job_ids = set()

        for row in jobs:
            jid = row.get("job_id_legacy")
            if not jid or jid in reconstructable_job_ids:
                continue
            latest_status = latest_status_by_job.get(jid, {})
            latest_analysis = latest_analysis_by_job.get(jid, {})
            analysis_status = str(latest_analysis.get("status") or "").strip().lower()
            status = str(latest_status.get("status") or analysis_status or "").strip().lower()
            snapshot_payload = latest_analysis.get("results_payload")
            if status not in {"done", "error", "stopped"}:
                continue
            if isinstance(snapshot_payload, dict) and snapshot_payload:
                continue
            reconstructable_probe_by_job[jid] = _can_reconstruct_job_results(
                client,
                job_id_legacy=jid,
                preferred_mode=row.get("input_mode"),
                snapshot_payload=snapshot_payload if isinstance(snapshot_payload, dict) else None,
            )

        out: list[dict[str, Any]] = []
        for row in jobs:
            jid = row.get("job_id_legacy")
            if not jid:
                continue

            latest_status = latest_status_by_job.get(jid, {})
            latest_analysis = latest_analysis_by_job.get(jid, {})
            run_config = row.get("run_config") or {}
            worker_state = _extract_worker_state(run_config)
            worker_status, worker_progress, worker_active = _resolved_worker_state(worker_state)
            latest_status_ts = _parse_timestamp(latest_status.get("created_at"))
            latest_analysis_ts = _parse_timestamp(latest_analysis.get("created_at"))
            worker_terminal_ts = _worker_state_terminal_timestamp(worker_state)
            analysis_status = str(latest_analysis.get("status") or "").strip().lower()
            status = latest_status.get("status") or analysis_status or worker_status or "pending"
            progress = latest_status.get("progress") or worker_progress
            snapshot_payload = latest_analysis.get("results_payload")
            if worker_active and analysis_status not in {"done", "error", "stopped"}:
                status = worker_status if worker_status != "claimed" else "running"
                progress = worker_progress or progress
            elif analysis_status in {"done", "error", "stopped"} and (
                latest_status_ts is None or (latest_analysis_ts is not None and latest_analysis_ts >= latest_status_ts)
            ):
                status = analysis_status
                if isinstance(snapshot_payload, dict):
                    progress = snapshot_payload.get("job_message") or progress
            elif worker_status in WORKER_TERMINAL_STATUSES and (
                latest_status_ts is None or (worker_terminal_ts is not None and worker_terminal_ts >= latest_status_ts)
            ):
                status = worker_status
                progress = worker_progress or progress
            elif worker_status == "interrupted" and analysis_status not in {"done", "error", "stopped"}:
                status = "interrupted"
                progress = worker_progress or progress
            elif worker_status in WORKER_TERMINAL_STATUSES and str(status).strip().lower() not in {"done", "error", "stopped"}:
                status = worker_status
                progress = worker_progress or progress
            elif analysis_status in {"done", "error", "stopped"} and str(status).strip().lower() not in {"done", "error", "stopped"}:
                status = analysis_status
                if isinstance(snapshot_payload, dict):
                    progress = snapshot_payload.get("job_message") or progress
            created_at = latest_analysis.get("created_at") or row.get("created_at")
            has_results = (
                bool(isinstance(snapshot_payload, dict) and snapshot_payload)
                or jid in reconstructable_job_ids
                or reconstructable_probe_by_job.get(jid) is True
            )


            out.append(
                {
                    "job_id": jid,
                    "status": status,
                    "progress": progress,
                    "created_at": created_at,
                    "input_mode": row.get("input_mode"),
                    "use_web_search": row.get("use_web_search"),
                    "run_name": run_config.get("run_name") if isinstance(run_config, dict) else None,
                    "started_by_user_id": row.get("started_by_user_id") or run_config.get("started_by_user_id"),
                    "started_by_email": row.get("started_by_email") or run_config.get("started_by_email"),
                    "started_by_display_name": row.get("started_by_display_name") or run_config.get("started_by_display_name"),
                    "started_by_label": row.get("started_by_label") or run_config.get("started_by_label"),
                    "run_config": run_config,
                    "results": None,
                    "has_results": has_results,
                    "worker_active": worker_active,
                }
            )

        return out
    except Exception as exc:
        _log_supabase_error("list_saved_jobs", "jobs,job_status_history,analyses", exc)
        return []


def _fetch_company_run_rows(client: Client, limit_runs: int) -> list[dict[str, Any]]:
    select_clause = (
        "company_key, company_name, startup_slug, job_id_legacy, decision, total_score, "
        "composite_score, bucket, mode, input_order, run_created_at, created_at, result_payload, "
        "started_by_user_id, started_by_email, started_by_display_name, started_by_label"
    )
    limits_to_try: list[int] = []
    for candidate in (limit_runs, min(limit_runs, 250), min(limit_runs, 100), 50):
        if candidate > 0 and candidate not in limits_to_try:
            limits_to_try.append(candidate)

    last_error: Exception | None = None
    for current_limit in limits_to_try:
        try:
            rows = (
                client.table("company_runs")
                .select(select_clause)
                .order("run_created_at", desc=True)
                .limit(current_limit)
                .execute()
            )
            return rows.data or []
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    return []


def _fetch_chat_company_run_rows(client: Client, limit_runs: int = 2000) -> list[dict[str, Any]]:
    try:
        rows = (
            client.table("company_runs")
            .select(
                "company_id, company_key, company_name, startup_slug, job_id_legacy, decision, "
                "total_score, composite_score, bucket, mode, input_order, run_created_at, created_at, result_payload"
            )
            .order("run_created_at", desc=True)
            .limit(limit_runs)
            .execute()
        )
        return rows.data or []
    except Exception:
        return []


def _fetch_analyses_for_company_ids(
    client: Client,
    company_ids: list[str],
) -> list[dict[str, Any]]:
    if not company_ids:
        return []
    try:
        rows = (
            client.table("analyses")
            .select("company_id, pitch_deck_id, job_id_legacy, created_at")
            .in_("company_id", company_ids)
            .order("created_at", desc=True)
            .limit(2000)
            .execute()
        )
        return rows.data or []
    except Exception:
        return []


def _fetch_chunks_for_pitch_deck_ids(
    client: Client,
    pitch_deck_ids: list[str],
) -> list[dict[str, Any]]:
    if not pitch_deck_ids:
        return []
    try:
        rows = (
            client.table("chunks")
            .select("pitch_deck_id, chunk_id, text, source_file, page_or_slide, sort_order, created_at")
            .in_("pitch_deck_id", pitch_deck_ids)
            .order("sort_order")
            .limit(10000)
            .execute()
        )
        return rows.data or []
    except Exception:
        return []


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
            .select(
                "job_id_legacy, input_mode, started_by_user_id, started_by_email, "
                "started_by_display_name, started_by_label"
            )
            .in_("job_id_legacy", missing_job_ids)
            .limit(len(missing_job_ids))
            .execute()
        )
        job_meta_by_job_id = {
            row.get("job_id_legacy"): row
            for row in (jobs.data or [])
            if row.get("job_id_legacy")
        }

        for row in missing_analyses:
            job_id_legacy = row.get("job_id_legacy")
            payload = row.get("results_payload")
            if not job_id_legacy or not payload:
                continue
            for run in _extract_company_runs_from_payload(
                job_id_legacy,
                payload,
                created_at=row.get("created_at"),
                mode=(job_meta_by_job_id.get(job_id_legacy) or {}).get("input_mode"),
                started_by=job_meta_by_job_id.get(job_id_legacy),
            ):
                try:
                    client.table("company_runs").upsert(
                        run,
                        on_conflict="job_id_legacy,company_key",
                    ).execute()
                    inserted += 1
                except Exception as exc:
                    _log_supabase_error(
                        "reconcile_missing_company_runs.upsert",
                        "company_runs",
                        exc,
                        job_id_legacy=job_id_legacy,
                    )
    except Exception as exc:
        _log_supabase_error("reconcile_missing_company_runs", "analyses,jobs,company_runs", exc)
        return inserted

    return inserted


def list_company_histories(
    limit_runs: int = 1000,
    *,
    perform_maintenance: bool = True,
) -> list[dict[str, Any]]:
    """Return grouped company histories with per-run records."""
    client = _get_client()
    if not client:
        return []

    try:
        rows_data = _fetch_company_run_rows(client, limit_runs)
        if perform_maintenance:
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
                "started_by_user_id": row.get("started_by_user_id"),
                "started_by_email": row.get("started_by_email"),
                "started_by_display_name": row.get("started_by_display_name"),
                "started_by_label": row.get("started_by_label"),
                "results": _compact_company_run_payload(row.get("result_payload")),
            }
            entry = grouped.setdefault(
                group_key,
                {
                    "company_key": company_key,
                    "company_lookup_key": group_key,
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
    except Exception as exc:
        _log_supabase_error("list_company_histories", "company_runs,analyses,jobs", exc)
        return []


def _compact_company_run_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    serialized = _serialize(payload or {})
    summary_row = ((serialized.get("summary_rows") or [{}]) or [{}])[0]
    ranking = _serialize(serialized.get("ranking_result") or {})
    return {
        "mode": serialized.get("mode"),
        "source_mode": serialized.get("source_mode"),
        "startup_slug": serialized.get("startup_slug") or summary_row.get("startup_slug"),
        "company_name": serialized.get("company_name") or summary_row.get("company_name"),
        "decision": serialized.get("decision") or summary_row.get("decision"),
        "total_score": serialized.get("total_score") or summary_row.get("total_score"),
        "avg_pro": serialized.get("avg_pro"),
        "avg_contra": serialized.get("avg_contra"),
        "summary_rows": [summary_row] if summary_row else [],
        "qa_provenance_rows": _serialize(serialized.get("qa_provenance_rows") or []),
        "argument_rows": _serialize(serialized.get("argument_rows") or []),
        "founders": serialized.get("founders") or summary_row.get("founders") or [],
        "team_members": serialized.get("team_members") or summary_row.get("team_members") or [],
        "ranking_result": {
            "rank": ranking.get("rank") or summary_row.get("rank"),
            "percentile": ranking.get("percentile") or summary_row.get("percentile"),
            "composite_score": ranking.get("composite_score") or summary_row.get("composite_score"),
            "strategy_fit_score": ranking.get("strategy_fit_score") or summary_row.get("strategy_fit_score"),
            "team_score": ranking.get("team_score") or summary_row.get("team_score"),
            "upside_score": ranking.get("upside_score") or summary_row.get("upside_score"),
            "bucket": ranking.get("bucket") or summary_row.get("bucket"),
            "strategy_fit_summary": ranking.get("strategy_fit_summary") or summary_row.get("strategy_fit_summary"),
            "team_summary": ranking.get("team_summary") or summary_row.get("team_summary"),
            "potential_summary": ranking.get("potential_summary") or summary_row.get("potential_summary"),
            "key_points": ranking.get("key_points") or summary_row.get("key_points"),
            "red_flags": ranking.get("red_flags") or summary_row.get("red_flags"),
        },
    }


def load_company_chat_context(company_lookup_key: str) -> dict[str, Any] | None:
    """Load all persisted runs and evidence for one grouped company identity."""
    client = _get_client()
    if not client:
        return None

    normalized_lookup_key = (company_lookup_key or "").strip().lower()
    if not normalized_lookup_key:
        return None

    rows = _fetch_chat_company_run_rows(client, limit_runs=2000)
    matching_rows = [
        row for row in rows
        if _company_history_group_key(row) == normalized_lookup_key
    ]
    if not matching_rows:
        return None

    company_ids = list({
        row.get("company_id")
        for row in matching_rows
        if row.get("company_id")
    })
    analyses = _fetch_analyses_for_company_ids(client, company_ids)
    pitch_deck_ids = list({
        row.get("pitch_deck_id")
        for row in analyses
        if row.get("pitch_deck_id")
    })
    chunks = _fetch_chunks_for_pitch_deck_ids(client, pitch_deck_ids)

    analyses_by_pitch_deck: dict[str, dict[str, Any]] = {}
    for row in analyses:
        pitch_deck_id = row.get("pitch_deck_id")
        if not pitch_deck_id:
            continue
        existing = analyses_by_pitch_deck.get(pitch_deck_id)
        current_ts = str(row.get("created_at") or "")
        if existing and str(existing.get("created_at") or "") >= current_ts:
            continue
        analyses_by_pitch_deck[pitch_deck_id] = row

    chunks_by_job: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        analysis_row = analyses_by_pitch_deck.get(chunk.get("pitch_deck_id"))
        if not analysis_row:
            continue
        job_id = analysis_row.get("job_id_legacy")
        if not job_id:
            continue
        row_payload = {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "source_file": chunk.get("source_file"),
            "page_or_slide": chunk.get("page_or_slide"),
            "pitch_deck_id": chunk.get("pitch_deck_id"),
        }
        chunks_by_job.setdefault(job_id, []).append(row_payload)

    runs: list[dict[str, Any]] = []
    latest_name = None
    for row in sorted(
        matching_rows,
        key=lambda item: str(item.get("run_created_at") or item.get("created_at") or ""),
        reverse=True,
    ):
        result_payload = _serialize(row.get("result_payload") or {})
        latest_name = latest_name or row.get("company_name") or result_payload.get("company_name")
        runs.append(
            {
                "job_id": row.get("job_id_legacy"),
                "company_id": row.get("company_id"),
                "company_name": row.get("company_name") or result_payload.get("company_name"),
                "startup_slug": row.get("startup_slug") or result_payload.get("startup_slug"),
                "decision": row.get("decision"),
                "created_at": row.get("run_created_at") or row.get("created_at"),
                "results": result_payload,
                "chunks": chunks_by_job.get(row.get("job_id_legacy"), []),
            }
        )

    latest_result = (runs[0].get("results") if runs else {}) or {}
    return {
        "company_lookup_key": normalized_lookup_key,
        "company_name": latest_name or "Unknown company",
        "domain": latest_result.get("domain"),
        "runs": runs,
    }


def load_company_chat_session(company_lookup_key: str) -> dict[str, Any] | None:
    """Load the shared persisted chat session for a company."""
    client = _get_client()
    if not client:
        return None

    normalized_lookup_key = (company_lookup_key or "").strip().lower()
    if not normalized_lookup_key:
        return None

    try:
        rows = (
            client.table("company_chat_sessions")
            .select("company_key, company_name, selection, summary, transcript, model_executions, created_at, updated_at")
            .eq("company_key", normalized_lookup_key)
            .limit(1)
            .execute()
        )
        if not rows.data:
            return None
        row = rows.data[0] or {}
        return {
            "company_lookup_key": row.get("company_key") or normalized_lookup_key,
            "company_name": row.get("company_name"),
            "selection": _serialize(row.get("selection") or {}),
            "summary": str(row.get("summary") or ""),
            "transcript": _serialize(row.get("transcript") or []),
            "model_executions": _serialize(row.get("model_executions") or []),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }
    except Exception as exc:
        _log_supabase_error("load_company_chat_session", "company_chat_sessions", exc)
        return None


def persist_company_chat_session(
    *,
    company_lookup_key: str,
    company_name: str | None,
    selection: dict[str, Any] | None,
    summary: str,
    transcript: list[dict[str, Any]],
    model_executions: list[dict[str, Any]] | None = None,
) -> bool:
    """Persist the shared chat transcript for a company."""
    client = _get_client()
    if not client:
        return False

    normalized_lookup_key = (company_lookup_key or "").strip().lower()
    if not normalized_lookup_key:
        return False

    try:
        client.table("company_chat_sessions").upsert(
            {
                "company_key": normalized_lookup_key,
                "company_name": company_name or normalized_lookup_key,
                "selection": _serialize(selection or {}),
                "summary": str(summary or ""),
                "transcript": _serialize(transcript or []),
                "model_executions": _serialize(model_executions or []),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="company_key",
        ).execute()
        return True
    except Exception as exc:
        _log_supabase_error("persist_company_chat_session", "company_chat_sessions", exc)
        return False


def delete_company_chat_session(company_lookup_key: str) -> bool:
    """Delete the shared persisted chat transcript for a company."""
    client = _get_client()
    if not client:
        return False

    normalized_lookup_key = (company_lookup_key or "").strip().lower()
    if not normalized_lookup_key:
        return False

    try:
        client.table("company_chat_sessions").delete().eq("company_key", normalized_lookup_key).execute()
        return True
    except Exception as exc:
        _log_supabase_error("delete_company_chat_session", "company_chat_sessions", exc)
        return False


def load_app_setting(key: str) -> str | None:
    """Load a shared app setting value by key."""
    client = _get_client()
    if not client:
        return None

    normalized_key = (key or "").strip()
    if not normalized_key:
        return None

    try:
        rows = (
            client.table("app_settings")
            .select("setting_key, value_text")
            .eq("setting_key", normalized_key)
            .limit(1)
            .execute()
        )
        if not rows.data:
            return ""
        row = rows.data[0] or {}
        return str(row.get("value_text") or "")
    except Exception as exc:
        _log_supabase_error("load_app_setting", "app_settings", exc)
        return None


def save_app_setting(key: str, value: str | None) -> bool:
    """Persist a shared app setting value by key."""
    client = _get_client()
    if not client:
        return False

    normalized_key = (key or "").strip()
    if not normalized_key:
        return False

    try:
        client.table("app_settings").upsert(
            {
                "setting_key": normalized_key,
                "value_text": str(value or ""),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="setting_key",
        ).execute()
        return True
    except Exception as exc:
        _log_supabase_error("save_app_setting", "app_settings", exc)
        return False


def _extract_company_runs_from_payload(
    job_id_legacy: str,
    payload: dict[str, Any],
    *,
    created_at: str | None,
    mode: str | None,
    started_by: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    serialized = _serialize(payload or {})
    payload_mode = serialized.get("mode")
    started_by_fields = _extract_started_by_fields(started_by)
    if payload_mode == "single":
        company_name = serialized.get("company_name") or serialized.get("startup_slug") or job_id_legacy
        company_key = _normalize_company_key(company_name, None, serialized.get("startup_slug"))
        ranking = _serialize(serialized.get("ranking_result") or {})
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
            "result_payload": _compact_company_run_payload(serialized),
            **started_by_fields,
        }]

    rows: list[dict[str, Any]] = []
    for summary_row in serialized.get("summary_rows") or []:
        company_name = summary_row.get("company_name") or summary_row.get("startup_slug") or job_id_legacy
        startup_slug = summary_row.get("startup_slug")
        company_key = _normalize_company_key(company_name, None, startup_slug)
        row_payload = _compact_company_run_payload(
            {
                "mode": "single",
                "source_mode": payload_mode or "batch",
                "startup_slug": startup_slug,
                "company_name": company_name,
                "decision": summary_row.get("decision"),
                "total_score": summary_row.get("total_score"),
                "avg_pro": summary_row.get("avg_pro"),
                "avg_contra": summary_row.get("avg_contra"),
                "summary_rows": [summary_row],
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
        )
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
            **started_by_fields,
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
    except Exception as exc:
        _log_supabase_error("backfill_company_runs_from_analyses.check_existing", "company_runs", exc)
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
            .select(
                "job_id_legacy, input_mode, started_by_user_id, started_by_email, "
                "started_by_display_name, started_by_label"
            )
            .limit(limit_jobs)
            .execute()
        )
        job_meta_by_job_id = {
            row.get("job_id_legacy"): row
            for row in (jobs.data or [])
            if row.get("job_id_legacy")
        }
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
                mode=(job_meta_by_job_id.get(job_id_legacy) or {}).get("input_mode"),
                started_by=job_meta_by_job_id.get(job_id_legacy),
            ):
                try:
                    client.table("company_runs").upsert(
                        run,
                        on_conflict="job_id_legacy,company_key",
                    ).execute()
                    inserted += 1
                except Exception as exc:
                    _log_supabase_error(
                        "backfill_company_runs_from_analyses.upsert",
                        "company_runs",
                        exc,
                        job_id_legacy=job_id_legacy,
                    )
    except Exception as exc:
        _log_supabase_error("backfill_company_runs_from_analyses", "analyses,jobs,company_runs", exc)
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
            .select(
                "job_id_legacy, input_mode, started_by_user_id, started_by_email, "
                "started_by_display_name, started_by_label"
            )
            .limit(limit_jobs)
            .execute()
        )
        job_meta_by_job_id = {
            row.get("job_id_legacy"): row
            for row in (jobs.data or [])
            if row.get("job_id_legacy")
        }
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
                mode=(job_meta_by_job_id.get(job_id_legacy) or {}).get("input_mode"),
                started_by=job_meta_by_job_id.get(job_id_legacy),
            ):
                try:
                    client.table("company_runs").upsert(
                        run,
                        on_conflict="job_id_legacy,company_key",
                    ).execute()
                    inserted += 1
                except Exception as exc:
                    _log_supabase_error(
                        "backfill_all_company_runs_from_analyses.upsert",
                        "company_runs",
                        exc,
                        job_id_legacy=job_id_legacy,
                    )
    except Exception as exc:
        _log_supabase_error("backfill_all_company_runs_from_analyses", "analyses,jobs,company_runs", exc)
        return inserted

    return inserted


def download_excel_bytes(job_id_legacy: str) -> bytes | None:
    """Excel export has been removed."""
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
    except Exception as exc:
        _log_supabase_error("load_analyses_by_company.company_runs", "company_runs", exc)

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
    except Exception as exc:
        _log_supabase_error("load_analyses_by_company", "companies,analyses", exc)
    return out


def ensure_excel_bucket() -> None:
    """Excel export has been removed."""
    return None
