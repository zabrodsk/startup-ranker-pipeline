"""Rockaway Deal Intelligence web application.

FastAPI backend serving the Rockaway-branded UI with password protection.
Provider API keys stay server-side only and are never exposed to the client.
"""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.batch import (
    build_argument_rows,
    build_evidence_rows,
    build_failed_rows,
    build_qa_provenance_rows,
    build_summary_rows,
    evaluate_from_specter,
    evaluate_startup,
    rank_batch_companies,
)
from agent.llm_catalog import (
    available_models_payload,
    current_default_selection,
    model_label,
    pricing_catalog_payload,
    serialize_selection,
    validate_requested_selection,
)
from agent.llm_policy import (
    build_phase_model_policy,
    build_phase_policy_display_label,
    build_pipeline_policy,
    build_tier_display_label,
    coerce_phase_models_payload,
    normalize_phase_models,
    normalize_premium_phase_models,
    normalize_quality_tier,
    phase_model_defaults_payload,
    premium_phase_options_payload,
    quality_tiers_payload,
    resolve_effective_phase_choices,
    resolve_effective_phase_models,
)
from agent.run_context import RunTelemetryCollector, use_run_context
from agent.dataclasses.company import Company
from agent.ingest import ingest_startup_folder
from agent.ingest.store import Chunk, EvidenceStore
from agent.ingest.specter_ingest import ingest_specter

try:
    import web.db as db
except ImportError:
    db = None
from agent.person_intel.models import (
    BulkFounderJobRequest,
    PersonProfileJobRequest,
)
from agent.person_intel.service import PersonIntelService

app = FastAPI(title="Rockaway Deal Intelligence", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_PASSWORD = os.getenv("APP_PASSWORD", "9876")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me-session-secret")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24 * 14)))
SESSION_STORE_PATH = Path(
    os.getenv(
        "SESSION_STORE_PATH",
        str(Path(tempfile.gettempdir()) / "startup_ranker_sessions.json"),
    ),
)
JOBS_STORE_PATH = Path(
    os.getenv(
        "JOBS_STORE_PATH",
        str(Path(tempfile.gettempdir()) / "startup_ranker_jobs.json"),
    ),
)
_sessions: dict[str, float] = {}
_results_cache: dict[str, dict[str, Any]] = {}

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def _get_llm_display() -> str:
    """Return a display string for the configured default LLM."""
    selection = current_default_selection()
    return selection["label"]


def _pipeline_meta_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    phase_models = payload.get("phase_models")
    if isinstance(phase_models, dict):
        effective_phase_models = payload.get("effective_phase_models")
        if not isinstance(effective_phase_models, dict):
            effective_phase_models = None
        return {
            "phase_models": normalize_phase_models(phase_models),
            "quality_tier": None,
            "premium_phase_models": None,
            "effective_phase_models": effective_phase_models,
        }
    quality_tier = normalize_quality_tier(payload.get("quality_tier"))
    if not quality_tier:
        return None
    premium_phase_models = (
        normalize_premium_phase_models(payload.get("premium_phase_models"))
        if quality_tier == "premium"
        else None
    )
    effective_phase_models = payload.get("effective_phase_models")
    if not isinstance(effective_phase_models, dict):
        effective_phase_models = None
    return {
        "phase_models": None,
        "quality_tier": quality_tier,
        "premium_phase_models": premium_phase_models,
        "effective_phase_models": effective_phase_models,
    }


def _selection_from_payload(payload: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    selection = payload.get("llm_selection")
    if isinstance(selection, dict) and selection.get("provider") and selection.get("model"):
        return serialize_selection(selection.get("provider"), selection.get("model"))
    provider = payload.get("llm_provider") or payload.get("provider")
    model = payload.get("llm_model") or payload.get("model")
    if provider and model:
        return serialize_selection(provider, model)
    return None


def _resolve_job_llm_selection(job_id: str, *, results: dict[str, Any] | None = None) -> dict[str, str]:
    cache = _results_cache.get(job_id, {})
    for candidate in (
        _selection_from_payload(results),
        _selection_from_payload(cache.get("results")),
        _selection_from_payload(cache.get("llm_selection")),
        _selection_from_payload(cache.get("run_config")),
    ):
        if candidate:
            return candidate
    return current_default_selection()


def _resolve_job_pipeline_meta(job_id: str, *, results: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cache = _results_cache.get(job_id, {})
    for candidate in (
        _pipeline_meta_from_payload(results),
        _pipeline_meta_from_payload(cache.get("results")),
        _pipeline_meta_from_payload(cache.get("run_config")),
    ):
        if candidate:
            return candidate
    return None


def _resolve_job_llm_label(job_id: str, *, results: dict[str, Any] | None = None) -> str:
    pipeline_meta = _resolve_job_pipeline_meta(job_id, results=results)
    if pipeline_meta:
        if pipeline_meta.get("phase_models"):
            return build_phase_policy_display_label(
                pipeline_meta.get("effective_phase_models") or pipeline_meta["phase_models"]
            )
        return build_tier_display_label(
            pipeline_meta["quality_tier"],
            effective_phase_models=pipeline_meta.get("effective_phase_models"),
        )
    if isinstance(results, dict):
        explicit = (results.get("llm") or "").strip()
        if explicit and not _selection_from_payload(results):
            return explicit
    return _resolve_job_llm_selection(job_id, results=results)["label"]


def _resolve_batch_chunking_selection(job_id: str) -> dict[str, str]:
    pipeline_meta = _resolve_job_pipeline_meta(job_id)
    if pipeline_meta:
        effective = pipeline_meta.get("effective_phase_models")
        if isinstance(effective, dict):
            for phase_name in (
                "answering",
                "evaluation",
                "ranking",
                "generation",
                "decomposition",
                "critique",
                "refinement",
            ):
                selection = effective.get(phase_name)
                if isinstance(selection, dict):
                    provider = (selection.get("provider") or "").strip().lower()
                    model = (selection.get("model") or "").strip()
                    if provider == "anthropic":
                        return serialize_selection(provider, model)
                elif selection == "claude":
                    return serialize_selection("anthropic", "claude-haiku-4-5-20251001")
    return _resolve_job_llm_selection(job_id)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _batch_chunking_config(
    job_id: str,
    *,
    total_items: int,
    mode: str,
) -> dict[str, Any]:
    selection = _resolve_batch_chunking_selection(job_id)
    provider = selection.get("provider")
    default_threshold = 4 if provider == "anthropic" else 10_000
    default_chunk_size = 2 if provider == "anthropic" else total_items
    default_cooldown = 20 if provider == "anthropic" else 0

    threshold = _read_positive_int_env("BATCH_CHUNKING_THRESHOLD", default_threshold)
    chunk_size = _read_positive_int_env("BATCH_CHUNKING_SIZE", default_chunk_size)
    cooldown_seconds = _read_positive_int_env("BATCH_CHUNKING_COOLDOWN_SECONDS", default_cooldown)
    enabled = total_items >= threshold and chunk_size < total_items
    total_chunks = (total_items + chunk_size - 1) // chunk_size if enabled and chunk_size > 0 else 1
    return {
        "enabled": enabled,
        "threshold": threshold,
        "chunk_size": chunk_size if chunk_size > 0 else total_items,
        "cooldown_seconds": cooldown_seconds if enabled else 0,
        "total_chunks": total_chunks,
        "mode": mode,
        "provider": provider,
        "model": selection.get("model"),
        "label": selection.get("label"),
        "reason": "anthropic_large_batch" if enabled and provider == "anthropic" else "",
    }


def _chunk_items(items: list[Any], chunk_size: int) -> list[list[Any]]:
    size = max(1, chunk_size)
    return [items[idx: idx + size] for idx in range(0, len(items), size)]


class LoginRequest(BaseModel):
    password: str


def _ensure_str(val: Any) -> str:
    """Normalize to str; handle list to avoid 'list' has no attribute 'strip'."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val) if val else ""
    return str(val)


class AnalyzeRequest(BaseModel):
    use_web_search: bool = False
    instructions: str | None = None
    input_mode: str = "pitchdeck"  # pitchdeck | specter | original
    vc_investment_strategy: str | None = None
    phase_models: dict[str, dict[str, str]] | None = None
    quality_tier: Literal["cheap", "premium"] | None = None
    premium_phase_models: dict[str, Literal["claude", "gpt5"]] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None

    @field_validator(
        "instructions",
        "vc_investment_strategy",
        "llm_provider",
        "llm_model",
        mode="before",
    )
    @classmethod
    def _coerce_str(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, list):
            return " ".join(str(x) for x in v).strip() or None
        s = str(v).strip()
        return s if s else None

    @field_validator("phase_models", mode="before")
    @classmethod
    def _coerce_phase_models(cls, v: Any) -> dict[str, dict[str, str]] | None:
        if v is None:
            return None
        normalized = coerce_phase_models_payload(v, require_all=True)
        return {
            phase: {
                "provider": selection["provider"],
                "model": selection["model"],
            }
            for phase, selection in normalized.items()
        }

    @field_validator("premium_phase_models", mode="before")
    @classmethod
    def _coerce_premium_phase_models(cls, v: Any) -> dict[str, str] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("premium_phase_models must be an object.")
        normalized = normalize_premium_phase_models(v)
        allowed = {"decomposition", "generation", "evaluation", "ranking"}
        invalid_keys = [key for key in v.keys() if key not in allowed]
        if invalid_keys:
            raise ValueError("premium_phase_models contains unsupported phases.")
        return normalized


class AnalysisStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "stopped" | "done" | "error"
    progress: str = ""
    progress_log: list[str] = []
    results: Any = None


_jobs: dict[str, AnalysisStatus] = {}
_job_controls: dict[str, dict[str, bool]] = {}


class _JobStoppedError(Exception):
    """Raised when a running job is stopped by the user."""


class JobControlRequest(BaseModel):
    action: str  # pause | resume | stop

    @field_validator("action")
    @classmethod
    def _validate_action(cls, v: str) -> str:
        action = (v or "").strip().lower()
        if action not in {"pause", "resume", "stop"}:
            raise ValueError("action must be one of: pause, resume, stop")
        return action


class PersonProfileJobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    progress: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


_person_jobs: dict[str, PersonProfileJobStatus] = {}
_person_service = PersonIntelService()


def _runtime_versions() -> dict[str, str]:
    return {
        "app_version": os.getenv("APP_VERSION", "dev"),
        "prompt_version": os.getenv("PROMPT_VERSION", "v1"),
        "pipeline_version": os.getenv("PIPELINE_VERSION", "v1"),
        "schema_version": os.getenv("SCHEMA_VERSION", "20260306000000"),
    }


def _run_costs_from_cache(job_id: str) -> dict[str, Any]:
    cache = _results_cache.get(job_id, {})
    collector = cache.get("telemetry_collector")
    if isinstance(collector, RunTelemetryCollector):
        return collector.build_run_costs()
    return {
        "currency": "USD",
        "status": "unavailable",
        "total_usd": None,
        "llm_usd": None,
        "perplexity_usd": None,
        "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "perplexity_search": {"requests": 0, "total_usd": 0.0},
        "by_model": [],
    }


def _llm_telemetry_base() -> dict[str, Any]:
    """Backward-compatible test hook for legacy telemetry scaffolding."""
    return {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _ensure_job_control(job_id: str) -> dict[str, bool]:
    return _job_controls.setdefault(
        job_id,
        {"pause_requested": False, "stop_requested": False},
    )


def _append_progress(job_id: str, msg: str, *, allow_stopped: bool = False) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    if job.status == "stopped" and not allow_stopped:
        return
    job.progress = msg
    log = getattr(job, "progress_log", []) or []
    job.progress_log = log + [msg]
    if db and db.is_configured():
        try:
            db.insert_analysis_event(job_id, message=msg, event_type="progress")
        except Exception:
            pass


def _set_job_status(job_id: str, status: str, progress: str | None = None, source: str = "app") -> None:
    if not _jobs.get(job_id):
        return
    current = _jobs[job_id].status
    if current == "stopped" and source not in {"control", "stop_finalize"}:
        return
    _jobs[job_id].status = status
    if progress is not None:
        _jobs[job_id].progress = progress
    if db and db.is_configured():
        try:
            db.insert_job_status_history(
                job_id,
                status=status,
                progress=_jobs[job_id].progress,
                source=source,
            )
        except Exception:
            pass


def _persist_person_job(job_id: str, request_payload: dict[str, Any] | None = None) -> None:
    if not (db and db.is_configured()):
        return
    job = _person_jobs.get(job_id)
    if not job:
        return
    db.upsert_person_profile_job(
        job_id,
        status=job.status,
        progress=job.progress,
        request_payload=request_payload,
        result_payload=job.result,
        error=job.error,
        company_slug=(request_payload or {}).get("company_slug"),
        person_key=(request_payload or {}).get("person_key"),
    )


def _is_stop_requested(job_id: str) -> bool:
    return bool(_job_controls.get(job_id, {}).get("stop_requested"))


def _raise_if_stopped(job_id: str) -> None:
    if _is_stop_requested(job_id):
        raise _JobStoppedError("Job stopped by user")


async def _wait_if_paused(job_id: str) -> None:
    paused_once = False
    while _job_controls.get(job_id, {}).get("pause_requested") and not _is_stop_requested(job_id):
        paused_once = True
        if _jobs.get(job_id) and _jobs[job_id].status != "paused":
            _set_job_status(job_id, "paused", source="wait_if_paused")
            _append_progress(job_id, "Paused by user.")
        await asyncio.sleep(0.35)

    if paused_once and not _is_stop_requested(job_id) and _jobs.get(job_id):
        if _jobs[job_id].status == "paused":
            _set_job_status(job_id, "running", source="wait_if_paused")
            _append_progress(job_id, "Resumed by user.")


async def _cooperate_with_job_control(job_id: str) -> None:
    _raise_if_stopped(job_id)
    await _wait_if_paused(job_id)
    _raise_if_stopped(job_id)


def _finalize_stopped_results(
    job_id: str,
    upload_dir: Path,
    results_list: list[dict],
    *,
    total: int,
    source: str,
) -> bool:
    evaluated = [r for r in results_list if not r.get("skipped")]
    if not evaluated:
        return False

    _build_results_payload(results_list, job_id, upload_dir)
    message = (
        "Stopped by user. Partial results ready — "
        f"{len(evaluated)}/{total} companies ranked"
        if total > 1
        else "Stopped by user. Completed analysis is available."
    )
    _results_cache[job_id].setdefault("results", {})
    _results_cache[job_id]["results"]["job_status"] = "stopped"
    _results_cache[job_id]["results"]["job_message"] = message
    _jobs[job_id].results = _results_cache[job_id]["results"]
    _append_progress(job_id, "Finalizing partial results...", allow_stopped=True)
    _append_progress(job_id, message, allow_stopped=True)
    _set_job_status(job_id, "stopped", message, source="stop_finalize")
    _persist_jobs()
    _persist_results_to_db(job_id, results_list)
    if db and db.is_configured():
        db.upsert_job_control(
            job_id,
            pause_requested=False,
            stop_requested=True,
            last_action="stop",
        )
    return True


def _persist_sessions() -> None:
    """Persist active sessions to disk."""
    try:
        SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_STORE_PATH.write_text(json.dumps(_sessions))
    except Exception:
        # Best-effort persistence only.
        pass


def _load_sessions() -> None:
    """Load persisted sessions, dropping expired entries."""
    global _sessions
    if not SESSION_STORE_PATH.exists():
        return
    try:
        raw = json.loads(SESSION_STORE_PATH.read_text())
        if not isinstance(raw, dict):
            return
        now = time.time()
        cleaned: dict[str, float] = {}
        for sid, expiry in raw.items():
            try:
                expiry_ts = float(expiry)
            except (TypeError, ValueError):
                continue
            if expiry_ts > now:
                cleaned[str(sid)] = expiry_ts
        _sessions = cleaned
        _persist_sessions()
    except Exception:
        _sessions = {}


_load_sessions()


def _persist_jobs() -> None:
    """Persist completed jobs to disk so they survive server restarts."""
    try:
        to_save = {}
        for job_id, job in _jobs.items():
            if job.status == "done" and job.results is not None:
                to_save[job_id] = {
                    "status": job.status,
                    "progress": job.progress,
                    "progress_log": getattr(job, "progress_log", []) or [],
                    "results": job.results,
                }
        if not to_save:
            return
        JOBS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        JOBS_STORE_PATH.write_text(json.dumps(to_save, default=str))
    except Exception:
        pass


def _load_jobs() -> None:
    """Load persisted jobs from disk and Supabase on startup."""
    global _jobs, _results_cache

    if JOBS_STORE_PATH.exists():
        try:
            raw = json.loads(JOBS_STORE_PATH.read_text())
            if isinstance(raw, dict):
                for job_id, data in raw.items():
                    if not isinstance(data, dict) or data.get("status") != "done":
                        continue
                    results = data.get("results")
                    if results is None:
                        continue
                    _jobs[job_id] = AnalysisStatus(
                        job_id=job_id,
                        status="done",
                        progress=data.get("progress", "Analysis complete"),
                        progress_log=data.get("progress_log") or [],
                        results=results,
                    )
                    _results_cache[job_id] = {"results": results}
        except Exception:
            pass

    if db and db.is_configured():
        try:
            for job_id, entry in db.load_all_completed_jobs().items():
                if job_id in _jobs:
                    continue
                results = entry.get("results")
                if results is None:
                    continue
                _jobs[job_id] = AnalysisStatus(
                    job_id=job_id,
                    status="done",
                    progress="Analysis complete",
                    progress_log=[],
                    results=results,
                )
                _results_cache[job_id] = {
                    "results": results,
                    "excel_storage_path": entry.get("excel_storage_path"),
                }
        except Exception:
            pass


_load_jobs()


def _get_job_summary(job_id: str, job: AnalysisStatus) -> dict[str, Any]:
    results = _results_cache.get(job_id, {}).get("results")
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "created_at": None,
        "input_mode": results.get("mode") if isinstance(results, dict) else None,
        "use_web_search": None,
        "results": None,
        "llm": _resolve_job_llm_label(job_id, results=results),
    }


def _list_jobs_for_ui() -> list[dict[str, Any]]:
    jobs_by_id: dict[str, dict[str, Any]] = {}

    for job_id, job in _jobs.items():
        jobs_by_id[job_id] = _get_job_summary(job_id, job)

    if db and db.is_configured():
        try:
            for entry in db.list_saved_jobs():
                job_id = entry.get("job_id")
                if not job_id:
                    continue
                existing = jobs_by_id.get(job_id)
                merged = {
                    "job_id": job_id,
                    "status": entry.get("status") or (existing or {}).get("status") or "pending",
                    "progress": entry.get("progress") or (existing or {}).get("progress") or "",
                    "created_at": entry.get("created_at") or (existing or {}).get("created_at"),
                    "input_mode": entry.get("input_mode") or (existing or {}).get("input_mode"),
                    "use_web_search": entry.get("use_web_search"),
                    "results": None,
                    "llm": _resolve_job_llm_label(
                        job_id,
                        results=entry.get("results") or {},
                    ) if entry.get("results") else (existing or {}).get("llm") or _get_llm_display(),
                }
                jobs_by_id[job_id] = merged
        except Exception:
            pass

    def _sort_key(item: dict[str, Any]) -> tuple[str, str]:
        created_at = str(item.get("created_at") or "")
        return (created_at, item.get("job_id") or "")

    return sorted(jobs_by_id.values(), key=_sort_key, reverse=True)


def _check_session(session_id: str | None) -> bool:
    if not session_id:
        return False
    # Preferred path: stateless signed token (works across restarts/instances).
    if "." in session_id:
        raw_id, provided_sig = session_id.rsplit(".", 1)
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(
                SESSION_SECRET.encode("utf-8"),
                raw_id.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8").rstrip("=")
        if hmac.compare_digest(provided_sig, expected_sig):
            return True

    # Backward compatibility: legacy in-memory/file session ids.
    expiry = _sessions.get(session_id)
    if not expiry:
        return False
    if expiry <= time.time():
        _sessions.pop(session_id, None)
        _persist_sessions()
        return False
    return True


def _parse_max_startups_from_instructions(instructions: str | None) -> int | None:
    """Extract max_startups only from explicit limit instructions.

    Examples:
    - "limit to 20 companies"
    - "only rank the first 20 companies"
    - "rank only 20 companies"
    """
    text_raw = _ensure_str(instructions)
    if not text_raw or not text_raw.strip():
        return None
    text = text_raw.lower().strip()
    patterns = [
        r"only\s+(?:rank\s+)?(?:the\s+)?first\s+(\d+)\s*companies?",
        r"limit\s+to\s+(\d+)\s*companies?",
        r"rank\s+only\s+(\d+)\s*companies?",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            n = int(m.group(1))
            if n > 0:
                return n
    return None


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")
    raw_id = secrets.token_urlsafe(32)
    sig = base64.urlsafe_b64encode(
        hmac.new(
            SESSION_SECRET.encode("utf-8"),
            raw_id.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8").rstrip("=")
    session_id = f"{raw_id}.{sig}"

    # Keep legacy store warm for old clients still sending unsigned ids.
    _sessions[raw_id] = time.time() + SESSION_TTL_SECONDS
    _persist_sessions()
    return {"session_id": session_id}


@app.get("/api/check-session")
async def check_session(session_id: str | None = Cookie(default=None)):
    return {"authenticated": _check_session(session_id)}


@app.get("/api/web-search-available")
async def web_search_available(session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    pplx = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
    brave = os.getenv("BRAVE_SEARCH_API_KEY")
    has_key = bool(
        (pplx and pplx != "your_perplexity_api_key_here")
        or brave
    )
    provider = os.getenv("WEB_SEARCH_PROVIDER", "sonar")
    return {"available": has_key, "provider": provider}


def _detect_specter_csvs(upload_dir: Path, filenames: list[str]) -> dict | None:
    """Detect Specter uploads from filenames first, then tabular headers.

    Returns a dict with a required ``companies`` path and an optional ``people`` path.
    """
    def _is_tabular(name: str) -> bool:
        lower = name.lower()
        return lower.endswith(".csv") or lower.endswith(".xlsx") or lower.endswith(".xls")

    def _sniff_specter_kind(path: Path) -> str | None:
        try:
            if path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path, nrows=3)
            else:
                df = pd.read_csv(path, nrows=3)
        except Exception:
            return None

        headers = {str(col).strip().lower() for col in df.columns}
        if not headers:
            return None

        company_markers = {
            "company name",
            "founders",
            "founder highlights",
            "industry",
            "domain",
        }
        people_markers = {
            "specter - person id",
            "full name",
            "current position title",
            "current position company name",
        }

        if len(company_markers & headers) >= 2:
            return "companies"
        if len(people_markers & headers) >= 2:
            return "people"
        return None

    companies_file = None
    people_file = None
    for name in filenames:
        lower = name.lower()
        if not _is_tabular(name):
            continue
        if "people" in lower:
            people_file = upload_dir / name
        elif "company" in lower or "comapny" in lower:
            companies_file = upload_dir / name

    # Fall back to inspecting headers so valid Specter exports still work even
    # when filenames are generic or only the companies file is provided.
    for name in filenames:
        path = upload_dir / name
        if not _is_tabular(name):
            continue
        kind = _sniff_specter_kind(path)
        if kind == "companies" and not companies_file:
            companies_file = path
        elif kind == "people" and not people_file:
            people_file = path

    if companies_file:
        payload = {"companies": str(companies_file)}
        if people_file:
            payload["people"] = str(people_file)
        return payload
    return None


@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(tempfile.mkdtemp()) / job_id
    upload_dir.mkdir(parents=True)

    saved = []
    for f in files:
        dest = upload_dir / f.filename
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved.append(
            {
                "name": f.filename,
                "size": dest.stat().st_size,
                "mime_type": f.content_type or "",
                "sha256": _sha256_file(dest),
                "local_path": str(dest),
            }
        )

    specter = _detect_specter_csvs(upload_dir, [f["name"] for f in saved])

    _jobs[job_id] = AnalysisStatus(
        job_id=job_id, status="pending", progress="Files uploaded"
    )
    _results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": saved,
        "specter": specter,
    }
    if db and db.is_configured():
        db.insert_analysis_event(
            job_id,
            message="Files uploaded",
            event_type="upload",
            payload={"num_files": len(saved)},
        )
        db.insert_job_status_history(
            job_id,
            status="pending",
            progress="Files uploaded",
            source="upload",
        )

    return {
        "job_id": job_id,
        "files": saved,
        "mode": "specter" if specter else "documents",
    }


@app.post("/api/analyze/{job_id}")
async def start_analysis(
    job_id: str,
    req: AnalyzeRequest = AnalyzeRequest(),
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    quality_tier = normalize_quality_tier(req.quality_tier)
    pipeline_policy = None
    phase_models = normalize_phase_models(req.phase_models) if req.phase_models is not None else None
    premium_phase_models = normalize_premium_phase_models(req.premium_phase_models)
    if req.phase_models:
        try:
            pipeline_policy = build_phase_model_policy(phase_models)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        llm_selection = dict(pipeline_policy.answering)
        effective_phase_models = resolve_effective_phase_models(pipeline_policy)
        llm_display = build_phase_policy_display_label(effective_phase_models)
        quality_tier = None
        premium_phase_models = None
    elif quality_tier:
        try:
            pipeline_policy = build_pipeline_policy(
                quality_tier,
                premium_phase_models if quality_tier == "premium" else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        llm_selection = dict(pipeline_policy.answering)
        effective_phase_models = resolve_effective_phase_choices(pipeline_policy)
        llm_display = build_tier_display_label(
            quality_tier,
            effective_phase_models=effective_phase_models,
        )
    else:
        try:
            selected_entry = validate_requested_selection(req.llm_provider, req.llm_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        llm_selection = serialize_selection(
            (selected_entry.provider if selected_entry else None),
            (selected_entry.model if selected_entry else None),
        )
        effective_phase_models = None
        llm_display = llm_selection["label"]

    _set_job_status(job_id, "running", "Starting analysis...", source="start_analysis")
    _job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    _results_cache[job_id]["input_mode"] = req.input_mode
    _results_cache[job_id]["vc_investment_strategy"] = req.vc_investment_strategy
    _results_cache[job_id]["use_web_search"] = req.use_web_search
    _results_cache[job_id]["instructions"] = req.instructions
    _results_cache[job_id]["llm_selection"] = llm_selection
    _results_cache[job_id]["phase_models"] = phase_models if req.phase_models else None
    _results_cache[job_id]["quality_tier"] = quality_tier
    _results_cache[job_id]["premium_phase_models"] = (
        premium_phase_models if quality_tier == "premium" else None
    )
    _results_cache[job_id]["effective_phase_models"] = effective_phase_models
    _results_cache[job_id]["run_config"] = {
        "input_mode": req.input_mode,
        "vc_investment_strategy": req.vc_investment_strategy,
        "instructions": req.instructions,
        "use_web_search": req.use_web_search,
        "phase_models": phase_models if req.phase_models else None,
        "quality_tier": quality_tier,
        "premium_phase_models": premium_phase_models if quality_tier == "premium" else None,
        "effective_phase_models": effective_phase_models,
        "llm_provider": llm_selection["provider"],
        "llm_model": llm_selection["model"],
        "llm": llm_display,
    }
    _results_cache[job_id]["model_executions"] = []
    _results_cache[job_id]["versions"] = _runtime_versions()

    if db and db.is_configured():
        run_config = dict(_results_cache[job_id]["run_config"])
        db.upsert_job(job_id, run_config=run_config, versions=_runtime_versions())
        db.upsert_job_control(
            job_id,
            pause_requested=False,
            stop_requested=False,
            last_action="start",
        )

    vc_str = _ensure_str(req.vc_investment_strategy).strip() or None
    inst = _ensure_str(req.instructions).strip() or None
    # Run the analysis loop in a dedicated thread/event-loop to keep
    # the main FastAPI loop responsive for pause/resume/stop controls.
    threading.Thread(
        target=lambda: asyncio.run(
            _run_analysis(
                job_id,
                use_web_search=req.use_web_search,
                instructions=inst,
                input_mode=req.input_mode,
                vc_investment_strategy=vc_str,
                llm_selection=llm_selection,
                pipeline_policy=pipeline_policy,
            )
        ),
        daemon=True,
    ).start()
    return {"status": "running", "use_web_search": req.use_web_search, "llm": llm_display}


@app.post("/api/jobs/{job_id}/control")
async def control_analysis_job(
    job_id: str,
    req: JobControlRequest,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.status in {"done", "error", "stopped"}:
        return {"job_id": job_id, "status": job.status, "progress": job.progress}

    control = _ensure_job_control(job_id)
    action = req.action

    if action == "pause":
        control["pause_requested"] = True
        if job.status in {"running", "pending"}:
            _set_job_status(job_id, "paused", source="control")
            _append_progress(job_id, "Paused by user.")
    elif action == "resume":
        control["pause_requested"] = False
        if job.status in {"paused", "pending"}:
            _set_job_status(job_id, "running", source="control")
            _append_progress(job_id, "Resumed by user.")
    else:  # stop
        control["pause_requested"] = False
        control["stop_requested"] = True
        _append_progress(job_id, "Stopped by user.", allow_stopped=True)
        _set_job_status(job_id, "stopped", "Stopped by user.", source="control")

    if db and db.is_configured():
        db.upsert_job_control(
            job_id,
            pause_requested=bool(control.get("pause_requested")),
            stop_requested=bool(control.get("stop_requested")),
            last_action=action,
        )

    return {"job_id": job_id, "status": job.status, "progress": job.progress}


def _build_results_payload(
    results_list: list[dict],
    job_id: str,
    upload_dir: Path,
    *,
    write_excel: bool = False,
) -> None:
    """Compute scores and populate _results_cache for the job."""
    payload = _compose_db_backed_results_payload(results_list, job_id)
    _results_cache[job_id]["results"] = payload


def _report_mode_hint(job_id: str, results_list: list[dict[str, Any]]) -> str:
    cache = _results_cache.get(job_id, {})
    input_mode = cache.get("input_mode")
    files = cache.get("files") or []
    specter = cache.get("specter")
    if specter:
        return "batch"
    if input_mode == "original":
        return "single"
    if input_mode == "pitchdeck":
        return "single" if len(files) <= 1 else "batch"
    if len(results_list) == 1 and not any(item.get("skipped") for item in results_list):
        return "single"
    return "batch"


def _merge_runtime_payload_metadata(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    pipeline_meta = _resolve_job_pipeline_meta(job_id, results=merged)
    llm_selection = _resolve_job_llm_selection(job_id, results=merged)
    if pipeline_meta:
        merged["phase_models"] = pipeline_meta.get("phase_models")
        merged["quality_tier"] = pipeline_meta["quality_tier"]
        merged["premium_phase_models"] = pipeline_meta["premium_phase_models"]
        merged["effective_phase_models"] = pipeline_meta.get("effective_phase_models")
    merged["llm"] = _resolve_job_llm_label(job_id, results=merged)
    merged["llm_selection"] = llm_selection
    run_costs = _run_costs_from_cache(job_id)
    if run_costs.get("status") != "unavailable" or "run_costs" not in merged:
        merged["run_costs"] = run_costs
    chunking = _results_cache.get(job_id, {}).get("batch_chunking")
    if isinstance(chunking, dict):
        merged["batch_chunking"] = chunking
    return merged


def _compose_db_backed_results_payload(
    results_list: list[dict[str, Any]],
    job_id: str,
) -> dict[str, Any]:
    if db and db.is_configured():
        preferred_mode = _report_mode_hint(job_id, results_list)
        loaded = db.load_job_results(job_id, preferred_mode=preferred_mode)
        if loaded and isinstance(loaded.get("results"), dict):
            return _merge_runtime_payload_metadata(job_id, loaded["results"])
    return _compose_results_payload(results_list, job_id)


def _compose_results_payload(
    results_list: list[dict],
    job_id: str,
) -> dict[str, Any]:
    """Compute API/UI payload for the current job state."""
    collector = _results_cache.get(job_id, {}).get("telemetry_collector")
    if isinstance(collector, RunTelemetryCollector):
        _results_cache[job_id]["model_executions"] = collector.snapshot_model_executions()

    input_order_map: dict[str, int] = {}
    input_order_counter = 0
    for item in results_list:
        if item.get("skipped"):
            continue
        slug = item.get("slug")
        if not slug:
            continue
        if slug not in input_order_map:
            input_order_map[slug] = input_order_counter
            input_order_counter += 1

    results_list = rank_batch_companies(results_list)
    summary_rows = build_summary_rows(results_list)
    argument_rows = build_argument_rows(results_list)
    qa_provenance_rows = build_qa_provenance_rows(results_list)

    evaluated = [r for r in results_list if not r.get("skipped")]

    if len(results_list) == 1 and len(evaluated) == 1:
        r = evaluated[0]
        final_state = r["final_state"]
        company = r["company"]
        final_args = final_state.get("final_arguments", [])
        pro_args = sorted(
            [a for a in final_args if a.argument_type == "pro"],
            key=lambda a: a.score, reverse=True,
        )
        contra_args = sorted(
            [a for a in final_args if a.argument_type == "contra"],
            key=lambda a: a.score, reverse=True,
        )
        avg_pro = (sum(a.score for a in pro_args) / len(pro_args)) if pro_args else 0
        avg_contra = (sum(a.score for a in contra_args) / len(contra_args)) if contra_args else 0
        total_score = avg_pro - avg_contra

        ranking = final_state.get("ranking_result")
        ranking_result = None
        if ranking:
            ranking_result = {
                "rank": ranking.rank,
                "percentile": ranking.percentile,
                "composite_score": ranking.composite_score,
                "strategy_fit_score": ranking.strategy_fit_score,
                "team_score": ranking.team_score,
                "upside_score": ranking.upside_score,
                "bucket": ranking.bucket,
                "strategy_fit_summary": getattr(ranking, "strategy_fit_summary", "") or "",
                "team_summary": getattr(ranking, "team_summary", "") or "",
                "potential_summary": getattr(ranking, "potential_summary", "") or "",
                "key_points": getattr(ranking, "key_points", []) or [],
                "red_flags": getattr(ranking, "red_flags", []) or [],
                "dimension_scores": [
                    {
                        "dimension": d.dimension,
                        "adjusted_score": d.adjusted_score,
                        "confidence": d.confidence,
                        "evidence_snippets": d.evidence_snippets,
                        "critical_gaps": d.critical_gaps,
                    }
                    for d in ranking.dimension_scores
                ],
            }

        payload = {
            "mode": "single",
            "startup_slug": r.get("slug", company.name),
            "company_name": company.name,
            "industry": company.industry or "N/A",
            "tagline": company.tagline or "",
            "about": company.about or "",
            "decision": final_state.get("final_decision", "unknown"),
            "total_score": round(total_score, 2),
            "avg_pro": round(avg_pro, 2),
            "avg_contra": round(avg_contra, 2),
            "ranking_result": ranking_result,
            "num_documents": len(_results_cache[job_id].get("files", [])),
            "num_chunks": len(r["evidence_store"].chunks),
            "num_arguments": len(final_args),
            "pro_arguments": [
                {"text": a.refined_content or a.content, "score": a.score, "critique": a.critique or ""}
                for a in pro_args[:5]
            ],
            "contra_arguments": [
                {"text": a.refined_content or a.content, "score": a.score, "critique": a.critique or ""}
                for a in contra_args[:5]
            ],
            "summary_rows": summary_rows,
            "argument_rows": argument_rows,
            "qa_provenance_rows": qa_provenance_rows,
            "founders": _extract_founders_from_company(company, r.get("slug", company.name)),
            "team_members": _extract_founders_from_company(company, r.get("slug", company.name)),
        }
    else:
        failed_rows = build_failed_rows(results_list)
        founders_by_slug: dict[str, list[dict[str, Any]]] = {}
        for item in evaluated:
            slug = item.get("slug", "")
            founders_by_slug[slug] = _extract_founders_from_company(
                item.get("company"),
                slug,
            )

        for row in summary_rows:
            row_slug = row.get("startup_slug", "")
            row["founders"] = founders_by_slug.get(row_slug, [])
            row["team_members"] = founders_by_slug.get(row_slug, [])
            row["specter_input_order"] = input_order_map.get(row_slug, 10_000_000)

        payload = {
            "mode": "batch",
            "num_companies": len(evaluated),
            "num_skipped": len(results_list) - len(evaluated),
            "summary_rows": summary_rows,
            "argument_rows": argument_rows,
            "qa_provenance_rows": qa_provenance_rows,
            "failed_rows": failed_rows,
            "founders_by_slug": founders_by_slug,
            "team_members_by_slug": founders_by_slug,
        }

    llm_selection = _resolve_job_llm_selection(job_id, results=payload)
    payload["llm"] = llm_selection["label"]
    payload["llm_selection"] = llm_selection
    payload["run_costs"] = _run_costs_from_cache(job_id)
    chunking = _results_cache.get(job_id, {}).get("batch_chunking")
    if isinstance(chunking, dict):
        payload["batch_chunking"] = chunking
    return payload


def _single_result_payload(job_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return _compose_results_payload([result], job_id)


def _run_config_from_cache(job_id: str) -> dict[str, Any]:
    cache = _results_cache.get(job_id, {})
    run_config = dict(cache.get("run_config") or {})
    if run_config:
        return run_config

    llm_selection = _resolve_job_llm_selection(job_id, results=cache.get("results"))
    return {
        "input_mode": cache.get("input_mode", "pitchdeck"),
        "vc_investment_strategy": cache.get("vc_investment_strategy"),
        "instructions": cache.get("instructions"),
        "use_web_search": cache.get("use_web_search", False),
        "phase_models": cache.get("phase_models"),
        "quality_tier": cache.get("quality_tier"),
        "premium_phase_models": cache.get("premium_phase_models"),
        "effective_phase_models": cache.get("effective_phase_models"),
        "llm_provider": llm_selection["provider"],
        "llm_model": llm_selection["model"],
        "llm": _resolve_job_llm_label(job_id, results=cache.get("results")),
    }


def _failure_result_payload(
    job_id: str,
    *,
    company: Company,
    store: EvidenceStore,
    slug: str,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    founders = _extract_founders_from_company(company, slug)
    payload = {
        "mode": "single",
        "startup_slug": slug,
        "company_name": company.name,
        "industry": company.industry or "N/A",
        "tagline": company.tagline or "",
        "about": company.about or "",
        "decision": status,
        "total_score": None,
        "avg_pro": None,
        "avg_contra": None,
        "ranking_result": None,
        "num_documents": len(_results_cache.get(job_id, {}).get("files", [])),
        "num_chunks": len(getattr(store, "chunks", []) or []),
        "num_arguments": 0,
        "pro_arguments": [],
        "contra_arguments": [],
        "summary_rows": [{
            "startup_slug": slug,
            "company_name": company.name,
            "decision": status,
            "total_score": None,
            "avg_pro": None,
            "avg_contra": None,
        }],
        "argument_rows": [],
        "qa_provenance_rows": [],
        "failed_rows": [{
            "startup_slug": slug,
            "company_name": company.name,
            "error": error_message,
        }],
        "founders": founders,
        "team_members": founders,
        "job_status": status,
        "job_message": error_message,
    }
    return _merge_runtime_payload_metadata(job_id, payload)


def _score_summary_from_result(result: dict[str, Any]) -> tuple[float | None, float | None]:
    final_state = result.get("final_state") or {}
    ranking = final_state.get("ranking_result")
    composite_score = getattr(ranking, "composite_score", None)
    if composite_score is None and isinstance(ranking, dict):
        composite_score = ranking.get("composite_score")
    final_args = final_state.get("final_arguments", []) or []
    pro_scores = [float(arg.score) for arg in final_args if getattr(arg, "argument_type", "") == "pro"]
    contra_scores = [float(arg.score) for arg in final_args if getattr(arg, "argument_type", "") == "contra"]
    total_score = None
    if pro_scores or contra_scores:
        avg_pro = (sum(pro_scores) / len(pro_scores)) if pro_scores else 0.0
        avg_contra = (sum(contra_scores) / len(contra_scores)) if contra_scores else 0.0
        total_score = round(avg_pro - avg_contra, 2)
    return total_score, composite_score


def _minimize_completed_result_for_memory(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("skipped"):
        return result
    company = result.get("company")
    slug = result.get("slug") or getattr(company, "name", None) or "unknown"
    company_name = getattr(company, "name", None) or result.get("company_name") or slug
    decision = (result.get("final_state") or {}).get("final_decision")
    total_score, composite_score = _score_summary_from_result(result)
    minimized = {
        "slug": slug,
        "company_name": company_name,
        "skipped": False,
        "persisted": True,
        "decision": decision,
        "total_score": total_score,
        "composite_score": composite_score,
        "persisted_at": datetime.now(timezone.utc).isoformat(),
    }
    result.clear()
    result.update(minimized)
    return result


def _update_partial_results_cache(
    job_id: str,
    upload_dir: Path,
    results_list: list[dict[str, Any]],
) -> None:
    try:
        _build_results_payload(results_list, job_id, upload_dir, write_excel=False)
    except TypeError:
        # Compatibility for tests monkeypatching the older 3-arg helper.
        _build_results_payload(results_list, job_id, upload_dir)
    payload = _results_cache[job_id].setdefault("results", {})
    payload["job_status"] = _jobs.get(job_id).status if _jobs.get(job_id) else "running"
    payload["job_message"] = _jobs.get(job_id).progress if _jobs.get(job_id) else ""
    if _jobs.get(job_id):
        _jobs[job_id].results = payload


def _persist_company_result_to_db(job_id: str, result: dict[str, Any]) -> bool:
    if not (db and db.is_configured()) or result.get("skipped"):
        return False
    cache = _results_cache.get(job_id, {})
    run_config = _run_config_from_cache(job_id)
    company_payload = _single_result_payload(job_id, result)
    return bool(db.persist_company_result(
        job_id_legacy=job_id,
        result_row=result,
        company_payload=company_payload,
        run_config=run_config,
        versions=cache.get("versions") or _runtime_versions(),
    ))


def _persist_failed_company_result_to_db(
    job_id: str,
    *,
    company: Company,
    store: EvidenceStore,
    slug: str,
    error_message: str,
    status: str,
) -> bool:
    if not (db and db.is_configured()):
        return False

    cache = _results_cache.get(job_id, {})
    run_config = _run_config_from_cache(job_id)

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
    company_payload = _failure_result_payload(
        job_id,
        company=company,
        store=store,
        slug=slug,
        status=status,
        error_message=error_message,
    )
    return bool(db.persist_company_failure_result(
        job_id_legacy=job_id,
        result_row=result_row,
        company_payload=company_payload,
        run_config=run_config,
        versions=cache.get("versions") or _runtime_versions(),
    ))


def _persist_results_to_db(job_id: str, results_list: list[dict]) -> None:
    """Best-effort DB persistence — must not block job completion."""
    if not (db and db.is_configured()):
        return
    try:
        cache = _results_cache.get(job_id, {})
        run_config = _run_config_from_cache(job_id)
        db.persist_analysis(
            job_id_legacy=job_id,
            results_list=results_list,
            results_payload=cache.get("results", {}),
            run_config=run_config,
            versions=cache.get("versions") or _runtime_versions(),
            source_files=cache.get("files") or [],
            model_executions=cache.get("model_executions") or [],
        )
    except Exception:
        import traceback
        traceback.print_exc()


def _extract_founders_from_company(company: Any, slug: str) -> list[dict[str, Any]]:
    """Build team-member metadata for on-demand person intelligence."""
    founders: list[dict[str, Any]] = []
    team = getattr(company, "team", None) or []
    for idx, person in enumerate(team, start=1):
        name = getattr(person, "name", None)
        profile_url = getattr(person, "profile_url", None)
        founders.append(
            {
                "person_key": f"{slug}:founder:{idx}",
                "full_name": name or f"Team Member {idx}",
                "primary_profile_url": profile_url or "",
                "current_company": getattr(company, "name", None) or "",
                "role": getattr(person, "title", None) or "",
                "known_aliases": [],
            }
        )
    return founders


async def _run_analysis(
    job_id: str,
    use_web_search: bool = False,
    instructions: str | None = None,
    input_mode: str = "pitchdeck",
    vc_investment_strategy: str | None = None,
    llm_selection: dict[str, str] | None = None,
    pipeline_policy: Any = None,
):
    try:
        selection = llm_selection or _resolve_job_llm_selection(job_id)
        collector = RunTelemetryCollector(selected_llm=selection)
        _results_cache[job_id]["telemetry_collector"] = collector

        with use_run_context(
            llm_selection=selection,
            telemetry_collector=collector,
            pipeline_policy=pipeline_policy,
        ):
            await _cooperate_with_job_control(job_id)
            upload_dir = Path(_results_cache[job_id]["upload_dir"])
            specter = _results_cache[job_id].get("specter")

            if input_mode == "specter":
                await _cooperate_with_job_control(job_id)
                specter_detected = _results_cache[job_id].get("specter")
                files = _results_cache[job_id].get("files", [])
                if specter_detected:
                    specter_paths = specter_detected
                elif len(files) >= 1:
                    specter_paths = {
                        "companies": str(upload_dir / files[0]["name"]),
                    }
                    if len(files) >= 2:
                        specter_paths["people"] = str(upload_dir / files[1]["name"])
                else:
                    _set_job_status(
                        job_id,
                        "error",
                        "Specter mode requires at least 1 tabular file.",
                        source="run_analysis",
                    )
                    return
                await _run_specter_analysis(
                    job_id, upload_dir, specter_paths, use_web_search, instructions,
                    vc_investment_strategy=vc_investment_strategy,
                )
            elif input_mode == "original":
                await _cooperate_with_job_control(job_id)
                await _run_document_analysis(
                    job_id, upload_dir, use_web_search, one_company=True,
                    vc_investment_strategy=vc_investment_strategy,
                )
            else:
                await _cooperate_with_job_control(job_id)
                await _run_document_analysis(
                    job_id, upload_dir, use_web_search, one_company=False,
                    vc_investment_strategy=vc_investment_strategy,
                )

    except _JobStoppedError:
        if _jobs.get(job_id):
            collector = _results_cache.get(job_id, {}).get("telemetry_collector")
            if isinstance(collector, RunTelemetryCollector):
                _results_cache[job_id]["model_executions"] = collector.snapshot_model_executions()
            _set_job_status(job_id, "stopped", source="run_analysis")
            if "Stopped by user" not in (_jobs[job_id].progress or ""):
                _append_progress(job_id, "Stopped by user.")
            if db and db.is_configured():
                db.upsert_job_control(
                    job_id,
                    pause_requested=False,
                    stop_requested=True,
                    last_action="stop",
                )
    except Exception as exc:
        if _jobs.get(job_id) and _jobs[job_id].status != "stopped":
            collector = _results_cache.get(job_id, {}).get("telemetry_collector")
            if isinstance(collector, RunTelemetryCollector):
                _results_cache[job_id]["model_executions"] = collector.snapshot_model_executions()
            _set_job_status(job_id, "error", f"Analysis failed: {exc}", source="run_analysis")
            if db and db.is_configured():
                db.insert_analysis_error(
                    job_id,
                    message=str(exc)[:1000],
                    stage="run_analysis",
                    error_type=type(exc).__name__,
                )


def _make_progress_callback(job_id: str):
    def on_progress(msg: str) -> None:
        _raise_if_stopped(job_id)
        _append_progress(job_id, msg)
    return on_progress


def _sanitize_slug(name: str) -> str:
    """Make a filesystem-safe slug from a filename."""
    base = Path(name).stem
    safe = re.sub(r"[^a-z0-9\-]", "-", base.lower()).strip("-")
    return safe or "doc"


def _merge_evidence_stores(
    primary: EvidenceStore,
    secondary: EvidenceStore,
    startup_slug: str,
) -> EvidenceStore:
    """Combine two evidence stores and normalize chunk IDs."""
    merged_chunks: list[Chunk] = []
    for idx, chunk in enumerate([*primary.chunks, *secondary.chunks]):
        merged_chunks.append(
            Chunk(
                chunk_id=f"chunk_{idx}",
                text=chunk.text,
                source_file=chunk.source_file,
                page_or_slide=chunk.page_or_slide,
            )
        )
    return EvidenceStore(startup_slug=startup_slug, chunks=merged_chunks)


def _build_single_company_specter_overlay(
    upload_dir: Path,
    specter: dict | None,
) -> tuple[Company | None, EvidenceStore | None, int | None]:
    """Build a merged one-company dossier when a single-company Specter export is present."""
    if not specter:
        return None, None, None

    try:
        company_store_pairs = ingest_specter(
            specter["companies"],
            specter.get("people"),
        )
    except Exception:
        return None, None, None

    parsed_count = len(company_store_pairs)
    if parsed_count != 1:
        return None, None, parsed_count

    specter_company, specter_store = company_store_pairs[0]
    excluded = {Path(specter["companies"]).name}
    if specter.get("people"):
        excluded.add(Path(specter["people"]).name)
    document_store = ingest_startup_folder(upload_dir, exclude_files=excluded)
    merged_store = _merge_evidence_stores(
        document_store,
        specter_store,
        startup_slug=specter_store.startup_slug or upload_dir.name,
    )
    return specter_company, merged_store, parsed_count


async def _run_document_analysis(
    job_id: str,
    upload_dir: Path,
    use_web_search: bool,
    one_company: bool = False,
    vc_investment_strategy: str | None = None,
) -> None:
    """Analyze uploaded documents.

    one_company=True (Original): All files = one company (folder per company).
    one_company=False (Pitch deck): Each file = separate company (batch).
    """
    files = _results_cache[job_id].get("files", [])
    file_count = len(files)
    if file_count == 0:
        _set_job_status(job_id, "error", "No files found.", source="run_document_analysis")
        return

    if one_company or file_count == 1:
        seed_company = None
        seed_store = None
        specter = _results_cache[job_id].get("specter")
        if specter:
            seed_company, seed_store, parsed_count = _build_single_company_specter_overlay(upload_dir, specter)
            if seed_company and seed_store:
                _append_progress(
                    job_id,
                    "Detected a single-company Specter export. Merging structured Specter evidence into the dossier.",
                )
            elif one_company and parsed_count and parsed_count > 1:
                _append_progress(
                    job_id,
                    "Multi-file mode stays enabled. Structured Specter overlay was skipped because the export contains multiple companies.",
                )
        result = await evaluate_startup(
            upload_dir, k=8, use_web_search=use_web_search,
            on_progress=_make_progress_callback(job_id),
            on_cooperate=lambda: _cooperate_with_job_control(job_id),
            vc_investment_strategy=vc_investment_strategy,
            initial_store=seed_store,
            initial_company=seed_company,
        )
        if result.get("skipped"):
            if _is_stop_requested(job_id):
                raise _JobStoppedError("Job stopped by user")
            _set_job_status(
                job_id,
                "error",
                "No extractable content found in uploaded files.",
                source="run_document_analysis",
            )
            return
        _persist_company_result_to_db(job_id, result)
        _append_progress(job_id, "Finalizing results...")
        _build_results_payload([result], job_id, upload_dir)
        if _is_stop_requested(job_id):
            _results_cache[job_id].setdefault("results", {})
            _results_cache[job_id]["results"]["job_status"] = "stopped"
            _results_cache[job_id]["results"]["job_message"] = "Stopped by user. Completed analysis is available."
            _jobs[job_id].results = _results_cache[job_id]["results"]
            _append_progress(job_id, "Stopped by user. Completed analysis is available.", allow_stopped=True)
            _set_job_status(
                job_id,
                "stopped",
                "Stopped by user. Completed analysis is available.",
                source="stop_finalize",
            )
            _persist_jobs()
            _persist_results_to_db(job_id, [result])
            return
        _append_progress(job_id, "Finalizing complete.")
        _set_job_status(job_id, "done", "Analysis complete", source="run_document_analysis")
        _jobs[job_id].results = _results_cache[job_id]["results"]
        _persist_jobs()
        _persist_results_to_db(job_id, [result])
        return

    results_list: list[dict] = []
    total = file_count
    chunking = _batch_chunking_config(job_id, total_items=total, mode="documents")
    _results_cache[job_id]["batch_chunking"] = chunking
    file_chunks = _chunk_items(files, chunking["chunk_size"]) if chunking["enabled"] else [files]
    if chunking["enabled"]:
        _append_progress(
            job_id,
            "Large batch chunking enabled "
            f"for {chunking['label']} — {chunking['total_chunks']} chunks of up to "
            f"{chunking['chunk_size']} company(ies), cooldown {chunking['cooldown_seconds']}s.",
        )
    try:
        processed = 0
        for chunk_idx, file_chunk in enumerate(file_chunks, 1):
            if chunking["enabled"]:
                chunk_start = processed + 1
                chunk_end = processed + len(file_chunk)
                _append_progress(
                    job_id,
                    f"Starting chunk {chunk_idx}/{chunking['total_chunks']} — companies {chunk_start}-{chunk_end} of {total}.",
                )
            for finfo in file_chunk:
                await _cooperate_with_job_control(job_id)
                processed += 1
                fname = finfo["name"]
                prefix = f"Chunk {chunk_idx}/{chunking['total_chunks']} — Analyzing {fname} ({processed}/{total})" if chunking["enabled"] else f"Analyzing {fname} ({processed}/{total})"
                _append_progress(job_id, f"{prefix} — Starting...")

                doc_dir = upload_dir / _sanitize_slug(fname)
                doc_dir.mkdir(exist_ok=True)
                src = upload_dir / fname
                if src.exists():
                    shutil.copy2(src, doc_dir / fname)

                def make_progress(msg: str) -> None:
                    _raise_if_stopped(job_id)
                    full_msg = f"{prefix} — {msg}"
                    _append_progress(job_id, full_msg)

                try:
                    result = await evaluate_startup(
                        doc_dir, k=8, use_web_search=use_web_search,
                        on_progress=make_progress,
                        on_cooperate=lambda: _cooperate_with_job_control(job_id),
                        vc_investment_strategy=vc_investment_strategy,
                    )
                    await _cooperate_with_job_control(job_id)
                    results_list.append(result)
                    _append_progress(job_id, f"{prefix} — Persisting partial result...")
                    persisted_to_db = _persist_company_result_to_db(job_id, result)
                    if persisted_to_db:
                        _minimize_completed_result_for_memory(result)
                    _update_partial_results_cache(job_id, upload_dir, results_list)
                    _append_progress(
                        job_id,
                        f"Partial results updated — {len([r for r in results_list if not r.get('skipped')])}/{total} companies completed.",
                    )
                except _JobStoppedError:
                    raise
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    if db and db.is_configured():
                        db.insert_analysis_error(
                            job_id,
                            message=str(exc)[:1000],
                            stage="evaluate_startup",
                            error_type=type(exc).__name__,
                            company_slug=_sanitize_slug(fname),
                        )
                        _persist_failed_company_result_to_db(
                            job_id,
                            company=Company(name=fname),
                            store=EvidenceStore(startup_slug=_sanitize_slug(fname), chunks=[]),
                            slug=_sanitize_slug(fname),
                            error_message=str(exc)[:1000],
                            status="timeout" if isinstance(exc, TimeoutError) else "error",
                        )
                    results_list.append({
                        "slug": _sanitize_slug(fname),
                        "skipped": True,
                        "error": str(exc)[:500],
                        "company_name": fname,
                    })

            if (
                chunking["enabled"]
                and chunk_idx < chunking["total_chunks"]
                and chunking["cooldown_seconds"] > 0
            ):
                _append_progress(
                    job_id,
                    f"Chunk {chunk_idx}/{chunking['total_chunks']} complete — cooling down for {chunking['cooldown_seconds']}s before next chunk.",
                )
                await asyncio.sleep(chunking["cooldown_seconds"])
    except _JobStoppedError:
        if _finalize_stopped_results(
            job_id,
            upload_dir,
            results_list,
            total=total,
            source="run_document_analysis",
        ):
            return
        raise

    evaluated = [r for r in results_list if not r.get("skipped")]
    if not evaluated:
        if _is_stop_requested(job_id):
            raise _JobStoppedError("Job stopped by user")
        _set_job_status(
            job_id,
            "error",
            "No startups were successfully evaluated.",
            source="run_document_analysis",
        )
        return

    _append_progress(job_id, "Finalizing batch results...")
    _build_results_payload(results_list, job_id, upload_dir)
    _append_progress(job_id, "Finalizing complete.")
    if _is_stop_requested(job_id):
        if _finalize_stopped_results(
            job_id,
            upload_dir,
            results_list,
            total=total,
            source="run_document_analysis",
        ):
            return
        raise _JobStoppedError("Job stopped by user")
    _set_job_status(
        job_id,
        "done",
        f"Analysis complete — {len(evaluated)}/{total} companies ranked",
        source="run_document_analysis",
    )
    _jobs[job_id].results = _results_cache[job_id]["results"]
    _persist_jobs()
    _persist_results_to_db(job_id, results_list)


async def _run_specter_analysis(
    job_id: str,
    upload_dir: Path,
    specter: dict,
    use_web_search: bool,
    instructions: str | None = None,
    vc_investment_strategy: str | None = None,
) -> None:
    """Batch Specter analysis from company + people CSVs."""
    await _cooperate_with_job_control(job_id)
    _append_progress(job_id, "Parsing Specter CSV files...")

    company_store_pairs = ingest_specter(
        specter["companies"],
        specter.get("people"),
    )
    parsed_total = len(company_store_pairs)
    print(f"Specter ingest: parsed {parsed_total} companies.")

    max_startups = _parse_max_startups_from_instructions(instructions)
    if max_startups is not None:
        print(
            f"Applying explicit instruction limit: "
            f"first {max_startups} company(ies) out of {parsed_total}.",
        )
        company_store_pairs = company_store_pairs[:max_startups]

    if not company_store_pairs:
        _set_job_status(job_id, "error", "No companies found in Specter data.", source="run_specter_analysis")
        return

    total = len(company_store_pairs)
    results_list: list[dict] = []
    chunking = _batch_chunking_config(job_id, total_items=total, mode="specter")
    _results_cache[job_id]["batch_chunking"] = chunking
    company_chunks = _chunk_items(company_store_pairs, chunking["chunk_size"]) if chunking["enabled"] else [company_store_pairs]
    if chunking["enabled"]:
        _append_progress(
            job_id,
            "Large batch chunking enabled "
            f"for {chunking['label']} — {chunking['total_chunks']} chunks of up to "
            f"{chunking['chunk_size']} company(ies), cooldown {chunking['cooldown_seconds']}s.",
        )

    last_error: str | None = None
    try:
        processed = 0
        for chunk_idx, company_chunk in enumerate(company_chunks, 1):
            if chunking["enabled"]:
                chunk_start = processed + 1
                chunk_end = processed + len(company_chunk)
                _append_progress(
                    job_id,
                    f"Starting chunk {chunk_idx}/{chunking['total_chunks']} — companies {chunk_start}-{chunk_end} of {total}.",
                )
            for company, store in company_chunk:
                await _cooperate_with_job_control(job_id)
                processed += 1
                prefix = (
                    f"Chunk {chunk_idx}/{chunking['total_chunks']} — Evaluating {company.name} ({processed}/{total})"
                    if chunking["enabled"]
                    else f"Evaluating {company.name} ({processed}/{total})"
                )
                _append_progress(job_id, f"{prefix} — Starting...")

                def make_specter_progress(p: str) -> None:
                    _raise_if_stopped(job_id)
                    full_msg = f"{prefix} — {p}"
                    _append_progress(job_id, full_msg)

                try:
                    result = await evaluate_from_specter(
                        company, store, k=8, use_web_search=use_web_search,
                        on_progress=make_specter_progress,
                        on_cooperate=lambda: _cooperate_with_job_control(job_id),
                        vc_investment_strategy=vc_investment_strategy,
                    )
                    await _cooperate_with_job_control(job_id)
                    results_list.append(result)
                    _append_progress(job_id, f"{prefix} — Persisting partial result...")
                    persisted_to_db = _persist_company_result_to_db(job_id, result)
                    if persisted_to_db:
                        _minimize_completed_result_for_memory(result)
                    _update_partial_results_cache(job_id, upload_dir, results_list)
                    _append_progress(
                        job_id,
                        f"Partial results updated — {len([r for r in results_list if not r.get('skipped')])}/{total} companies completed.",
                    )
                except _JobStoppedError:
                    raise
                except Exception as exc:
                    import traceback
                    last_error = str(exc)
                    print(f"  ERROR evaluating {company.name}: {exc}")
                    traceback.print_exc()
                    if db and db.is_configured():
                        db.insert_analysis_error(
                            job_id,
                            message=str(exc)[:1000],
                            stage="evaluate_from_specter",
                            error_type=type(exc).__name__,
                            company_slug=store.startup_slug,
                        )
                        _persist_failed_company_result_to_db(
                            job_id,
                            company=company,
                            store=store,
                            slug=store.startup_slug,
                            error_message=str(exc)[:1000],
                            status="timeout" if isinstance(exc, TimeoutError) else "error",
                        )
                    results_list.append({
                        "slug": store.startup_slug,
                        "skipped": True,
                        "error": str(exc)[:500],
                        "company_name": company.name,
                    })

            if (
                chunking["enabled"]
                and chunk_idx < chunking["total_chunks"]
                and chunking["cooldown_seconds"] > 0
            ):
                _append_progress(
                    job_id,
                    f"Chunk {chunk_idx}/{chunking['total_chunks']} complete — cooling down for {chunking['cooldown_seconds']}s before next chunk.",
                )
                await asyncio.sleep(chunking["cooldown_seconds"])
    except _JobStoppedError:
        if _finalize_stopped_results(
            job_id,
            upload_dir,
            results_list,
            total=total,
            source="run_specter_analysis",
        ):
            return
        raise

    evaluated = [r for r in results_list if not r.get("skipped")]
    if not evaluated:
        if _is_stop_requested(job_id):
            raise _JobStoppedError("Job stopped by user")
        msg = "No startups were successfully evaluated."
        if last_error:
            msg += f" Last error: {last_error[:200]}"
        _set_job_status(job_id, "error", msg, source="run_specter_analysis")
        return

    _append_progress(job_id, "Finalizing batch results...")
    _build_results_payload(results_list, job_id, upload_dir)
    _append_progress(job_id, "Finalizing complete.")
    if _is_stop_requested(job_id):
        if _finalize_stopped_results(
            job_id,
            upload_dir,
            results_list,
            total=total,
            source="run_specter_analysis",
        ):
            return
        raise _JobStoppedError("Job stopped by user")
    _set_job_status(
        job_id,
        "done",
        f"Analysis complete — {len(evaluated)}/{total} companies ranked",
        source="run_specter_analysis",
    )
    _jobs[job_id].results = _results_cache[job_id]["results"]
    _persist_jobs()
    _persist_results_to_db(job_id, results_list)


async def _run_person_profile_job(job_id: str, req: PersonProfileJobRequest) -> None:
    """Execute asynchronous person profile generation job."""
    req_payload = req.model_dump()
    try:
        _person_jobs[job_id].status = "running"
        _person_jobs[job_id].progress = "Collecting evidence..."
        _persist_person_job(job_id, req_payload)
        built = await _person_service.build_profile(req)
        diagnostics: dict[str, Any] = {}
        if isinstance(built, tuple) and len(built) == 4:
            profile_json, profile_markdown, cache_key, diagnostics = built
        else:
            profile_json, profile_markdown, cache_key = built  # backward compatibility
            diagnostics = {}
        evidence_sources: dict[str, int] = {}
        for claim in getattr(profile_json, "claims", []) or []:
            for ev in getattr(claim, "evidence", []) or []:
                st = getattr(ev, "source_type", None) or "unknown"
                evidence_sources[st] = evidence_sources.get(st, 0) + 1
        diagnostics["evidence_sources"] = evidence_sources
        _person_jobs[job_id].status = "done"
        _person_jobs[job_id].progress = "Profile completed"
        _person_jobs[job_id].result = {
            "profile_json": profile_json.model_dump(),
            "profile_markdown": profile_markdown,
            "cache_key": cache_key,
            "diagnostics": diagnostics,
        }
        _persist_person_job(job_id, req_payload)
    except Exception as exc:
        _person_jobs[job_id].status = "error"
        _person_jobs[job_id].progress = "Profile generation failed"
        _person_jobs[job_id].error = str(exc)
        _persist_person_job(job_id, req_payload)


@app.post("/api/person-profile/jobs")
async def create_person_profile_job(
    req: PersonProfileJobRequest,
    session_id: str | None = Cookie(default=None),
):
    """Create one person intelligence job."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    job_id = str(uuid.uuid4())[:8]
    _person_jobs[job_id] = PersonProfileJobStatus(
        job_id=job_id,
        status="pending",
        progress="Queued",
    )
    _persist_person_job(job_id, req.model_dump())
    asyncio.create_task(_run_person_profile_job(job_id, req))
    return {"job_id": job_id, "status": "running"}


@app.get("/api/person-profile/status/{job_id}")
async def get_person_profile_status(
    job_id: str,
    session_id: str | None = Cookie(default=None),
):
    """Fetch person intelligence job state/result."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if job_id not in _person_jobs:
        if db and db.is_configured():
            loaded = db.load_person_profile_job(job_id)
            if loaded:
                _person_jobs[job_id] = PersonProfileJobStatus(
                    job_id=job_id,
                    status=loaded.get("status", "pending"),
                    progress=loaded.get("progress") or "",
                    result=loaded.get("result_payload"),
                    error=loaded.get("error"),
                )
            else:
                raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=404, detail="Job not found")

    job = _person_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
    }


@app.post("/api/person-profile/jobs/bulk-founders")
async def create_bulk_founder_jobs(
    req: BulkFounderJobRequest,
    session_id: str | None = Cookie(default=None),
):
    """Create one enrichment job per founder candidate."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for idx, founder in enumerate(req.founders):
        person_key = founder.person_key or f"{req.company_slug}:founder:{idx + 1}"
        if not founder.primary_profile_url:
            skipped.append(
                {
                    "person_key": person_key,
                    "full_name": founder.full_name or f"Founder {idx + 1}",
                    "status": "needs_input",
                    "reason": "missing_primary_profile_url",
                }
            )
            continue

        profile_req = PersonProfileJobRequest(
            primary_profile_url=founder.primary_profile_url,
            full_name=founder.full_name,
            location=founder.location,
            current_company=founder.current_company,
            role=founder.role,
            known_aliases=founder.known_aliases,
            user_uploaded_text=req.user_uploaded_text,
            user_uploaded_images=req.user_uploaded_images,
            company_slug=req.company_slug,
            person_key=person_key,
        )
        job_id = str(uuid.uuid4())[:8]
        _person_jobs[job_id] = PersonProfileJobStatus(
            job_id=job_id,
            status="pending",
            progress="Queued",
        )
        _persist_person_job(job_id, profile_req.model_dump())
        asyncio.create_task(_run_person_profile_job(job_id, profile_req))
        created.append(
            {
                "person_key": person_key,
                "full_name": founder.full_name or "",
                "job_id": job_id,
                "status": "running",
            }
        )

    return {
        "company_slug": req.company_slug,
        "jobs": created,
        "skipped": skipped,
    }


@app.get("/api/config")
async def get_config(session_id: str | None = Cookie(default=None)):
    """Return current app config (e.g. LLM) for backfilling past runs."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    default_llm = current_default_selection()
    return {
        "llm": default_llm["label"],
        "default_llm": default_llm,
        "available_models": available_models_payload(),
        "pricing_catalog": pricing_catalog_payload(),
        "phase_model_defaults": phase_model_defaults_payload(),
        "quality_tiers": quality_tiers_payload(),
        "premium_phase_options": premium_phase_options_payload(),
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if job_id in _jobs:
        job = _jobs[job_id]
        results = _results_cache.get(job_id, {}).get("results")
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "progress_log": getattr(job, "progress_log", []) or [],
            "results": results,
            "llm": _resolve_job_llm_label(job_id, results=results),
        }

    if db and db.is_configured():
        loaded = db.load_job_results(job_id)
        if loaded:
            results = loaded.get("results")
            loaded_status = (results or {}).get("job_status") or "done"
            loaded_progress = (results or {}).get("job_message") or "Analysis complete"
            _jobs[job_id] = AnalysisStatus(
                job_id=job_id,
                status=loaded_status,
                progress=loaded_progress,
                progress_log=[],
                results=results,
            )
            _results_cache[job_id] = {
                "results": results,
            }
            return {
                "job_id": job_id,
                "status": loaded_status,
                "progress": loaded_progress,
                "progress_log": [],
                "results": results,
                "llm": _selection_from_payload(results)["label"]
                if _selection_from_payload(results)
                else _get_llm_display(),
            }

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/download/{job_id}")
async def download_excel(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    raise HTTPException(status_code=410, detail="Excel export has been removed")


@app.get("/api/analyses/{job_id}")
async def get_analysis(job_id: str, session_id: str | None = Cookie(default=None)):
    """Return analysis results for a completed job. Uses in-memory cache or Supabase."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    cache = _results_cache.get(job_id, {})
    results = cache.get("results")
    if results:
        return {"job_id": job_id, "results": results}

    if db and db.is_configured():
        loaded = db.load_job_results(job_id)
        if loaded:
            return {"job_id": job_id, "results": loaded.get("results")}

    raise HTTPException(status_code=404, detail="Analysis not found")


@app.get("/api/jobs")
async def list_jobs(session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {"jobs": _list_jobs_for_ui()}


@app.get("/api/company-runs")
async def list_company_runs(session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not db or not db.is_configured():
        return {"companies": []}

    return {"companies": db.list_company_histories(perform_maintenance=False)}


@app.get("/api/companies/{company_name}/analyses")
async def get_company_analyses(
    company_name: str,
    session_id: str | None = Cookie(default=None),
):
    """Return analyses for a company by name. Requires Supabase."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not db or not db.is_configured():
        raise HTTPException(status_code=501, detail="Supabase not configured")

    analyses = db.load_analyses_by_company(company_name)
    return {"company_name": company_name, "analyses": analyses}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
