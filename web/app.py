"""Rockaway Deal Intelligence web application.

FastAPI backend serving the Rockaway-branded UI with password protection.
Provider API keys stay server-side only and are never exposed to the client.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import gc
import hashlib
import hmac
import json
import logging
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


# ---------------------------------------------------------------------------
# In-memory log buffer — captures all Python logging for the admin live log.
# Thread-safe circular buffer; survives for the lifetime of the process.
# ---------------------------------------------------------------------------

class _InMemoryLogHandler(logging.Handler):
    """Append log records to a shared deque so the admin portal can poll them."""

    _records: collections.deque = collections.deque(maxlen=500)
    _lock: threading.Lock = threading.Lock()

    LEVEL_COLOURS = {
        "DEBUG": "muted",
        "INFO": "info",
        "WARNING": "warning",
        "ERROR": "error",
        "CRITICAL": "error",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": self.format(record),
            }
            with self._lock:
                self._records.append(entry)
        except Exception:
            pass  # never crash the app due to logging

    @classmethod
    def get_entries(cls, since_ts: str | None = None, level: str | None = None) -> list[dict]:
        with cls._lock:
            records = list(cls._records)
        if since_ts:
            records = [r for r in records if r["ts"] > since_ts]
        if level and level != "ALL":
            lvl = getattr(logging, level, logging.DEBUG)
            records = [r for r in records if getattr(logging, r["level"], 0) >= lvl]
        return records


# Install the handler on the root logger once at import time.
_mem_handler = _InMemoryLogHandler()
_mem_handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
_mem_handler.setLevel(logging.DEBUG)
_root_logger = logging.getLogger()
if not any(isinstance(h, _InMemoryLogHandler) for h in _root_logger.handlers):
    _root_logger.addHandler(_mem_handler)
    if _root_logger.level == logging.NOTSET or _root_logger.level > logging.DEBUG:
        _root_logger.setLevel(logging.DEBUG)

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, Cookie, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.llm_catalog import (
    available_models_payload,
    available_chat_models_payload,
    model_label,
    current_default_selection,
    normalize_creativity,
    normalize_provider,
    pricing_catalog_payload,
    serialize_selection,
    validate_chat_requested_selection,
    validate_requested_selection,
)
from agent.llm_policy import (
    PHASE_LABELS,
    PHASE_SHORT_LABELS,
    build_default_phase_model_policy,
    build_phase_model_policy,
    build_phase_policy_display_label,
    build_pipeline_policy,
    build_tier_display_label,
    coerce_phase_models_payload,
    default_phase_model_selections,
    normalize_phase_models,
    normalize_premium_phase_models,
    normalize_quality_tier,
    phase_model_defaults_payload,
    premium_phase_options_payload,
    quality_tiers_payload,
    resolve_effective_phase_choices,
    resolve_effective_phase_models,
)
from agent.company_chat import answer_company_question, build_company_chat_store
from agent.run_context import (
    RunTelemetryCollector,
    build_run_costs_from_model_executions,
    use_company_context,
    use_run_context,
)

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
try:
    MAX_PROGRESS_LOG_ENTRIES = max(1, int(os.getenv("MAX_PROGRESS_LOG_ENTRIES", "200")))
except Exception:
    MAX_PROGRESS_LOG_ENTRIES = 200
try:
    PERSISTED_STATUS_SYNC_INTERVAL_SECONDS = max(
        1, int(os.getenv("PERSISTED_STATUS_SYNC_INTERVAL_SECONDS", "10"))
    )
except Exception:
    PERSISTED_STATUS_SYNC_INTERVAL_SECONDS = 10
RESTART_ON_IDLE_AFTER_ANALYSIS = os.getenv("RESTART_ON_IDLE_AFTER_ANALYSIS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_CHUNKED_SPECTER_PERSISTENCE = os.getenv("ENABLE_CHUNKED_SPECTER_PERSISTENCE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_SPECTER_SUBPROCESS_CHUNKS = os.getenv("ENABLE_SPECTER_SUBPROCESS_CHUNKS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_SPECTER_WORKER_SERVICE = os.getenv("ENABLE_SPECTER_WORKER_SERVICE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
try:
    RESTART_ON_IDLE_DELAY_SECONDS = max(5, int(os.getenv("RESTART_ON_IDLE_DELAY_SECONDS", "15")))
except Exception:
    RESTART_ON_IDLE_DELAY_SECONDS = 15
try:
    JOBS_OVERVIEW_CACHE_SECONDS = max(1, int(os.getenv("JOBS_OVERVIEW_CACHE_SECONDS", "5")))
except Exception:
    JOBS_OVERVIEW_CACHE_SECONDS = 5
try:
    COMPANY_RUNS_CACHE_SECONDS = max(5, int(os.getenv("COMPANY_RUNS_CACHE_SECONDS", "30")))
except Exception:
    COMPANY_RUNS_CACHE_SECONDS = 30
try:
    COMPANY_RUNS_OVERVIEW_LIMIT = max(50, int(os.getenv("COMPANY_RUNS_OVERVIEW_LIMIT", "250")))
except Exception:
    COMPANY_RUNS_OVERVIEW_LIMIT = 250
_sessions: dict[str, float] = {}
_results_cache: dict[str, dict[str, Any]] = {}
_restart_timer: threading.Timer | None = None
_restart_lock = threading.Lock()
_overview_cache_lock = threading.Lock()
_jobs_overview_cache: dict[str, Any] = {"expires_at": 0.0, "payload": None}
_company_runs_cache: dict[str, Any] = {"expires_at": 0.0, "payload": None}
SPECTER_CHUNK_EVENT_PREFIX = "__SPECTER_CHUNK_EVENT__"
SPECTER_COMPANY_EVENT_PREFIX = "__SPECTER_COMPANY_EVENT__"
_company_chat_sessions: dict[tuple[str, str], dict[str, Any]] = {}

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
VENDOR_DIR = PROJECT_ROOT / "node_modules"
if VENDOR_DIR.exists():
    app.mount("/vendor", StaticFiles(directory=str(VENDOR_DIR)), name="vendor")


def _lazy_import_pandas():
    import pandas as pd

    return pd


def _lazy_import_batch():
    from agent import batch as batch_module

    return batch_module


def _lazy_import_company():
    from agent.dataclasses.company import Company

    return Company


def _lazy_import_ingest_store():
    from agent.ingest.store import Chunk, EvidenceStore

    return Chunk, EvidenceStore


def _lazy_import_ingest_startup_folder():
    from agent.ingest import ingest_startup_folder

    return ingest_startup_folder


def _lazy_import_ingest_specter():
    from agent.ingest.specter_ingest import ingest_specter

    return ingest_specter


def _lazy_import_list_specter_companies():
    from agent.ingest.specter_ingest import list_specter_companies

    return list_specter_companies


def _lazy_import_ingest_specter_company():
    from agent.ingest.specter_ingest import ingest_specter_company

    return ingest_specter_company


def build_argument_rows(results_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _lazy_import_batch().build_argument_rows(results_list)


def build_failed_rows(results_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _lazy_import_batch().build_failed_rows(results_list)


def build_qa_provenance_rows(results_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _lazy_import_batch().build_qa_provenance_rows(results_list)


def build_summary_rows(results_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _lazy_import_batch().build_summary_rows(results_list)


async def evaluate_from_specter(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return await _lazy_import_batch().evaluate_from_specter(*args, **kwargs)


async def evaluate_startup(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return await _lazy_import_batch().evaluate_startup(*args, **kwargs)


def ingest_specter(*args: Any, **kwargs: Any) -> list[tuple[Any, Any]]:
    return _lazy_import_ingest_specter()(*args, **kwargs)


def list_specter_companies(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return _lazy_import_list_specter_companies()(*args, **kwargs)


def ingest_specter_company(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    return _lazy_import_ingest_specter_company()(*args, **kwargs)


def ingest_startup_folder(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import_ingest_startup_folder()(*args, **kwargs)


def rank_batch_companies(results_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _lazy_import_batch().rank_batch_companies(results_list)


def _set_no_store_headers(response: Response | None) -> None:
    if response is None:
        return
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

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


def _selection_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    selection = payload.get("llm_selection")
    if isinstance(selection, dict) and selection.get("provider") and selection.get("model"):
        return serialize_selection(
            selection.get("provider"),
            selection.get("model"),
            creativity=selection.get("creativity"),
        )
    provider = payload.get("llm_provider") or payload.get("provider")
    model = payload.get("llm_model") or payload.get("model")
    if provider and model:
        return serialize_selection(provider, model, creativity=payload.get("creativity"))
    return None


def _llm_label_from_payload(payload: dict[str, Any] | None) -> str | None:
    pipeline_meta = _pipeline_meta_from_payload(payload)
    if pipeline_meta:
        if pipeline_meta.get("phase_models"):
            return build_phase_policy_display_label(
                pipeline_meta.get("effective_phase_models") or pipeline_meta["phase_models"]
            )
        return build_tier_display_label(
            pipeline_meta["quality_tier"],
            effective_phase_models=pipeline_meta.get("effective_phase_models"),
        )
    if isinstance(payload, dict):
        explicit = (payload.get("llm") or "").strip()
        if explicit and not _selection_from_payload(payload):
            return explicit
    selection = _selection_from_payload(payload)
    if selection:
        return selection["label"]
    return None


def _resolve_job_llm_selection(job_id: str, *, results: dict[str, Any] | None = None) -> dict[str, Any]:
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
    cache = _results_cache.get(job_id, {})
    for candidate in (
        _llm_label_from_payload(results),
        _llm_label_from_payload(cache.get("results")),
        _llm_label_from_payload(cache.get("run_config")),
        _llm_label_from_payload(cache.get("llm_selection")),
    ):
        if candidate:
            return candidate
    return _resolve_job_llm_selection(job_id, results=results)["label"]


def _resolve_batch_chunking_selection(job_id: str) -> dict[str, Any]:
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


def _promote_results_metadata(job_id: str, results: dict[str, Any] | None) -> None:
    if not isinstance(results, dict):
        return
    cache = _results_cache.setdefault(job_id, {})

    mode = results.get("mode")
    if isinstance(mode, str) and mode:
        cache["input_mode"] = mode

    selection = _selection_from_payload(results)
    if selection:
        cache["llm_selection"] = selection

    pipeline_meta = _pipeline_meta_from_payload(results)
    if pipeline_meta:
        run_config = dict(cache.get("run_config") or {})
        run_config["phase_models"] = pipeline_meta.get("phase_models")
        run_config["quality_tier"] = pipeline_meta.get("quality_tier")
        run_config["premium_phase_models"] = pipeline_meta.get("premium_phase_models")
        run_config["effective_phase_models"] = pipeline_meta.get("effective_phase_models")
        cache["run_config"] = run_config

    started_by = _started_by_from_payload(results)
    if started_by:
        job = _jobs.get(job_id)
        if job:
            job.started_by_user_id = started_by.get("started_by_user_id")
            job.started_by_email = started_by.get("started_by_email")
            job.started_by_display_name = started_by.get("started_by_display_name")
            job.started_by_label = started_by.get("started_by_label")
        run_config = dict(cache.get("run_config") or {})
        run_config.update(started_by)
        cache["run_config"] = run_config


def _load_local_job_results(job_id: str) -> dict[str, Any] | None:
    if not JOBS_STORE_PATH.exists():
        return None
    try:
        raw = json.loads(JOBS_STORE_PATH.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    data = raw.get(job_id)
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, dict):
        return None
    return {"results": results}


def _load_persisted_job_results(
    job_id: str,
    *,
    preferred_mode: str | None = None,
) -> dict[str, Any] | None:
    loaded = None
    if db and db.is_configured():
        try:
            loaded = db.load_job_results(job_id, preferred_mode=preferred_mode)
        except TypeError:
            loaded = db.load_job_results(job_id)
        except Exception:
            loaded = None
    return loaded or _load_local_job_results(job_id)


def _is_compact_results_payload(results: dict[str, Any] | None) -> bool:
    return bool(isinstance(results, dict) and results.get("_memory_compact"))


def _completed_count_from_results_payload(results: dict[str, Any] | None) -> int:
    if not isinstance(results, dict):
        return 0
    if isinstance(results.get("summary_rows_count"), int):
        return int(results["summary_rows_count"])
    summary_rows = results.get("summary_rows")
    if isinstance(summary_rows, list):
        return len(summary_rows)
    if results.get("mode") == "single" and (results.get("company_name") or results.get("startup_slug")):
        return 1
    if isinstance(results.get("num_companies"), int):
        return int(results["num_companies"])
    return 0


def _compact_results_for_runtime(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    compact: dict[str, Any] = {
        "_memory_compact": True,
        "mode": payload.get("mode"),
        "job_status": payload.get("job_status"),
        "job_message": payload.get("job_message"),
        "llm": payload.get("llm"),
        "llm_selection": payload.get("llm_selection"),
        "run_costs": payload.get("run_costs"),
        "batch_chunking": payload.get("batch_chunking"),
        "num_companies": payload.get("num_companies"),
        "num_skipped": payload.get("num_skipped"),
        "summary_rows_count": _completed_count_from_results_payload(payload),
        "failed_rows_count": len(payload.get("failed_rows") or []) if isinstance(payload.get("failed_rows"), list) else int(payload.get("failed_rows_count") or 0),
        "argument_rows_count": len(payload.get("argument_rows") or []) if isinstance(payload.get("argument_rows"), list) else int(payload.get("argument_rows_count") or 0),
        "qa_provenance_rows_count": len(payload.get("qa_provenance_rows") or []) if isinstance(payload.get("qa_provenance_rows"), list) else int(payload.get("qa_provenance_rows_count") or 0),
    }
    if payload.get("mode") == "single":
        for key in (
            "company_name",
            "startup_slug",
            "decision",
            "total_score",
            "avg_pro",
            "avg_contra",
        ):
            if key in payload:
                compact[key] = payload.get(key)
    return compact


def _release_job_runtime_resources(job_id: str, *, drop_results: bool) -> None:
    cache = _results_cache.get(job_id)
    if cache is None:
        _job_controls.pop(job_id, None)
        return

    _promote_results_metadata(job_id, cache.get("results"))

    upload_dir = cache.pop("upload_dir", None)
    if isinstance(upload_dir, str) and upload_dir:
        with contextlib.suppress(Exception):
            shutil.rmtree(upload_dir, ignore_errors=True)

    for key in (
        "files",
        "specter",
        "telemetry_collector",
        "model_executions",
        "run_costs_aggregate",
        "versions",
    ):
        cache.pop(key, None)

    job = _jobs.get(job_id)
    if job is not None:
        log = getattr(job, "progress_log", []) or []
        job.progress_log = log[-MAX_PROGRESS_LOG_ENTRIES:]
        if drop_results:
            job.results = None

    if drop_results:
        cache.pop("results", None)

    _job_controls.pop(job_id, None)
    gc.collect()


def _has_active_analysis_jobs() -> bool:
    return any(job.status in {"pending", "running", "paused"} for job in _jobs.values())


def _maybe_promote_terminal_persisted_results(
    job_id: str,
    *,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    job = _jobs.get(job_id)
    if not job or job.status in {"done", "error", "stopped"}:
        return None
    if not db or not db.is_configured():
        return None

    now = time.monotonic()
    last_check = float(getattr(job, "last_persisted_status_check_at", 0.0) or 0.0)
    if now - last_check < PERSISTED_STATUS_SYNC_INTERVAL_SECONDS:
        return None
    job.last_persisted_status_check_at = now

    try:
        persisted_status = db.load_job_status(job_id)
    except Exception:
        persisted_status = None
    if not isinstance(persisted_status, dict):
        return None

    status = str(persisted_status.get("status") or "").strip().lower()
    if status not in {"done", "error", "stopped"}:
        return None

    loaded = _load_persisted_job_results(
        job_id,
        preferred_mode=(cache or {}).get("input_mode"),
    )
    results = (loaded or {}).get("results")
    if not isinstance(results, dict):
        return None

    progress = (
        str(results.get("job_message") or "").strip()
        or str(persisted_status.get("progress") or "").strip()
        or ("Analysis complete" if status == "done" else job.progress)
    )
    job.status = status
    job.progress = progress
    job.persistence_complete = True
    _results_cache.setdefault(job_id, {})["results"] = results
    job.results = results
    _promote_results_metadata(job_id, results)
    return results


def _has_active_person_jobs() -> bool:
    return any(job.status in {"pending", "running"} for job in _person_jobs.values())


def _cancel_scheduled_restart() -> None:
    global _restart_timer
    with _restart_lock:
        timer = _restart_timer
        _restart_timer = None
        if timer and timer.is_alive():
            timer.cancel()


def _restart_process_for_memory_reset() -> None:
    print("Restarting process after analysis completion to reclaim memory.")
    os._exit(1)


def _schedule_idle_restart_if_enabled(job_id: str) -> None:
    global _restart_timer
    job = _jobs.get(job_id)
    if not job or not job.restart_pending or not job.terminal_results_served or not job.persistence_complete:
        return
    if not RESTART_ON_IDLE_AFTER_ANALYSIS:
        return
    if not (db and db.is_configured()):
        return
    if _has_active_analysis_jobs() or _has_active_person_jobs():
        return

    with _restart_lock:
        if _restart_timer and _restart_timer.is_alive():
            return
        print(
            f"Scheduling process restart in {RESTART_ON_IDLE_DELAY_SECONDS}s "
            f"after terminal analysis job {job_id}.",
        )
        timer = threading.Timer(
            RESTART_ON_IDLE_DELAY_SECONDS,
            _restart_process_for_memory_reset,
        )
        timer.daemon = True
        _restart_timer = timer
        timer.start()


def _mark_terminal_results_served(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job or job.status not in {"done", "stopped"}:
        return
    job.terminal_results_served = True
    _schedule_idle_restart_if_enabled(job_id)


def _mark_terminal_persistence_complete(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job or job.status not in {"done", "stopped"}:
        return
    job.persistence_complete = True
    _schedule_idle_restart_if_enabled(job_id)


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


def _persisted_company_key(name: str | None, slug: str | None) -> str:
    base = (slug or name or "").strip().lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return base or "unknown"


def _load_persisted_company_keys(job_id: str) -> set[str]:
    if not (db and db.is_configured()):
        return set()
    try:
        rows = db.load_job_company_runs(job_id)
    except Exception:
        return set()

    keys: set[str] = set()
    for row in rows or []:
        keys.add(
            _persisted_company_key(
                row.get("company_name"),
                row.get("startup_slug"),
            )
        )
    return keys


class LoginRequest(BaseModel):
    password: str


# ---------------------------------------------------------------------------
# Sprint 1 — Request / Response schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    """New user registration payload for the startup/vc portal."""

    email: str
    password: str
    role: Literal["startup", "vc"]
    display_name: str | None = None
    organization: str | None = None


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    role: str
    message: str


class VerifyDomainRequest(BaseModel):
    email: str
    company_id: str


class VerifyDomainResponse(BaseModel):
    verified: bool
    domain: str | None = None
    message: str


class FundraisingToggleRequest(BaseModel):
    fundraising: bool


# Sprint 2 — VC portal schemas

class VCProfileRequest(BaseModel):
    """Create or update a VC profile."""
    firm_name: str
    investment_thesis: str | None = None


class VCThesisRequest(BaseModel):
    """Update only the VC's investment thesis text."""
    investment_thesis: str


class VCThresholdsRequest(BaseModel):
    """Update match score thresholds (0–100 each)."""
    min_strategy_fit: int = Field(default=0, ge=0, le=100)
    min_team: int = Field(default=0, ge=0, le=100)
    min_potential: int = Field(default=0, ge=0, le=100)


class MatchActionRequest(BaseModel):
    """VC action on a match."""
    action: Literal["viewed", "interested", "passed", "in_debate"]


# ---------------------------------------------------------------------------

def _ensure_str(val: Any) -> str:
    """Normalize to str; handle list to avoid 'list' has no attribute 'strip'."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val) if val else ""
    return str(val)


class AnalyzeRequest(BaseModel):
    use_web_search: bool = False
    use_specter_mcp: bool = True
    instructions: str | None = None
    input_mode: str = "pitchdeck"  # pitchdeck | specter | original
    run_name: str | None = None
    vc_investment_strategy: str | None = None
    phase_models: dict[str, dict[str, Any]] | None = None
    quality_tier: Literal["cheap", "premium"] | None = None
    premium_phase_models: dict[str, Literal["claude", "gpt5"]] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None

    @field_validator(
        "instructions",
        "run_name",
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
    def _coerce_phase_models(cls, v: Any) -> dict[str, dict[str, Any]] | None:
        if v is None:
            return None
        normalized = coerce_phase_models_payload(v, require_all=True)
        phase_models: dict[str, dict[str, Any]] = {}
        for phase, selection in normalized.items():
            creativity = normalize_creativity(selection.get("creativity"))
            temperature = selection.get("temperature")
            reasoning_effort = selection.get("reasoning_effort")
            phase_models[phase] = {
                "provider": selection["provider"],
                "model": selection["model"],
                **({"creativity": creativity} if creativity is not None else {}),
                **({"temperature": temperature} if temperature is not None else {}),
                **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
            }
        return phase_models

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
    progress_log: list[str] = Field(default_factory=list)
    results: object | None = None
    started_by_user_id: str | None = None
    started_by_email: str | None = None
    started_by_display_name: str | None = None
    started_by_label: str | None = None
    terminal_results_served: bool = False
    restart_pending: bool = False
    persistence_complete: bool = False
    last_persisted_status_check_at: float = 0.0


AnalysisStatus.model_rebuild()

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


PersonProfileJobStatus.model_rebuild()

_person_jobs: dict[str, PersonProfileJobStatus] = {}
_person_job_tasks: dict[str, asyncio.Task[Any]] = {}
_person_service = PersonIntelService()


class CompanyChatRequest(BaseModel):
    message: str
    active_job_id: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None

    @field_validator("message")
    @classmethod
    def _validate_message(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("message is required")
        return text[:4000]


class VcStrategyRequest(BaseModel):
    vc_investment_strategy: str = ""

    @field_validator("vc_investment_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        return (v or "").strip()[:8000]


class CompanyChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str
    llm_label: str | None = None
    run_costs: dict[str, Any] | None = None


class CompanyChatResponse(BaseModel):
    company_lookup_key: str
    transcript: list[CompanyChatMessage]
    run_count: int = 0
    source_counts: dict[str, int] = Field(default_factory=dict)
    web_search_enabled: bool = True
    model_label: str = "Gemini 3.1 Flash Lite"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.1-flash-lite-preview"
    session_run_costs: dict[str, Any] = Field(default_factory=dict)
    available_models: list[dict[str, Any]] = Field(default_factory=list)
    used_run_ids: list[str] = Field(default_factory=list)
    used_web_search: bool = False
    web_search_query: str | None = None


CompanyChatMessage.model_rebuild()
CompanyChatResponse.model_rebuild()


def _chat_session_key(session_id: str, company_lookup_key: str) -> tuple[str, str]:
    return (session_id, company_lookup_key.strip().lower())


COMPANY_CHAT_DB_TIMEOUT_SEC = 3.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_summary(text: str, limit: int = 4000) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _compress_company_chat_session(session: dict[str, Any], max_messages: int = 24) -> None:
    transcript = session.setdefault("transcript", [])
    while len(transcript) > max_messages:
        oldest = transcript.pop(0)
        summary = session.get("summary") or ""
        summary_line = f"{oldest.get('role', 'unknown')}: {oldest.get('content', '')}"
        session["summary"] = _truncate_summary(f"{summary}\n{summary_line}".strip())


def _default_company_chat_session() -> dict[str, Any]:
    return {
        "company_name": None,
        "summary": "",
        "transcript": [],
        "selection": serialize_selection("gemini", "gemini-3.1-flash-lite-preview"),
        "model_executions": [],
    }


def _normalize_company_chat_transcript(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            normalized.append(CompanyChatMessage(**item).model_dump())
        except Exception:
            continue
    return normalized


def _normalize_company_chat_session_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    session = _default_company_chat_session()
    source = payload or {}
    session["company_name"] = source.get("company_name")
    session["summary"] = str(source.get("summary") or "")
    session["transcript"] = _normalize_company_chat_transcript(source.get("transcript") or [])
    session["model_executions"] = list(source.get("model_executions") or [])
    session["selection"] = _company_chat_selection(source)
    _compress_company_chat_session(session)
    return session


def _get_or_create_company_chat_session(session_id: str, company_lookup_key: str) -> dict[str, Any]:
    return _company_chat_sessions.setdefault(_chat_session_key(session_id, company_lookup_key), _default_company_chat_session())


async def _load_company_chat_session(session_id: str, company_lookup_key: str) -> dict[str, Any]:
    if db and db.is_configured():
        load_persisted = getattr(db, "load_company_chat_session", None)
        persisted = await _call_company_chat_db(load_persisted, company_lookup_key)
        if isinstance(persisted, dict):
            session = _normalize_company_chat_session_payload(persisted)
            _company_chat_sessions[_chat_session_key(session_id, company_lookup_key)] = session
            return session
    return _normalize_company_chat_session_payload(_get_or_create_company_chat_session(session_id, company_lookup_key))


async def _persist_company_chat_session(
    session_id: str,
    company_lookup_key: str,
    company_name: str,
    chat_session: dict[str, Any],
) -> None:
    normalized = _normalize_company_chat_session_payload(chat_session)
    normalized["company_name"] = company_name
    _company_chat_sessions[_chat_session_key(session_id, company_lookup_key)] = normalized
    if db and db.is_configured():
        persist_persisted = getattr(db, "persist_company_chat_session", None)
        await _call_company_chat_db(
            persist_persisted,
            company_lookup_key=company_lookup_key,
            company_name=company_name,
            selection=normalized.get("selection"),
            summary=normalized.get("summary") or "",
            transcript=normalized.get("transcript") or [],
            model_executions=normalized.get("model_executions") or [],
        )


async def _clear_company_chat_session(session_id: str, company_lookup_key: str) -> dict[str, Any]:
    current = await _load_company_chat_session(session_id, company_lookup_key)
    selection = _company_chat_selection(current)
    company_name = current.get("company_name") or company_lookup_key
    normalized_key = (company_lookup_key or "").strip().lower()
    for key in list(_company_chat_sessions.keys()):
        if key[1] == normalized_key:
            _company_chat_sessions.pop(key, None)
    cleared = _default_company_chat_session()
    cleared["selection"] = selection
    cleared["company_name"] = company_name
    _company_chat_sessions[_chat_session_key(session_id, company_lookup_key)] = cleared
    if db and db.is_configured():
        persist_persisted = getattr(db, "persist_company_chat_session", None)
        await _call_company_chat_db(
            persist_persisted,
            company_lookup_key=company_lookup_key,
            company_name=company_name,
            selection=selection,
            summary="",
            transcript=[],
            model_executions=[],
        )
    return cleared


async def _call_company_chat_db(func: Any, *args: Any, timeout: float = COMPANY_CHAT_DB_TIMEOUT_SEC, **kwargs: Any) -> Any:
    if not callable(func):
        return None
    try:
        return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=timeout)
    except Exception:
        return None


def _company_chat_selection(session: dict[str, Any] | None = None) -> dict[str, str]:
    selection = (session or {}).get("selection") if isinstance(session, dict) else None
    if isinstance(selection, dict) and selection.get("provider") and selection.get("model"):
        return serialize_selection(selection.get("provider"), selection.get("model"))
    default = serialize_selection("gemini", "gemini-3.1-flash-lite-preview")
    try:
        validate_chat_requested_selection(default.get("provider"), default.get("model"))
        return default
    except Exception:
        fallback = current_default_selection()
        return serialize_selection(fallback.get("provider"), fallback.get("model"))


def _company_chat_model_label(session: dict[str, Any] | None = None) -> str:
    selection = _company_chat_selection(session)
    return model_label(selection.get("provider"), selection.get("model"))


def _resolve_requested_company_chat_selection(provider: str | None, model: str | None) -> dict[str, str]:
    validation_error: ValueError | None = None
    try:
        selected_entry = validate_chat_requested_selection(provider, model)
    except ValueError as exc:
        validation_error = exc
        selected_entry = None
    if selected_entry:
        return serialize_selection(selected_entry.provider, selected_entry.model)

    provider_norm = normalize_provider(provider)
    model_norm = (model or "").strip()
    for item in available_chat_models_payload():
        if item.get("provider") != provider_norm or item.get("model") != model_norm:
            continue
        if not item.get("available"):
            raise ValueError(f'{item.get("label") or model_norm} is not available in this environment.')
        if item.get("selectable") is False:
            raise ValueError(f'{item.get("label") or model_norm} is not selectable for company chat.')
        return serialize_selection(provider_norm, model_norm)

    if validation_error is not None:
        raise validation_error
    raise ValueError("Unknown chat LLM model selection.")


def _company_chat_session_costs(session: dict[str, Any] | None) -> dict[str, Any]:
    transcript = list((session or {}).get("transcript") or [])
    aggregate: dict[str, Any] | None = None
    for item in transcript:
        if item.get("role") != "assistant":
            continue
        run_costs = item.get("run_costs")
        if not isinstance(run_costs, dict):
            continue
        aggregate = _merge_run_cost_summaries(aggregate, run_costs)
    if isinstance(aggregate, dict):
        return aggregate
    rows = list((session or {}).get("model_executions") or [])
    return build_run_costs_from_model_executions(rows)


async def _load_shared_vc_strategy() -> str | None:
    if db and db.is_configured():
        load_setting = getattr(db, "load_app_setting", None)
        value = await _call_company_chat_db(load_setting, "vc_investment_strategy")
        if value is None:
            return None
        return str(value or "")
    return None


async def _save_shared_vc_strategy(value: str) -> bool:
    if db and db.is_configured():
        save_setting = getattr(db, "save_app_setting", None)
        result = await _call_company_chat_db(save_setting, "vc_investment_strategy", value)
        return bool(result)
    return False


def _runtime_versions() -> dict[str, str]:
    return {
        "app_version": os.getenv("APP_VERSION", "dev"),
        "prompt_version": os.getenv("PROMPT_VERSION", "v1"),
        "pipeline_version": os.getenv("PIPELINE_VERSION", "v1"),
        "schema_version": os.getenv("SCHEMA_VERSION", "20260306000000"),
    }


def _run_costs_from_cache(job_id: str) -> dict[str, Any]:
    cache = _results_cache.get(job_id, {})
    aggregate = cache.get("run_costs_aggregate")
    collector = cache.get("telemetry_collector")
    if isinstance(collector, RunTelemetryCollector):
        current = collector.build_run_costs()
        if isinstance(aggregate, dict):
            return _merge_run_cost_summaries(aggregate, current)
        return current
    if isinstance(aggregate, dict):
        return aggregate
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


def _empty_run_costs_summary() -> dict[str, Any]:
    return {
        "currency": "USD",
        "status": "unavailable",
        "total_usd": None,
        "llm_usd": None,
        "perplexity_usd": 0.0,
        "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "perplexity_search": {"requests": 0, "total_usd": 0.0},
        "by_model": [],
    }


def _merge_run_cost_summaries(base: dict[str, Any] | None, delta: dict[str, Any] | None) -> dict[str, Any]:
    merged = _empty_run_costs_summary()
    base = base or {}
    delta = delta or {}

    status_order = {"complete": 0, "partial": 1, "unavailable": 2}
    base_status = base.get("status") or "unavailable"
    delta_status = delta.get("status") or "unavailable"
    merged["status"] = (
        base_status
        if status_order.get(base_status, 2) <= status_order.get(delta_status, 2)
        else delta_status
    )

    for side in (base, delta):
        llm_tokens = side.get("llm_tokens") or {}
        merged["llm_tokens"]["prompt"] += int(llm_tokens.get("prompt") or 0)
        merged["llm_tokens"]["completion"] += int(llm_tokens.get("completion") or 0)
        merged["llm_tokens"]["total"] += int(llm_tokens.get("total") or 0)
        merged["perplexity_search"]["requests"] += int((side.get("perplexity_search") or {}).get("requests") or 0)
        merged["perplexity_search"]["total_usd"] += float((side.get("perplexity_search") or {}).get("total_usd") or 0.0)
        merged["perplexity_usd"] += float(side.get("perplexity_usd") or 0.0)

    def _sum_optional(left: Any, right: Any) -> float | None:
        known = [float(v) for v in (left, right) if isinstance(v, (int, float))]
        return round(sum(known), 8) if known else None

    merged["llm_usd"] = _sum_optional(base.get("llm_usd"), delta.get("llm_usd"))
    merged["total_usd"] = _sum_optional(base.get("total_usd"), delta.get("total_usd"))
    merged["perplexity_usd"] = round(merged["perplexity_usd"], 8)
    merged["perplexity_search"]["total_usd"] = round(merged["perplexity_search"]["total_usd"], 8)

    by_model: dict[tuple[str, str], dict[str, Any]] = {}
    for side in (base, delta):
        for item in side.get("by_model") or []:
            key = (str(item.get("provider") or ""), str(item.get("model") or ""))
            current = by_model.setdefault(
                key,
                {
                    "provider": item.get("provider"),
                    "model": item.get("model"),
                    "label": item.get("label"),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "usd": 0.0 if item.get("usd") is not None else None,
                    "pricing_available": item.get("pricing_available", True),
                    "partial": item.get("partial", False),
                },
            )
            current["prompt_tokens"] += int(item.get("prompt_tokens") or 0)
            current["completion_tokens"] += int(item.get("completion_tokens") or 0)
            current["total_tokens"] += int(item.get("total_tokens") or 0)
            current["pricing_available"] = bool(current.get("pricing_available")) and bool(item.get("pricing_available", True))
            current["partial"] = bool(current.get("partial")) or bool(item.get("partial", False))
            if current["usd"] is None or item.get("usd") is None:
                current["usd"] = None
            else:
                current["usd"] += float(item.get("usd") or 0.0)

    merged["by_model"] = sorted(
        [
            {**item, "usd": round(item["usd"], 8) if isinstance(item.get("usd"), float) else item.get("usd")}
            for item in by_model.values()
        ],
        key=lambda item: (str(item.get("provider") or ""), str(item.get("model") or "")),
    )
    return merged


def _flush_chunk_telemetry(job_id: str) -> None:
    cache = _results_cache.get(job_id, {})
    collector = cache.get("telemetry_collector")
    if not isinstance(collector, RunTelemetryCollector):
        return

    rows = collector.drain_model_executions()
    if not rows:
        return

    delta_costs = build_run_costs_from_model_executions(
        rows,
        missing_llm_usage=collector.missing_llm_usage,
    )
    cache["run_costs_aggregate"] = _merge_run_cost_summaries(
        cache.get("run_costs_aggregate"),
        delta_costs,
    )

    persisted = False
    if db and db.is_configured():
        persisted = bool(db.persist_model_executions(
            job_id,
            rows,
            run_config=_run_config_from_cache(job_id),
            versions=cache.get("versions") or _runtime_versions(),
        ))

    if persisted:
        cache["model_executions"] = collector.snapshot_model_executions()
    else:
        pending_rows = list(cache.get("model_executions") or [])
        pending_rows.extend(rows)
        cache["model_executions"] = pending_rows


def _refresh_persisted_batch_results(
    job_id: str,
    *,
    progress_message: str | None = None,
    full: bool = False,
) -> bool:
    if not (db and db.is_configured()):
        return False
    cache = _results_cache.get(job_id, {})
    if full:
        loaded = db.load_job_results(job_id, preferred_mode=cache.get("input_mode"))
    else:
        loaded = (
            db.load_job_progress_snapshot(job_id, preferred_mode=cache.get("input_mode"))
            or db.load_job_results(job_id, preferred_mode=cache.get("input_mode"))
        )
    if not loaded or not isinstance(loaded.get("results"), dict):
        return False

    payload = _merge_runtime_payload_metadata(job_id, loaded["results"])
    if progress_message:
        payload["job_status"] = _jobs.get(job_id).status if _jobs.get(job_id) else "running"
        payload["job_message"] = progress_message
    cache["results"] = payload if full else (_compact_results_for_runtime(payload) or payload)
    if _jobs.get(job_id):
        _jobs[job_id].results = cache["results"]
    return True


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
    trimmed = log[-(MAX_PROGRESS_LOG_ENTRIES - 1):] if MAX_PROGRESS_LOG_ENTRIES > 1 else []
    job.progress_log = trimmed + [msg]
    cache = _results_cache.get(job_id)
    if isinstance(cache, dict) and isinstance(cache.get("results"), dict):
        cache["results"]["job_message"] = msg
        cache["results"]["job_status"] = job.status
        job.results = cache["results"]
    if db and db.is_configured():
        try:
            db.insert_analysis_event(job_id, message=msg, event_type="progress")
        except Exception:
            pass


def _append_progress_and_log(job_id: str, msg: str, *, allow_stopped: bool = False) -> None:
    _append_progress(job_id, msg, allow_stopped=allow_stopped)
    print(msg)


def _set_job_status(job_id: str, status: str, progress: str | None = None, source: str = "app") -> None:
    if not _jobs.get(job_id):
        return
    current = _jobs[job_id].status
    if current == "stopped" and source not in {"control", "stop_finalize"}:
        return
    _jobs[job_id].status = status
    if progress is not None:
        _jobs[job_id].progress = progress
    cache = _results_cache.get(job_id)
    if isinstance(cache, dict) and isinstance(cache.get("results"), dict):
        cache["results"]["job_status"] = status
        if progress is not None:
            cache["results"]["job_message"] = progress
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
    company_slug = (request_payload or {}).get("company_slug")
    person_key = (request_payload or {}).get("person_key")
    if job.status in {"done", "error"} and company_slug and person_key:
        db.prune_person_profile_jobs(
            company_slug,
            person_key,
            keep_person_job_id=job_id,
        )


def _register_person_job_task(job_id: str, task: asyncio.Task[Any]) -> None:
    _person_job_tasks[job_id] = task

    def _cleanup(completed_task: asyncio.Task[Any]) -> None:
        if _person_job_tasks.get(job_id) is completed_task:
            _person_job_tasks.pop(job_id, None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            completed_task.result()

    task.add_done_callback(_cleanup)


def _schedule_person_profile_job(job_id: str, req: PersonProfileJobRequest) -> None:
    task = _person_job_tasks.get(job_id)
    if task and not task.done():
        return
    _register_person_job_task(job_id, asyncio.create_task(_run_person_profile_job(job_id, req)))


def _resume_person_profile_job_if_needed(
    job_id: str,
    loaded: dict[str, Any] | None = None,
) -> None:
    task = _person_job_tasks.get(job_id)
    if task and not task.done():
        return

    job = _person_jobs.get(job_id)
    if not job or job.status not in {"pending", "running"}:
        return

    loaded = loaded or (db.load_person_profile_job(job_id) if db and db.is_configured() else None)
    request_payload = (loaded or {}).get("request_payload") or {}
    if not isinstance(request_payload, dict) or not request_payload:
        job.status = "error"
        job.progress = "Profile generation failed"
        job.error = "Job payload missing; unable to resume person profile job."
        _persist_person_job(job_id)
        return

    try:
        req = PersonProfileJobRequest.model_validate(request_payload)
    except Exception as exc:
        job.status = "error"
        job.progress = "Profile generation failed"
        job.error = f"Invalid persisted person profile payload: {exc}"
        _persist_person_job(job_id, request_payload)
        return

    job.status = "pending"
    job.progress = "Resuming after restart..."
    job.error = None
    _persist_person_job(job_id, request_payload)
    print(f"Resuming persisted person profile job {job_id}.")
    _schedule_person_profile_job(job_id, req)


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
        if not _refresh_persisted_batch_results(job_id, full=True):
            return False

    if evaluated:
        _build_results_payload(results_list, job_id, upload_dir)
    message = (
        "Stopped by user. Partial results ready — "
        f"{len(evaluated) if evaluated else _completed_count_from_results_payload(_results_cache.get(job_id, {}).get('results'))}/{total} companies ranked"
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
    _mark_terminal_persistence_complete(job_id)
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
                    "progress_log": (getattr(job, "progress_log", []) or [])[-MAX_PROGRESS_LOG_ENTRIES:],
                    "results": job.results,
                    "started_by_user_id": job.started_by_user_id,
                    "started_by_email": job.started_by_email,
                    "started_by_display_name": job.started_by_display_name,
                    "started_by_label": job.started_by_label,
                }
        if not to_save:
            return
        JOBS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        JOBS_STORE_PATH.write_text(json.dumps(to_save, default=str))
    except Exception:
        pass


def _load_jobs() -> None:
    """Load lightweight completed-job metadata from local persistence on startup."""
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
                        progress_log=(data.get("progress_log") or [])[-MAX_PROGRESS_LOG_ENTRIES:],
                        results=None,
                        started_by_user_id=data.get("started_by_user_id"),
                        started_by_email=data.get("started_by_email"),
                        started_by_display_name=data.get("started_by_display_name"),
                        started_by_label=data.get("started_by_label"),
                        persistence_complete=True,
                    )
                    _results_cache[job_id] = {}
                    _promote_results_metadata(job_id, results)
        except Exception:
            pass


_load_jobs()


def _get_job_summary(job_id: str, job: AnalysisStatus) -> dict[str, Any]:
    cache = _results_cache.get(job_id, {})
    results = cache.get("results")
    has_results = bool(results) and not _is_compact_results_payload(results)
    run_config = cache.get("run_config") or {}
    is_active = job.status in {"pending", "running", "paused"}
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "created_at": None,
        "input_mode": (
            results.get("mode")
            if isinstance(results, dict)
            else cache.get("input_mode")
        ),
        "use_web_search": None,
        "run_name": cache.get("run_name") or run_config.get("run_name"),
        "started_by_user_id": job.started_by_user_id or run_config.get("started_by_user_id"),
        "started_by_email": job.started_by_email or run_config.get("started_by_email"),
        "started_by_display_name": job.started_by_display_name or run_config.get("started_by_display_name"),
        "started_by_label": job.started_by_label or run_config.get("started_by_label"),
        "results": None,
        "has_results": has_results,
        "can_open_results": job.status == "done" and has_results,
        "can_view_log": is_active,
        "llm": _resolve_job_llm_label(job_id, results=results),
    }


def _get_cached_overview_payload(cache: dict[str, Any]) -> Any:
    now = time.monotonic()
    with _overview_cache_lock:
        if float(cache.get("expires_at") or 0.0) > now:
            return cache.get("payload")
    return None


def _set_cached_overview_payload(cache: dict[str, Any], payload: Any, ttl_seconds: int) -> Any:
    with _overview_cache_lock:
        cache["payload"] = payload
        cache["expires_at"] = time.monotonic() + max(ttl_seconds, 1)
    return payload


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
                    "run_name": entry.get("run_name") or (existing or {}).get("run_name"),
                    "started_by_user_id": entry.get("started_by_user_id") or (existing or {}).get("started_by_user_id"),
                    "started_by_email": entry.get("started_by_email") or (existing or {}).get("started_by_email"),
                    "started_by_display_name": entry.get("started_by_display_name") or (existing or {}).get("started_by_display_name"),
                    "started_by_label": entry.get("started_by_label") or (existing or {}).get("started_by_label"),
                    "results": None,
                    "has_results": entry.get("has_results") or (existing or {}).get("has_results") or False,
                    "can_open_results": False,
                    "can_view_log": False,
                    "llm": (
                        _llm_label_from_payload(entry.get("results"))
                        or _llm_label_from_payload(entry.get("run_config"))
                        or (existing or {}).get("llm")
                        or _get_llm_display()
                    ),
                }
                has_live_job = bool(existing) and (existing or {}).get("status") in {"pending", "running", "paused"}
                worker_active = bool(entry.get("worker_active"))
                merged["status"] = _normalize_worker_status(merged["status"])
                if not has_live_job and not worker_active and merged["status"] in {"pending", "running", "paused"}:
                    merged["status"] = "interrupted"
                    merged["progress"] = "Run interrupted before completion."
                    merged["has_results"] = False
                merged["can_open_results"] = merged["status"] == "done" and merged["has_results"] is True
                merged["can_view_log"] = (has_live_job or worker_active) and merged["status"] in {"pending", "running", "paused"}
                jobs_by_id[job_id] = merged
        except Exception:
            pass

    def _sort_key(item: dict[str, Any]) -> tuple[str, str]:
        created_at = str(item.get("created_at") or "")
        return (created_at, item.get("job_id") or "")

    return sorted(jobs_by_id.values(), key=_sort_key, reverse=True)


def _list_company_runs_for_ui() -> list[dict[str, Any]]:
    if not db or not db.is_configured():
        return []
    return db.list_company_histories(
        limit_runs=COMPANY_RUNS_OVERVIEW_LIMIT,
        perform_maintenance=False,
    )


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


def _supabase_public_auth_config() -> dict[str, Any]:
    url = (os.getenv("SUPABASE_URL") or "").strip()
    anon_key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    db_ready = bool(callable(getattr(db, "is_configured", None)) and db.is_configured())
    redirect_url = (
        os.getenv("SUPABASE_AUTH_REDIRECT_URL")
        or os.getenv("PUBLIC_BASE_URL")
        or ""
    ).strip()
    return {
        "configured": bool(url and anon_key),
        "required": bool(url and anon_key and db_ready),
        "url": url,
        "anon_key": anon_key,
        "redirect_url": redirect_url,
    }


def _extract_bearer_token(authorization: str | None) -> str | None:
    header = (authorization or "").strip()
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _display_name_from_supabase_user(user: dict[str, Any] | None) -> str | None:
    metadata = user.get("user_metadata") if isinstance(user, dict) else None
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("full_name", "name", "display_name", "user_name"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return None


def _build_started_by_identity(user: dict[str, Any] | None) -> dict[str, str | None]:
    payload = user if isinstance(user, dict) else {}
    email = str(payload.get("email") or "").strip() or None
    display_name = _display_name_from_supabase_user(payload)
    return {
        "started_by_user_id": str(payload.get("id") or "").strip() or None,
        "started_by_email": email,
        "started_by_display_name": display_name,
        "started_by_label": display_name or email,
    }


# ---------------------------------------------------------------------------
# Sprint 1 — Domain validation helpers
# ---------------------------------------------------------------------------

_FREE_EMAIL_PROVIDERS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "hotmail.com",
    "hotmail.co.uk", "outlook.com", "live.com", "msn.com", "icloud.com",
    "me.com", "mac.com", "protonmail.com", "proton.me", "tutanota.com",
    "zoho.com", "yandex.com", "yandex.ru", "aol.com",
})


def _normalize_domain(raw: str) -> str:
    """Lowercase, strip scheme, www prefix, and trailing slashes/paths."""
    s = (raw or "").strip().lower()
    # Strip scheme
    for scheme in ("https://", "http://"):
        if s.startswith(scheme):
            s = s[len(scheme):]
    # Strip www.
    if s.startswith("www."):
        s = s[4:]
    # Keep only the host (drop path, query, fragment)
    s = s.split("/")[0].split("?")[0].split("#")[0]
    return s.strip()


def _extract_email_domain(email: str) -> str:
    """Extract and normalise the domain part of an email address.

    Raises ValueError if the email is malformed.
    """
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError(f"Invalid email address: {email!r}")
    _, _, domain_part = email.partition("@")
    domain_part = domain_part.strip()
    if not domain_part or "." not in domain_part:
        raise ValueError(f"Invalid email domain: {domain_part!r}")
    return _normalize_domain(domain_part)


def _root_domain(domain: str) -> str:
    """Return the registrable root domain (last two labels).

    e.g. mail.company.com → company.com
    """
    parts = domain.split(".")
    if len(parts) <= 2:
        return domain
    return ".".join(parts[-2:])


def verify_email_domain(email: str, company_domain: str | None) -> tuple[bool, str]:
    """Check whether *email* belongs to the same domain as *company_domain*.

    Returns (verified: bool, reason: str).
    Rejects free email providers even if the domain technically matches.
    Returns (False, reason) when company has no domain set.
    """
    try:
        email_domain = _extract_email_domain(email)
    except ValueError as exc:
        return False, str(exc)

    if email_domain in _FREE_EMAIL_PROVIDERS:
        return False, f"{email_domain} is a free email provider and cannot be used for domain verification."

    if not company_domain or not company_domain.strip():
        return False, "This company has no domain set — an admin must approve your claim manually."

    normalized_company = _normalize_domain(company_domain)

    # Exact match or subdomain match (compare root domains).
    if email_domain == normalized_company:
        return True, f"Email domain {email_domain!r} matches company domain."

    if _root_domain(email_domain) == _root_domain(normalized_company):
        return True, f"Email domain {email_domain!r} matches company root domain {_root_domain(normalized_company)!r}."

    return False, (
        f"Email domain {email_domain!r} does not match company domain {normalized_company!r}."
    )


# ---------------------------------------------------------------------------
# Sprint 1 — Supabase JWT auth dependency (new endpoints only)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from fastapi import Depends


@dataclass
class CurrentUser:
    """Resolved identity for requests authenticated via Supabase JWT."""

    id: str
    email: str | None
    role: str          # 'admin' | 'vc' | 'startup'
    approved: bool
    display_name: str | None


async def get_current_user(
    authorization: str | None = Header(default=None),
) -> CurrentUser:
    """FastAPI dependency: validate Supabase Bearer token and resolve users_profile.

    Used exclusively on new Sprint 1+ endpoints. Existing endpoints continue
    to use _check_session() and are not affected.
    """
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    get_user_fn = getattr(db, "get_authenticated_supabase_user", None) if db else None
    if not callable(get_user_fn):
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    supabase_user = await asyncio.to_thread(get_user_fn, token)
    if not supabase_user or not supabase_user.get("id"):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    get_profile_fn = getattr(db, "get_user_profile", None) if db else None
    if not callable(get_profile_fn):
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    profile = await asyncio.to_thread(get_profile_fn, supabase_user["id"])
    if not profile:
        raise HTTPException(
            status_code=403,
            detail="No user profile found. Please complete registration first.",
        )

    return CurrentUser(
        id=supabase_user["id"],
        email=supabase_user.get("email"),
        role=profile["role"],
        approved=bool(profile.get("approved", False)),
        display_name=profile.get("display_name"),
    )


async def _require_startup(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Dependency: allow only users with role='startup'."""
    if user.role != "startup":
        raise HTTPException(status_code=403, detail="Startup role required.")
    return user


async def _require_vc(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Dependency: allow users with role='vc' or role='admin' (admin can access VC endpoints)."""
    if user.role not in ("vc", "admin"):
        raise HTTPException(status_code=403, detail="VC role required.")
    return user


async def _require_admin(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Dependency: allow only users with role='admin'."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


# ---------------------------------------------------------------------------

def _started_by_from_payload(payload: dict[str, Any] | None) -> dict[str, str | None]:
    if not isinstance(payload, dict):
        return {}
    values = {
        "started_by_user_id": payload.get("started_by_user_id"),
        "started_by_email": payload.get("started_by_email"),
        "started_by_display_name": payload.get("started_by_display_name"),
        "started_by_label": payload.get("started_by_label"),
    }
    return {key: value for key, value in values.items() if value}


async def _require_supabase_identity(authorization: str | None) -> dict[str, str | None]:
    config = _supabase_public_auth_config()
    if not config["required"]:
        return {}
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Supabase sign-in required before starting an analysis.")
    if not callable(getattr(db, "is_configured", None)) or not db.is_configured():
        raise HTTPException(status_code=503, detail="Supabase storage is not configured.")
    get_user = getattr(db, "get_authenticated_supabase_user", None)
    if not callable(get_user):
        raise HTTPException(status_code=503, detail="Supabase auth validation is unavailable.")
    user = await asyncio.to_thread(get_user, token)
    identity = _build_started_by_identity(user)
    if not identity.get("started_by_user_id") or not identity.get("started_by_label"):
        raise HTTPException(status_code=401, detail="Supabase session is invalid or expired.")
    return identity


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


@app.get("/portal", response_class=HTMLResponse)
async def portal():
    """Serve the startup / VC self-service portal (separate from the internal tool)."""
    return (STATIC_DIR / "portal.html").read_text()


class PortalLoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/portal/login")
async def portal_login(req: PortalLoginRequest) -> dict[str, Any]:
    """Server-side sign-in for the portal. Returns Supabase JWT tokens.

    Avoids requiring the Supabase JS SDK on the frontend — the backend holds
    the credentials and proxies the token exchange.
    """
    if not db or not db.is_configured():
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    # sign_in_with_password must use the anon key, not the service role key.
    supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
    anon_key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    if not supabase_url or not anon_key:
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    try:
        from supabase import create_client as _create_client
        anon_client = _create_client(supabase_url, anon_key)
    except Exception:
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    try:
        response = await asyncio.to_thread(
            lambda: anon_client.auth.sign_in_with_password(
                {"email": req.email, "password": req.password}
            )
        )
        session = getattr(response, "session", None)
        user = getattr(response, "user", None)
        if not session or not getattr(session, "access_token", None):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
    except HTTPException:
        raise
    except Exception as exc:
        err_str = str(exc).lower()
        if "invalid" in err_str or "credentials" in err_str or "password" in err_str:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "user": {
            "id": str(user.id) if user else None,
            "email": user.email if user else req.email,
        },
    }


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


@app.get("/api/public-auth-config")
async def public_auth_config():
    return {"supabase_auth": _supabase_public_auth_config()}


@app.get("/api/check-session")
async def check_session(session_id: str | None = Cookie(default=None)):
    return {"authenticated": _check_session(session_id)}


# ---------------------------------------------------------------------------
# Sprint 1 — Auth endpoints (portal users: startup / vc)
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=RegisterResponse, status_code=201)
async def auth_register(req: RegisterRequest) -> RegisterResponse:
    """Register a new startup or VC user via Supabase Auth.

    Creates the Supabase auth user, then inserts a users_profile row with the
    requested role.  The profile starts with approved=False; admin approval is
    required before VC users gain access to matches.

    If SUPABASE_BYPASS_EMAIL_CONFIRMATION=true the admin client is used so that
    no confirmation email is sent (useful during development).
    """
    if not db or not db.is_configured():
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    supabase_client = getattr(db, "_get_client", None)
    if not callable(supabase_client):
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    client = supabase_client()
    if not client:
        raise HTTPException(status_code=503, detail="Auth service unavailable.")

    # Always use admin.create_user — the backend holds the service role key,
    # so using admin is correct and ensures the auth row is fully committed
    # before we insert the users_profile FK reference.
    # email_confirm=True skips the confirmation email in all environments;
    # set SUPABASE_REQUIRE_EMAIL_CONFIRMATION=true to enforce it in production.
    require_confirmation = os.getenv(
        "SUPABASE_REQUIRE_EMAIL_CONFIRMATION", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    email_confirm = not require_confirmation

    try:
        response = await asyncio.to_thread(
            lambda: client.auth.admin.create_user(
                {"email": req.email, "password": req.password, "email_confirm": email_confirm}
            )
        )
        user = getattr(response, "user", None)
    except Exception as exc:
        err_str = str(exc).lower()
        if any(kw in err_str for kw in (
            "already registered", "already exists", "duplicate",
            "not allowed", "user already",
        )):
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        raise HTTPException(status_code=400, detail=f"Registration failed: {exc}")

    if not user or not getattr(user, "id", None):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user_id = str(user.id)
    profile = await asyncio.to_thread(
        db.create_user_profile,
        user_id,
        req.role,
        req.display_name,
        req.organization,
    )
    if not profile:
        # Roll back the auth user so the email can be re-registered cleanly.
        try:
            await asyncio.to_thread(lambda: client.auth.admin.delete_user(user_id))
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Registration failed — please try again.",
        )

    return RegisterResponse(
        user_id=user_id,
        email=req.email,
        role=req.role,
        message=(
            "Registration successful. Please check your email to confirm your account."
            if require_confirmation
            else "Registration successful."
        ),
    )


@app.post("/api/auth/verify-domain", response_model=VerifyDomainResponse)
async def auth_verify_domain(
    req: VerifyDomainRequest,
    user: CurrentUser = Depends(get_current_user),
) -> VerifyDomainResponse:
    """Verify that the user's email domain matches the claimed company's domain.

    On success, creates a user_company_links row and stamps claimed_at on the
    company (if not already claimed).
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    company = await asyncio.to_thread(db.get_company_by_id, req.company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    verified, reason = verify_email_domain(req.email, company.get("domain"))

    if verified:
        await asyncio.to_thread(
            db.create_user_company_link, user.id, req.company_id, None
        )
        await asyncio.to_thread(db.set_company_claimed_at, req.company_id)

    return VerifyDomainResponse(
        verified=verified,
        domain=_normalize_domain(company.get("domain") or "") or None,
        message=reason,
    )


@app.get("/api/auth/me")
async def auth_me(user: CurrentUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return the authenticated user's profile and linked companies."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    links = await asyncio.to_thread(db.get_user_company_links, user.id)

    companies = []
    for link in links:
        company_data = link.get("companies") or {}
        if isinstance(company_data, dict) and company_data:
            companies.append({
                "id": company_data.get("id"),
                "name": company_data.get("name"),
                "industry": company_data.get("industry"),
                "fundraising": company_data.get("fundraising", False),
                "role_in_company": link.get("role_in_company"),
            })

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "approved": user.approved,
        "display_name": user.display_name,
        "companies": companies,
    }


# ---------------------------------------------------------------------------
# Sprint 1 — Startup portal endpoints
# ---------------------------------------------------------------------------

def _first_linked_company_id(user: CurrentUser) -> str:
    """Resolve the company_id for a startup user. Raises 404 if none linked."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    links = db.get_user_company_links(user.id)
    if not links:
        raise HTTPException(
            status_code=404,
            detail="No company linked to your account. Please verify your domain first.",
        )
    company_data = (links[0].get("companies") or {})
    company_id = company_data.get("id") if isinstance(company_data, dict) else None
    if not company_id:
        raise HTTPException(status_code=404, detail="Linked company data unavailable.")
    return company_id


def _safe_analysis_summary(results_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract only the startup-safe fields from a results_payload.

    Deliberately excludes VC thesis context and any Specter-sourced content.
    """
    if not isinstance(results_payload, dict):
        return None
    ranking = results_payload.get("ranking_result") or {}
    return {
        "composite_score": ranking.get("composite_score"),
        "strategy_fit_score": ranking.get("strategy_fit_score"),
        "team_score": ranking.get("team_score"),
        "upside_score": ranking.get("upside_score"),
        "bucket": ranking.get("bucket"),
        "key_points": ranking.get("key_points") or [],
        "red_flags": ranking.get("red_flags") or [],
    }


@app.get("/api/startup/profile")
async def startup_profile(user: CurrentUser = Depends(_require_startup)) -> dict[str, Any]:
    """Return the startup's company profile and latest analysis summary."""
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    company = await asyncio.to_thread(db.get_company_by_id, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    analysis = await asyncio.to_thread(db.get_company_latest_analysis, company_id)
    analysis_summary = None
    if analysis:
        analysis_summary = _safe_analysis_summary(analysis.get("results_payload"))

    return {
        "company": {
            "id": company.get("id"),
            "name": company.get("name"),
            "industry": company.get("industry"),
            "tagline": company.get("tagline"),
            "about": company.get("about"),
            "fundraising": company.get("fundraising", False),
            "fundraising_updated_at": company.get("fundraising_updated_at"),
            "claimed_at": company.get("claimed_at"),
            "data_room_enabled": company.get("data_room_enabled", False),
        },
        "analysis": analysis_summary,
    }


@app.put("/api/startup/fundraising")
async def startup_fundraising(
    req: FundraisingToggleRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Toggle the fundraising flag for the startup's linked company.

    When fundraising is toggled ON, triggers async matching against all active
    VC profiles as a background task.
    """
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    updated = await asyncio.to_thread(db.set_company_fundraising, company_id, req.fundraising)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update fundraising flag.")

    if req.fundraising:
        background_tasks.add_task(_run_matching_background, company_id)

    return {
        "fundraising": updated.get("fundraising"),
        "fundraising_updated_at": updated.get("fundraising_updated_at"),
        "matching_triggered": req.fundraising,
    }


@app.get("/api/startup/evidence")
async def startup_evidence(user: CurrentUser = Depends(_require_startup)) -> dict[str, Any]:
    """Return evidence chunks for the startup's company, excluding Specter-sourced files."""
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    chunks = await asyncio.to_thread(db.get_company_chunks, company_id)
    return {
        "chunks": [
            {
                "id": c.get("chunk_id") or c.get("id"),
                "text": c.get("text"),
                "source_file": c.get("source_file"),
                "page_or_slide": c.get("page_or_slide"),
            }
            for c in chunks
        ]
    }


_SPECTER_SOURCE_FILES = {"companies.csv", "people.csv"}


def _safe_evidence_log(results_payload: dict, state: dict | None = None) -> dict:
    """Extract evidence log data safe for startup-facing responses.

    Filters:
    - Strips vc_context from any QA pair
    - Redacts chunk previews from Specter-sourced files (companies.csv, people.csv)
    - Removes internal fields (startup_slug, company_name)
    """
    # Get qa_provenance_rows — prefer results_payload, fall back to state
    qa_rows = results_payload.get("qa_provenance_rows")
    if not qa_rows and state:
        # Fallback: reconstruct from state.all_qa_pairs for pre-upgrade analyses
        qa_rows, _ = _build_evidence_provenance(state)

    arg_rows = results_payload.get("argument_rows")
    if not arg_rows and state:
        _, arg_rows = _build_evidence_provenance(state)

    # Filter QA rows: strip vc_context, redact Specter chunk previews
    safe_qa: list[dict] = []
    for qa in (qa_rows or []):
        row = dict(qa)
        row.pop("vc_context", None)
        row.pop("startup_slug", None)
        row.pop("company_name", None)

        # Redact Specter-sourced chunk previews
        preview = row.get("chunks_preview", "")
        if preview:
            lines = preview.split("\n---\n")
            safe_lines = []
            for line in lines:
                is_specter = any(sf in line for sf in _SPECTER_SOURCE_FILES)
                if is_specter:
                    # Keep chunk ID reference but redact text
                    bracket_end = line.find("]: ")
                    if bracket_end > 0:
                        safe_lines.append(line[: bracket_end + 3] + "[External data source]")
                    else:
                        safe_lines.append("[External data source]")
                else:
                    safe_lines.append(line)
            row["chunks_preview"] = "\n---\n".join(safe_lines)

        safe_qa.append(row)

    # Filter argument rows: strip internal fields
    safe_args: list[dict] = []
    for arg in (arg_rows or []):
        row = dict(arg)
        row.pop("startup_slug", None)
        row.pop("company_name", None)
        safe_args.append(row)

    # Deduplicate QA pairs (safety net for analyses stored before this fix).
    # Keyed on (question, answer) — same evidence seen twice is always redundant.
    seen_qa: set[tuple[str, str]] = set()
    deduped_qa: list[dict] = []
    for row in safe_qa:
        key = (row.get("question", ""), row.get("answer", ""))
        if key not in seen_qa:
            seen_qa.add(key)
            deduped_qa.append(row)

    # Build decision/scores from ranking_result
    ranking = results_payload.get("ranking_result") or {}
    return {
        "qa_pairs": deduped_qa,
        "arguments": safe_args,
        "decision": results_payload.get("decision"),
        "scores": {
            "composite_score": ranking.get("composite_score"),
            "strategy_fit_score": ranking.get("strategy_fit_score"),
            "team_score": ranking.get("team_score"),
            "upside_score": ranking.get("upside_score"),
            "bucket": ranking.get("bucket"),
        },
    }


@app.get("/api/startup/evidence-log")
async def startup_evidence_log(user: CurrentUser = Depends(_require_startup)) -> dict[str, Any]:
    """Return the evidence log (QA pairs, arguments, decision) for the startup's latest analysis.

    Provides audit-log-style evidence grounding so startups can see what the AI
    found and how it reasoned. All VC thesis content and Specter-sourced data
    are filtered before returning.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    analysis = await asyncio.to_thread(db.get_latest_analysis_full, company_id)
    if not analysis:
        return {"qa_pairs": [], "arguments": [], "decision": None, "scores": {}, "analyzed_at": None}

    results_payload = analysis.get("results_payload") or {}
    state = analysis.get("state")
    evidence = _safe_evidence_log(results_payload, state)
    evidence["analyzed_at"] = analysis.get("created_at")
    evidence["analysis_id"] = analysis.get("id")
    return evidence


@app.get("/api/startup/matches")
async def startup_matches(user: CurrentUser = Depends(_require_startup)) -> dict[str, Any]:
    """Return all VC matches for the startup's company.

    Only shows the VC firm name and scores — never the VC's private thesis.
    """
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    matches = await asyncio.to_thread(db.get_matches_for_company, company_id)
    return {
        "matches": [
            {
                "id": m.get("id"),
                "firm_name": (m.get("vc_profiles") or {}).get("firm_name"),
                "strategy_fit_score": m.get("strategy_fit_score"),
                "team_score": m.get("team_score"),
                "potential_score": m.get("potential_score"),
                "composite_score": m.get("composite_score"),
                "bucket": m.get("bucket"),
                "status": m.get("status"),
                "created_at": m.get("created_at"),
            }
            for m in matches
        ]
    }


# ---------------------------------------------------------------------------
# Sprint 2 — VC portal helpers
# ---------------------------------------------------------------------------

def _get_vc_profile_or_404(user: CurrentUser) -> dict[str, Any]:
    """Fetch the VC's profile row or raise 404."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    profile = db.get_vc_profile(user.id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="VC profile not found. Please complete your profile first.",
        )
    return profile


def _blocking_match_for_company(company_id: str, *, force_refresh: bool = False) -> int:
    """Run matching synchronously inside a dedicated thread with its own event loop.

    This isolates the matching pipeline (which contains synchronous LLM .invoke()
    calls) from the main FastAPI event loop and its shared thread pool.  Without
    this isolation, long-running sync LLM calls saturate the default thread pool
    and cause asyncio.to_thread() calls (e.g. Supabase auth) to queue up, making
    login and other endpoints time-out while a matching job is running.

    When ``force_refresh=True``, existing matches are re-scored in place via the
    upsert in ``db.create_match`` (preserving match ids → preserving debate FK
    links). Used by the re-evaluation flow so a fresh analysis updates scores
    without destroying debates.
    """
    import asyncio as _aio  # noqa: PLC0415
    from agent.matching.engine import trigger_matching_for_company  # noqa: PLC0415

    loop = _aio.new_event_loop()
    _aio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            trigger_matching_for_company(company_id, db, force_refresh=force_refresh)
        )
    finally:
        loop.close()
        _aio.set_event_loop(None)


async def _run_matching_background(company_id: str, *, force_refresh: bool = False) -> None:
    """Background task: run matching for a company against all active VCs.

    Delegates to _blocking_match_for_company() via asyncio.to_thread() so the
    matching pipeline runs in an isolated thread, leaving the FastAPI event loop
    free to handle other requests (login, health-checks, etc.).

    ``force_refresh`` is forwarded so the re-evaluation path can update existing
    matches in place without deleting them (which would cascade-destroy debates).
    """
    try:
        count = await asyncio.to_thread(
            _blocking_match_for_company, company_id, force_refresh=force_refresh
        )
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).info(
            "Matching complete for company=%s: %d matches created/refreshed", company_id, count
        )
    except Exception as exc:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).error(
            "Matching background task failed for company=%s: %s", company_id, exc
        )


def _blocking_rematch_for_vc(vc_profile_id: str) -> int:
    """Run VC re-matching synchronously inside a dedicated thread with its own event loop.

    Same isolation rationale as _blocking_match_for_company — prevents sync LLM
    calls inside the pipeline from blocking the FastAPI event loop.
    """
    import asyncio as _aio  # noqa: PLC0415

    async def _async_body() -> int:
        import logging as _log_module  # noqa: PLC0415
        from agent.matching.engine import run_matching_for_pair  # noqa: PLC0415
        _logger = _log_module.getLogger(__name__)

        deleted = await _aio.to_thread(db.delete_matches_for_vc, vc_profile_id)
        _logger.info("VC re-matching: deleted %d stale matches for vc=%s", deleted, vc_profile_id)

        vc_profile = await _aio.to_thread(db.get_vc_profile_by_id, vc_profile_id)
        if not vc_profile:
            _logger.warning("VC re-matching: vc_profile %s not found", vc_profile_id)
            return 0

        vc_thesis: str = vc_profile.get("investment_thesis") or ""
        min_strategy: float = float(vc_profile.get("min_strategy_fit") or 0)
        min_team: float = float(vc_profile.get("min_team") or 0)
        min_potential: float = float(vc_profile.get("min_potential") or 0)

        companies = await _aio.to_thread(db.get_fundraising_companies)
        _logger.info(
            "VC re-matching: evaluating %d fundraising companies against vc=%s",
            len(companies), vc_profile_id,
        )

        created = 0
        for company_row in companies:
            company_id: str = company_row.get("id") or ""
            if not company_id:
                continue

            chunks = await _aio.to_thread(db.get_company_chunks, company_id)
            all_qa_pairs = await _aio.to_thread(db.get_analysis_qa_pairs, company_id)

            analysis_final = (
                await _aio.to_thread(db.get_analysis_final_state, company_id)
                if hasattr(db, "get_analysis_final_state")
                else None
            )
            final_arguments = analysis_final.get("final_arguments") if analysis_final else None
            final_decision = analysis_final.get("final_decision") if analysis_final else None

            if not final_arguments and not all_qa_pairs:
                _logger.debug("VC re-matching: skipping company=%s (no analysis data)", company_id)
                continue

            try:
                scores = await run_matching_for_pair(
                    company_row=company_row,
                    chunks=chunks,
                    all_qa_pairs=all_qa_pairs,
                    vc_thesis=vc_thesis,
                    final_arguments=final_arguments,
                    final_decision=final_decision,
                )
            except Exception as exc:
                _logger.error(
                    "VC re-matching error vc=%s company=%s: %s", vc_profile_id, company_id, exc
                )
                continue

            if not scores:
                continue

            strategy_fit: float = float(scores.get("strategy_fit_score") or 0)
            team: float = float(scores.get("team_score") or 0)
            potential: float = float(scores.get("upside_score") or 0)

            if strategy_fit >= min_strategy and team >= min_team and potential >= min_potential:
                latest = await _aio.to_thread(db.get_company_latest_analysis, company_id)
                analysis_id = latest.get("id") if latest else None
                await _aio.to_thread(
                    db.create_match,
                    vc_profile_id=vc_profile_id,
                    company_id=company_id,
                    analysis_id=analysis_id,
                    scores={
                        "strategy_fit_score": strategy_fit,
                        "team_score": team,
                        "potential_score": potential,
                        "composite_score": float(scores.get("composite_score") or 0),
                        "bucket": scores.get("bucket"),
                    },
                )
                created += 1
                _logger.info(
                    "VC re-matching: match created vc=%s company=%s scores=%.1f/%.1f/%.1f",
                    vc_profile_id, company_id, strategy_fit, team, potential,
                )

        _logger.info(
            "VC re-matching complete for vc=%s: %d matches created from %d companies",
            vc_profile_id, created, len(companies),
        )
        return created

    loop = _aio.new_event_loop()
    _aio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_async_body())
    finally:
        loop.close()
        _aio.set_event_loop(None)


async def _run_vc_rematching_background(vc_profile_id: str) -> None:
    """Background task: re-run matching for a VC against all actively fundraising companies.

    Called when a VC updates their investment thesis or score thresholds.  Clears
    all existing matches for this VC, then re-evaluates every company that currently
    has fundraising=True so the match list reflects the latest thesis and thresholds.

    Delegates to _blocking_rematch_for_vc() via asyncio.to_thread() so the pipeline
    runs in an isolated thread and the FastAPI event loop stays responsive.
    """
    import logging as _log_module  # noqa: PLC0415
    _logger = _log_module.getLogger(__name__)

    try:
        await asyncio.to_thread(_blocking_rematch_for_vc, vc_profile_id)
    except Exception as exc:
        _logger.error(
            "VC re-matching background task failed for vc=%s: %s", vc_profile_id, exc, exc_info=True
        )
        return

    # (success logging is inside _blocking_rematch_for_vc / _async_body)


# ---------------------------------------------------------------------------
# Sprint 2 — VC portal endpoints
# ---------------------------------------------------------------------------

@app.get("/api/vc/profile")
async def vc_get_profile(user: CurrentUser = Depends(_require_vc)) -> dict[str, Any]:
    """Return the VC's profile (firm name, thesis, thresholds)."""
    profile = await asyncio.to_thread(_get_vc_profile_or_404, user)
    return {
        "id": profile.get("id"),
        "firm_name": profile.get("firm_name"),
        "investment_thesis": profile.get("investment_thesis"),
        "min_strategy_fit": profile.get("min_strategy_fit", 0),
        "min_team": profile.get("min_team", 0),
        "min_potential": profile.get("min_potential", 0),
        "active": profile.get("active", True),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
    }


@app.put("/api/vc/profile")
async def vc_update_profile(
    req: VCProfileRequest,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Create or update the VC's profile (firm name + optional thesis)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    existing = await asyncio.to_thread(db.get_vc_profile, user.id)
    if existing:
        updates: dict[str, Any] = {"firm_name": req.firm_name.strip()}
        if req.investment_thesis is not None:
            updates["investment_thesis"] = req.investment_thesis.strip()
        updated = await asyncio.to_thread(db.update_vc_profile, existing["id"], updates)
        return updated or existing
    else:
        created = await asyncio.to_thread(
            db.create_vc_profile,
            user.id,
            req.firm_name,
            req.investment_thesis,
        )
        if not created:
            raise HTTPException(status_code=500, detail="Failed to create VC profile.")
        return created


@app.put("/api/vc/thesis")
async def vc_update_thesis(
    req: VCThesisRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Update only the VC's investment thesis text.

    Triggers background re-matching so all fundraising startups are re-scored
    against the updated thesis.
    """
    profile = await asyncio.to_thread(_get_vc_profile_or_404, user)
    updated = await asyncio.to_thread(
        db.update_vc_profile,
        profile["id"],
        {"investment_thesis": req.investment_thesis.strip()},
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update thesis.")
    background_tasks.add_task(_run_vc_rematching_background, profile["id"])
    return {"investment_thesis": updated.get("investment_thesis")}


@app.put("/api/vc/thresholds")
async def vc_update_thresholds(
    req: VCThresholdsRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Update the VC's minimum score thresholds for matching.

    Triggers background re-matching so the match list immediately reflects the
    new thresholds: stale matches are deleted and every fundraising startup is
    re-evaluated against the updated criteria.
    """
    profile = await asyncio.to_thread(_get_vc_profile_or_404, user)
    updates = {
        "min_strategy_fit": req.min_strategy_fit,
        "min_team": req.min_team,
        "min_potential": req.min_potential,
    }
    updated = await asyncio.to_thread(db.update_vc_profile, profile["id"], updates)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update thresholds.")
    background_tasks.add_task(_run_vc_rematching_background, profile["id"])
    return {
        "min_strategy_fit": updated.get("min_strategy_fit"),
        "min_team": updated.get("min_team"),
        "min_potential": updated.get("min_potential"),
    }


@app.get("/api/vc/matches")
async def vc_get_matches(user: CurrentUser = Depends(_require_vc)) -> dict[str, Any]:
    """Return all matches for the VC, ordered by composite score descending."""
    profile = await asyncio.to_thread(_get_vc_profile_or_404, user)
    matches = await asyncio.to_thread(db.get_matches_for_vc, profile["id"])

    # Enrich with data room file counts (batch to avoid N+1).
    company_ids = list({(m.get("companies") or {}).get("id") for m in matches} - {None})
    dr_counts: dict[str, int] = {}
    for cid in company_ids:
        dr_counts[cid] = await asyncio.to_thread(db.count_data_room_files, cid)

    return {
        "matches": [
            {
                "id": m.get("id"),
                "company": {
                    "id": (m.get("companies") or {}).get("id"),
                    "name": (m.get("companies") or {}).get("name"),
                    "industry": (m.get("companies") or {}).get("industry"),
                    "tagline": (m.get("companies") or {}).get("tagline"),
                    "about": (m.get("companies") or {}).get("about"),
                },
                "strategy_fit_score": m.get("strategy_fit_score"),
                "team_score": m.get("team_score"),
                "potential_score": m.get("potential_score"),
                "composite_score": m.get("composite_score"),
                "bucket": m.get("bucket"),
                "status": m.get("status"),
                "created_at": m.get("created_at"),
                "data_room_enabled": (m.get("companies") or {}).get("data_room_enabled", False),
                "data_room_file_count": dr_counts.get((m.get("companies") or {}).get("id"), 0),
            }
            for m in matches
        ]
    }


@app.post("/api/vc/matches/{match_id}/action")
async def vc_match_action(
    match_id: str,
    req: MatchActionRequest,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Update match status (viewed / interested / passed / in_debate)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    updated = await asyncio.to_thread(db.update_match_status, match_id, req.action)
    if not updated:
        raise HTTPException(status_code=404, detail="Match not found or update failed.")
    return {"id": match_id, "status": req.action}


# ---------------------------------------------------------------------------
# Admin portal endpoints
# ---------------------------------------------------------------------------

@app.get("/api/admin/overview")
async def admin_overview(user: CurrentUser = Depends(_require_admin)) -> dict[str, Any]:
    """Return aggregate counts for the admin dashboard."""
    companies, vc_profiles, matches, analyses, users = await asyncio.gather(
        asyncio.to_thread(db.admin_get_all_companies),
        asyncio.to_thread(db.admin_get_all_vc_profiles),
        asyncio.to_thread(db.admin_get_all_matches),
        asyncio.to_thread(db.admin_get_recent_analyses, 40),
        asyncio.to_thread(db.admin_get_all_users),
    )
    fundraising = [c for c in companies if c.get("fundraising")]
    analysed = [c for c in companies if c.get("analysis_status") == "done"]
    running = [a for a in analyses if a.get("status") not in ("done", "error", None)]
    return {
        "counts": {
            "companies": len(companies),
            "fundraising": len(fundraising),
            "analysed": len(analysed),
            "vc_profiles": len(vc_profiles),
            "matches": len(matches),
            "analyses": len(analyses),
            "users": len(users),
            "running_analyses": len(running),
        },
        "recent_analyses": analyses[:10],
        "recent_matches": matches[:10],
    }


@app.get("/api/admin/companies")
async def admin_companies(user: CurrentUser = Depends(_require_admin)) -> dict[str, Any]:
    """Return all companies with latest analysis scores."""
    companies = await asyncio.to_thread(db.admin_get_all_companies)
    return {"companies": companies}


@app.get("/api/admin/vc-profiles")
async def admin_vc_profiles(user: CurrentUser = Depends(_require_admin)) -> dict[str, Any]:
    """Return all VC profiles with match counts."""
    profiles = await asyncio.to_thread(db.admin_get_all_vc_profiles)
    return {"vc_profiles": profiles}


@app.get("/api/admin/matches")
async def admin_matches(user: CurrentUser = Depends(_require_admin)) -> dict[str, Any]:
    """Return all matches with company and VC names."""
    matches = await asyncio.to_thread(db.admin_get_all_matches)
    return {"matches": matches}


@app.get("/api/admin/analyses")
async def admin_analyses(user: CurrentUser = Depends(_require_admin)) -> dict[str, Any]:
    """Return recent analyses with status, scores, per-stage models, and costs.

    Overlays in-memory ``_jobs`` status so currently-running portal
    re-evaluations appear with live progress messages even when the DB row
    is still in ``running`` status from a previous poll.

    Also surfaces active in-memory jobs that have not yet been persisted
    to the ``analyses`` table (e.g. a just-started re-evaluation) so the
    admin UI shows them immediately.
    """
    analyses = await asyncio.to_thread(db.admin_get_recent_analyses, 40)

    by_job_id: dict[str, dict[str, Any]] = {}
    for row in analyses:
        jid = row.get("job_id_legacy")
        if jid:
            by_job_id[jid] = row

    # Overlay live status + progress from the in-memory job store.
    for jid, job in list(_jobs.items()):
        if not jid.startswith("re-"):
            continue
        row = by_job_id.get(jid)
        if row is not None:
            row["live_progress"] = getattr(job, "progress", None)
            row["live_progress_log"] = list(getattr(job, "progress_log", []) or [])
            if job.status in ("running", "pending"):
                row["status"] = job.status
        else:
            analyses.insert(0, {
                "id": None,
                "company_id": None,
                "company_name": None,
                "job_id_legacy": jid,
                "status": job.status,
                "created_at": None,
                "error": None,
                "composite_score": None,
                "bucket": None,
                "run_config": None,
                "phase_models": None,
                "started_by": {
                    "started_by_user_id": job.started_by_user_id,
                    "started_by_email": job.started_by_email,
                    "started_by_display_name": job.started_by_display_name,
                    "started_by_label": job.started_by_label,
                },
                "cost_summary": None,
                "live_progress": job.progress,
                "live_progress_log": list(job.progress_log or []),
            })

    return {"analyses": analyses}


@app.get("/api/admin/users")
async def admin_users(user: CurrentUser = Depends(_require_admin)) -> dict[str, Any]:
    """Return all registered users."""
    users = await asyncio.to_thread(db.admin_get_all_users)
    return {"users": users}


@app.post("/api/admin/users/{user_id}/approve")
async def admin_approve_user(
    user_id: str,
    body: dict[str, Any],
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Approve or unapprove a user account."""
    approved: bool = bool(body.get("approved", True))
    ok = await asyncio.to_thread(db.admin_set_user_approved, user_id, approved)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update user approval.")
    return {"user_id": user_id, "approved": approved}


@app.post("/api/admin/trigger-matching/{company_id}")
async def admin_trigger_matching(
    company_id: str,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Manually trigger matching for a company against all active VC profiles."""
    background_tasks.add_task(_run_matching_background, company_id)
    return {"triggered": True, "company_id": company_id}


@app.get("/api/admin/matching-debug/{company_id}")
async def admin_matching_debug(
    company_id: str,
    key: str | None = None,
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Diagnostic endpoint: check matching preconditions without running LLMs.

    Returns a full checklist so you can see exactly why matching succeeds or fails.
    """
    result: dict[str, Any] = {"company_id": company_id, "checks": {}}

    company = await asyncio.to_thread(db.get_company_by_id, company_id)
    result["checks"]["company_found"] = bool(company)
    result["checks"]["company_name"] = (company or {}).get("name")
    result["checks"]["fundraising"] = (company or {}).get("fundraising", False)

    chunks = await asyncio.to_thread(db.get_company_chunks, company_id)
    result["checks"]["chunk_count"] = len(chunks)

    qa_pairs = await asyncio.to_thread(db.get_analysis_qa_pairs, company_id)
    result["checks"]["qa_pairs_count"] = len(qa_pairs)

    analysis_final = None
    if hasattr(db, "get_analysis_final_state"):
        analysis_final = await asyncio.to_thread(db.get_analysis_final_state, company_id)
    result["checks"]["final_arguments_count"] = len((analysis_final or {}).get("final_arguments") or [])
    result["checks"]["final_decision"] = (analysis_final or {}).get("final_decision")
    result["checks"]["stage8_shortcut_available"] = bool(
        analysis_final and analysis_final.get("final_arguments") and analysis_final.get("final_decision")
    )

    vc_profiles = await asyncio.to_thread(db.get_active_vc_profiles)
    result["checks"]["active_vc_count"] = len(vc_profiles)
    result["vc_profiles"] = []
    for vc in vc_profiles:
        vc_id = vc.get("id", "")
        exists = await asyncio.to_thread(db.match_exists, vc_id, company_id)
        result["vc_profiles"].append({
            "id": vc_id,
            "firm_name": vc.get("firm_name"),
            "min_strategy_fit": vc.get("min_strategy_fit"),
            "min_team": vc.get("min_team"),
            "min_potential": vc.get("min_potential"),
            "thesis_length": len(vc.get("investment_thesis") or ""),
            "match_already_exists": exists,
        })

    will_attempt = (
        bool(company)
        and (bool(analysis_final) or bool(qa_pairs))
        and len(vc_profiles) > 0
    )
    result["will_attempt_matching"] = will_attempt
    if not will_attempt:
        if not company:
            result["skip_reason"] = "Company not found"
        elif not analysis_final and not qa_pairs:
            result["skip_reason"] = "No QA pairs or final_arguments — company needs a completed analysis first"
        elif not vc_profiles:
            result["skip_reason"] = "No active VC profiles"
    else:
        result["skip_reason"] = None

    return result


@app.get("/api/admin/logs")
async def admin_logs(
    since: str | None = None,
    level: str | None = None,
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Return recent in-process log entries for the admin live log.

    Args:
        since: ISO-8601 timestamp — only return entries after this time.
        level: Minimum level to include: DEBUG | INFO | WARNING | ERROR (default ALL).
    """
    entries = _InMemoryLogHandler.get_entries(since_ts=since, level=level)
    return {"entries": entries, "total": len(entries)}


@app.get("/api/admin/runs/{job_id}/log")
async def admin_run_log(
    job_id: str,
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Return progress log and cost breakdown for a single pipeline run.

    Resolution order:
      1. In-memory ``_jobs`` entry (live view for currently-running runs).
      2. Persisted ``analysis_events`` rows (fallback for completed/old runs).

    Always attempts to include ``run_costs`` via ``db.load_run_costs``.
    """
    job = _jobs.get(job_id)
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": None,
        "progress": None,
        "progress_log": [],
        "run_costs": None,
        "phase_models": None,
        "started_by": None,
        "company": None,
        "created_at": None,
        "error": None,
    }

    if job is not None:
        payload["status"] = job.status
        payload["progress"] = job.progress
        payload["progress_log"] = list(job.progress_log or [])
        payload["started_by"] = {
            "started_by_user_id": job.started_by_user_id,
            "started_by_email": job.started_by_email,
            "started_by_display_name": job.started_by_display_name,
            "started_by_label": job.started_by_label,
        }

    # Enrich with persisted fields (analysis row + run_costs) regardless of
    # whether the run is still live — live runs benefit from the live cost
    # snapshot once rows start getting persisted to model_executions.
    try:
        run_costs = await asyncio.to_thread(db.load_run_costs, job_id)
    except Exception:
        run_costs = None
    payload["run_costs"] = run_costs

    # Look up the analyses row (if persisted) to fill in company + phase_models
    # + error fields, and pull the persisted progress log when we don't have a
    # live in-memory copy.
    try:
        all_rows = await asyncio.to_thread(db.admin_get_recent_analyses, 100)
    except Exception:
        all_rows = []
    matching_row = next(
        (r for r in all_rows if r.get("job_id_legacy") == job_id),
        None,
    )
    if matching_row:
        payload["company"] = {
            "id": matching_row.get("company_id"),
            "name": matching_row.get("company_name"),
        }
        payload["phase_models"] = matching_row.get("phase_models")
        payload["created_at"] = matching_row.get("created_at")
        payload["error"] = matching_row.get("error")
        if not payload["status"]:
            payload["status"] = matching_row.get("status")
        if not payload["started_by"] and matching_row.get("started_by"):
            payload["started_by"] = matching_row.get("started_by")

    # Fallback log source when no in-memory job survived.
    if not payload["progress_log"]:
        try:
            persisted = await asyncio.to_thread(db.load_analysis_events, job_id, 500)
        except Exception:
            persisted = []
        payload["progress_log"] = list(persisted or [])
        if persisted and not payload["progress"]:
            payload["progress"] = persisted[-1]

    if (
        job is None
        and not matching_row
        and not payload["progress_log"]
        and not payload["run_costs"]
    ):
        raise HTTPException(status_code=404, detail="Run not found.")

    return payload


@app.get("/api/admin/pipeline-models")
async def admin_get_pipeline_models(
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Return persisted admin pipeline model defaults + catalog for the editor."""
    try:
        stored = await asyncio.to_thread(db.admin_get_pipeline_model_defaults)
    except Exception:
        stored = None
    try:
        factory = default_phase_model_selections()
    except Exception:
        factory = {}

    if isinstance(stored, dict):
        try:
            phase_models = coerce_phase_models_payload(stored, require_all=True)
        except Exception:
            phase_models = coerce_phase_models_payload(factory, require_all=True)
    else:
        phase_models = coerce_phase_models_payload(factory, require_all=True)

    catalog_list = available_models_payload()
    # Only offer models that are selectable (credentials present + supports
    # structured output).  Wrap in a dict so the frontend can evolve without
    # breaking on a bare list.
    selectable = [m for m in catalog_list if m.get("selectable", False)]
    return {
        "phase_models": phase_models,
        "factory_defaults": coerce_phase_models_payload(factory, require_all=True),
        "catalog": {"entries": selectable},
        "labels": PHASE_LABELS,
        "short_labels": PHASE_SHORT_LABELS,
    }


@app.put("/api/admin/pipeline-models")
async def admin_put_pipeline_models(
    payload: dict[str, Any] = Body(default_factory=dict),
    user: CurrentUser = Depends(_require_admin),
) -> dict[str, Any]:
    """Persist new admin pipeline model defaults. Validates all 5 stages."""
    raw = payload.get("phase_models") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=400,
            detail="phase_models (dict) is required.",
        )
    try:
        phase_models = coerce_phase_models_payload(raw, require_all=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    updated_by = user.email or user.display_name or user.id
    stored = await asyncio.to_thread(
        db.admin_set_pipeline_model_defaults,
        phase_models,
        updated_by=updated_by,
    )
    if stored is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to persist pipeline model defaults.",
        )
    return {"phase_models": stored}


# ---------------------------------------------------------------------------
# Sprint 3 — Debate engine
# ---------------------------------------------------------------------------

# Active debate tasks: debate_id → asyncio.Task (so we can cancel on pause)
_active_debate_tasks: dict[str, asyncio.Task] = {}

# WebSocket connections per debate: debate_id → list of WebSocket
_debate_ws_clients: dict[str, list[WebSocket]] = {}

# Re-evaluation ↔ paused-debate linkage so success/failure system_notes land in
# the right debates. Keyed by re-eval job_id; value is the list of paused
# debate ids snapshotted at the start of the re-evaluation run.
_reeval_debate_links: dict[str, list[str]] = {}


def _safe_analysis_for_debate(company_id: str) -> dict[str, Any]:
    """Return the full ranking_result dict from the latest analysis."""
    if not db:
        return {}
    analysis = db.get_company_latest_analysis(company_id)
    if not analysis:
        return {}
    payload = analysis.get("results_payload") or {}
    return payload.get("ranking_result") or {}


async def _broadcast_debate_message(debate_id: str, message: dict[str, Any]) -> None:
    """Send a message to all WebSocket clients watching this debate."""
    clients = _debate_ws_clients.get(debate_id, [])
    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


async def _run_debate_task(
    debate_id: str,
    company_name: str,
    vc_thesis: str,
    analysis_summary: dict[str, Any],
    chunks: list[dict[str, Any]],
    existing_messages: list[dict[str, Any]],
    current_round: int,
    max_rounds: int,
    *,
    context_notes: dict[str, Any] | None = None,
) -> None:
    """Background task that drives the debate loop and broadcasts via WebSocket."""
    import logging as _logging  # noqa: PLC0415
    _logger = _logging.getLogger(__name__)
    try:
        from agent.debate.orchestrator import (  # noqa: PLC0415
            DebatePausedForEvidence,
            run_debate,
        )
        async for message in run_debate(
            debate_id=debate_id,
            company_name=company_name,
            vc_thesis=vc_thesis,
            analysis_summary=analysis_summary,
            chunks=chunks,
            existing_messages=existing_messages,
            current_round=current_round,
            max_rounds=max_rounds,
            db_module=db,
            context_notes=context_notes,
        ):
            await _broadcast_debate_message(debate_id, message)
    except asyncio.CancelledError:
        _logger.info("Debate task %s cancelled (paused)", debate_id)
        await asyncio.to_thread(db.pause_debate, debate_id)
    except DebatePausedForEvidence:
        # Orchestrator already persisted the evidence_request message and
        # flipped debates.status='paused' + awaiting_input_from='founder'.
        # Exit cleanly — this is not an error.
        _logger.info("Debate %s paused awaiting founder evidence", debate_id)
    except Exception as exc:
        _logger.error("Debate task %s error: %s", debate_id, exc)
    finally:
        _active_debate_tasks.pop(debate_id, None)


@app.post("/api/debates", status_code=201)
async def create_debate(
    match_id: str,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Start a new debate for a match. The VC must have actioned the match as 'in_debate'
    before calling this endpoint.

    Kicks off the debate background task immediately.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    # Verify the match belongs to this VC
    vc_profile = await asyncio.to_thread(_get_vc_profile_or_404, user)
    vc_profile_id: str = vc_profile["id"]

    matches = await asyncio.to_thread(db.get_matches_for_vc, vc_profile_id)
    match = next((m for m in matches if m["id"] == match_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")

    company_id: str = (match.get("companies") or {}).get("id") or ""
    company_name: str = (match.get("companies") or {}).get("name") or "Unknown"
    if not company_id:
        raise HTTPException(status_code=400, detail="Company data unavailable.")

    # Idempotent — return existing debate if already created
    existing = await asyncio.to_thread(db.get_debate_by_match, match_id)
    if existing:
        return existing

    debate = await asyncio.to_thread(
        db.create_debate,
        match_id=match_id,
        company_id=company_id,
        vc_profile_id=vc_profile_id,
    )
    if not debate:
        raise HTTPException(status_code=500, detail="Failed to create debate.")

    debate_id: str = debate["id"]

    # Mark match as in_debate
    await asyncio.to_thread(db.update_match_status, match_id, "in_debate")

    # Start background task
    vc_thesis: str = vc_profile.get("investment_thesis") or ""
    analysis_summary = await asyncio.to_thread(_safe_analysis_for_debate, company_id)
    chunks = await asyncio.to_thread(db.get_company_chunks, company_id)

    task = asyncio.create_task(
        _run_debate_task(
            debate_id=debate_id,
            company_name=company_name,
            vc_thesis=vc_thesis,
            analysis_summary=analysis_summary,
            chunks=chunks,
            existing_messages=[],
            current_round=1,
            max_rounds=3,
        )
    )
    _active_debate_tasks[debate_id] = task

    return debate


@app.get("/api/debates/{debate_id}")
async def get_debate(
    debate_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Return a debate and all its messages."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    debate = await asyncio.to_thread(db.get_debate_by_id, debate_id)
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found.")
    messages = await asyncio.to_thread(db.get_debate_messages, debate_id)
    return {**debate, "messages": messages}


@app.post("/api/debates/{debate_id}/pause")
async def pause_debate(
    debate_id: str,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Pause a running debate (cancels the background task; state is persisted)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    task = _active_debate_tasks.get(debate_id)
    if task and not task.done():
        task.cancel()
    else:
        await asyncio.to_thread(db.pause_debate, debate_id)
    return {"debate_id": debate_id, "status": "paused"}


async def _spawn_debate_task(
    debate_id: str,
    *,
    context_notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load debate context and spawn a background ``_run_debate_task``.

    Shared by the VC resume endpoint and the founder respond endpoint so both
    code paths are guaranteed to hydrate the orchestrator the same way.
    Returns a dict ``{debate_id, status, current_round}`` on success.

    Raises HTTPException(404) if the debate does not exist, HTTPException(400)
    if it is already completed.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    debate = await asyncio.to_thread(db.get_debate_by_id, debate_id)
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found.")
    if debate.get("status") == "completed":
        raise HTTPException(status_code=400, detail="Debate is already completed.")

    company_id: str = debate["company_id"]
    vc_profile_id: str | None = debate.get("vc_profile_id")
    vc_thesis: str = ""
    if vc_profile_id:
        vc_profile = await asyncio.to_thread(db.get_vc_profile_by_id, vc_profile_id)
        if vc_profile:
            vc_thesis = vc_profile.get("investment_thesis") or ""

    company_row = await asyncio.to_thread(db.get_company_by_id, company_id)
    company_name: str = (company_row or {}).get("name") or "Unknown"

    analysis_summary = await asyncio.to_thread(_safe_analysis_for_debate, company_id)
    chunks = await asyncio.to_thread(db.get_company_chunks, company_id)
    existing_messages = await asyncio.to_thread(db.get_debate_messages, debate_id)
    current_round: int = debate.get("current_round") or 1
    max_rounds: int = debate.get("max_rounds") or 3

    await asyncio.to_thread(db.resume_debate, debate_id)

    task = asyncio.create_task(
        _run_debate_task(
            debate_id=debate_id,
            company_name=company_name,
            vc_thesis=vc_thesis,
            analysis_summary=analysis_summary,
            chunks=chunks,
            existing_messages=existing_messages,
            current_round=current_round,
            max_rounds=max_rounds,
            context_notes=context_notes,
        )
    )
    _active_debate_tasks[debate_id] = task

    return {"debate_id": debate_id, "status": "active", "current_round": current_round}


@app.post("/api/debates/{debate_id}/resume")
async def resume_debate(
    debate_id: str,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """Resume a paused debate from where it left off (VC-initiated)."""
    return await _spawn_debate_task(debate_id, context_notes=None)


@app.get("/api/vc/debates")
async def vc_list_debates(user: CurrentUser = Depends(_require_vc)) -> dict[str, Any]:
    """Return all debates for the VC."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    profile = await asyncio.to_thread(_get_vc_profile_or_404, user)
    debates = await asyncio.to_thread(db.get_debates_for_vc, profile["id"])
    return {"debates": debates}


@app.get("/api/startup/debates")
async def startup_list_debates(user: CurrentUser = Depends(_require_startup)) -> dict[str, Any]:
    """Return all debates for the startup's company.

    For any debate that is currently paused awaiting founder input, we
    additionally load the most recent ``evidence_request`` message so the
    portal can render the topic / questions / rationale without a second
    round-trip per debate.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    debates = await asyncio.to_thread(db.get_debates_for_company, company_id)

    for debate in debates:
        if (
            debate.get("status") == "paused"
            and debate.get("awaiting_input_from") == "founder"
        ):
            try:
                latest_req = await asyncio.to_thread(
                    db.get_latest_evidence_request_message, debate["id"]
                )
            except Exception:
                latest_req = None
            debate["latest_evidence_request"] = latest_req

    return {"debates": debates}


class StartupDebateResponseRequest(BaseModel):
    """Payload for the founder's respond endpoint.

    ``response_type='uploaded'`` means the founder uploaded new evidence and
    (optionally) kicked off a re-evaluation via ``reeval_job_id``. The debate
    resumes immediately; the VC agent will pick up the new analysis on its
    next turn.

    ``response_type='unavailable'`` means the founder cannot provide the
    requested evidence. The debate resumes with a ``gap_note`` injected into
    the first startup turn so the Startup Agent acknowledges the gap instead
    of fabricating data.
    """

    response_type: Literal["uploaded", "unavailable"]
    reeval_job_id: str | None = None
    note: str | None = None


@app.post("/api/startup/debates/{debate_id}/respond")
async def startup_respond_to_debate(
    debate_id: str,
    body: StartupDebateResponseRequest,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Founder response to an evidence request — resumes the paused debate.

    Authorisation: the debate must belong to the founder's linked company.
    On a cross-company attempt we return 404 (not 403) so we don't leak
    existence of other founders' debates.

    Concurrency: we atomically claim the ``awaiting_input_from='founder'``
    flag via ``db.claim_founder_response`` so a double-click (or a race with
    upload-complete vs. decline) cannot produce two founder_response messages.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    debate = await asyncio.to_thread(db.get_debate_by_id, debate_id)
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found.")

    # Auth: the debate must belong to the founder's linked company.
    founder_company_id = await asyncio.to_thread(_first_linked_company_id, user)
    if debate.get("company_id") != founder_company_id:
        # Return 404 instead of 403 — do not leak existence of other debates.
        raise HTTPException(status_code=404, detail="Debate not found.")

    if debate.get("status") != "paused" or debate.get("awaiting_input_from") != "founder":
        raise HTTPException(
            status_code=409,
            detail="Debate is not currently awaiting founder input.",
        )

    # Atomic claim: exactly one founder response wins.
    claimed = await asyncio.to_thread(db.claim_founder_response, debate_id)
    if not claimed:
        raise HTTPException(
            status_code=409,
            detail="This debate has already been responded to.",
        )

    # Find the topic from the latest evidence_request so we can pass a gap_note
    # if the founder declined to provide it.
    latest_req = await asyncio.to_thread(db.get_latest_evidence_request_message, debate_id)
    topic: str | None = None
    if latest_req and isinstance(latest_req.get("info_request"), dict):
        topic_raw = latest_req["info_request"].get("topic")
        if isinstance(topic_raw, str) and topic_raw.strip():
            topic = topic_raw.strip()

    if body.response_type == "uploaded":
        content = (
            body.note
            or "Founder uploaded new evidence"
            + (f" and kicked off re-evaluation {body.reeval_job_id}"
               if body.reeval_job_id else "")
            + "."
        )
    else:
        content = body.note or (
            f"Founder declined to provide evidence: {topic}." if topic
            else "Founder stated they cannot provide the requested evidence."
        )

    current_round = int(debate.get("current_round") or 1)

    saved_msg = await asyncio.to_thread(
        db.save_debate_message,
        debate_id=debate_id,
        round=current_round,
        speaker="system",
        content=content,
        citations=[],
        message_type="founder_response",
        founder_response_type=body.response_type,
        linked_reeval_job_id=body.reeval_job_id,
    )

    # Broadcast so any live WebSocket subscriber sees the response in real time.
    if saved_msg:
        await _broadcast_debate_message(debate_id, saved_msg)

    context_notes: dict[str, Any] | None = None
    if body.response_type == "unavailable" and topic:
        context_notes = {"gap_note": topic}

    return await _spawn_debate_task(debate_id, context_notes=context_notes)


@app.get("/api/startup/reeval-status/{job_id}")
async def startup_reeval_status(
    job_id: str,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Lightweight polling endpoint for a re-evaluation triggered from the
    founder portal. Returns the in-memory job status when available.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    progress_log = getattr(job, "progress_log", None) or []
    return {
        "job_id": job_id,
        "status": getattr(job, "status", None),
        "progress": getattr(job, "progress", None),
        "progress_log": progress_log[-20:],  # cap tail for bandwidth
    }


@app.websocket("/ws/debates/{debate_id}")
async def debate_websocket(websocket: WebSocket, debate_id: str):
    """WebSocket endpoint for real-time debate message streaming.

    Clients connect here to receive new messages as the agents generate them.
    Authentication via ?token=<bearer_token> query param.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001)
        return

    # Validate token
    try:
        user = await get_current_user(authorization=f"Bearer {token}")
    except HTTPException:
        await websocket.close(code=4001)
        return

    await websocket.accept()

    # Register client
    if debate_id not in _debate_ws_clients:
        _debate_ws_clients[debate_id] = []
    _debate_ws_clients[debate_id].append(websocket)

    # Send existing messages immediately on connect
    if db:
        messages = await asyncio.to_thread(db.get_debate_messages, debate_id)
        for msg in messages:
            await websocket.send_json(msg)

    try:
        while True:
            # Keep connection alive; client sends pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients = _debate_ws_clients.get(debate_id, [])
        if websocket in clients:
            clients.remove(websocket)


# ---------------------------------------------------------------------------
# Sprint 4 — Evidence upload, re-analysis, new company onboarding
# ---------------------------------------------------------------------------

class CreateCompanyRequest(BaseModel):
    """Request body for creating a new company via the startup portal."""

    name: str
    industry: str | None = None
    tagline: str | None = None
    about: str | None = None
    domain: str | None = None


@app.post("/api/startup/company")
async def startup_create_company(
    req: CreateCompanyRequest,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Create a new company record for a startup that has no existing DB entry.

    Fails if the user already has a linked company. After creation, the company
    is linked to the user via user_company_links. The startup can then upload
    documents and run a full analysis.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    # Prevent creating a second company.
    links = await asyncio.to_thread(db.get_user_company_links, user.id)
    if links:
        raise HTTPException(
            status_code=400,
            detail="You already have a linked company. Use the upload endpoint to add more evidence.",
        )

    company_id = await asyncio.to_thread(
        db.create_company_from_portal,
        name=req.name,
        industry=req.industry,
        tagline=req.tagline,
        about=req.about,
        domain=req.domain,
    )
    if not company_id:
        raise HTTPException(status_code=500, detail="Failed to create company record.")

    # Link the user to the new company.
    link_ok = await asyncio.to_thread(
        db.create_user_company_link,
        user_id=user.id,
        company_id=company_id,
    )
    if not link_ok:
        raise HTTPException(status_code=500, detail="Company created but failed to link to user.")

    return {"company_id": company_id, "name": req.name}


_SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".pptx", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md"}


@app.post("/api/startup/upload")
async def startup_upload(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Upload one or more documents and ingest them as evidence chunks.

    Supports: PDF, PPTX, DOCX, XLSX, XLS, CSV, TXT, MD.
    Files are saved to Supabase Storage and chunks are inserted into the DB.
    A new pitch_decks row is created for each uploaded file.

    Returns: list of {filename, pitch_deck_id, chunks_count} for each file.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    company_id = await asyncio.to_thread(_first_linked_company_id, user)

    import tempfile  # noqa: PLC0415

    from agent.ingest import EvidenceStore  # noqa: PLC0415
    from agent.ingest import _EXTENSION_MAP, _TEXT_EXTENSIONS  # noqa: PLC0415
    from agent.ingest.chunking import smart_chunk_texts  # noqa: PLC0415

    results: list[dict[str, Any]] = []
    tmp_dir = tempfile.mkdtemp(prefix="startup_upload_")

    try:
        for upload_file in files:
            filename = upload_file.filename or "upload"
            suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            if suffix not in _SUPPORTED_UPLOAD_EXTENSIONS:
                results.append({
                    "filename": filename,
                    "error": f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(_SUPPORTED_UPLOAD_EXTENSIONS))}",
                })
                continue

            file_bytes = await upload_file.read()
            if not file_bytes:
                results.append({"filename": filename, "error": "Empty file."})
                continue

            # Save to temp dir for parsing.
            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_bytes(file_bytes)

            # Parse using ingest pipeline.
            try:
                extractor = _EXTENSION_MAP.get(suffix)
                if extractor is not None:
                    raw_items = extractor(tmp_path)
                elif suffix in _TEXT_EXTENSIONS:
                    raw_items = [{"text": tmp_path.read_text(errors="replace").strip(), "page_or_slide": "N/A", "source_file": filename}]
                else:
                    raw_items = []

                if not raw_items:
                    results.append({"filename": filename, "error": "No content could be extracted."})
                    continue

                chunks = smart_chunk_texts(raw_items)
            except Exception as exc:
                _log = logging.getLogger(__name__)
                _log.warning("Failed to parse upload %s: %s", filename, exc)
                results.append({"filename": filename, "error": f"Parse failed: {exc}"})
                continue

            # Upload bytes to Supabase Storage (best-effort).
            storage_path = await asyncio.to_thread(
                db.upload_startup_file_bytes,
                company_id,
                filename,
                file_bytes,
            ) or f"startup_uploads/{company_id}/{filename}"

            # Create pitch_decks row and insert chunks.
            pitch_deck_id = await asyncio.to_thread(
                db.create_pitch_deck_for_upload,
                company_id,
                storage_path,
                filename,
            )
            if not pitch_deck_id:
                results.append({"filename": filename, "error": "Failed to create pitch deck record."})
                continue

            chunk_count = await asyncio.to_thread(
                db.insert_chunks_for_pitch_deck,
                pitch_deck_id,
                chunks,
            )

            results.append({
                "filename": filename,
                "pitch_deck_id": pitch_deck_id,
                "chunks_count": chunk_count,
            })
    finally:
        import shutil  # noqa: PLC0415
        shutil.rmtree(tmp_dir, ignore_errors=True)

    files_processed = sum(1 for r in results if "pitch_deck_id" in r)
    return {
        "files_processed": files_processed,
        "results": results,
    }


@app.post("/api/startup/re-evaluate")
async def startup_re_evaluate(
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] = Body(default_factory=dict),
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Trigger a re-evaluation of the startup's company using all available evidence.

    Re-evaluation is additive: all existing chunks (from Specter, original pitch deck,
    and any prior uploads) are combined with any new upload into a single EvidenceStore.
    Stage 1 (decomposition) is skipped if a prior analysis exists — the cached
    question trees are reused. Stages 2-8 are re-run. Old matches are deleted
    and re-computed against all active VCs.

    Optional body: {"use_web_search": bool} — when true, Stage 2 may call the
    configured web search provider (Perplexity/Brave) to fill gaps where the
    document evidence does not answer a question.

    Returns immediately. Status can be polled via GET /api/startup/profile.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    use_web_search = bool(payload.get("use_web_search", False))
    job_id = "re-" + uuid.uuid4().hex[:6]
    background_tasks.add_task(
        _run_re_evaluation,
        company_id,
        use_web_search,
        triggered_by=user,
        job_id=job_id,
    )
    return {
        "status": "re_evaluation_started",
        "company_id": company_id,
        "use_web_search": use_web_search,
        "job_id": job_id,
    }


# ---------------------------------------------------------------------------
# Data Room endpoints
# ---------------------------------------------------------------------------

_DATA_ROOM_CATEGORIES = {"pitch_deck", "financials", "legal", "team", "product", "other"}


@app.put("/api/startup/data-room/toggle")
async def startup_data_room_toggle(
    user: CurrentUser = Depends(_require_startup),
    enabled: bool = Body(..., embed=True),
) -> dict[str, Any]:
    """Enable or disable the data room for the startup's company."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    ok = await asyncio.to_thread(db.set_data_room_enabled, company_id, enabled)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update data room setting.")
    return {"data_room_enabled": enabled}


@app.post("/api/startup/data-room/upload")
async def startup_data_room_upload(
    file: UploadFile,
    user: CurrentUser = Depends(_require_startup),
    category: str = Form("other"),
    also_evidence: bool = Form(False),
) -> dict[str, Any]:
    """Upload a file to the startup's data room.

    If also_evidence is True, the file is additionally parsed, chunked and
    ingested as evidence for the AI pipeline (same as POST /api/startup/upload).
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in _SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(_SUPPORTED_UPLOAD_EXTENSIONS))}",
        )

    if category not in _DATA_ROOM_CATEGORIES:
        category = "other"

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Upload to Supabase Storage under data_room/ sub-path.
    safe_name = __import__("re").sub(r"[^a-zA-Z0-9._\-]+", "-", filename).strip("-") or "upload"
    storage_path = f"startup_uploads/{company_id}/data_room/{safe_name}"
    try:
        from web.db import ensure_source_files_bucket, SOURCE_FILES_BUCKET  # noqa: PLC0415
        await asyncio.to_thread(ensure_source_files_bucket)
        client = db._get_client()
        options: dict[str, Any] = {"upsert": "true"}
        content_type = file.content_type
        if content_type:
            options["content-type"] = content_type
        await asyncio.to_thread(
            client.storage.from_(SOURCE_FILES_BUCKET).upload,
            storage_path, file_bytes, options,
        )
    except Exception as exc:
        _log = logging.getLogger(__name__)
        _log.warning("Data room storage upload failed for %s: %s", filename, exc)
        # Proceed anyway — record the row so the user sees the file.

    # Optionally ingest as evidence.
    pitch_deck_id: str | None = None
    chunks_count = 0
    if also_evidence:
        import tempfile  # noqa: PLC0415
        from agent.ingest import _EXTENSION_MAP, _TEXT_EXTENSIONS  # noqa: PLC0415
        from agent.ingest.chunking import smart_chunk_texts  # noqa: PLC0415

        tmp_dir = tempfile.mkdtemp(prefix="dr_upload_")
        try:
            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_bytes(file_bytes)

            extractor = _EXTENSION_MAP.get(suffix)
            if extractor is not None:
                raw_items = extractor(tmp_path)
            elif suffix in _TEXT_EXTENSIONS:
                raw_items = [{"text": tmp_path.read_text(errors="replace").strip(), "page_or_slide": "N/A", "source_file": filename}]
            else:
                raw_items = []

            if raw_items:
                chunks = smart_chunk_texts(raw_items)
                # Reuse the data_room storage_path — no second upload needed.
                pitch_deck_id = await asyncio.to_thread(
                    db.create_pitch_deck_for_upload, company_id, storage_path, filename,
                )
                if pitch_deck_id:
                    chunks_count = await asyncio.to_thread(
                        db.insert_chunks_for_pitch_deck, pitch_deck_id, chunks,
                    )
        except Exception as exc:
            _log = logging.getLogger(__name__)
            _log.warning("Data room evidence ingest failed for %s: %s", filename, exc)
        finally:
            import shutil  # noqa: PLC0415
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Create the data_room_files row.
    row = await asyncio.to_thread(
        db.create_data_room_file,
        company_id=company_id,
        storage_path=storage_path,
        original_filename=filename,
        file_size_bytes=len(file_bytes),
        mime_type=file.content_type,
        category=category,
        also_evidence=also_evidence,
        pitch_deck_id=pitch_deck_id,
        uploaded_by=user.id,
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create data room record.")

    return {
        "file": {
            "id": row.get("id"),
            "original_filename": row.get("original_filename"),
            "file_size_bytes": row.get("file_size_bytes"),
            "mime_type": row.get("mime_type"),
            "category": row.get("category"),
            "also_evidence": row.get("also_evidence"),
            "created_at": row.get("created_at"),
        },
        "evidence_created": also_evidence and pitch_deck_id is not None,
        "chunks_count": chunks_count,
    }


@app.get("/api/startup/data-room/files")
async def startup_data_room_files(
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """List all data room files for the startup's company."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    files = await asyncio.to_thread(db.list_data_room_files, company_id)
    return {"files": files}


@app.delete("/api/startup/data-room/files/{file_id}")
async def startup_data_room_delete(
    file_id: str,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Delete a data room file (DB row + Storage object)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)

    # Verify ownership.
    file_row = await asyncio.to_thread(db.get_data_room_file, file_id)
    if not file_row or file_row.get("company_id") != company_id:
        raise HTTPException(status_code=404, detail="File not found.")

    # Delete from Storage (best-effort).
    await asyncio.to_thread(db.delete_storage_file, file_row.get("storage_path", ""))

    # If also_evidence and linked pitch_deck, delete the pitch_deck (chunks cascade).
    if file_row.get("also_evidence") and file_row.get("pitch_deck_id"):
        try:
            def _delete_pitch_deck(pd_id: str) -> None:
                client = db._get_client()
                if client:
                    client.table("pitch_decks").delete().eq("id", pd_id).execute()
            await asyncio.to_thread(_delete_pitch_deck, file_row["pitch_deck_id"])
        except Exception:
            pass  # best-effort

    # Delete the data_room_files row.
    await asyncio.to_thread(db.delete_data_room_file, file_id)
    return {"deleted": True}


@app.get("/api/vc/matches/{match_id}/data-room")
async def vc_match_data_room(
    match_id: str,
    user: CurrentUser = Depends(_require_vc),
) -> dict[str, Any]:
    """List data room files for a matched company. Verifies VC ownership + data_room_enabled."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    files = await asyncio.to_thread(db.list_data_room_files_for_match, match_id, user.id)
    if files is None:
        raise HTTPException(status_code=403, detail="Access denied or data room not enabled.")
    return {"files": files}


@app.get("/api/data-room/download/{file_id}")
async def data_room_download(
    file_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Generate a signed download URL for a data room file.

    Accessible by:
    - Startup users who own the company
    - VC users with an active match to the company (and data_room_enabled)
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    file_row = await asyncio.to_thread(db.get_data_room_file, file_id)
    if not file_row:
        raise HTTPException(status_code=404, detail="File not found.")

    company_id = file_row.get("company_id")

    # Check access.
    allowed = False
    if user.role == "startup":
        try:
            linked = await asyncio.to_thread(_first_linked_company_id, user)
            allowed = linked == company_id
        except HTTPException:
            pass
    elif user.role in ("vc", "admin"):
        # Check if VC has any active match with this company.
        profile = await asyncio.to_thread(db.get_vc_profile, user.id)
        if profile:
            matches = await asyncio.to_thread(db.get_matches_for_vc, profile["id"])
            for m in matches:
                co = m.get("companies") or {}
                if co.get("id") == company_id and co.get("data_room_enabled"):
                    allowed = True
                    break
    elif user.role == "admin":
        allowed = True

    if not allowed:
        raise HTTPException(status_code=403, detail="Access denied.")

    url = await asyncio.to_thread(db.create_signed_download_url, file_row.get("storage_path", ""))
    if not url:
        raise HTTPException(status_code=500, detail="Failed to generate download URL.")

    return {"download_url": url, "filename": file_row.get("original_filename")}


@app.post("/api/startup/analyze")
async def startup_analyze(
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(_require_startup),
) -> dict[str, Any]:
    """Trigger a full pipeline analysis for a newly onboarded company.

    Runs all 8 stages (decomposition through ranking). Use this for companies
    that have uploaded documents but have no prior analysis record.
    Does NOT auto-trigger VC matching — the startup must explicitly toggle
    fundraising ON after reviewing their scores.

    Returns immediately. Status can be polled via GET /api/startup/profile.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    company_id = await asyncio.to_thread(_first_linked_company_id, user)
    background_tasks.add_task(_run_full_analysis, company_id)
    return {"status": "analysis_started", "company_id": company_id}


_DIMENSION_BY_ASPECT = {
    "general_company": "strategy_fit",
    "team": "team",
    "market": "upside",
    "product": "upside",
}


def _build_evidence_provenance(
    final_state: dict,
    all_qa_pairs: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build qa_provenance_rows and argument_rows from a completed pipeline state.

    Mirrors the format produced by build_qa_provenance_rows() / build_argument_rows()
    in src/agent/batch.py but works with a single company's final_state dict.

    Args:
        final_state: The graph's final state dict (used for final_arguments).
        all_qa_pairs: Pre-built Q&A pairs to use. If None, reads from final_state.
            Prefer passing this directly from the caller's local variable to avoid
            LangGraph state-merging artefacts (duplicate list items).

    Returns:
        (qa_provenance_rows, argument_rows)
    """
    # --- QA provenance rows ---
    # Prefer the caller-supplied list; fall back to final_state for backward compat.
    _qa_source = all_qa_pairs if all_qa_pairs is not None else (final_state.get("all_qa_pairs") or [])
    qa_provenance_rows: list[dict] = []
    for qa in _qa_source:
        chunk_ids = qa.get("chunk_ids")
        if isinstance(chunk_ids, list):
            chunk_ids_str = ", ".join(str(c) for c in chunk_ids)
        else:
            chunk_ids_str = str(chunk_ids) if chunk_ids else ""
        aspect = str(qa.get("aspect") or "")
        qa_provenance_rows.append({
            "aspect": aspect,
            "dimension": _DIMENSION_BY_ASPECT.get(aspect.strip(), ""),
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "chunk_ids": chunk_ids_str,
            "chunks_preview": qa.get("chunks_preview", ""),
            "web_search_query": qa.get("web_search_query") or "",
            "web_search_results": qa.get("web_search_results") or "",
            "web_search_used": bool(qa.get("web_search_used")),
            "web_search_decision": qa.get("web_search_decision") or "",
        })

    # --- Argument rows ---
    argument_rows: list[dict] = []
    for arg in (final_state.get("final_arguments") or []):
        arg_dict = arg.model_dump() if hasattr(arg, "model_dump") else arg
        # Build qa_pairs_used summary text (same format as batch.py)
        qa_pairs_used = ""
        if arg_dict.get("qa_pairs"):
            qa_pairs_used = "\n---\n".join(
                f"Q: {q.get('question', '')}\nA: {q.get('answer', '')}"
                for q in arg_dict["qa_pairs"]
            )
        argument_rows.append({
            "type": arg_dict.get("argument_type", ""),
            "score": arg_dict.get("score"),
            "argument_text": arg_dict.get("content", ""),
            "critique_text": arg_dict.get("critique") or "",
            "refined_text": arg_dict.get("refined_content") or "",
            "argument_feedback": arg_dict.get("argument_feedback") or "",
            "qa_pairs_used": qa_pairs_used,
            "qa_indices": arg_dict.get("qa_indices", []),
        })

    return qa_provenance_rows, argument_rows


async def _run_re_evaluation(
    company_id: str,
    use_web_search: bool = False,
    *,
    triggered_by: CurrentUser | None = None,
    job_id: str | None = None,
) -> None:
    """Background task: re-evaluate a company with all available evidence.

    Evidence is additive — combines Specter chunks, original pitch deck chunks,
    and all portal-uploaded chunks. Reuses cached question_trees to skip Stage 1.

    Per-stage models are loaded from admin_settings.pipeline_model_defaults (or
    factory defaults when absent), and all 8 pipeline stages read them via
    ``use_run_context(pipeline_policy=...)``.

    A ``RunTelemetryCollector`` captures every LLM call and Perplexity search;
    results are persisted to ``model_executions`` and the aggregated
    ``run_costs`` block is embedded in ``results_payload``.

    If ``use_web_search`` is True, Stage 2 may call the configured web search
    provider (Perplexity) to fill gaps where document evidence is missing.
    """
    import logging as _log_module  # noqa: PLC0415
    _logger = _log_module.getLogger(__name__)

    # 0. Mint a trackable job id and register with the in-memory job store so
    #    the admin Analyses tab + log modal can poll progress.
    if not job_id:
        job_id = "re-" + uuid.uuid4().hex[:6]

    started_by_user_id = getattr(triggered_by, "id", None)
    started_by_email = getattr(triggered_by, "email", None)
    started_by_display_name = getattr(triggered_by, "display_name", None)
    started_by_label = started_by_display_name or started_by_email or "portal"

    _jobs[job_id] = AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting re-evaluation",
        progress_log=[],
        started_by_user_id=started_by_user_id,
        started_by_email=started_by_email,
        started_by_display_name=started_by_display_name,
        started_by_label=started_by_label,
    )

    _logger.info(
        "Re-evaluation started job=%s company=%s use_web_search=%s user=%s",
        job_id, company_id, use_web_search, started_by_email or "?",
    )

    # Load admin-configured per-stage model defaults (fall back to factory).
    try:
        stored_defaults = await asyncio.to_thread(db.admin_get_pipeline_model_defaults)
    except Exception as exc:
        _logger.warning("Re-evaluation: failed to load admin pipeline defaults: %s", exc)
        stored_defaults = None
    try:
        factory_defaults = default_phase_model_selections()
    except Exception:
        factory_defaults = {}

    phase_models_raw = stored_defaults if isinstance(stored_defaults, dict) else factory_defaults
    try:
        phase_models = coerce_phase_models_payload(phase_models_raw, require_all=True)
    except Exception as exc:
        _logger.warning(
            "Re-evaluation: stored phase_models invalid (%s); falling back to factory defaults",
            exc,
        )
        phase_models = coerce_phase_models_payload(factory_defaults, require_all=True)

    try:
        pipeline_policy = build_phase_model_policy(phase_models)
    except Exception as exc:
        _logger.warning("Re-evaluation: failed to build pipeline policy: %s", exc)
        pipeline_policy = None

    # NB: RunTelemetryCollector only takes `selected_llm`; company slug and job
    # id are propagated through contextvars by `use_company_context` and the
    # collector+pipeline policy bound inside `use_run_context` below.
    collector = RunTelemetryCollector()

    run_costs: dict[str, Any] | None = None
    analysis_id: str | None = None
    final_state_error: Exception | None = None

    try:
        from agent.dataclasses.company import Company  # noqa: PLC0415
        from agent.dataclasses.config import Config  # noqa: PLC0415
        from agent.dataclasses.question_tree import QuestionTree  # noqa: PLC0415
        from agent.evidence_answering import (  # noqa: PLC0415
            answer_all_trees_from_evidence,
            enrich_qa_pairs_with_new_evidence,
            write_qa_pairs_into_trees,
        )
        from agent.ingest.store import Chunk, EvidenceStore  # noqa: PLC0415
        from agent.pipeline.graph import graph  # noqa: PLC0415
        from agent.pipeline.state.investment_story import IterativeInvestmentStoryState  # noqa: PLC0415

        _append_progress(job_id, "Loading company and evidence chunks")

        # 1. Load company row.
        company_row = await asyncio.to_thread(db.get_company_by_id, company_id)
        if not company_row:
            _logger.error("Re-evaluation: company %s not found", company_id)
            _append_progress(job_id, f"ERROR: company {company_id} not found")
            _set_job_status(job_id, "error")
            return

        # 1a. Snapshot paused debates awaiting founder input so success/failure
        #     system_notes can land in the right rows. Stored in a module-level
        #     map keyed by job_id so the except block below can reach it too.
        try:
            paused_debates_snapshot = await asyncio.to_thread(
                db.get_paused_debates_for_company, company_id
            )
        except Exception as exc:
            _logger.warning(
                "Re-evaluation: failed to snapshot paused debates for %s: %s",
                company_id, exc,
            )
            paused_debates_snapshot = []
        paused_debate_ids = [d.get("id") for d in paused_debates_snapshot if d.get("id")]
        if paused_debate_ids:
            _reeval_debate_links[job_id] = paused_debate_ids
            _logger.info(
                "Re-evaluation job=%s linked to %d paused debate(s): %s",
                job_id, len(paused_debate_ids), paused_debate_ids,
            )

        company = Company(
            name=company_row.get("name") or "Unknown",
            industry=company_row.get("industry"),
            tagline=company_row.get("tagline"),
            about=company_row.get("about"),
            domain=company_row.get("domain"),
        )

        # 2. Load ALL chunks (no Specter filter) — additive corpus.
        all_chunk_rows = await asyncio.to_thread(db.get_all_company_chunks, company_id)
        if not all_chunk_rows:
            _logger.warning("Re-evaluation: no chunks found for company=%s — aborting", company_id)
            _append_progress(job_id, "ERROR: no evidence chunks found for company")
            _set_job_status(job_id, "error")
            return

        store = EvidenceStore(startup_slug=company_id)
        for idx, c in enumerate(all_chunk_rows):
            store.chunks.append(Chunk(
                chunk_id=c.get("chunk_id") or f"chunk_{idx}",
                text=c.get("text") or "",
                source_file=c.get("source_file") or "",
                page_or_slide=c.get("page_or_slide") or "N/A",
            ))
        _append_progress(job_id, f"Loaded {len(store.chunks)} evidence chunks")

        # 3. Try to load cached question_trees to skip Stage 1.
        cached_trees_raw = await asyncio.to_thread(db.get_analysis_question_trees, company_id)
        question_trees: dict[str, QuestionTree] | None = None
        if cached_trees_raw:
            try:
                question_trees = {
                    aspect: QuestionTree.model_validate(tree_data)
                    for aspect, tree_data in cached_trees_raw.items()
                }
                _append_progress(
                    job_id,
                    f"Stage 1: reusing {len(question_trees)} cached question trees",
                )
            except Exception as exc:
                _logger.warning("Re-evaluation: failed to reconstruct question trees: %s", exc)
                question_trees = None

        config = Config(
            n_pro_arguments=3,
            n_contra_arguments=3,
            k_best_arguments_per_iteration=[3, 1],
            max_iterations=1,
        )

        # 3a. Load previous analysis snapshot for INCREMENTAL ENRICHMENT.
        # Re-evaluation must preserve good previous answers verbatim when
        # the newly uploaded evidence is not relevant to a question — and
        # enrich (not replace) them when it is. See plan file Part B.
        prev_analysis = await asyncio.to_thread(db.get_latest_analysis_full, company_id)
        previous_all_qa_pairs: list[dict] | None = None
        prev_created_at: str | None = None
        if prev_analysis and isinstance(prev_analysis.get("state"), dict):
            raw_qa = prev_analysis["state"].get("all_qa_pairs")
            if isinstance(raw_qa, list) and raw_qa:
                previous_all_qa_pairs = [dict(p) for p in raw_qa if isinstance(p, dict)]
            prev_created_at = prev_analysis.get("created_at")

        # 3b. Identify chunks uploaded AFTER the previous analysis.
        new_chunk_rows: list[dict] = []
        if prev_created_at:
            try:
                new_chunk_rows = await asyncio.to_thread(
                    db.get_chunks_created_after, company_id, prev_created_at,
                )
            except Exception as exc:
                _logger.warning(
                    "Re-evaluation: get_chunks_created_after failed: %s", exc,
                )
                new_chunk_rows = []
        new_chunks_list: list[Chunk] = [
            Chunk(
                chunk_id=c.get("chunk_id") or f"new_chunk_{i}",
                text=c.get("text") or "",
                source_file=c.get("source_file") or "",
                page_or_slide=c.get("page_or_slide") or "N/A",
            )
            for i, c in enumerate(new_chunk_rows)
        ]
        _append_progress(
            job_id,
            f"Found {len(new_chunks_list)} new evidence chunks since previous analysis"
            + (f" (prev={prev_created_at})" if prev_created_at else " (no previous analysis)"),
        )

        # Wrap the whole pipeline in a run context so stages read the admin
        # pipeline policy and the collector captures token usage + perplexity.
        # `use_run_context` only takes pipeline_policy / telemetry_collector /
        # llm_selection — company slug is propagated via `use_company_context`
        # so `get_current_company_slug()` returns the right value inside the
        # telemetry callback.
        with use_company_context(company_id), use_run_context(
            pipeline_policy=pipeline_policy,
            telemetry_collector=collector,
        ):
            if question_trees:
                if previous_all_qa_pairs and new_chunks_list:
                    _append_progress(
                        job_id,
                        f"Stage 2: enriching {len(previous_all_qa_pairs)} previous answers "
                        f"with {len(new_chunks_list)} new chunks",
                    )
                    all_qa_pairs = await enrich_qa_pairs_with_new_evidence(
                        previous_qa_pairs=previous_all_qa_pairs,
                        question_trees=question_trees,
                        new_chunks=new_chunks_list,
                        company=company,
                        vc_context="",
                    )
                elif previous_all_qa_pairs and not new_chunks_list:
                    _append_progress(
                        job_id,
                        f"Stage 2: no new evidence since previous analysis — reusing "
                        f"{len(previous_all_qa_pairs)} previous answers verbatim",
                    )
                    all_qa_pairs = [dict(p) for p in previous_all_qa_pairs]
                    write_qa_pairs_into_trees(question_trees, all_qa_pairs)
                else:
                    _append_progress(
                        job_id,
                        f"Stage 2: no previous answers cached — answering "
                        f"{len(question_trees)} trees against {len(store.chunks)} chunks",
                    )
                    all_qa_pairs = await answer_all_trees_from_evidence(
                        question_trees, company, store, use_web_search=use_web_search,
                    )

                _append_progress(job_id, "Stages 3-6: argument generation + critique + evaluation + refinement")
                final_state = await graph.ainvoke(
                    {
                        "company": company,
                        "config": config,
                        "all_qa_pairs": all_qa_pairs,
                        "vc_context": "",
                        "slug": company_id,
                        "prompt_overrides": {},
                    },
                    config={"recursion_limit": 100},
                )
            else:
                _append_progress(job_id, "Stage 1: decomposing investment questions")
                temp_state = IterativeInvestmentStoryState(
                    company=company,
                    config=config,
                    prompt_overrides={},
                )
                decomp_result = await _decompose_questions_safe(temp_state)
                if not decomp_result:
                    _append_progress(job_id, "ERROR: decomposition failed")
                    _set_job_status(job_id, "error")
                    return

                retrieved_trees = decomp_result.get("question_trees", {})
                _append_progress(
                    job_id,
                    f"Stage 2: answering {len(retrieved_trees)} trees against {len(store.chunks)} chunks",
                )
                all_qa_pairs = await answer_all_trees_from_evidence(
                    retrieved_trees, company, store, use_web_search=use_web_search,
                )
                _append_progress(job_id, "Stages 3-6: argument generation + critique + evaluation + refinement")
                final_state = await graph.ainvoke(
                    {
                        "company": company,
                        "config": config,
                        "all_qa_pairs": all_qa_pairs,
                        "vc_context": "",
                        "slug": company_id,
                        "prompt_overrides": {},
                    },
                    config={"recursion_limit": 100},
                )

            _append_progress(job_id, "Stages 7-8: decision + ranking complete")

        # 5. Build results_payload from the final state (including evidence provenance).
        ranking = final_state.get("ranking_result")
        ranking_dict = ranking.model_dump() if hasattr(ranking, "model_dump") else (ranking or {})
        qa_provenance_rows, argument_rows = _build_evidence_provenance(
            final_state, all_qa_pairs=all_qa_pairs
        )

        # Drain + persist telemetry BEFORE saving the analysis so that
        # `results_payload.run_costs` is populated and `model_executions`
        # rows exist for the join.
        _append_progress(job_id, "Persisting telemetry and costs")
        try:
            execution_rows = collector.drain_model_executions()
        except Exception as exc:
            _logger.error("Re-evaluation: drain_model_executions failed: %s", exc, exc_info=True)
            execution_rows = []

        if execution_rows:
            try:
                await asyncio.to_thread(
                    db.persist_model_executions,
                    job_id,
                    execution_rows,
                    run_config={
                        "source": "portal_re_evaluate",
                        "company_id": company_id,
                        "phase_models": phase_models,
                    },
                )
            except Exception as exc:
                _logger.error(
                    "Re-evaluation: persist_model_executions failed: %s", exc, exc_info=True
                )

        try:
            run_costs = build_run_costs_from_model_executions(execution_rows)
        except Exception as exc:
            _logger.error(
                "Re-evaluation: build_run_costs_from_model_executions failed: %s", exc
            )
            run_costs = None

        results_payload = {
            "mode": "single",
            "ranking_result": ranking_dict,
            "qa_provenance_rows": qa_provenance_rows,
            "argument_rows": argument_rows,
            "decision": final_state.get("final_decision"),
            "run_costs": run_costs,
        }

        # 6. Build state snapshot for storage (includes question_trees for future reuse).
        state_snapshot = {
            "question_trees": {
                k: v.model_dump() if hasattr(v, "model_dump") else v
                for k, v in (final_state.get("question_trees") or {}).items()
            },
            "all_qa_pairs": final_state.get("all_qa_pairs") or [],
            "final_arguments": [
                a.model_dump() if hasattr(a, "model_dump") else a
                for a in (final_state.get("final_arguments") or [])
            ],
            "final_decision": final_state.get("final_decision"),
        }

        run_config = {
            "source": "portal_re_evaluate",
            "use_web_search": use_web_search,
            "phase_models": phase_models,
            "started_by_user_id": started_by_user_id,
            "started_by_email": started_by_email,
            "started_by_display_name": started_by_display_name,
            "started_by_label": started_by_label,
        }

        _append_progress(job_id, "Saving analysis record")
        analysis_id = await asyncio.to_thread(
            db.create_analysis_record,
            company_id=company_id,
            pitch_deck_id=None,  # multi-deck scenario; no single pitch_deck
            state=state_snapshot,
            results_payload=results_payload,
            status="done",
            run_config=run_config,
            job_id_legacy=job_id,
        )
        _logger.info(
            "Re-evaluation: analysis saved, id=%s job_id=%s for company=%s",
            analysis_id, job_id, company_id,
        )

        # 8. Re-score existing VC matches in place (force_refresh upserts so
        #    match ids stay stable → debate FK links survive). We DO NOT call
        #    delete_matches_for_company here any more — doing so would cascade
        #    through debates.match_id and destroy every debate for the company.
        company_fresh = await asyncio.to_thread(db.get_company_by_id, company_id)
        if company_fresh and company_fresh.get("fundraising"):
            _append_progress(job_id, "Re-running VC matching (force_refresh)")
            await _run_matching_background(company_id, force_refresh=True)
        else:
            _logger.info(
                "Re-evaluation: fundraising=false for company=%s — matching not triggered", company_id
            )

        # 8a. Post a system_note into every debate that was paused awaiting
        #     founder input at the start of this run, so founders can see the
        #     new analysis is ready and return to the debate.
        if paused_debate_ids:
            note_content = (
                f"New analysis {analysis_id} saved (re-evaluation job {job_id}). "
                f"Founder may return to the debate to confirm."
            )
            for debate_id in paused_debate_ids:
                try:
                    saved = await asyncio.to_thread(
                        db.save_debate_message,
                        debate_id=debate_id,
                        round=0,
                        speaker="system",
                        content=note_content,
                        citations=[],
                        message_type="system_note",
                        linked_reeval_job_id=job_id,
                    )
                    if saved:
                        await _broadcast_debate_message(debate_id, saved)
                except Exception as exc:
                    _logger.warning(
                        "Re-evaluation: failed to post success system_note to debate=%s: %s",
                        debate_id, exc,
                    )

        _append_progress(job_id, "Re-evaluation complete")
        _set_job_status(job_id, "done")

    except Exception as exc:
        final_state_error = exc
        import logging as _lm  # noqa: PLC0415
        _lm.getLogger(__name__).error(
            "Re-evaluation failed for company=%s job=%s: %s",
            company_id, job_id, exc, exc_info=True,
        )
        try:
            _append_progress(job_id, f"ERROR: {exc}", allow_stopped=True)
        except Exception:
            pass
        _set_job_status(job_id, "error")

        # Surface the failure into any paused debates so the founder sees a
        # retry/decline prompt inside the debate panel instead of the run
        # hanging silently.
        failure_debate_ids = _reeval_debate_links.get(job_id, [])
        if failure_debate_ids:
            failure_note = (
                f"Re-evaluation {job_id} failed: {exc}. "
                f"Please try uploading again, or decline to provide the evidence."
            )
            for debate_id in failure_debate_ids:
                try:
                    saved = await asyncio.to_thread(
                        db.save_debate_message,
                        debate_id=debate_id,
                        round=0,
                        speaker="system",
                        content=failure_note,
                        citations=[],
                        message_type="system_note",
                        linked_reeval_job_id=job_id,
                    )
                    if saved:
                        await _broadcast_debate_message(debate_id, saved)
                except Exception as note_exc:
                    _lm.getLogger(__name__).warning(
                        "Re-evaluation: failed to post failure system_note to debate=%s: %s",
                        debate_id, note_exc,
                    )
    finally:
        # Drop the job → debates mapping so we don't leak memory over time.
        _reeval_debate_links.pop(job_id, None)


async def _run_full_analysis(company_id: str) -> None:
    """Background task: run a full 8-stage pipeline analysis for a new company.

    Used when a startup has uploaded documents but has no prior analysis record.
    All chunks are combined into one EvidenceStore. Stages 1-8 run in sequence.
    Does NOT auto-trigger VC matching — the startup must toggle fundraising ON.
    """
    import logging as _log_module  # noqa: PLC0415
    _logger = _log_module.getLogger(__name__)
    _logger.info("Full analysis started for company=%s", company_id)

    try:
        from agent.dataclasses.company import Company  # noqa: PLC0415
        from agent.dataclasses.config import Config  # noqa: PLC0415
        from agent.evidence_answering import answer_all_trees_from_evidence  # noqa: PLC0415
        from agent.ingest.store import Chunk, EvidenceStore  # noqa: PLC0415
        from agent.pipeline.graph import graph  # noqa: PLC0415
        from agent.pipeline.state.investment_story import IterativeInvestmentStoryState  # noqa: PLC0415

        # 1. Load company row.
        company_row = await asyncio.to_thread(db.get_company_by_id, company_id)
        if not company_row:
            _logger.error("Full analysis: company %s not found", company_id)
            return

        company = Company(
            name=company_row.get("name") or "Unknown",
            industry=company_row.get("industry"),
            tagline=company_row.get("tagline"),
            about=company_row.get("about"),
            domain=company_row.get("domain"),
        )

        # 2. Load ALL chunks (no Specter filter).
        all_chunk_rows = await asyncio.to_thread(db.get_all_company_chunks, company_id)
        if not all_chunk_rows:
            _logger.warning("Full analysis: no chunks found for company=%s — aborting", company_id)
            return

        store = EvidenceStore(startup_slug=company_id)
        for idx, c in enumerate(all_chunk_rows):
            store.chunks.append(Chunk(
                chunk_id=c.get("chunk_id") or f"chunk_{idx}",
                text=c.get("text") or "",
                source_file=c.get("source_file") or "",
                page_or_slide=c.get("page_or_slide") or "N/A",
            ))
        _logger.info("Full analysis: %d chunks for company=%s", len(store.chunks), company_id)

        config = Config(
            n_pro_arguments=3,
            n_contra_arguments=3,
            k_best_arguments_per_iteration=[3, 1],
            max_iterations=1,
        )

        # 3. Stage 1: Decompose questions.
        _logger.info("Full analysis: running Stage 1 (decomposition) for company=%s", company_id)
        temp_state = IterativeInvestmentStoryState(
            company=company,
            config=config,
            prompt_overrides={},
        )
        decomp_result = await _decompose_questions_safe(temp_state)
        if not decomp_result:
            _logger.error("Full analysis: decomposition failed for company=%s", company_id)
            return

        question_trees = decomp_result.get("question_trees", {})
        _logger.info("Full analysis: %d question trees for company=%s", len(question_trees), company_id)

        # 4. Stage 2: Answer from evidence.
        _logger.info("Full analysis: running Stage 2 (answering) for company=%s", company_id)
        all_qa_pairs = await answer_all_trees_from_evidence(question_trees, company, store)
        _logger.info("Full analysis: %d Q&A pairs for company=%s", len(all_qa_pairs), company_id)

        # 5. Stages 3-8: Argument generation through ranking.
        _logger.info("Full analysis: running Stages 3-8 for company=%s", company_id)
        final_state = await graph.ainvoke(
            {
                "company": company,
                "config": config,
                "all_qa_pairs": all_qa_pairs,
                "vc_context": "",
                "slug": company_id,
                "prompt_overrides": {},
            },
            config={"recursion_limit": 100},
        )

        # 6. Build results_payload (including evidence provenance) and state snapshot.
        ranking = final_state.get("ranking_result")
        ranking_dict = ranking.model_dump() if hasattr(ranking, "model_dump") else (ranking or {})
        # Pass local all_qa_pairs directly to avoid LangGraph state-merge artefacts.
        qa_provenance_rows, argument_rows = _build_evidence_provenance(
            final_state, all_qa_pairs=all_qa_pairs
        )
        results_payload = {
            "mode": "single",
            "ranking_result": ranking_dict,
            "qa_provenance_rows": qa_provenance_rows,
            "argument_rows": argument_rows,
            "decision": final_state.get("final_decision"),
        }

        state_snapshot = {
            "question_trees": {
                k: v.model_dump() if hasattr(v, "model_dump") else v
                for k, v in (final_state.get("question_trees") or {}).items()
            },
            "all_qa_pairs": all_qa_pairs,
            "final_arguments": [
                a.model_dump() if hasattr(a, "model_dump") else a
                for a in (final_state.get("final_arguments") or [])
            ],
            "final_decision": final_state.get("final_decision"),
        }

        # 7. Persist analysis record.
        analysis_id = await asyncio.to_thread(
            db.create_analysis_record,
            company_id=company_id,
            pitch_deck_id=None,
            state=state_snapshot,
            results_payload=results_payload,
            status="done",
            run_config={"source": "portal_full_analyze"},
        )
        _logger.info(
            "Full analysis: complete, analysis_id=%s for company=%s", analysis_id, company_id
        )
        # NOTE: Do NOT trigger matching here. Startup must explicitly toggle fundraising ON.

    except Exception as exc:
        import logging as _lm  # noqa: PLC0415
        _lm.getLogger(__name__).error(
            "Full analysis failed for company=%s: %s", company_id, exc, exc_info=True
        )


async def _decompose_questions_safe(state: Any) -> dict[str, Any] | None:
    """Run Stage 1 (decompose_all_questions) and return the updated state dict.

    Returns None if decomposition fails. Isolated here to avoid code duplication
    between _run_re_evaluation and _run_full_analysis.
    """
    try:
        from agent.pipeline.stages.parallel_decomposition import decompose_all_questions  # noqa: PLC0415
        result = await decompose_all_questions(state)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        import logging as _lm  # noqa: PLC0415
        _lm.getLogger(__name__).error("Decomposition failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------

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
        pd = _lazy_import_pandas()
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


_URL_LIKE_RE = re.compile(r"^(?:https?://)?[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}(?:/.*)?$")


def _normalize_url_for_intake(raw: str) -> str | None:
    s = (raw or "").strip()
    if not s:
        return None
    # Strip leading "@" (from copy-pasted social handles) and surrounding quotes.
    s = s.lstrip("@").strip("'\"")
    if not _URL_LIKE_RE.match(s):
        return None
    return s


@app.post("/api/upload-urls")
async def upload_urls(
    body: dict[str, Any] = Body(...),
    session_id: str | None = Cookie(default=None),
):
    """Create a job for URL-based Specter intake (no files).

    Body shape::

        {"urls": ["acme.com", "https://example.io", ...]}

    Each URL becomes a Specter MCP lookup at analyze time. Reviews / Glassdoor /
    awards / multi-period growth metrics are NOT included via MCP — upload the
    matching CSVs through /api/upload if you need those signals.
    """
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    raw_urls = body.get("urls")
    if not isinstance(raw_urls, list):
        raise HTTPException(status_code=400, detail="`urls` must be a list of strings")

    cleaned: list[str] = []
    invalid: list[str] = []
    for raw in raw_urls:
        if not isinstance(raw, str):
            invalid.append(repr(raw))
            continue
        normalized = _normalize_url_for_intake(raw)
        if normalized is None:
            invalid.append(raw)
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=f"No valid URLs provided. Invalid entries: {invalid[:5]!r}",
        )

    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(tempfile.mkdtemp()) / job_id
    upload_dir.mkdir(parents=True)

    _jobs[job_id] = AnalysisStatus(
        job_id=job_id, status="pending", progress="URLs received"
    )
    _results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [],
        "specter": {},
        "specter_urls": cleaned,
    }
    if db and db.is_configured():
        db.insert_analysis_event(
            job_id,
            message=f"Received {len(cleaned)} URLs for Specter MCP intake",
            event_type="upload",
            payload={"num_urls": len(cleaned)},
        )
        db.insert_job_status_history(
            job_id,
            status="pending",
            progress="URLs received",
            source="upload",
        )

    return {
        "job_id": job_id,
        "urls": cleaned,
        "invalid": invalid,
        "mode": "specter",
    }


@app.post("/api/analyze/{job_id}")
async def start_analysis(
    job_id: str,
    req: AnalyzeRequest = AnalyzeRequest(),
    session_id: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    _cancel_scheduled_restart()
    started_by = await _require_supabase_identity(authorization)

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
    elif not (req.llm_provider or req.llm_model):
        try:
            pipeline_policy = build_default_phase_model_policy()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        phase_models = phase_model_defaults_payload()
        llm_selection = dict(pipeline_policy.answering)
        effective_phase_models = resolve_effective_phase_models(pipeline_policy)
        llm_display = build_phase_policy_display_label(effective_phase_models)
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
    cache = _results_cache[job_id]
    _jobs[job_id].started_by_user_id = started_by.get("started_by_user_id")
    _jobs[job_id].started_by_email = started_by.get("started_by_email")
    _jobs[job_id].started_by_display_name = started_by.get("started_by_display_name")
    _jobs[job_id].started_by_label = started_by.get("started_by_label")
    cache["started_by_user_id"] = started_by.get("started_by_user_id")
    cache["started_by_email"] = started_by.get("started_by_email")
    cache["started_by_display_name"] = started_by.get("started_by_display_name")
    cache["started_by_label"] = started_by.get("started_by_label")
    cache["input_mode"] = req.input_mode
    cache["run_name"] = req.run_name
    cache["vc_investment_strategy"] = req.vc_investment_strategy
    cache["use_web_search"] = req.use_web_search
    cache["use_specter_mcp"] = req.use_specter_mcp
    cache["instructions"] = req.instructions
    cache["llm_selection"] = llm_selection
    cache["phase_models"] = phase_models if (req.phase_models or pipeline_policy is not None and quality_tier is None) else None
    cache["quality_tier"] = quality_tier
    cache["premium_phase_models"] = (
        premium_phase_models if quality_tier == "premium" else None
    )
    cache["effective_phase_models"] = effective_phase_models
    cache["run_config"] = {
        "input_mode": req.input_mode,
        "run_name": req.run_name,
        "vc_investment_strategy": req.vc_investment_strategy,
        "instructions": req.instructions,
        "use_web_search": req.use_web_search,
        "use_specter_mcp": req.use_specter_mcp,
        "phase_models": phase_models if (req.phase_models or pipeline_policy is not None and quality_tier is None) else None,
        "quality_tier": quality_tier,
        "premium_phase_models": premium_phase_models if quality_tier == "premium" else None,
        "effective_phase_models": effective_phase_models,
        "llm_provider": llm_selection["provider"],
        "llm_model": llm_selection["model"],
        "llm": llm_display,
        **started_by,
    }
    cache["model_executions"] = []
    cache["run_costs_aggregate"] = _empty_run_costs_summary()
    cache["versions"] = _runtime_versions()

    if db and db.is_configured():
        run_config = dict(cache["run_config"])
        db.upsert_job(job_id, run_config=run_config, versions=_runtime_versions())
        db.upsert_job_control(
            job_id,
            pause_requested=False,
            stop_requested=False,
            last_action="start",
        )

    vc_str = _ensure_str(req.vc_investment_strategy).strip() or None
    inst = _ensure_str(req.instructions).strip() or None
    if req.input_mode == "specter" and ENABLE_SPECTER_WORKER_SERVICE:
        queued, worker_message = _queue_worker_backed_specter_job(job_id)
        if queued:
            _set_job_status(job_id, "running", worker_message or "Queued for worker...", source="start_analysis")
            _append_progress_and_log(job_id, worker_message or "Queued for worker...")
            return {"status": "running", "use_web_search": req.use_web_search, "llm": llm_display}
        queue_error = worker_message or "Unknown worker queue error."
        _set_job_status(job_id, "error", f"Worker queue failed. {queue_error}", source="start_analysis")
        _append_progress_and_log(job_id, f"Worker queue failed — {queue_error}")
        raise HTTPException(status_code=503, detail=f"Specter worker queue failed: {queue_error}")

    # Run the analysis loop in a dedicated thread/event-loop to keep
    # the main FastAPI loop responsive for pause/resume/stop controls.
    threading.Thread(
        target=lambda: asyncio.run(
            _run_analysis(
                job_id,
                use_web_search=req.use_web_search,
                use_specter_mcp=req.use_specter_mcp,
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
    persisted_run_costs = merged.get("run_costs") if isinstance(merged.get("run_costs"), dict) else None
    runtime_run_costs = _run_costs_from_cache(job_id)
    job = _jobs.get(job_id)
    if (
        persisted_run_costs
        and job is not None
        and (job.status in {"done", "error", "stopped"} or job.persistence_complete)
    ):
        merged["run_costs"] = persisted_run_costs
    elif runtime_run_costs.get("status") != "unavailable" or "run_costs" not in merged:
        merged["run_costs"] = runtime_run_costs
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
            dimension_scores_payload = []
            for d in ranking.dimension_scores:
                display_score = d.raw_score if d.dimension == "upside" else d.adjusted_score
                dimension_scores_payload.append(
                    {
                        "dimension": d.dimension,
                        "adjusted_score": display_score,
                        "confidence": d.confidence,
                        "evidence_snippets": d.evidence_snippets,
                        "critical_gaps": d.critical_gaps,
                    }
                )
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
                "dimension_scores": dimension_scores_payload,
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
    payload.update(_started_by_from_payload(_run_config_from_cache(job_id)))
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
        for key in ("started_by_user_id", "started_by_email", "started_by_display_name", "started_by_label"):
            if cache.get(key) and not run_config.get(key):
                run_config[key] = cache.get(key)
        return run_config

    llm_selection = _resolve_job_llm_selection(job_id, results=cache.get("results"))
    return {
        "input_mode": cache.get("input_mode", "pitchdeck"),
        "run_name": cache.get("run_name"),
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
        "started_by_user_id": cache.get("started_by_user_id"),
        "started_by_email": cache.get("started_by_email"),
        "started_by_display_name": cache.get("started_by_display_name"),
        "started_by_label": cache.get("started_by_label"),
    }


def _is_worker_backed_job(job_id: str) -> bool:
    return bool((_results_cache.get(job_id) or {}).get("worker_backed"))


def _normalize_worker_status(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"queued", "claimed", "running", "finalizing"}:
        return "running"
    if normalized == "interrupted":
        return "stopped"
    return normalized or "pending"


def _is_terminal_job_status(status: str | None) -> bool:
    return _normalize_worker_status(status) in {"done", "error", "stopped"}


def _load_worker_progress_log(job_id: str) -> list[str]:
    if not (db and db.is_configured()):
        return []
    load_events = getattr(db, "load_analysis_events", None)
    if not callable(load_events):
        return []
    try:
        return load_events(job_id, limit=MAX_PROGRESS_LOG_ENTRIES) or []
    except Exception:
        return []


def _prepare_worker_source_files(job_id: str) -> list[dict[str, Any]] | None:
    if not (db and db.is_configured()):
        return None
    cache = _results_cache.get(job_id, {})
    files = list(cache.get("files") or [])
    if not files:
        return None

    prepared: list[dict[str, Any]] = []
    for file_info in files:
        current = dict(file_info)
        storage_path = current.get("storage_path")
        if not storage_path:
            local_path = current.get("local_path")
            if not local_path:
                return None
            storage_path = db.upload_source_file(
                job_id,
                local_path,
                file_name=current.get("name"),
                mime_type=current.get("mime_type"),
            )
            if not storage_path:
                return None
            current["storage_path"] = storage_path
        prepared.append(current)

    cache["files"] = prepared
    specter = cache.get("specter") or {}
    for file_info in prepared:
        name = file_info.get("name")
        local_path = file_info.get("local_path")
        if specter.get("companies") == local_path:
            specter["companies_storage_path"] = file_info.get("storage_path")
        if specter.get("people") == local_path:
            specter["people_storage_path"] = file_info.get("storage_path")
    cache["specter"] = specter
    return prepared


def _queue_worker_backed_specter_job(job_id: str) -> tuple[bool, str | None]:
    def _failure(message: str) -> tuple[bool, str]:
        print(f"Specter worker queue failed for job {job_id}: {message}")
        return (False, message)

    if not ENABLE_SPECTER_WORKER_SERVICE:
        return (False, None)
    if not (db and db.is_configured()):
        return _failure("Database is not configured for worker-backed Specter runs.")

    cache = _results_cache.get(job_id, {})
    specter = cache.get("specter") or {}
    specter_urls: list[str] = list(cache.get("specter_urls") or [])
    companies_csv = specter.get("companies")
    companies_storage_path = specter.get("companies_storage_path")

    if not companies_csv and not specter_urls:
        return _failure(
            "Neither Specter CSV exports nor URL list provided for this job."
        )

    source_files: list[dict[str, Any]] = []
    if companies_csv:
        prepared = _prepare_worker_source_files(job_id)
        if not prepared:
            return _failure("Could not prepare shared Specter source files.")
        source_files = prepared

        companies_storage_path = companies_storage_path or (
            next(
                (
                    item.get("storage_path")
                    for item in source_files
                    if item.get("local_path") == companies_csv
                ),
                None,
            )
        )
        if not companies_storage_path:
            return _failure(
                "Could not upload Specter company export to shared storage."
            )
        specter["companies_storage_path"] = companies_storage_path

        if specter.get("people"):
            people_storage_path = specter.get("people_storage_path") or next(
                (
                    item.get("storage_path")
                    for item in source_files
                    if item.get("local_path") == specter.get("people")
                ),
                None,
            )
            if not people_storage_path:
                return _failure(
                    "Could not upload Specter people export to shared storage."
                )
            specter["people_storage_path"] = people_storage_path

    csv_descriptors = list_specter_companies(companies_csv) if companies_csv else []
    csv_domains = {
        (d.get("domain") or "").strip().lower()
        for d in csv_descriptors
        if d.get("domain")
    }
    deduped_urls = [u for u in specter_urls if u.lower() not in csv_domains]

    max_startups = _parse_max_startups_from_instructions(cache.get("instructions"))
    if max_startups is not None:
        # Apply the cap across the union of CSV + URL companies, prefer CSV.
        remaining = max(0, max_startups - len(csv_descriptors))
        deduped_urls = deduped_urls[:remaining]
        csv_descriptors = csv_descriptors[:max_startups]
    total_companies = len(csv_descriptors) + len(deduped_urls)
    if total_companies <= 0:
        return _failure("No companies found in Specter data.")

    cache["specter"] = specter
    run_config = dict(_run_config_from_cache(job_id))
    if companies_csv:
        run_config["specter_worker_files"] = {
            "companies_storage_path": specter.get("companies_storage_path"),
            "people_storage_path": specter.get("people_storage_path"),
            "companies_name": Path(companies_csv).name if companies_csv else None,
            "people_name": Path(str(specter.get("people") or "")).name
            if specter.get("people") else None,
        }
    if deduped_urls:
        run_config["specter_urls"] = deduped_urls
    cache["run_config"] = run_config
    progress = f"Queued for worker — 0/{total_companies} companies completed."

    if not db.persist_source_files(
        job_id,
        source_files,
        run_config=run_config,
        versions=cache.get("versions") or _runtime_versions(),
        worker_state={
            "status": "queued",
            "progress": progress,
            "worker_service_enabled": True,
            "total_companies": total_companies,
            "completed_companies": 0,
            "failed_companies": 0,
        },
    ):
        return _failure("Could not persist Specter source file metadata.")

    if not db.queue_specter_worker_job(
        job_id,
        run_config=run_config,
        versions=cache.get("versions") or _runtime_versions(),
        total_companies=total_companies,
        progress=progress,
    ):
        return _failure("Could not queue Specter worker job.")

    cache["worker_backed"] = True
    cache["results"] = None
    return (True, progress)


def _chunk_worker_config_path(upload_dir: Path, job_id: str, chunk_idx: int) -> Path:
    return upload_dir / f".specter-chunk-{job_id}-{chunk_idx}.json"


def _write_chunk_worker_config(upload_dir: Path, job_id: str, chunk_idx: int) -> Path:
    path = _chunk_worker_config_path(upload_dir, job_id, chunk_idx)
    payload = {
        "run_config": _run_config_from_cache(job_id),
        "versions": _results_cache.get(job_id, {}).get("versions") or _runtime_versions(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _handle_specter_chunk_worker_event(
    job_id: str,
    *,
    chunk_idx: int,
    total_chunks: int,
    total: int,
    event: dict[str, Any],
) -> None:
    event_type = str(event.get("type") or "").strip().lower()
    company_name = str(event.get("company_name") or "").strip()
    absolute_index = int(event.get("absolute_index") or 0)
    prefix = (
        f"Chunk {chunk_idx}/{total_chunks} — Evaluating {company_name} ({absolute_index}/{total})"
        if company_name and absolute_index > 0
        else f"Chunk {chunk_idx}/{total_chunks}"
    )

    if event_type == "progress":
        message = str(event.get("message") or "").strip()
        if message:
            _append_progress(job_id, f"{prefix} — {message}")
        return

    if event_type == "company_complete":
        refreshed = _refresh_persisted_batch_results(
            job_id,
            progress_message=f"{prefix} — Persisting partial result...",
        )
        completed_count = (
            _completed_count_from_results_payload(_results_cache.get(job_id, {}).get("results"))
            if refreshed
            else max(absolute_index, 0)
        )
        _append_progress(
            job_id,
            f"Partial results updated — {completed_count}/{total} companies completed.",
        )
        error_message = str(event.get("error") or "").strip()
        status = str(event.get("status") or "").strip().lower()
        if error_message and status in {"error", "timeout"}:
            _append_progress(job_id, f"{prefix} — {status}: {error_message}")
        return

    if event_type == "chunk_complete":
        print(
            f"Chunk worker finished chunk {chunk_idx}/{total_chunks} "
            f"for job {job_id}.",
        )


async def _run_specter_company_subprocess(
    job_id: str,
    *,
    upload_dir: Path,
    specter: dict[str, Any],
    company_index: int,
    absolute_index: int,
    chunk_idx: int,
    total_chunks: int,
    total: int,
    use_web_search: bool,
    vc_investment_strategy: str | None,
) -> None:
    config_path = _write_chunk_worker_config(upload_dir, job_id, absolute_index)
    cmd = [
        sys.executable,
        "-m",
        "agent.specter_company_worker",
        "--job-id",
        job_id,
        "--specter-companies",
        str(specter["companies"]),
        "--company-index",
        str(company_index),
        "--absolute-index",
        str(absolute_index),
        "--config-path",
        str(config_path),
    ]
    if specter.get("people"):
        cmd.extend(["--specter-people", str(specter["people"])])
    if use_web_search:
        cmd.append("--use-web-search")
    if vc_investment_strategy:
        cmd.extend(["--vc-investment-strategy", vc_investment_strategy])

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        assert process.stdout is not None
        while True:
            if _is_stop_requested(job_id):
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=10)
                raise _JobStoppedError("Job stopped by user")

            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=0.5)
            except asyncio.TimeoutError:
                await _wait_if_paused(job_id)
                continue

            if not line:
                break

            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue

            if text.startswith(SPECTER_COMPANY_EVENT_PREFIX):
                try:
                    event = json.loads(text[len(SPECTER_COMPANY_EVENT_PREFIX):])
                except Exception:
                    print(text)
                else:
                    _handle_specter_chunk_worker_event(
                        job_id,
                        chunk_idx=chunk_idx,
                        total_chunks=total_chunks,
                        total=total,
                        event=event,
                    )
            else:
                print(text)

        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Specter company worker exited with code {return_code} "
                f"for chunk {chunk_idx}/{total_chunks}, company {absolute_index}/{total}."
            )
    finally:
        with contextlib.suppress(Exception):
            config_path.unlink()


def _failure_result_payload(
    job_id: str,
    *,
    company: Any,
    store: Any,
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
    payload = _compact_results_for_runtime(_results_cache[job_id].get("results")) or {}
    _results_cache[job_id]["results"] = payload
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
    company: Any,
    store: Any,
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
    use_specter_mcp: bool = True,
    instructions: str | None = None,
    input_mode: str = "pitchdeck",
    vc_investment_strategy: str | None = None,
    llm_selection: dict[str, Any] | None = None,
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
                    use_specter_mcp=use_specter_mcp,
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
    finally:
        job = _jobs.get(job_id)
        if job and job.status in {"done", "error", "stopped"}:
            _release_job_runtime_resources(
                job_id,
                drop_results=bool(db and db.is_configured()),
            )
            if job.status in {"done", "stopped"}:
                job.restart_pending = True


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
    primary: Any,
    secondary: Any,
    startup_slug: str,
) -> Any:
    """Combine two evidence stores and normalize chunk IDs."""
    Chunk, EvidenceStore = _lazy_import_ingest_store()
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
) -> tuple[Any | None, Any | None, int | None]:
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


_GENERIC_DECK_STEMS: frozenset[str] = frozenset(
    {
        # Deck-type words
        "deck", "pitch", "pitchdeck", "pitchdeckextended", "pitch_deck",
        "presentation", "slides", "slidedeck", "slide_deck",
        # Lifecycle / version words
        "final", "draft", "latest", "current", "updated",
        "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        # Audience / purpose words
        "company", "investor", "investors", "memo", "intro", "introduction",
        "overview", "summary", "executive", "exec",
        # Variant / scope qualifiers
        "extended", "extension", "long", "short", "full", "brief",
        "public", "private", "redacted", "anonymized", "sanitized",
        "teaser", "teaserdeck", "teaser_deck",
        "confidential", "nda",
        # Date-y words
        "year", "annual", "quarter", "q1", "q2", "q3", "q4",
    }
)


def _tentative_name_from_filename(fname: str) -> str | None:
    """Derive a candidate company name from an uploaded deck filename.

    Heuristic: take the file stem, drop generic deck-ish tokens, and return the
    FIRST remaining alphabetic word (>=3 chars). Brand names lead deck
    filenames in the overwhelming majority of cases (e.g. ``Zaitra PitchDeck
    Extended.pdf`` → "Zaitra"), so first-position beats longest-token. Returns
    None when nothing plausible remains — the MCP client then falls back to
    the domain-root check alone, which is still a strong disambiguation
    safeguard.
    """
    try:
        stem = Path(fname).stem
    except Exception:
        return None
    if not stem:
        return None
    # Split on common separators; keep alphabetic word-like tokens.
    tokens = re.split(r"[\s_\-.()\[\]]+", stem)
    for t in tokens:
        if (
            t
            and t.lower() not in _GENERIC_DECK_STEMS
            and t.isalpha()
            and len(t) >= 3
        ):
            return t
    return None


async def _run_document_analysis(
    job_id: str,
    upload_dir: Path,
    use_web_search: bool,
    one_company: bool = False,
    vc_investment_strategy: str | None = None,
    use_specter_mcp: bool = False,
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

        # Pitch-deck single-file path: augment with Specter MCP via the deck's URL.
        # Skipped when one_company=True (Original mode is out of scope) and when a
        # CSV Specter overlay already populated seed_store/seed_company above.
        if (
            use_specter_mcp
            and not one_company
            and seed_store is None
            and seed_company is None
        ):
            try:
                from agent.ingest import ingest_startup_folder
                from agent.ingest.specter_augmentation import augment_with_specter
                tentative_name = None
                if files:
                    tentative_name = _tentative_name_from_filename(files[0].get("name") or "")
                deck_store = ingest_startup_folder(upload_dir)
                seed_store, seed_company = augment_with_specter(
                    deck_store,
                    slug=upload_dir.name,
                    expected_name=tentative_name,
                    fetch_full_team=False,
                    on_log=lambda m: (_append_progress(job_id, m), print(m)),
                )
            except Exception as exc:  # noqa: BLE001 — augmentation is best-effort
                print(f"specter-augment: pre-ingest failed for {upload_dir.name}: {exc}")
                seed_store = None
                seed_company = None

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
            _mark_terminal_persistence_complete(job_id)
            return
        _append_progress(job_id, "Finalizing complete.")
        _set_job_status(job_id, "done", "Analysis complete", source="run_document_analysis")
        _jobs[job_id].results = _results_cache[job_id]["results"]
        _persist_jobs()
        _persist_results_to_db(job_id, [result])
        _mark_terminal_persistence_complete(job_id)
        return

    results_list: list[dict] = []
    total = file_count
    chunking = _batch_chunking_config(job_id, total_items=total, mode="documents")
    _results_cache[job_id]["batch_chunking"] = chunking
    file_chunks = _chunk_items(files, chunking["chunk_size"]) if chunking["enabled"] else [files]
    if chunking["enabled"]:
        _append_progress_and_log(
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
                _append_progress_and_log(
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

                # Optional: pre-ingest the deck and augment with Specter MCP
                # data extracted via the deck's company URL. Helper never
                # raises — falls back to deck-only on any failure.
                seed_store = None
                seed_company = None
                if use_specter_mcp:
                    try:
                        from agent.ingest import ingest_startup_folder
                        from agent.ingest.specter_augmentation import (
                            augment_with_specter,
                        )
                        deck_store = ingest_startup_folder(doc_dir)
                        seed_store, seed_company = augment_with_specter(
                            deck_store,
                            slug=doc_dir.name,
                            expected_name=_tentative_name_from_filename(fname),
                            fetch_full_team=False,
                            on_log=lambda m: (_append_progress(job_id, m), print(m)),
                        )
                    except Exception as exc:  # noqa: BLE001 — augmentation is best-effort
                        print(f"specter-augment: pre-ingest failed for {fname}: {exc}")
                        seed_store = None
                        seed_company = None

                try:
                    result = await evaluate_startup(
                        doc_dir, k=8, use_web_search=use_web_search,
                        on_progress=make_progress,
                        on_cooperate=lambda: _cooperate_with_job_control(job_id),
                        vc_investment_strategy=vc_investment_strategy,
                        initial_store=seed_store,
                        initial_company=seed_company,
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
                            company=_lazy_import_company()(name=fname),
                            store=_lazy_import_ingest_store()[1](startup_slug=_sanitize_slug(fname), chunks=[]),
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
                _append_progress_and_log(
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
    _mark_terminal_persistence_complete(job_id)


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

    company_descriptors = list_specter_companies(specter["companies"])
    parsed_total = len(company_descriptors)
    print(f"Specter ingest: parsed {parsed_total} companies.")

    max_startups = _parse_max_startups_from_instructions(instructions)
    if max_startups is not None:
        print(
            f"Applying explicit instruction limit: "
            f"first {max_startups} company(ies) out of {parsed_total}.",
        )
        company_descriptors = company_descriptors[:max_startups]

    if not company_descriptors:
        _set_job_status(job_id, "error", "No companies found in Specter data.", source="run_specter_analysis")
        return

    total = len(company_descriptors)
    results_list: list[dict] = []
    chunking = _batch_chunking_config(job_id, total_items=total, mode="specter")
    chunked_db_mode = bool(
        ENABLE_CHUNKED_SPECTER_PERSISTENCE
        and chunking["enabled"]
        and db
        and db.is_configured()
    )
    subprocess_company_mode = bool(
        chunked_db_mode
        and ENABLE_SPECTER_SUBPROCESS_CHUNKS
    )
    _results_cache[job_id]["batch_chunking"] = chunking
    if subprocess_company_mode:
        company_chunks = (
            _chunk_items(company_descriptors, chunking["chunk_size"])
            if chunking["enabled"]
            else [company_descriptors]
        )
    else:
        company_store_pairs = ingest_specter(
            specter["companies"],
            specter.get("people"),
        )
        if max_startups is not None:
            company_store_pairs = company_store_pairs[:max_startups]
        company_chunks = (
            _chunk_items(company_store_pairs, chunking["chunk_size"])
            if chunking["enabled"]
            else [company_store_pairs]
        )
    if chunking["enabled"]:
        _append_progress_and_log(
            job_id,
            "Large batch chunking enabled "
            f"for {chunking['label']} — {chunking['total_chunks']} chunks of up to "
            f"{chunking['chunk_size']} company(ies), cooldown {chunking['cooldown_seconds']}s.",
        )

    last_error: str | None = None
    persisted_company_keys = _load_persisted_company_keys(job_id) if chunked_db_mode else set()
    if persisted_company_keys:
        _refresh_persisted_batch_results(
            job_id,
            progress_message=(
                f"Resuming batch — {len(persisted_company_keys)}/{total} companies already persisted."
            ),
        )
    try:
        processed = 0
        for chunk_idx, company_chunk in enumerate(company_chunks, 1):
            if chunking["enabled"]:
                chunk_start = processed + 1
                chunk_end = processed + len(company_chunk)
                _append_progress_and_log(
                    job_id,
                    f"Starting chunk {chunk_idx}/{chunking['total_chunks']} — companies {chunk_start}-{chunk_end} of {total}.",
                )
            if subprocess_company_mode:
                for descriptor in company_chunk:
                    processed += 1
                    company_key = _persisted_company_key(
                        descriptor.get("name"),
                        descriptor.get("slug"),
                    )
                    if company_key in persisted_company_keys:
                        continue
                    await _run_specter_company_subprocess(
                        job_id,
                        upload_dir=upload_dir,
                        specter=specter,
                        company_index=int(descriptor["index"]),
                        absolute_index=processed,
                        chunk_idx=chunk_idx,
                        total_chunks=chunking["total_chunks"],
                        total=total,
                        use_web_search=use_web_search,
                        vc_investment_strategy=vc_investment_strategy,
                    )
                    persisted_company_keys.add(company_key)
                if (
                    chunking["enabled"]
                    and chunk_idx < chunking["total_chunks"]
                    and chunking["cooldown_seconds"] > 0
                ):
                    _refresh_persisted_batch_results(
                        job_id,
                        progress_message=(
                            f"Chunk {chunk_idx}/{chunking['total_chunks']} complete — "
                            f"cooling down for {chunking['cooldown_seconds']}s before next chunk."
                        ),
                    )
                    gc.collect()
                    _append_progress_and_log(
                        job_id,
                        f"Chunk {chunk_idx}/{chunking['total_chunks']} complete — cooling down for {chunking['cooldown_seconds']}s before next chunk.",
                    )
                    await asyncio.sleep(chunking["cooldown_seconds"])
                continue
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
                    if not chunked_db_mode:
                        results_list.append(result)
                    _append_progress(job_id, f"{prefix} — Persisting partial result...")
                    persisted_to_db = _persist_company_result_to_db(job_id, result)
                    if persisted_to_db:
                        _minimize_completed_result_for_memory(result)
                    if chunked_db_mode:
                        refreshed = _refresh_persisted_batch_results(
                            job_id,
                            progress_message=f"Partial results updated — {processed}/{total} companies processed.",
                        )
                        completed_count = (
                            _completed_count_from_results_payload(_results_cache.get(job_id, {}).get("results"))
                            if refreshed
                            else processed
                        )
                    else:
                        _update_partial_results_cache(job_id, upload_dir, results_list)
                        completed_count = len([r for r in results_list if not r.get('skipped')])
                    _append_progress(
                        job_id,
                        f"Partial results updated — {completed_count}/{total} companies completed.",
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
                    failed_result = {
                        "slug": store.startup_slug,
                        "skipped": True,
                        "error": str(exc)[:500],
                        "company_name": company.name,
                    }
                    if not chunked_db_mode:
                        results_list.append(failed_result)

            if (
                chunking["enabled"]
                and chunk_idx < chunking["total_chunks"]
                and chunking["cooldown_seconds"] > 0
            ):
                if chunked_db_mode:
                    _flush_chunk_telemetry(job_id)
                    _refresh_persisted_batch_results(
                        job_id,
                        progress_message=(
                            f"Chunk {chunk_idx}/{chunking['total_chunks']} complete — "
                            f"cooling down for {chunking['cooldown_seconds']}s before next chunk."
                        ),
                    )
                    gc.collect()
                _append_progress_and_log(
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
    if chunked_db_mode:
        loaded = db.load_job_results(job_id, preferred_mode=_results_cache.get(job_id, {}).get("input_mode")) if db and db.is_configured() else None
        summary_rows = ((loaded or {}).get("results") or {}).get("summary_rows") or []
        failed_rows = ((loaded or {}).get("results") or {}).get("failed_rows") or []
        evaluated_count = len(summary_rows)
        total_failures = len(failed_rows)
    else:
        evaluated_count = len(evaluated)
        total_failures = len(results_list) - len(evaluated)
    if (chunked_db_mode and evaluated_count == 0) or (not chunked_db_mode and not evaluated):
        if _is_stop_requested(job_id):
            raise _JobStoppedError("Job stopped by user")
        msg = "No startups were successfully evaluated."
        if last_error:
            msg += f" Last error: {last_error[:200]}"
        _set_job_status(job_id, "error", msg, source="run_specter_analysis")
        return

    if chunked_db_mode:
        _flush_chunk_telemetry(job_id)
    if chunking["enabled"]:
        _append_progress_and_log(
            job_id,
            f"Chunked batch processing complete — finalizing {evaluated_count}/{total} company result(s).",
        )
    _append_progress(job_id, "Finalizing batch results...")
    if chunked_db_mode:
        if not _refresh_persisted_batch_results(
            job_id,
            progress_message="Finalizing batch results...",
            full=True,
        ):
            raise RuntimeError("Failed to reconstruct chunked batch results from persistence.")
    else:
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
        f"Analysis complete — {evaluated_count}/{total} companies ranked",
        source="run_specter_analysis",
    )
    _jobs[job_id].results = _results_cache[job_id]["results"]
    _persist_jobs()
    if chunked_db_mode:
        _persist_results_to_db(job_id, [])
    else:
        _persist_results_to_db(job_id, results_list)
    _mark_terminal_persistence_complete(job_id)


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

    _cancel_scheduled_restart()

    job_id = str(uuid.uuid4())[:8]
    _person_jobs[job_id] = PersonProfileJobStatus(
        job_id=job_id,
        status="pending",
        progress="Queued",
    )
    _persist_person_job(job_id, req.model_dump())
    _schedule_person_profile_job(job_id, req)
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
                _resume_person_profile_job_if_needed(job_id, loaded)
            else:
                raise HTTPException(status_code=404, detail="Job not found")
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    else:
        _resume_person_profile_job_if_needed(job_id)

    job = _person_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
    }


@app.get("/api/person-profile/latest")
async def get_latest_person_profile(
    company_slug: str,
    person_key: str,
    session_id: str | None = Cookie(default=None),
):
    """Fetch the latest shared person intelligence result for a team member."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not (db and db.is_configured()):
        raise HTTPException(status_code=503, detail="Database not configured")

    loaded = db.load_latest_person_profile_job(company_slug, person_key)
    if not loaded:
        raise HTTPException(status_code=404, detail="Person profile not found")

    job_id = loaded.get("person_job_id") or ""
    return {
        "job_id": job_id,
        "company_slug": loaded.get("company_slug") or company_slug,
        "person_key": loaded.get("person_key") or person_key,
        "status": loaded.get("status") or "pending",
        "progress": loaded.get("progress") or "",
        "result": loaded.get("result_payload"),
        "error": loaded.get("error"),
        "request_payload": loaded.get("request_payload") or {},
        "created_at": loaded.get("created_at"),
        "updated_at": loaded.get("updated_at"),
    }


@app.post("/api/person-profile/jobs/bulk-founders")
async def create_bulk_founder_jobs(
    req: BulkFounderJobRequest,
    session_id: str | None = Cookie(default=None),
):
    """Create one enrichment job per founder candidate."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    _cancel_scheduled_restart()

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
        _schedule_person_profile_job(job_id, profile_req)
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
    shared_vc_strategy = ""
    vc_strategy_source = "local"
    if db and db.is_configured():
        vc_strategy_source = "supabase"
        loaded_strategy = await _load_shared_vc_strategy()
        if isinstance(loaded_strategy, str):
            shared_vc_strategy = loaded_strategy
    return {
        "llm": default_llm["label"],
        "default_llm": default_llm,
        "available_models": available_models_payload(),
        "pricing_catalog": pricing_catalog_payload(),
        "phase_model_defaults": phase_model_defaults_payload(),
        "quality_tiers": quality_tiers_payload(),
        "premium_phase_options": premium_phase_options_payload(),
        "vc_investment_strategy": shared_vc_strategy,
        "vc_strategy_source": vc_strategy_source,
        "supabase_auth": _supabase_public_auth_config(),
    }


@app.post("/api/settings/vc-strategy")
async def save_vc_strategy(
    req: VcStrategyRequest,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not db or not db.is_configured():
        raise HTTPException(status_code=503, detail="Shared settings storage is not configured.")
    strategy = (req.vc_investment_strategy or "").strip()
    ok = await _save_shared_vc_strategy(strategy)
    if not ok:
        raise HTTPException(status_code=503, detail="Could not save VC investment strategy.")
    return {
        "vc_investment_strategy": strategy,
        "vc_strategy_source": "supabase",
    }


@app.get("/api/status/{job_id}")
async def get_status(
    job_id: str,
    response: Response,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    _set_no_store_headers(response)

    if job_id in _jobs:
        job = _jobs[job_id]
        cache = _results_cache.get(job_id, {})
        if _is_worker_backed_job(job_id) and db and db.is_configured():
            persisted_status = db.load_job_status(job_id)
            if persisted_status:
                job.status = _normalize_worker_status(persisted_status.get("status"))
                job.progress = persisted_status.get("progress") or job.progress
                job.progress_log = _load_worker_progress_log(job_id)
        results = cache.get("results")
        if _is_compact_results_payload(results) and job.status in {"done", "error", "stopped"}:
            results = None
        if results is None and job.status in {"pending", "running", "paused"}:
            results = _maybe_promote_terminal_persisted_results(job_id, cache=cache)
            job = _jobs[job_id]
        if results is None and job.status in {"done", "error", "stopped"}:
            loaded = _load_persisted_job_results(
                job_id,
                preferred_mode=cache.get("input_mode"),
            )
            if loaded:
                results = loaded.get("results")
                _promote_results_metadata(job_id, results)
        if results is not None and not _is_terminal_job_status(job.status):
            results = None
        if results is not None and not _is_compact_results_payload(results):
            _mark_terminal_results_served(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "progress_log": getattr(job, "progress_log", []) or [],
            "results": results,
            "llm": _resolve_job_llm_label(job_id, results=results),
        }

    if db and db.is_configured():
        persisted_status = db.load_job_status(job_id)
        if persisted_status:
            loaded_status = _normalize_worker_status(persisted_status.get("status"))
            loaded_progress = persisted_status.get("progress") or "Analysis in progress"
            progress_log = _load_worker_progress_log(job_id)
            worker_active = bool(persisted_status.get("worker_active"))
            if loaded_status in {"pending", "running", "paused"} and worker_active:
                return {
                    "job_id": job_id,
                    "status": loaded_status,
                    "progress": loaded_progress,
                    "progress_log": progress_log,
                    "results": None,
                    "llm": _resolve_job_llm_label(job_id, results=None),
                }
            if loaded_status in {"pending", "running", "paused"}:
                return {
                    "job_id": job_id,
                    "status": "stopped",
                    "progress": "Run interrupted before completion.",
                    "progress_log": [],
                    "results": None,
                    "llm": _resolve_job_llm_label(job_id, results=None),
                }
            loaded = _load_persisted_job_results(job_id)
            if loaded and _is_terminal_job_status(loaded_status):
                results = loaded.get("results")
                _jobs[job_id] = AnalysisStatus(
                    job_id=job_id,
                    status=loaded_status,
                    progress=loaded_progress,
                    progress_log=[],
                    results=None,
                    persistence_complete=True,
                )
                _results_cache[job_id] = {}
                _promote_results_metadata(job_id, results)
                _mark_terminal_results_served(job_id)
                return {
                    "job_id": job_id,
                    "status": loaded_status,
                    "progress": loaded_progress,
                    "progress_log": [],
                    "results": results,
                    "llm": _resolve_job_llm_label(job_id, results=results),
                }
            return {
                "job_id": job_id,
                "status": loaded_status,
                "progress": loaded_progress,
                "progress_log": [],
                "results": None,
                "llm": _resolve_job_llm_label(job_id, results=None),
            }
        saved_job = db.load_saved_job(job_id)
        if saved_job:
            return {
                "job_id": job_id,
                "status": "stopped",
                "progress": "Run interrupted before completion.",
                "progress_log": [],
                "results": None,
                "llm": saved_job.get("llm") or _resolve_job_llm_label(job_id, results=None),
            }

    loaded = _load_persisted_job_results(job_id)
    if loaded:
        results = loaded.get("results")
        loaded_status = (results or {}).get("job_status") or "done"
        loaded_progress = (results or {}).get("job_message") or "Analysis complete"
        _jobs[job_id] = AnalysisStatus(
            job_id=job_id,
            status=loaded_status,
            progress=loaded_progress,
            progress_log=[],
            results=None,
            persistence_complete=True,
        )
        _results_cache[job_id] = {}
        _promote_results_metadata(job_id, results)
        _mark_terminal_results_served(job_id)
        return {
            "job_id": job_id,
            "status": loaded_status,
            "progress": loaded_progress,
            "progress_log": [],
            "results": results,
            "llm": _resolve_job_llm_label(job_id, results=results),
        }

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/jobs/{job_id}/log")
async def get_job_log(
    job_id: str,
    response: Response,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    _set_no_store_headers(response)

    job = _jobs.get(job_id)
    if job and job.status in {"pending", "running", "paused"}:
        if _is_worker_backed_job(job_id) and db and db.is_configured():
            persisted_status = db.load_job_status(job_id)
            if persisted_status:
                job.status = _normalize_worker_status(persisted_status.get("status"))
                job.progress = persisted_status.get("progress") or job.progress
                job.progress_log = _load_worker_progress_log(job_id)
        return {
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "progress_log": getattr(job, "progress_log", []) or [],
        }

    if db and db.is_configured():
        persisted_status = db.load_job_status(job_id)
        progress_log = _load_worker_progress_log(job_id) if persisted_status else []
        if (
            persisted_status
            and persisted_status.get("status") in {"pending", "running", "paused"}
            and progress_log
        ):
            return {
                "job_id": job_id,
                "status": _normalize_worker_status(persisted_status.get("status")),
                "progress": persisted_status.get("progress") or "Analysis in progress",
                "progress_log": progress_log,
            }

    raise HTTPException(status_code=409, detail="Run is no longer active.")


@app.get("/api/download/{job_id}")
async def download_excel(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    raise HTTPException(status_code=410, detail="Excel export has been removed")


@app.get("/api/analyses/{job_id}")
async def get_analysis(
    job_id: str,
    response: Response,
    session_id: str | None = Cookie(default=None),
):
    """Return analysis results for a completed job. Uses in-memory cache or Supabase."""
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    _set_no_store_headers(response)

    cache = _results_cache.get(job_id, {})
    results = cache.get("results")
    if results and not _is_compact_results_payload(results):
        _mark_terminal_results_served(job_id)
        return {"job_id": job_id, "results": results}

    saved_job = None
    if db and db.is_configured():
        load_saved_job = getattr(db, "load_saved_job", None)
        load_job_status = getattr(db, "load_job_status", None)
        if callable(load_saved_job):
            saved_job = load_saved_job(job_id)
        if saved_job:
            saved_status = _normalize_worker_status(saved_job.get("status"))
            if not _is_terminal_job_status(saved_status):
                raise HTTPException(status_code=409, detail="Analysis is still in progress.")
            if job_id in _jobs:
                _jobs[job_id].status = saved_status
                _jobs[job_id].progress = saved_job.get("progress") or _jobs[job_id].progress
            else:
                _jobs[job_id] = AnalysisStatus(
                    job_id=job_id,
                    status=saved_status,
                    progress=saved_job.get("progress") or "Analysis complete",
                    progress_log=[],
                )
        else:
            persisted_status = load_job_status(job_id) if callable(load_job_status) else None
            if persisted_status and not _is_terminal_job_status(persisted_status.get("status")):
                raise HTTPException(status_code=409, detail="Analysis is still in progress.")
            if job_id in _jobs and not _is_terminal_job_status(_jobs[job_id].status):
                raise HTTPException(status_code=409, detail="Analysis is still in progress.")
    elif job_id in _jobs and not _is_terminal_job_status(_jobs[job_id].status):
        raise HTTPException(status_code=409, detail="Analysis is still in progress.")

    loaded = _load_persisted_job_results(job_id, preferred_mode=cache.get("input_mode"))
    if loaded:
        results = loaded.get("results")
        _promote_results_metadata(job_id, results)
        _mark_terminal_results_served(job_id)
        return {"job_id": job_id, "results": results}

    raise HTTPException(status_code=404, detail="Analysis not found")


@app.get("/api/jobs")
async def list_jobs(session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    cached = _get_cached_overview_payload(_jobs_overview_cache)
    if isinstance(cached, dict):
        return cached

    payload = {"jobs": _list_jobs_for_ui()}
    return _set_cached_overview_payload(
        _jobs_overview_cache,
        payload,
        JOBS_OVERVIEW_CACHE_SECONDS,
    )


@app.get("/api/company-runs")
async def list_company_runs(session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    cached = _get_cached_overview_payload(_company_runs_cache)
    if isinstance(cached, dict):
        return cached

    payload = {"companies": _list_company_runs_for_ui()}
    return _set_cached_overview_payload(
        _company_runs_cache,
        payload,
        COMPANY_RUNS_CACHE_SECONDS,
    )


@app.get("/api/companies/{company_lookup_key}/chat")
async def get_company_chat(
    company_lookup_key: str,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not db or not db.is_configured():
        raise HTTPException(status_code=501, detail="Supabase not configured")

    context = db.load_company_chat_context(company_lookup_key)
    if not context:
        raise HTTPException(status_code=404, detail="Company chat context not found")

    chat_session = await _load_company_chat_session(session_id, company_lookup_key)
    _company, _store, _citation_map, meta = build_company_chat_store(context)
    selection = _company_chat_selection(chat_session)
    return CompanyChatResponse(
        company_lookup_key=company_lookup_key,
        transcript=[CompanyChatMessage(**item) for item in chat_session.get("transcript") or []],
        run_count=meta["run_count"],
        source_counts=meta["source_counts"],
        web_search_enabled=True,
        model_label=_company_chat_model_label(chat_session),
        llm_provider=selection["provider"],
        llm_model=selection["model"],
        session_run_costs=_company_chat_session_costs(chat_session),
        available_models=available_chat_models_payload(),
    )


@app.post("/api/companies/{company_lookup_key}/chat")
async def post_company_chat(
    company_lookup_key: str,
    req: CompanyChatRequest,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not db or not db.is_configured():
        raise HTTPException(status_code=501, detail="Supabase not configured")

    context = db.load_company_chat_context(company_lookup_key)
    if not context:
        raise HTTPException(status_code=404, detail="Company chat context not found")

    chat_session = await _load_company_chat_session(session_id, company_lookup_key)
    if req.llm_provider or req.llm_model:
        selection = _resolve_requested_company_chat_selection(req.llm_provider, req.llm_model)
        chat_session["selection"] = selection
    else:
        selection = _company_chat_selection(chat_session)
        chat_session["selection"] = selection

    transcript = list(chat_session.get("transcript") or [])
    transcript.append(
        CompanyChatMessage(
            role="user",
            content=req.message,
            citations=[],
            created_at=_now_iso(),
        ).model_dump()
    )
    transient_session = {"transcript": transcript, "summary": chat_session.get("summary") or ""}
    _compress_company_chat_session(transient_session)

    result = await answer_company_question(
        context=context,
        transcript=transient_session["transcript"],
        conversation_summary=transient_session["summary"],
        question=req.message,
        use_web_search=True,
        active_job_id=req.active_job_id,
        llm_selection=selection,
    )

    chat_session["summary"] = transient_session["summary"]
    session_model_executions = chat_session.setdefault("model_executions", [])
    session_model_executions.extend(result.get("model_executions") or [])
    chat_session["transcript"] = transcript + [
        CompanyChatMessage(
            role="assistant",
            content=result["answer"],
            citations=result["citations"],
            created_at=_now_iso(),
            llm_label=result.get("model_label"),
            run_costs=result.get("run_costs"),
        ).model_dump()
    ]
    _compress_company_chat_session(chat_session)
    await _persist_company_chat_session(
        session_id,
        company_lookup_key,
        context.get("company_name") or "Unknown company",
        chat_session,
    )
    session_run_costs = _company_chat_session_costs(chat_session)

    return {
        "company_lookup_key": company_lookup_key,
        "answer": result["answer"],
        "citations": result["citations"],
        "transcript": chat_session["transcript"],
        "run_count": result["run_count"],
        "source_counts": result["source_counts"],
        "web_search_enabled": True,
        "model_label": result["model_label"],
        "llm_provider": selection["provider"],
        "llm_model": selection["model"],
        "run_costs": result.get("run_costs"),
        "session_run_costs": session_run_costs,
        "available_models": available_chat_models_payload(),
        "used_run_ids": result["used_run_ids"],
        "used_web_search": result["used_web_search"],
        "web_search_query": result["web_search_query"],
    }


@app.delete("/api/companies/{company_lookup_key}/chat")
async def delete_company_chat(
    company_lookup_key: str,
    session_id: str | None = Cookie(default=None),
):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")
    await _clear_company_chat_session(session_id, company_lookup_key)
    return {"company_lookup_key": company_lookup_key, "cleared": True}


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
