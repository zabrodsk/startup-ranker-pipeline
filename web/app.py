"""Rockaway Deal Intelligence web application.

FastAPI backend serving the Rockaway-branded UI with password protection.
The Gemini API key stays server-side only — never exposed to the client.
"""

import asyncio
import base64
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
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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
    export_excel,
    rank_batch_companies,
)
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

# Friendly display names for common LLM model IDs
_LLM_DISPLAY_NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-3-flash-lite-preview": "Gemini 3 Flash Lite",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
}


def _get_llm_display() -> str:
    """Return a display string for the configured LLM (provider + model)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
    friendly = _LLM_DISPLAY_NAMES.get(model)
    return friendly if friendly else f"{provider} · {model}"


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

    @field_validator("instructions", "vc_investment_strategy", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, list):
            return " ".join(str(x) for x in v).strip() or None
        s = str(v).strip()
        return s if s else None


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


def _llm_telemetry_base() -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
    timeout_raw = os.getenv("LLM_REQUEST_TIMEOUT_SECONDS")
    retries_raw = os.getenv("LLM_MAX_RETRIES")
    timeout_s: float | None = None
    max_retries: int | None = None
    try:
        if timeout_raw is not None:
            timeout_s = float(timeout_raw)
    except Exception:
        timeout_s = None
    try:
        if retries_raw is not None:
            max_retries = int(retries_raw)
    except Exception:
        max_retries = None

    return {
        "provider": provider,
        "model": model,
        "request_timeout_seconds": timeout_s,
        "max_retries": max_retries,
    }


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
        db.insert_analysis_event(job_id, message=msg, event_type="progress")


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
        db.insert_job_status_history(
            job_id,
            status=status,
            progress=_jobs[job_id].progress,
            source=source,
        )


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
    _append_progress(job_id, "Finalizing partial results (ranking + Excel export)...", allow_stopped=True)
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
        "results": results,
        "llm": _get_llm_display(),
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
                    "results": entry.get("results") or (existing or {}).get("results"),
                    "llm": (existing or {}).get("llm") or _get_llm_display(),
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

    _set_job_status(job_id, "running", "Starting analysis...", source="start_analysis")
    _job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    _results_cache[job_id]["input_mode"] = req.input_mode
    _results_cache[job_id]["vc_investment_strategy"] = req.vc_investment_strategy
    _results_cache[job_id]["use_web_search"] = req.use_web_search
    _results_cache[job_id]["instructions"] = req.instructions
    _results_cache[job_id]["model_executions"] = []
    _results_cache[job_id]["versions"] = _runtime_versions()

    if db and db.is_configured():
        run_config = {
            "input_mode": req.input_mode,
            "vc_investment_strategy": req.vc_investment_strategy,
            "instructions": req.instructions,
            "use_web_search": req.use_web_search,
        }
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
            )
        ),
        daemon=True,
    ).start()
    return {"status": "running", "use_web_search": req.use_web_search}


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
) -> None:
    """Compute scores and populate _results_cache for the finished job."""
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

    excel_path = upload_dir / "results.xlsx"
    export_excel(results_list, str(excel_path))

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

        _results_cache[job_id]["results"] = {
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

        _results_cache[job_id]["results"] = {
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

    _results_cache[job_id]["excel_path"] = str(excel_path)


def _persist_results_to_db(job_id: str, results_list: list[dict]) -> None:
    """Best-effort DB persistence — must not block job completion."""
    if not (db and db.is_configured()):
        return
    try:
        cache = _results_cache.get(job_id, {})
        run_config = {
            "input_mode": cache.get("input_mode", "pitchdeck"),
            "vc_investment_strategy": cache.get("vc_investment_strategy"),
            "instructions": cache.get("instructions"),
            "use_web_search": cache.get("use_web_search", False),
        }
        db.persist_analysis(
            job_id_legacy=job_id,
            results_list=results_list,
            results_payload=cache.get("results", {}),
            excel_path=cache.get("excel_path", ""),
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
):
    try:
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
    telemetry_base = _llm_telemetry_base()

    if file_count == 0:
        _set_job_status(job_id, "error", "No files found.", source="run_document_analysis")
        return

    if one_company or file_count == 1:
        started = time.perf_counter()
        result = await evaluate_startup(
            upload_dir, k=8, use_web_search=use_web_search,
            on_progress=_make_progress_callback(job_id),
            on_cooperate=lambda: _cooperate_with_job_control(job_id),
            vc_investment_strategy=vc_investment_strategy,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        _results_cache[job_id].setdefault("model_executions", []).append(
            {
                **telemetry_base,
                "company_slug": result.get("slug"),
                "stage": "evaluate_startup",
                "latency_ms": latency_ms,
                "status": "done",
                "metadata": {"input_mode": "original" if one_company else "documents"},
            }
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
        _append_progress(job_id, "Finalizing results (ranking + Excel export)...")
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
    try:
        for i, finfo in enumerate(files, 1):
            await _cooperate_with_job_control(job_id)
            fname = finfo["name"]
            prefix = f"Analyzing {fname} ({i}/{total})"
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
                started = time.perf_counter()
                result = await evaluate_startup(
                    doc_dir, k=8, use_web_search=use_web_search,
                    on_progress=make_progress,
                    on_cooperate=lambda: _cooperate_with_job_control(job_id),
                    vc_investment_strategy=vc_investment_strategy,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                await _cooperate_with_job_control(job_id)
                results_list.append(result)
                _results_cache[job_id].setdefault("model_executions", []).append(
                    {
                        **telemetry_base,
                        "company_slug": result.get("slug"),
                        "stage": "evaluate_startup",
                        "latency_ms": latency_ms,
                        "status": "done",
                        "metadata": {"file_name": fname},
                    }
                )
            except _JobStoppedError:
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                _results_cache[job_id].setdefault("model_executions", []).append(
                    {
                        **telemetry_base,
                        "company_slug": _sanitize_slug(fname),
                        "stage": "evaluate_startup",
                        "status": "error",
                        "error_message": str(exc)[:500],
                        "metadata": {"file_name": fname},
                    }
                )
                if db and db.is_configured():
                    db.insert_analysis_error(
                        job_id,
                        message=str(exc)[:1000],
                        stage="evaluate_startup",
                        error_type=type(exc).__name__,
                        company_slug=_sanitize_slug(fname),
                    )
                results_list.append({
                    "slug": _sanitize_slug(fname),
                    "skipped": True,
                    "error": str(exc)[:500],
                    "company_name": fname,
                })
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

    _append_progress(job_id, "Finalizing batch results (ranking + Excel export)...")
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
    telemetry_base = _llm_telemetry_base()
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

    last_error: str | None = None
    try:
        for i, (company, store) in enumerate(company_store_pairs, 1):
            await _cooperate_with_job_control(job_id)
            prefix = f"Evaluating {company.name} ({i}/{total})"
            _append_progress(job_id, f"{prefix} — Starting...")

            def make_specter_progress(p: str) -> None:
                _raise_if_stopped(job_id)
                full_msg = f"{prefix} — {p}"
                _append_progress(job_id, full_msg)

            try:
                started = time.perf_counter()
                result = await evaluate_from_specter(
                    company, store, k=8, use_web_search=use_web_search,
                    on_progress=make_specter_progress,
                    on_cooperate=lambda: _cooperate_with_job_control(job_id),
                    vc_investment_strategy=vc_investment_strategy,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                await _cooperate_with_job_control(job_id)
                results_list.append(result)
                _results_cache[job_id].setdefault("model_executions", []).append(
                    {
                        **telemetry_base,
                        "company_slug": result.get("slug"),
                        "stage": "evaluate_from_specter",
                        "latency_ms": latency_ms,
                        "status": "done",
                    }
                )
            except _JobStoppedError:
                raise
            except Exception as exc:
                import traceback
                last_error = str(exc)
                print(f"  ERROR evaluating {company.name}: {exc}")
                traceback.print_exc()
                _results_cache[job_id].setdefault("model_executions", []).append(
                    {
                        **telemetry_base,
                        "company_slug": store.startup_slug,
                        "stage": "evaluate_from_specter",
                        "status": "error",
                        "error_message": str(exc)[:500],
                    }
                )
                if db and db.is_configured():
                    db.insert_analysis_error(
                        job_id,
                        message=str(exc)[:1000],
                        stage="evaluate_from_specter",
                        error_type=type(exc).__name__,
                        company_slug=store.startup_slug,
                    )
                results_list.append({
                    "slug": store.startup_slug,
                    "skipped": True,
                    "error": str(exc)[:500],
                    "company_name": company.name,
                })
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

    _append_progress(job_id, "Finalizing batch results (ranking + Excel export)...")
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
    return {"llm": _get_llm_display()}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if job_id in _jobs:
        job = _jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "progress_log": getattr(job, "progress_log", []) or [],
            "results": _results_cache.get(job_id, {}).get("results"),
            "llm": _get_llm_display(),
        }

    if db and db.is_configured():
        loaded = db.load_job_results(job_id)
        if loaded:
            results = loaded.get("results")
            _jobs[job_id] = AnalysisStatus(
                job_id=job_id,
                status="done",
                progress="Analysis complete",
                progress_log=[],
                results=results,
            )
            _results_cache[job_id] = {
                "results": results,
                "excel_storage_path": loaded.get("excel_storage_path"),
            }
            return {
                "job_id": job_id,
                "status": "done",
                "progress": "Analysis complete",
                "progress_log": [],
                "results": results,
                "llm": _get_llm_display(),
            }

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/download/{job_id}")
async def download_excel(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    cache = _results_cache.get(job_id, {})
    excel_path = cache.get("excel_path")
    if excel_path and Path(excel_path).exists():
        return FileResponse(
            excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="startup_ranking_results.xlsx",
        )

    if db and db.is_configured():
        data = db.download_excel_bytes(job_id)
        if data:
            return Response(
                content=data,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": 'attachment; filename="startup_ranking_results.xlsx"'},
            )

    raise HTTPException(status_code=404, detail="Results not ready")


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

    return {"companies": db.list_company_histories()}


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
