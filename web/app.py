"""Startup Ranker Web Application.

FastAPI backend serving the Rockaway-styled UI with password protection.
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
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

app = FastAPI(title="Startup Ranker", docs_url=None, redoc_url=None)

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
_sessions: dict[str, float] = {}
_results_cache: dict[str, dict[str, Any]] = {}

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    status: str  # "pending" | "running" | "done" | "error"
    progress: str = ""
    progress_log: list[str] = []
    results: Any = None


_jobs: dict[str, AnalysisStatus] = {}


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
    """Check if uploaded files are a Specter company + people pair (CSV or Excel)."""
    companies_file = None
    people_file = None
    for name in filenames:
        lower = name.lower()
        if not (lower.endswith(".csv") or lower.endswith(".xlsx") or lower.endswith(".xls")):
            continue
        if "people" in lower:
            people_file = upload_dir / name
        elif "company" in lower or "comapny" in lower:
            companies_file = upload_dir / name
    if companies_file and people_file:
        return {"companies": str(companies_file), "people": str(people_file)}
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
        saved.append({"name": f.filename, "size": dest.stat().st_size})

    specter = _detect_specter_csvs(upload_dir, [f["name"] for f in saved])

    _jobs[job_id] = AnalysisStatus(
        job_id=job_id, status="pending", progress="Files uploaded"
    )
    _results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": saved,
        "specter": specter,
    }

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

    _jobs[job_id].status = "running"
    _jobs[job_id].progress = "Starting analysis..."
    _results_cache[job_id]["input_mode"] = req.input_mode
    _results_cache[job_id]["vc_investment_strategy"] = req.vc_investment_strategy

    vc_str = _ensure_str(req.vc_investment_strategy).strip() or None
    inst = _ensure_str(req.instructions).strip() or None
    asyncio.create_task(_run_analysis(
        job_id,
        use_web_search=req.use_web_search,
        instructions=inst,
        input_mode=req.input_mode,
        vc_investment_strategy=vc_str,
    ))
    return {"status": "running", "use_web_search": req.use_web_search}


def _build_results_payload(
    results_list: list[dict],
    job_id: str,
    upload_dir: Path,
) -> None:
    """Compute scores and populate _results_cache for the finished job."""
    results_list = rank_batch_companies(results_list)
    summary_rows = build_summary_rows(results_list)
    argument_rows = build_argument_rows(results_list)
    qa_provenance_rows = build_qa_provenance_rows(results_list)

    excel_path = upload_dir / "results.xlsx"
    export_excel(results_list, str(excel_path))

    evaluated = [r for r in results_list if not r.get("skipped")]

    if len(evaluated) == 1:
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
        }
    else:
        failed_rows = build_failed_rows(results_list)
        _results_cache[job_id]["results"] = {
            "mode": "batch",
            "num_companies": len(evaluated),
            "num_skipped": len(results_list) - len(evaluated),
            "summary_rows": summary_rows,
            "argument_rows": argument_rows,
            "qa_provenance_rows": qa_provenance_rows,
            "failed_rows": failed_rows,
        }

    _results_cache[job_id]["excel_path"] = str(excel_path)


async def _run_analysis(
    job_id: str,
    use_web_search: bool = False,
    instructions: str | None = None,
    input_mode: str = "pitchdeck",
    vc_investment_strategy: str | None = None,
):
    try:
        upload_dir = Path(_results_cache[job_id]["upload_dir"])
        specter = _results_cache[job_id].get("specter")

        if input_mode == "specter":
            specter_detected = _results_cache[job_id].get("specter")
            files = _results_cache[job_id].get("files", [])
            if specter_detected:
                specter_paths = specter_detected
            elif len(files) >= 2:
                specter_paths = {
                    "companies": str(upload_dir / files[0]["name"]),
                    "people": str(upload_dir / files[1]["name"]),
                }
            else:
                _jobs[job_id].status = "error"
                _jobs[job_id].progress = "Specter mode requires 2 files (company + people CSV/Excel)."
                return
            await _run_specter_analysis(
                job_id, upload_dir, specter_paths, use_web_search, instructions,
                vc_investment_strategy=vc_investment_strategy,
            )
        elif input_mode == "original":
            await _run_document_analysis(
                job_id, upload_dir, use_web_search, one_company=True,
                vc_investment_strategy=vc_investment_strategy,
            )
        else:
            await _run_document_analysis(
                job_id, upload_dir, use_web_search, one_company=False,
                vc_investment_strategy=vc_investment_strategy,
            )

    except Exception as exc:
        _jobs[job_id].status = "error"
        _jobs[job_id].progress = f"Analysis failed: {exc}"


def _make_progress_callback(job_id: str):
    def on_progress(msg: str) -> None:
        _jobs[job_id].progress = msg
        log = getattr(_jobs[job_id], "progress_log", []) or []
        _jobs[job_id].progress_log = log + [msg]
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

    if file_count == 0:
        _jobs[job_id].status = "error"
        _jobs[job_id].progress = "No files found."
        return

    if one_company or file_count == 1:
        result = await evaluate_startup(
            upload_dir, k=8, use_web_search=use_web_search,
            on_progress=_make_progress_callback(job_id),
            vc_investment_strategy=vc_investment_strategy,
        )
        if result.get("skipped"):
            _jobs[job_id].status = "error"
            _jobs[job_id].progress = "No extractable content found in uploaded files."
            return
        _build_results_payload([result], job_id, upload_dir)
        _jobs[job_id].status = "done"
        _jobs[job_id].progress = "Analysis complete"
        _jobs[job_id].results = _results_cache[job_id]["results"]
        return

    results_list: list[dict] = []
    total = file_count
    for i, finfo in enumerate(files, 1):
        fname = finfo["name"]
        prefix = f"Analyzing {fname} ({i}/{total})"
        _jobs[job_id].progress = f"{prefix} — Starting..."

        doc_dir = upload_dir / _sanitize_slug(fname)
        doc_dir.mkdir(exist_ok=True)
        src = upload_dir / fname
        if src.exists():
            shutil.copy2(src, doc_dir / fname)

        def make_progress(msg: str) -> None:
            full_msg = f"{prefix} — {msg}"
            _jobs[job_id].progress = full_msg
            log = getattr(_jobs[job_id], "progress_log", []) or []
            _jobs[job_id].progress_log = log + [full_msg]

        try:
            result = await evaluate_startup(
                doc_dir, k=8, use_web_search=use_web_search,
                on_progress=make_progress,
                vc_investment_strategy=vc_investment_strategy,
            )
            results_list.append(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results_list.append({
                "slug": _sanitize_slug(fname),
                "skipped": True,
                "error": str(exc)[:500],
                "company_name": fname,
            })

    evaluated = [r for r in results_list if not r.get("skipped")]
    if not evaluated:
        _jobs[job_id].status = "error"
        _jobs[job_id].progress = "No startups were successfully evaluated."
        return

    _build_results_payload(results_list, job_id, upload_dir)
    _jobs[job_id].status = "done"
    _jobs[job_id].progress = f"Analysis complete — {len(evaluated)}/{total} companies ranked"
    _jobs[job_id].results = _results_cache[job_id]["results"]


async def _run_specter_analysis(
    job_id: str,
    upload_dir: Path,
    specter: dict,
    use_web_search: bool,
    instructions: str | None = None,
    vc_investment_strategy: str | None = None,
) -> None:
    """Batch Specter analysis from company + people CSVs."""
    _jobs[job_id].progress = "Parsing Specter CSV files..."

    company_store_pairs = ingest_specter(specter["companies"], specter["people"])
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
        _jobs[job_id].status = "error"
        _jobs[job_id].progress = "No companies found in Specter data."
        return

    total = len(company_store_pairs)
    results_list: list[dict] = []

    last_error: str | None = None
    for i, (company, store) in enumerate(company_store_pairs, 1):
        prefix = f"Evaluating {company.name} ({i}/{total})"
        _jobs[job_id].progress = f"{prefix} — Starting..."

        def make_specter_progress(p: str) -> None:
            full_msg = f"{prefix} — {p}"
            _jobs[job_id].progress = full_msg
            log = getattr(_jobs[job_id], "progress_log", []) or []
            _jobs[job_id].progress_log = log + [full_msg]

        try:
            result = await evaluate_from_specter(
                company, store, k=8, use_web_search=use_web_search,
                on_progress=make_specter_progress,
                vc_investment_strategy=vc_investment_strategy,
            )
            results_list.append(result)
        except Exception as exc:
            import traceback
            last_error = str(exc)
            print(f"  ERROR evaluating {company.name}: {exc}")
            traceback.print_exc()
            results_list.append({
                "slug": store.startup_slug,
                "skipped": True,
                "error": str(exc)[:500],
                "company_name": company.name,
            })

    evaluated = [r for r in results_list if not r.get("skipped")]
    if not evaluated:
        _jobs[job_id].status = "error"
        msg = "No startups were successfully evaluated."
        if last_error:
            msg += f" Last error: {last_error[:200]}"
        _jobs[job_id].progress = msg
        return

    _build_results_payload(results_list, job_id, upload_dir)
    _jobs[job_id].status = "done"
    _jobs[job_id].progress = f"Analysis complete — {len(evaluated)}/{total} companies ranked"
    _jobs[job_id].results = _results_cache[job_id]["results"]


@app.get("/api/status/{job_id}")
async def get_status(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "progress_log": getattr(job, "progress_log", []) or [],
        "results": _results_cache.get(job_id, {}).get("results"),
    }


@app.get("/api/download/{job_id}")
async def download_excel(job_id: str, session_id: str | None = Cookie(default=None)):
    if not _check_session(session_id):
        raise HTTPException(status_code=401, detail="Not authenticated")

    cache = _results_cache.get(job_id, {})
    excel_path = cache.get("excel_path")
    if not excel_path or not Path(excel_path).exists():
        raise HTTPException(status_code=404, detail="Results not ready")

    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="startup_ranking_results.xlsx",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
