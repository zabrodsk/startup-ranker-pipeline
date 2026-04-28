"""Dedicated Specter batch worker.

This process runs outside the web service and keeps long-running Specter batch
coordination out of the Railway web process. Company evaluation still happens
in per-company subprocesses so memory is reclaimed aggressively.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import traceback
from typing import Any

from agent.dataclasses.company import Company
from agent.ingest.specter_ingest import (
    _company_slug,
    ingest_specter_company,
    list_specter_companies,
)
from agent.ingest.store import EvidenceStore
from web import app as web_app
import web.db as db

EVENT_PREFIX = "__SPECTER_COMPANY_EVENT__"
POLL_SECONDS = max(1, int(os.getenv("SPECTER_WORKER_POLL_SECONDS", "5")))
HEARTBEAT_SECONDS = max(10, int(os.getenv("SPECTER_WORKER_HEARTBEAT_SECONDS", "20")))


def _log(message: str) -> None:
    print(f"[specter-worker] {message}", flush=True)


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _normalize_company_key(name: str | None, slug: str | None) -> str:
    return f"{(slug or name or 'unknown').strip().lower()}"


def _domain_root(value: str | None) -> str:
    if not value:
        return ""
    s = value.strip().lower().lstrip("@").strip("'\"")
    if "://" in s:
        s = s.split("://", 1)[1]
    if s.startswith("www."):
        s = s[4:]
    return s.split("/", 1)[0]


def _slug_from_url(url: str) -> str:
    root = _domain_root(url)
    if not root:
        return "unknown"
    return _company_slug(root.replace(".", "-"))


def _normalize_specter_urls(run_config: dict[str, Any]) -> list[dict[str, str]]:
    raw = run_config.get("specter_urls") or []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            url = (item.get("url") or "").strip()
            expected_name = (item.get("name") or "").strip() or None
        else:
            url = str(item or "").strip()
            expected_name = None
        if not url:
            continue
        domain_key = _domain_root(url) or url.lower()
        if domain_key in seen:
            continue
        seen.add(domain_key)
        out.append(
            {
                "url": url,
                "domain": domain_key,
                "expected_name": expected_name or "",
                "slug": _slug_from_url(url),
                "name": expected_name or domain_key,
            }
        )
    return out


def _build_company_tasks(
    run_config: dict[str, Any],
    companies_csv: Path | None,
) -> list[dict[str, Any]]:
    """Unified company task list across CSV and URL sources, dedup'd by domain.

    Each task is one of two shapes::

        {"mode": "csv", "index": int, "name": str, "slug": str, "domain": str | None}
        {"mode": "url", "url": str, "name": str, "slug": str, "domain": str, "expected_name": str}

    Dedup rule: when a CSV row and a URL share the same domain root, the CSV
    task wins (richer data for the hybrid path). URLs that don't collide are
    appended after CSV tasks.
    """
    tasks: list[dict[str, Any]] = []
    seen_domains: set[str] = set()

    if companies_csv is not None:
        descriptors = list_specter_companies(companies_csv)
        for d in descriptors:
            domain = _domain_root(d.get("domain"))
            if domain:
                seen_domains.add(domain)
            tasks.append({"mode": "csv", **d, "domain": domain or None})

    for url_task in _normalize_specter_urls(run_config):
        if url_task["domain"] in seen_domains:
            _log(
                f"deduping URL task {url_task['url']!r} — already covered by CSV row "
                f"with same domain root"
            )
            continue
        seen_domains.add(url_task["domain"])
        tasks.append({"mode": "url", **url_task})

    return tasks


def _final_job_outcome(
    *,
    completed_companies: int,
    failed_companies: int,
    total_companies: int,
) -> tuple[str, str]:
    if completed_companies <= 0 and failed_companies > 0:
        return (
            "error",
            f"No companies were successfully evaluated. {failed_companies}/{total_companies} failed.",
        )
    return (
        "done",
        f"Analysis complete — {completed_companies}/{total_companies} companies ranked",
    )


def _write_worker_config(
    work_dir: Path,
    *,
    job_id: str,
    absolute_index: int,
    run_config: dict[str, Any],
    versions: dict[str, Any],
) -> Path:
    path = work_dir / f".specter-worker-{job_id}-{absolute_index}.json"
    path.write_text(
        json.dumps({"run_config": run_config, "versions": versions}, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def _load_completed_company_keys(job_id: str) -> tuple[set[str], int, int]:
    rows = db.load_job_company_runs(job_id)
    completed: set[str] = set()
    failed = 0
    for row in rows:
        key = _normalize_company_key(row.get("company_name"), row.get("startup_slug"))
        completed.add(key)
        if str(row.get("decision") or "").strip().lower() in {"error", "timeout"}:
            failed += 1
    return completed, max(0, len(completed) - failed), failed


def _specter_files_from_run_config(
    run_config: dict[str, Any],
    source_files: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    manifest = run_config.get("specter_worker_files") or {}
    companies_storage_path = manifest.get("companies_storage_path")
    people_storage_path = manifest.get("people_storage_path")
    if companies_storage_path:
        return companies_storage_path, people_storage_path

    companies_name = manifest.get("companies_name")
    people_name = manifest.get("people_name")
    for file_info in source_files:
        if not companies_storage_path and file_info.get("name") == companies_name:
            companies_storage_path = file_info.get("storage_path")
        if people_name and not people_storage_path and file_info.get("name") == people_name:
            people_storage_path = file_info.get("storage_path")
    return companies_storage_path, people_storage_path


def _download_worker_inputs(
    job_id: str, run_config: dict[str, Any]
) -> tuple[Path, Path | None, Path | None]:
    """Materialize CSV inputs (if any) into a fresh temp work_dir.

    Returns (work_dir, companies_csv | None, people_csv | None). The CSV pair
    is None when the run is URL-only — the work_dir is still created because
    the per-company subprocess needs a place for its config.json.
    """
    work_dir = Path(tempfile.mkdtemp(prefix=f"specter-worker-{job_id}-"))

    source_files = db.load_source_files(job_id)
    companies_storage_path, people_storage_path = _specter_files_from_run_config(run_config, source_files)
    if not companies_storage_path:
        # URL-only batch (or hybrid with no CSV side). That's allowed when
        # run_config.specter_urls is non-empty.
        return work_dir, None, None

    companies_path = work_dir / "companies.csv"
    if not db.download_source_file_to_path(companies_storage_path, companies_path):
        raise RuntimeError("Failed to download shared Specter company export.")

    people_path: Path | None = None
    if people_storage_path:
        people_path = work_dir / "people.csv"
        if not db.download_source_file_to_path(people_storage_path, people_path):
            raise RuntimeError("Failed to download shared Specter people export.")

    return work_dir, companies_path, people_path


def _failure_result(
    job_id: str,
    *,
    company: Any,
    store: Any,
    slug: str,
    error_message: str,
    status: str,
) -> dict[str, Any]:
    web_app._results_cache[job_id] = web_app._results_cache.get(job_id, {})
    return web_app._failure_result_payload(
        job_id,
        company=company,
        store=store,
        slug=slug,
        status=status,
        error_message=error_message,
    )


def _persist_subprocess_failure(
    job_id: str,
    *,
    task: dict[str, Any],
    companies_csv: Path | None,
    people_csv: Path | None,
    run_config: dict[str, Any],
    versions: dict[str, Any],
    error_message: str,
) -> None:
    if task.get("mode") == "csv":
        if companies_csv is None:
            raise RuntimeError("CSV task encountered without a downloaded CSV file")
        company, store = ingest_specter_company(
            companies_csv, people_csv, company_index=int(task["index"])
        )
    else:
        # URL task failed before producing real evidence — synthesize a minimal
        # Company/EvidenceStore so the failure is persisted with stable identity.
        slug = task.get("slug") or "unknown"
        domain = task.get("domain") or None
        name = task.get("expected_name") or task.get("name") or domain or "Unknown"
        company = Company(name=name, domain=domain)
        store = EvidenceStore(startup_slug=slug, chunks=[])
    status = "error"
    payload = _failure_result(
        job_id,
        company=company,
        store=store,
        slug=store.startup_slug,
        error_message=error_message[:1000],
        status=status,
    )
    db.insert_analysis_error(
        job_id,
        message=error_message[:1000],
        stage="specter_batch_worker",
        error_type="WorkerSubprocessError",
        company_slug=store.startup_slug,
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
            "error": error_message[:1000],
            "skipped": False,
        },
        company_payload=payload,
        run_config=run_config,
        versions=versions,
    )


async def _run_company_subprocess(
    *,
    job_id: str,
    work_dir: Path,
    companies_csv: Path | None,
    people_csv: Path | None,
    company_descriptor: dict[str, Any],
    absolute_index: int,
    total_companies: int,
    run_config: dict[str, Any],
    versions: dict[str, Any],
    use_web_search: bool,
    vc_investment_strategy: str | None,
    worker_id: str,
    completed_companies: int,
    failed_companies: int,
) -> tuple[int, int]:
    mode = company_descriptor.get("mode", "csv")
    company_name = str(company_descriptor.get("name") or company_descriptor.get("slug") or "company")
    prefix = f"Worker evaluating {company_name} ({absolute_index}/{total_companies})"
    _log(f"{job_id}: starting company {absolute_index}/{total_companies} ({company_name})")
    db.insert_analysis_event(job_id, message=f"{prefix} — starting", event_type="worker_company", stage="company")
    heartbeat_state: dict[str, Any] = {
        "progress": f"{prefix} — starting",
        "completed_companies": completed_companies,
        "failed_companies": failed_companies,
    }
    db.heartbeat_specter_worker_job(
        job_id,
        status="running",
        progress=str(heartbeat_state["progress"]),
        active_company_slug=company_descriptor.get("slug"),
        active_company_index=absolute_index,
        completed_companies=completed_companies,
        failed_companies=failed_companies,
        total_companies=total_companies,
        worker_id=worker_id,
    )

    config_path = _write_worker_config(
        work_dir,
        job_id=job_id,
        absolute_index=absolute_index,
        run_config=run_config,
        versions=versions,
    )
    cmd = [
        sys.executable,
        "-m",
        "agent.specter_company_worker",
        "--job-id",
        job_id,
        "--absolute-index",
        str(absolute_index),
        "--config-path",
        str(config_path),
    ]
    if mode == "csv":
        if companies_csv is None:
            raise RuntimeError(
                f"CSV company task {company_name!r} but no CSV input was downloaded"
            )
        cmd.extend(["--specter-companies", str(companies_csv)])
        cmd.extend(["--company-index", str(int(company_descriptor["index"]))])
        if people_csv:
            cmd.extend(["--specter-people", str(people_csv)])
    elif mode == "url":
        url = str(company_descriptor.get("url") or "").strip()
        if not url:
            raise RuntimeError(f"URL task missing url field: {company_descriptor!r}")
        cmd.extend(["--specter-url", url])
        expected = str(company_descriptor.get("expected_name") or "").strip()
        if expected:
            cmd.extend(["--expected-name", expected])
    else:
        raise RuntimeError(f"Unknown company task mode: {mode!r}")
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

    heartbeat_stop = asyncio.Event()

    async def _heartbeat_loop() -> None:
        while True:
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=HEARTBEAT_SECONDS)
                return
            except asyncio.TimeoutError:
                db.heartbeat_specter_worker_job(
                    job_id,
                    status="running",
                    progress=str(heartbeat_state.get("progress") or f"{prefix} — working"),
                    active_company_slug=company_descriptor.get("slug"),
                    active_company_index=absolute_index,
                    completed_companies=int(heartbeat_state.get("completed_companies") or 0),
                    failed_companies=int(heartbeat_state.get("failed_companies") or 0),
                    total_companies=total_companies,
                    worker_id=worker_id,
                )

    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    saw_completion_event = False
    company_failed = False
    try:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            if text.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(text[len(EVENT_PREFIX):])
                except Exception:
                    db.insert_analysis_event(job_id, message=text, event_type="worker_stdout", stage="company")
                    continue
                event_type = str(event.get("type") or "").strip().lower()
                if event_type == "progress":
                    message = str(event.get("message") or "").strip()
                    if message:
                        full_message = f"{prefix} — {message}"
                        heartbeat_state["progress"] = full_message
                        db.insert_analysis_event(job_id, message=full_message, event_type="progress", stage="company")
                        db.heartbeat_specter_worker_job(
                            job_id,
                            status="running",
                            progress=full_message,
                            active_company_slug=company_descriptor.get("slug"),
                            active_company_index=absolute_index,
                            completed_companies=completed_companies,
                            failed_companies=failed_companies,
                            total_companies=total_companies,
                            worker_id=worker_id,
                        )
                elif event_type == "company_complete":
                    saw_completion_event = True
                    status = str(event.get("status") or "done").strip().lower()
                    if status in {"error", "timeout"}:
                        company_failed = True
                        failed_companies += 1
                        message = f"{prefix} — {status}: {str(event.get('error') or '').strip()}"
                        _log(f"{job_id}: company {absolute_index}/{total_companies} finished with status={status}")
                    else:
                        completed_companies += 1
                        message = f"Partial results updated — {completed_companies}/{total_companies} companies completed."
                        _log(
                            f"{job_id}: company {absolute_index}/{total_companies} completed "
                            f"(completed={completed_companies}, failed={failed_companies})"
                        )
                    heartbeat_state["progress"] = message
                    heartbeat_state["completed_companies"] = completed_companies
                    heartbeat_state["failed_companies"] = failed_companies
                    db.insert_analysis_event(job_id, message=message, event_type="company_complete", stage="company", payload=event)
                    db.heartbeat_specter_worker_job(
                        job_id,
                        status="running",
                        progress=message,
                        active_company_slug=company_descriptor.get("slug"),
                        active_company_index=absolute_index,
                        completed_companies=completed_companies,
                        failed_companies=failed_companies,
                        total_companies=total_companies,
                        worker_id=worker_id,
                    )
                continue
            db.insert_analysis_event(job_id, message=text, event_type="worker_stdout", stage="company")

        return_code = await process.wait()
        if return_code != 0 and not saw_completion_event:
            error_message = (
                f"Specter company worker exited with code {return_code} "
                f"for company {absolute_index}/{total_companies}."
            )
            _log(f"{job_id}: company {absolute_index}/{total_companies} worker exited with code {return_code}")
            _persist_subprocess_failure(
                job_id,
                task=company_descriptor,
                companies_csv=companies_csv,
                people_csv=people_csv,
                run_config=run_config,
                versions=versions,
                error_message=error_message,
            )
            failed_companies += 1
            heartbeat_state["progress"] = f"{prefix} — error: {error_message}"
            heartbeat_state["completed_companies"] = completed_companies
            heartbeat_state["failed_companies"] = failed_companies
            db.insert_analysis_event(job_id, message=f"{prefix} — error: {error_message}", event_type="company_complete", stage="company")
            db.heartbeat_specter_worker_job(
                job_id,
                status="running",
                progress=f"{prefix} — error: {error_message}",
                active_company_slug=company_descriptor.get("slug"),
                active_company_index=absolute_index,
                completed_companies=completed_companies,
                failed_companies=failed_companies,
                total_companies=total_companies,
                worker_id=worker_id,
            )
        return completed_companies, failed_companies
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        with contextlib.suppress(Exception):
            config_path.unlink()


async def _process_job(job: dict[str, Any], worker_id: str) -> None:
    job_id = str(job.get("job_id") or "")
    if not job_id:
        return
    _log(f"{job_id}: claimed by {worker_id}")

    run_config = dict(job.get("run_config") or {})
    versions = {
        "app_version": run_config.get("app_version"),
        "prompt_version": run_config.get("prompt_version"),
        "pipeline_version": run_config.get("pipeline_version"),
        "schema_version": run_config.get("schema_version"),
    }
    use_web_search = bool(run_config.get("use_web_search"))
    vc_investment_strategy = run_config.get("vc_investment_strategy")

    work_dir: Path | None = None
    try:
        work_dir, companies_csv, people_csv = _download_worker_inputs(job_id, run_config)
        _log(f"{job_id}: downloaded worker inputs (csv={companies_csv is not None})")
        tasks = _build_company_tasks(run_config, companies_csv)
        max_startups = web_app._parse_max_startups_from_instructions(run_config.get("instructions"))
        if max_startups is not None:
            tasks = tasks[:max_startups]

        completed_keys, completed_companies, failed_companies = _load_completed_company_keys(job_id)
        total_companies = len(tasks)
        if total_companies <= 0:
            raise RuntimeError(
                "No companies found — neither CSV exports nor URL list were provided."
            )
        _log(
            f"{job_id}: loaded {total_companies} company tasks "
            f"({sum(1 for t in tasks if t['mode'] == 'csv')} csv, "
            f"{sum(1 for t in tasks if t['mode'] == 'url')} url; "
            f"already_completed={len(completed_keys)}, failed={failed_companies})"
        )

        if completed_keys:
            message = f"Resuming worker batch — {len(completed_keys)}/{total_companies} companies already persisted."
            _log(f"{job_id}: resuming with {len(completed_keys)} completed companies")
            db.insert_analysis_event(job_id, message=message, event_type="worker_resume", stage="resume")
            db.heartbeat_specter_worker_job(
                job_id,
                status="running",
                progress=message,
                completed_companies=completed_companies,
                failed_companies=failed_companies,
                total_companies=total_companies,
                worker_id=worker_id,
            )

        for absolute_index, task in enumerate(tasks, start=1):
            company_key = _normalize_company_key(task.get("name"), task.get("slug"))
            if company_key in completed_keys:
                _log(f"{job_id}: skipping already persisted company {absolute_index}/{total_companies}")
                continue
            completed_companies, failed_companies = await _run_company_subprocess(
                job_id=job_id,
                work_dir=work_dir,
                companies_csv=companies_csv,
                people_csv=people_csv,
                company_descriptor=task,
                absolute_index=absolute_index,
                total_companies=total_companies,
                run_config=run_config,
                versions=versions,
                use_web_search=use_web_search,
                vc_investment_strategy=vc_investment_strategy,
                worker_id=worker_id,
                completed_companies=completed_companies,
                failed_companies=failed_companies,
            )
            completed_keys.add(company_key)
            gc.collect()

        db.insert_analysis_event(job_id, message="Finalizing batch results...", event_type="worker_finalizing", stage="finalize")
        _log(f"{job_id}: finalizing batch results")
        db.heartbeat_specter_worker_job(
            job_id,
            status="finalizing",
            progress="Finalizing batch results...",
            completed_companies=completed_companies,
            failed_companies=failed_companies,
            total_companies=total_companies,
            worker_id=worker_id,
        )
        loaded = db.load_job_results(job_id, preferred_mode="specter")
        if not loaded or not isinstance(loaded.get("results"), dict):
            raise RuntimeError("Could not reconstruct final Specter results from persisted state.")
        results = loaded["results"]
        final_status, final_message = _final_job_outcome(
            completed_companies=completed_companies,
            failed_companies=failed_companies,
            total_companies=total_companies,
        )
        results["job_status"] = final_status
        results["job_message"] = final_message
        if "run_costs" not in results:
            run_costs = db.load_run_costs(job_id)
            if isinstance(run_costs, dict):
                results["run_costs"] = run_costs

        if not db.persist_analysis_snapshot(
            job_id,
            results_payload=results,
            run_config=run_config,
            versions=versions,
            worker_state={
                "status": final_status,
                "progress": final_message,
                "completed_companies": completed_companies,
                "failed_companies": failed_companies,
                "total_companies": total_companies,
                "worker_service_enabled": True,
            },
        ):
            raise RuntimeError("Could not persist final worker-backed analysis snapshot.")

        db.finish_specter_worker_job(
            job_id,
            status=final_status,
            progress=final_message,
            completed_companies=completed_companies,
            failed_companies=failed_companies,
            total_companies=total_companies,
        )
        db.insert_analysis_event(
            job_id,
            message="Finalizing complete.",
            event_type="worker_done" if final_status == "done" else "worker_error",
            stage="finalize",
        )
        _log(f"{job_id}: finalization complete")
    except Exception as exc:
        _log(f"{job_id}: worker error {type(exc).__name__}: {exc}")
        traceback.print_exc()
        message = str(exc)[:1000]
        db.insert_analysis_error(job_id, message=message, stage="specter_batch_worker", error_type=type(exc).__name__)
        db.finish_specter_worker_job(job_id, status="error", progress=message)
        db.insert_analysis_event(job_id, message=f"Worker error: {message}", event_type="worker_error", stage="error")
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        web_app._results_cache.pop(job_id, None)
        gc.collect()
        _log(f"{job_id}: cleanup complete")


async def _worker_loop(run_once: bool = False) -> None:
    worker_id = _worker_id()
    _log(f"worker loop starting (worker_id={worker_id}, poll_seconds={POLL_SECONDS}, run_once={run_once})")
    idle_polls = 0
    while True:
        claimed = False
        candidates = db.list_claimable_specter_worker_jobs(limit=5)
        if candidates:
            idle_polls = 0
            candidate_ids = ", ".join(str(candidate.get("job_id") or "?") for candidate in candidates)
            _log(f"found {len(candidates)} claimable job(s): {candidate_ids}")
        for candidate in candidates:
            job = db.claim_specter_worker_job(str(candidate.get("job_id") or ""), worker_id=worker_id)
            if not job:
                _log(f"claim skipped for job {candidate.get('job_id')}")
                continue
            claimed = True
            _log(f"claim succeeded for job {job.get('job_id')}")
            await _process_job(job, worker_id)
            break
        if run_once:
            _log("run-once worker loop finished")
            return
        if not claimed:
            idle_polls += 1
            if idle_polls == 1 or idle_polls % 12 == 0:
                _log("no claimable jobs found; sleeping")
            await asyncio.sleep(POLL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    if not db.is_configured():
        _log("Supabase is not configured; worker cannot start")
        return 1
    asyncio.run(_worker_loop(run_once=args.run_once))
    return 0


if __name__ == "__main__":
    sys.exit(main())
