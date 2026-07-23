"""Frozen, non-persisting A/B/C benchmark for Deal Intelligence model profiles."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PRIMARY_PROFILE_IDS: tuple[str, ...] = (
    "gpt_current",
    "kimi_k26",
    "glm_deepseek_flash",
)
DEFAULT_CAMPAIGN_SEED = 20260722
OPENROUTER_PREFLIGHT_MODELS: tuple[str, ...] = (
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
)
OPENROUTER_APPROVED_PROVIDER_SLUGS: tuple[str, ...] = ("deepinfra",)
SPECTER_TEAM_CAPTURE_VERSION = 2


def build_run_schedule(
    company_ids: Sequence[str],
    *,
    repeats: int = 1,
    seed: int = DEFAULT_CAMPAIGN_SEED,
    profile_ids: Sequence[str] = PRIMARY_PROFILE_IDS,
) -> list[dict[str, Any]]:
    """Build a reproducibly randomized, balanced, sequential run schedule."""
    if repeats <= 0:
        raise ValueError("repeats must be greater than zero.")
    normalized_companies = [str(value).strip() for value in company_ids if str(value).strip()]
    if len(set(normalized_companies)) != len(normalized_companies):
        raise ValueError("company_ids must be unique.")
    normalized_profiles = [str(value).strip() for value in profile_ids if str(value).strip()]
    if not normalized_profiles:
        raise ValueError("At least one profile is required.")

    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for company_id in normalized_companies:
            randomized_profiles = list(normalized_profiles)
            rng.shuffle(randomized_profiles)
            for profile_id in randomized_profiles:
                schedule.append(
                    {
                        "sequence": len(schedule) + 1,
                        "repeat": repeat,
                        "company_id": company_id,
                        "profile_id": profile_id,
                    }
                )
    return schedule


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized or "company"


def freeze_corpus(
    campaign_dir: str | Path,
    companies: Sequence[dict[str, Any]],
    *,
    seed: int = DEFAULT_CAMPAIGN_SEED,
    repeats: int = 1,
    expected_company_count: int = 6,
) -> dict[str, Any]:
    """Freeze approved-candidate evidence without writing application records."""
    root = Path(campaign_dir)
    root.mkdir(parents=True, exist_ok=True)
    if len(companies) != expected_company_count:
        raise ValueError(
            f"Benchmark corpus must contain exactly {expected_company_count} companies."
        )

    company_ids = [str(item.get("company_id") or "").strip() for item in companies]
    if any(not value for value in company_ids):
        raise ValueError("Every corpus item must include company_id.")
    schedule = build_run_schedule(company_ids, repeats=repeats, seed=seed)

    tracked_files: list[Path] = []
    corpus_entries: list[dict[str, Any]] = []
    for item in companies:
        company_id = str(item["company_id"]).strip()
        company = item.get("company")
        chunks = item.get("chunks")
        if not isinstance(company, dict) or not str(company.get("name") or "").strip():
            raise ValueError(f"Corpus company {company_id} has no trustworthy name.")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"Corpus company {company_id} has no evidence chunks.")

        relative_dir = Path("corpus") / _safe_component(company_id)
        company_path = root / relative_dir / "company.json"
        chunks_path = root / relative_dir / "chunks.json"
        _write_json(company_path, company)
        _write_json(chunks_path, chunks)
        tracked_files.extend((company_path, chunks_path))
        corpus_entries.append(
            {
                "company_id": company_id,
                "company_name": str(company["name"]).strip(),
                "source_job_id": item.get("source_job_id"),
                "input_mode": item.get("input_mode"),
                "company_path": company_path.relative_to(root).as_posix(),
                "chunks_path": chunks_path.relative_to(root).as_posix(),
                "chunk_count": len(chunks),
            }
        )

    from agent.llm_catalog import find_model_entry

    profile_models = (
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-5.4-nano"),
        ("openrouter", "moonshotai/kimi-k2.6"),
        ("openrouter", "z-ai/glm-5.2"),
        ("openrouter", "deepseek/deepseek-v4-flash"),
    )
    pricing_snapshot: dict[str, dict[str, float] | None] = {}
    for provider, model in profile_models:
        entry = find_model_entry(provider, model)
        pricing_snapshot[f"{provider}:{model}"] = (
            {
                "input_per_million_tokens_usd": entry.pricing.input_per_million_tokens_usd,
                "output_per_million_tokens_usd": entry.pricing.output_per_million_tokens_usd,
            }
            if entry is not None and entry.pricing is not None
            else None
        )

    config = {
        "profiles": list(PRIMARY_PROFILE_IDS),
        "repeats": repeats,
        "seed": seed,
        "live_web_search": False,
        "live_specter_mcp": False,
        "sequential": True,
        "outer_llm_retries": 1,
        "inner_llm_retries": 0,
        "pricing_snapshot": pricing_snapshot,
        "schedule": schedule,
    }
    config_path = root / "campaign.json"
    _write_json(config_path, config)
    tracked_files.append(config_path)

    file_manifest = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(tracked_files)
    ]
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "company_count": len(corpus_entries),
        "run_count": len(schedule),
        "live_web_search": False,
        "live_specter_mcp": False,
        "companies": corpus_entries,
        "files": file_manifest,
        "approval": {
            "status": "pending",
            "approved_by": None,
            "approved_at": None,
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


class _RecordingSpecterClient:
    """Record the raw result of every MCP data call without changing its behavior."""

    _RECORDED_METHODS = {
        "find_company",
        "get_company_profile",
        "get_company_intelligence",
        "get_company_financials",
        "get_person_profile",
        "search_people",
    }

    def __init__(self, client: Any):
        self._client = client
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _serializable(value: Any) -> Any:
        return json.loads(json.dumps(value, default=str, ensure_ascii=True))

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._client, name)
        if name not in self._RECORDED_METHODS or not callable(target):
            return target

        def _recorded(*args: Any, **kwargs: Any) -> Any:
            event = {
                "method": name,
                "arguments": self._serializable({"args": args, "kwargs": kwargs}),
            }
            try:
                response = target(*args, **kwargs)
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                self.calls.append(event)
                raise
            event["response"] = self._serializable(response)
            self.calls.append(event)
            return response

        return _recorded


class _CachedSpecterCaptureClient:
    """Replay frozen Specter calls and delegate only missing calls to a live client."""

    def __init__(self, capture: dict[str, Any], live_client: Any):
        self._calls = list(capture.get("calls") or [])
        self._live_client = live_client

    @staticmethod
    def _arguments_match(
        event: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        recorded = event.get("arguments") or {}
        return (
            recorded.get("args") == json.loads(json.dumps(args, default=str))
            and recorded.get("kwargs") == json.loads(json.dumps(kwargs, default=str))
        )

    def _replay_or_call(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        for event in self._calls:
            if (
                event.get("method") == method
                and not event.get("error")
                and "response" in event
                and self._arguments_match(event, args, kwargs)
            ):
                return json.loads(json.dumps(event["response"]))
        return getattr(self._live_client, method)(*args, **kwargs)

    def find_company(self, identifier: str) -> dict[str, Any]:
        return self._replay_or_call("find_company", identifier)

    def get_company_profile(self, company_id: str) -> dict[str, Any]:
        return self._replay_or_call("get_company_profile", company_id)

    def get_company_intelligence(self, company_id: str) -> dict[str, Any]:
        return self._replay_or_call("get_company_intelligence", company_id)

    def get_company_financials(self, company_id: str) -> dict[str, Any]:
        return self._replay_or_call("get_company_financials", company_id)

    def get_person_profile(self, person_id: str) -> dict[str, Any]:
        return self._replay_or_call("get_person_profile", person_id)

    def search_people(self, query: str, *, limit: int = 20) -> dict[str, Any]:
        return self._replay_or_call("search_people", query, limit=limit)


def capture_specter_corpus(
    campaign_dir: str | Path,
    companies: Sequence[dict[str, Any]],
    *,
    client: Any | None = None,
    seed: int = DEFAULT_CAMPAIGN_SEED,
    expected_company_count: int = 6,
) -> dict[str, Any]:
    """Call Specter once per selected company, including deep leadership profiles.

    Raw MCP responses and their normalized Company/EvidenceStore projections are
    frozen under the ignored campaign directory. All model arms subsequently
    consume only the normalized frozen projection; no campaign run calls Specter.
    """
    if len(companies) != expected_company_count:
        raise ValueError(
            f"Benchmark corpus must contain exactly {expected_company_count} companies."
        )
    from dataclasses import asdict

    from agent.ingest.specter_mcp_client import (
        fetch_specter_company,
        get_default_client,
    )

    root = Path(campaign_dir)
    resolved_client = client
    frozen_companies: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in companies:
        company_id = str(item.get("company_id") or "").strip()
        identifier = str(item.get("identifier") or "").strip()
        if not company_id or not identifier:
            raise ValueError("Every Specter capture item requires company_id and identifier.")
        if company_id in seen_ids:
            raise ValueError("Specter capture company_ids must be unique.")
        seen_ids.add(company_id)
        raw_path = root / "raw-specter" / f"{_safe_component(company_id)}.json"
        capture: dict[str, Any] | None = None
        reusable_capture: dict[str, Any] | None = None
        if raw_path.is_file():
            try:
                candidate = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                candidate = None
            if (
                isinstance(candidate, dict)
                and candidate.get("company_id") == company_id
                and candidate.get("identifier") == identifier
                and candidate.get("fetch_full_team") is True
                and isinstance(candidate.get("company"), dict)
                and str(candidate["company"].get("name") or "").strip()
                and isinstance(candidate.get("chunks"), list)
                and bool(candidate["chunks"])
                and isinstance(candidate.get("calls"), list)
                and not any(call.get("error") for call in candidate["calls"])
            ):
                reusable_capture = candidate
                search_calls = [
                    call
                    for call in candidate["calls"]
                    if call.get("method") == "search_people"
                ]
                if (
                    candidate.get("team_capture_version")
                    == SPECTER_TEAM_CAPTURE_VERSION
                    and candidate.get("leadership_search_performed") is True
                    and len(search_calls) == 1
                ):
                    capture = candidate

        if capture is None:
            if resolved_client is None:
                resolved_client = get_default_client()
            recorder = _RecordingSpecterClient(resolved_client)
            capture_client: Any = recorder
            prior_calls: list[dict[str, Any]] = []
            if reusable_capture is not None:
                capture_client = _CachedSpecterCaptureClient(
                    reusable_capture,
                    recorder,
                )
                prior_calls = list(reusable_capture["calls"])
            company, store = fetch_specter_company(
                identifier,
                expected_name=str(item.get("expected_name") or "").strip() or None,
                fetch_full_team=True,
                client=capture_client,
            )
            calls = prior_calls + recorder.calls
            capture = {
                "company_id": company_id,
                "identifier": identifier,
                "fetch_full_team": True,
                "team_capture_version": SPECTER_TEAM_CAPTURE_VERSION,
                "leadership_search_performed": any(
                    call.get("method") == "search_people" and not call.get("error")
                    for call in calls
                ),
                "calls": calls,
                "company": company.model_dump(),
                "chunks": [asdict(chunk) for chunk in store.chunks],
            }
            _write_json(raw_path, capture)

        calls = capture["calls"]
        leadership_search_calls = sum(
            1 for call in calls if call.get("method") == "search_people"
        )
        leadership_search_errors = sum(
            1
            for call in calls
            if call.get("method") == "search_people" and call.get("error")
        )
        person_profile_errors = sum(
            1
            for call in calls
            if call.get("method") == "get_person_profile" and call.get("error")
        )
        person_profile_calls = sum(
            1 for call in calls if call.get("method") == "get_person_profile"
        )
        leadership_members = [
            {
                "name": str(person.get("name") or "").strip(),
                "title": str(person.get("title") or "").strip() or None,
            }
            for person in (capture["company"].get("team") or [])
            if isinstance(person, dict) and str(person.get("name") or "").strip()
        ]
        profile_coverage_complete = person_profile_calls >= len(leadership_members)
        if leadership_search_calls != 1 or leadership_search_errors:
            raise RuntimeError(
                f"Specter deep-team capture for {company_id} did not complete "
                "one successful leadership search."
            )
        if person_profile_errors:
            raise RuntimeError(
                f"Specter deep-team capture for {company_id} had "
                f"{person_profile_errors} person-profile error(s)."
            )
        raw_files.append(
            {
                "company_id": company_id,
                "identifier": identifier,
                "path": raw_path.relative_to(root).as_posix(),
                "size_bytes": raw_path.stat().st_size,
                "sha256": _sha256(raw_path),
                "mcp_call_count": len(calls),
                "leadership_search_calls": leadership_search_calls,
                "leadership_search_errors": leadership_search_errors,
                "leadership_member_count": len(leadership_members),
                "leadership_members": leadership_members,
                "person_profile_calls": person_profile_calls,
                "person_profile_errors": person_profile_errors,
                "profile_coverage_complete": profile_coverage_complete,
            }
        )
        frozen_companies.append(
            {
                "company_id": company_id,
                "source_job_id": None,
                "input_mode": "specter_frozen",
                "company": capture["company"],
                "chunks": capture["chunks"],
            }
        )

    manifest = freeze_corpus(
        root,
        frozen_companies,
        seed=seed,
        repeats=1,
        expected_company_count=expected_company_count,
    )
    manifest["specter_capture_once"] = True
    manifest["fetch_full_team"] = True
    manifest["specter_team_capture_version"] = SPECTER_TEAM_CAPTURE_VERSION
    manifest["leadership_search_complete"] = all(
        item["leadership_search_calls"] == 1
        and item["leadership_search_errors"] == 0
        for item in raw_files
    )
    manifest["deep_team_complete"] = all(
        item["person_profile_errors"] == 0
        and item["profile_coverage_complete"]
        for item in raw_files
    ) and manifest["leadership_search_complete"]
    manifest["raw_specter_responses"] = raw_files
    manifest["files"].extend(
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in raw_files
    )
    manifest["files"] = sorted(manifest["files"], key=lambda item: item["path"])
    _write_json(root / "manifest.json", manifest)
    return manifest


def prepare_staging_corpus(
    campaign_dir: str | Path,
    *,
    db_module: Any,
    base_job_id: str = "f20aa510",
    seed: int = DEFAULT_CAMPAIGN_SEED,
) -> dict[str, Any]:
    """Read a staging corpus from Supabase and freeze it without mutations."""
    if not db_module or not db_module.is_configured():
        raise RuntimeError("Staging Supabase is not configured.")

    recent = list(db_module.admin_get_recent_analyses(1000) or [])
    base_rows: list[dict[str, Any]] = []
    seen_base: set[str] = set()
    for row in recent:
        company_id = str(row.get("company_id") or "").strip()
        if (
            row.get("job_id_legacy") == base_job_id
            and str(row.get("status") or "").lower() == "done"
            and company_id
            and company_id not in seen_base
        ):
            seen_base.add(company_id)
            base_rows.append(row)
    if len(base_rows) != 10:
        raise RuntimeError(
            f"Expected 10 unique completed companies in staging job {base_job_id}; found {len(base_rows)}."
        )

    selected_rows = list(base_rows)
    selected_company_ids = set(seen_base)
    extras: dict[str, dict[str, Any]] = {}
    for mode in ("pitchdeck", "specter"):
        for row in recent:
            run_config = row.get("run_config") or {}
            company_id = str(row.get("company_id") or "").strip()
            legacy_job_id = str(row.get("job_id_legacy") or "").strip()
            input_mode = str(run_config.get("input_mode") or "").strip().lower()
            if (
                input_mode == mode
                and str(row.get("status") or "").lower() == "done"
                and company_id
                and company_id not in selected_company_ids
                and legacy_job_id != base_job_id
                # Test fixtures use human-readable job-* identifiers. They can
                # persist in staging when integration tests exercise the DB and
                # must never displace the latest real benchmark candidate.
                and not legacy_job_id.lower().startswith(("job-", "test-"))
            ):
                extras[mode] = row
                selected_rows.append(row)
                selected_company_ids.add(company_id)
                break
        if mode not in extras:
            raise RuntimeError(f"No distinct completed staging {mode} company is available.")

    companies: list[dict[str, Any]] = []
    for row in selected_rows:
        company_id = str(row["company_id"])
        company = db_module.get_company_by_id(company_id)
        chunks = db_module.get_all_company_chunks(company_id)
        if not isinstance(company, dict) or not str(company.get("name") or "").strip():
            raise RuntimeError(f"Staging company {company_id} has no trustworthy identity.")
        if not isinstance(chunks, list) or not chunks:
            raise RuntimeError(f"Staging company {company_id} has no frozen evidence chunks.")
        run_config = row.get("run_config") or {}
        companies.append(
            {
                "company_id": company_id,
                "source_job_id": row.get("job_id_legacy"),
                "input_mode": run_config.get("input_mode"),
                "company": company,
                "chunks": chunks,
            }
        )

    root = Path(campaign_dir)
    manifest = freeze_corpus(
        root,
        companies,
        seed=seed,
        repeats=2,
        expected_company_count=12,
    )
    selection = {
        "base_job_id": base_job_id,
        "extra_jobs": {
            mode: extras[mode].get("job_id_legacy")
            for mode in ("pitchdeck", "specter")
        },
        "company_ids": [str(row["company_id"]) for row in selected_rows],
    }
    selection_path = root / "selection.json"
    _write_json(selection_path, selection)

    source_file_manifest: list[dict[str, Any]] = []
    source_job_ids = sorted(
        {
            str(row.get("job_id_legacy") or "").strip()
            for row in selected_rows
            if str(row.get("job_id_legacy") or "").strip()
        }
    )
    for job_id in source_job_ids:
        for index, source in enumerate(db_module.load_source_files(job_id) or [], start=1):
            storage_path = str(source.get("storage_path") or "").strip()
            if not storage_path:
                continue
            file_name = _safe_component(str(source.get("name") or f"source-{index}"))
            destination = root / "source-files" / _safe_component(job_id) / f"{index:02d}-{file_name}"
            if not db_module.download_source_file_to_path(storage_path, destination):
                raise RuntimeError(f"Could not download staging source file {storage_path}.")
            source_file_manifest.append(
                {
                    "job_id": job_id,
                    "storage_path": storage_path,
                    "path": destination.relative_to(root).as_posix(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )

    manifest["selection"] = selection
    manifest["source_files"] = source_file_manifest
    manifest["files"].append(
        {
            "path": selection_path.relative_to(root).as_posix(),
            "size_bytes": selection_path.stat().st_size,
            "sha256": _sha256(selection_path),
        }
    )
    manifest["files"].extend(
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in source_file_manifest
    )
    manifest["files"] = sorted(manifest["files"], key=lambda item: item["path"])
    _write_json(root / "manifest.json", manifest)
    return manifest


def verify_manifest(
    campaign_dir: str | Path,
    *,
    require_approval: bool = False,
) -> dict[str, Any]:
    """Verify every frozen artifact hash and, optionally, human approval."""
    root = Path(campaign_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Benchmark manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_approval and (manifest.get("approval") or {}).get("status") != "approved":
        raise RuntimeError("Benchmark manifest has not been approved.")
    for item in manifest.get("files") or []:
        relative = Path(str(item.get("path") or ""))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Benchmark manifest contains an unsafe file path.") from exc
        if not candidate.is_file():
            raise RuntimeError(f"Benchmark artifact is missing: {relative.as_posix()}.")
        if _sha256(candidate) != item.get("sha256"):
            raise RuntimeError(f"Benchmark artifact hash mismatch: {relative.as_posix()}.")
    return manifest


def approve_manifest(
    campaign_dir: str | Path,
    *,
    approved_by: str,
) -> dict[str, Any]:
    """Record human approval after re-verifying the frozen manifest."""
    approver = approved_by.strip()
    if not approver:
        raise ValueError("approved_by is required.")
    root = Path(campaign_dir)
    manifest = verify_manifest(root, require_approval=False)
    manifest["approval"] = {
        "status": "approved",
        "approved_by": approver,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def evaluate_candidate_gates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-declared quality, reliability, cost, and speed gates."""
    def _number(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Benchmark metric {key} is missing.")
        return float(value)

    quality_passed = all(
        _number(candidate, key) >= -0.25
        for key in (
            "quality_ci_lower_delta",
            "factual_support_ci_lower_delta",
            "completeness_ci_lower_delta",
        )
    )
    unsupported_passed = (
        _number(candidate, "critical_unsupported_claims")
        <= _number(baseline, "critical_unsupported_claims")
        and candidate.get("new_systemic_omission_class") is False
    )
    ranking_passed = (
        _number(candidate, "ranking_spearman")
        >= _number(baseline, "ranking_spearman") - 0.05
    )
    stability_passed = (
        _number(candidate, "repeat_decision_agreement")
        >= _number(baseline, "repeat_decision_agreement") - 0.10
        and _number(candidate, "score_stddev")
        <= _number(baseline, "score_stddev") * 1.25
    )
    reliability_passed = (
        _number(candidate, "structured_success_rate") >= 0.99
        and _number(candidate, "incomplete_runs") == 0
    )
    cost_passed = (
        _number(candidate, "cost_per_company_usd")
        <= _number(baseline, "cost_per_company_usd") * 0.75
    )
    speed_passed = (
        _number(candidate, "p95_wall_clock_seconds")
        <= _number(baseline, "p95_wall_clock_seconds") * 1.20
    )
    gates = [
        {"id": "quality", "passed": quality_passed},
        {"id": "unsupported_claims", "passed": unsupported_passed},
        {"id": "ranking", "passed": ranking_passed},
        {"id": "repeat_stability", "passed": stability_passed},
        {"id": "structured_reliability", "passed": reliability_passed},
        {"id": "cost", "passed": cost_passed},
        {"id": "speed", "passed": speed_passed},
    ]
    return {"passed": all(item["passed"] for item in gates), "gates": gates}


def _openrouter_smoke_cases() -> list[dict[str, Any]]:
    """Return one synthetic call for every distinct experiment request mode."""
    from agent.model_profiles import resolve_model_profile

    kimi = resolve_model_profile("kimi_k26").policy
    hybrid = resolve_model_profile("glm_deepseek_flash").policy
    return [
        {
            "id": "kimi-thinking-on",
            "model": kimi.decomposition["model"],
            "selection": dict(kimi.decomposition),
            "stage": "decomposition",
            "temperature": 0.5,
        },
        {
            "id": "kimi-thinking-off",
            "model": kimi.answering["model"],
            "selection": dict(kimi.answering),
            "stage": "answering",
            "temperature": 0.2,
        },
        {
            "id": "glm-high-evaluation",
            "model": hybrid.evaluation["model"],
            "selection": dict(hybrid.evaluation),
            "stage": "evaluation",
            "temperature": 0.0,
        },
        {
            "id": "glm-thinking-off-upside",
            "model": hybrid.ranking["model"],
            "selection": dict(hybrid.ranking),
            "stage": "ranking_upside_score",
            "temperature": 0.7,
        },
        {
            "id": "deepseek-thinking-off-answering",
            "model": hybrid.answering["model"],
            "selection": dict(hybrid.answering),
            "stage": "answering",
            "temperature": 0.2,
        },
        {
            "id": "deepseek-high-refinement",
            "model": hybrid.refinement["model"],
            "selection": dict(hybrid.refinement),
            "stage": "refinement",
            "temperature": 0.7,
        },
        {
            "id": "deepseek-pro-high-admin",
            "model": "deepseek/deepseek-v4-pro",
            "selection": {
                "provider": "openrouter",
                "model": "deepseek/deepseek-v4-pro",
                "temperature": None,
                "reasoning_effort": "high",
            },
            "stage": "admin_smoke",
            "temperature": 0.7,
        },
    ]


async def _invoke_openrouter_smoke_case(
    case: dict[str, Any],
    routing: dict[str, Any],
) -> dict[str, Any]:
    from pydantic import BaseModel

    from agent.llm import create_llm
    from agent.run_context import (
        RunTelemetryCollector,
        use_phase_llm,
        use_run_context,
        use_stage_context,
    )

    class SmokeResponse(BaseModel):
        ok: bool
        marker: str

    collector = RunTelemetryCollector()
    marker = f"deal-intelligence-smoke-{case['id']}"
    selection = dict(case["selection"])
    selection["openrouter_routing"] = dict(routing)
    with use_run_context(telemetry_collector=collector, web_search_mode="off"):
        with use_phase_llm(selection), use_stage_context(str(case["stage"])):
            runnable = create_llm(
                temperature=case.get("temperature"),
            ).with_structured_output(SmokeResponse)
            response = await runnable.ainvoke(
                f"Return ok=true and marker='{marker}'. Do not add other fields."
            )
    rows = [
        row
        for row in collector.snapshot_model_executions()
        if row.get("service") == "llm" and row.get("status") == "done"
    ]
    metadata = (rows[-1].get("metadata") or {}) if rows else {}
    return {
        "structured_ok": bool(getattr(response, "ok", False))
        and getattr(response, "marker", None) == marker,
        "selected_provider": metadata.get("selected_provider"),
        "generation_id": metadata.get("generation_id"),
        "actual_cost_usd": metadata.get("actual_cost_usd"),
        "reasoning_tokens": metadata.get("reasoning_tokens"),
        "request_parameters": {
            key: metadata.get(key)
            for key in (
                "requested_temperature",
                "effective_temperature",
                "requested_reasoning_effort",
                "effective_reasoning_effort",
                "requested_reasoning_enabled",
                "effective_reasoning_enabled",
            )
        },
    }


async def run_openrouter_smoke(
    output_dir: str | Path,
    *,
    invoke_case: Any | None = None,
) -> dict[str, Any]:
    """Run small synthetic schema calls using the final B/C request settings."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    routing_policy = {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "only": list(OPENROUTER_APPROVED_PROVIDER_SLUGS),
        "allow_fallbacks": False,
    }
    invoker = invoke_case or _invoke_openrouter_smoke_case
    reports: list[dict[str, Any]] = []
    for case in _openrouter_smoke_cases():
        public_case = {
            key: value for key, value in case.items() if key != "selection"
        }
        try:
            result = invoker(dict(case), dict(routing_policy))
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise RuntimeError("Smoke invoker returned an invalid result.")
            selected_provider = str(result.get("selected_provider") or "").strip()
            eligible = (
                result.get("structured_ok") is True
                and selected_provider.lower() in OPENROUTER_APPROVED_PROVIDER_SLUGS
            )
            reports.append(
                {
                    **public_case,
                    **dict(result),
                    "routing_policy": dict(routing_policy),
                    "eligible": eligible,
                    "error": None,
                }
            )
        except Exception as exc:
            reports.append(
                {
                    **public_case,
                    "routing_policy": dict(routing_policy),
                    "eligible": False,
                    "error": str(exc),
                }
            )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True,
        "eligible": bool(reports) and all(item["eligible"] for item in reports),
        "routing_policy": routing_policy,
        "cases": reports,
    }
    _write_json(root / "smoke.json", report)
    return report


async def _invoke_openrouter_preflight(
    model: str,
    attempt: int,
    routing: dict[str, Any],
) -> dict[str, Any]:
    from pydantic import BaseModel

    from agent.llm import create_llm
    from agent.run_context import RunTelemetryCollector, use_run_context

    class PreflightResponse(BaseModel):
        ok: bool
        marker: str

    collector = RunTelemetryCollector()
    with use_run_context(
        llm_selection={
            "provider": "openrouter",
            "model": model,
            "reasoning_effort": "high",
            "openrouter_routing": dict(routing),
        },
        telemetry_collector=collector,
        web_search_mode="off",
    ):
        runnable = create_llm(temperature=None).with_structured_output(PreflightResponse)
        response = await runnable.ainvoke(
            f"Return ok=true and marker='deal-intelligence-preflight-{attempt}'."
        )
    rows = [
        row
        for row in collector.snapshot_model_executions()
        if row.get("service") == "llm" and row.get("status") == "done"
    ]
    metadata = (rows[-1].get("metadata") or {}) if rows else {}
    return {
        "structured_ok": bool(getattr(response, "ok", False)),
        "selected_provider": metadata.get("selected_provider"),
        "generation_id": metadata.get("generation_id"),
    }


async def run_openrouter_preflight(
    campaign_dir: str | Path,
    *,
    invoke_model: Any | None = None,
    conformance_calls: int = 2,
) -> dict[str, Any]:
    """Verify strict-schema/privacy conformance and freeze one provider per model."""
    if conformance_calls <= 0:
        raise ValueError("conformance_calls must be greater than zero.")
    root = Path(campaign_dir)
    verify_manifest(root, require_approval=True)
    invoker = invoke_model or _invoke_openrouter_preflight
    routing_policy = {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "only": list(OPENROUTER_APPROVED_PROVIDER_SLUGS),
        "allow_fallbacks": False,
    }
    model_reports: list[dict[str, Any]] = []
    provider_pins: dict[str, str] = {}
    for model in OPENROUTER_PREFLIGHT_MODELS:
        calls: list[dict[str, Any]] = []
        errors: list[str] = []
        for attempt in range(1, conformance_calls + 1):
            try:
                result = invoker(model, attempt, dict(routing_policy))
                if inspect.isawaitable(result):
                    result = await result
                if not isinstance(result, dict):
                    raise RuntimeError("Preflight invoker returned an invalid result.")
                calls.append(dict(result))
            except Exception as exc:
                errors.append(str(exc))
        providers = {
            str(call.get("selected_provider") or "").strip()
            for call in calls
            if str(call.get("selected_provider") or "").strip()
        }
        eligible = (
            not errors
            and len(calls) == conformance_calls
            and all(call.get("structured_ok") is True for call in calls)
            and len(providers) == 1
            and next(iter(providers)).lower() in OPENROUTER_APPROVED_PROVIDER_SLUGS
        )
        if eligible:
            provider_pins[model] = next(
                slug
                for slug in OPENROUTER_APPROVED_PROVIDER_SLUGS
                if slug == next(iter(providers)).lower()
            )
        model_reports.append(
            {
                "model": model,
                "eligible": eligible,
                "routing_policy": dict(routing_policy),
                "selected_provider": next(iter(providers)) if len(providers) == 1 else None,
                "calls": calls,
                "errors": errors,
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eligible": all(item["eligible"] for item in model_reports),
        "conformance_calls": conformance_calls,
        "provider_pins": provider_pins,
        "models": model_reports,
    }
    _write_json(root / "preflight.json", report)
    return report


def _pinned_profile_policy(profile_id: str, provider_pins: dict[str, str]):
    from agent.llm_policy import PipelineModelPolicy
    from agent.model_profiles import resolve_model_profile

    resolved = resolve_model_profile(profile_id)
    phase_models = resolved.policy.as_dict()
    for selection in phase_models.values():
        if selection.get("provider") != "openrouter":
            continue
        model = str(selection.get("model") or "")
        pin = str(provider_pins.get(model) or "").strip()
        if not pin:
            raise RuntimeError(f"No compliant provider pin is available for {model}.")
        selection["openrouter_routing"] = {
            "only": [pin],
            "allow_fallbacks": False,
        }
    return PipelineModelPolicy(**phase_models)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


async def _evaluate_frozen_run(
    *,
    company_id: str,
    company: dict[str, Any],
    chunks: list[dict[str, Any]],
    policy: Any,
    corpus_dir: Path,
    **_: Any,
) -> dict[str, Any]:
    import agent.rate_limit as rate_limit_module
    from agent.batch import evaluate_startup
    from agent.dataclasses.company import Company
    from agent.ingest.store import Chunk, EvidenceStore
    from agent.rate_limit import RetryPolicy
    from agent.run_context import RunTelemetryCollector, use_run_context

    evidence_store = EvidenceStore(
        startup_slug=company_id,
        chunks=[
            Chunk(
                chunk_id=str(item.get("chunk_id") or f"chunk-{index}"),
                text=str(item.get("text") or ""),
                source_file=str(item.get("source_file") or "frozen-evidence"),
                page_or_slide=str(item.get("page_or_slide") or "N/A"),
            )
            for index, item in enumerate(chunks, start=1)
            if str(item.get("text") or "").strip()
        ],
    )
    initial_company = Company.model_validate(company)
    collector = RunTelemetryCollector()
    previous_retry_policy = rate_limit_module._LLM_RETRY_POLICY
    previous_inner_retries = os.getenv("LLM_CLIENT_MAX_RETRIES")
    rate_limit_module._LLM_RETRY_POLICY = RetryPolicy(
        max_retries=1,
        base_delay_sec=previous_retry_policy.base_delay_sec,
        max_delay_sec=previous_retry_policy.max_delay_sec,
        jitter_sec=previous_retry_policy.jitter_sec,
    )
    os.environ["LLM_CLIENT_MAX_RETRIES"] = "0"
    started = time.monotonic()
    try:
        with use_run_context(
            llm_selection=dict(policy.answering),
            pipeline_policy=policy,
            telemetry_collector=collector,
            web_search_mode="off",
        ):
            result = await evaluate_startup(
                corpus_dir,
                use_web_search=False,
                initial_store=evidence_store,
                initial_company=initial_company,
            )
    finally:
        rate_limit_module._LLM_RETRY_POLICY = previous_retry_policy
        if previous_inner_retries is None:
            os.environ.pop("LLM_CLIENT_MAX_RETRIES", None)
        else:
            os.environ["LLM_CLIENT_MAX_RETRIES"] = previous_inner_retries

    rows = collector.snapshot_model_executions()
    final_state = result.get("final_state") or {}
    final_arguments = list(final_state.get("final_arguments") or [])
    final_arguments.sort(
        key=lambda argument: getattr(argument, "score", None) or float("-inf"),
        reverse=True,
    )
    return {
        "status": "done" if not result.get("skipped") else "incomplete",
        "structured_success": not bool(result.get("skipped")),
        "wall_clock_seconds": round(time.monotonic() - started, 6),
        "final_decision": final_state.get("final_decision", "unknown"),
        "ranking_result": _jsonable(final_state.get("ranking_result")),
        "questions_answers": _jsonable(final_state.get("all_qa_pairs") or []),
        "final_arguments": _jsonable(final_arguments),
        "top_final_arguments": _jsonable(final_arguments[:5]),
        "run_costs": collector.build_run_costs(),
        "model_executions": _jsonable(rows),
        "retry_count": (
            sum(1 for row in rows if row.get("status") == "retrying")
            + sum(
                int((row.get("metadata") or {}).get("provider_retry_count") or 0)
                for row in rows
                if row.get("service") == "llm"
            )
        ),
        "output_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 6)


def _profile_run_summary(profile_id: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == "done"]
    wall_times = [float(run.get("wall_clock_seconds") or 0.0) for run in completed]
    costs = [
        float((run.get("run_costs") or {}).get("total_usd"))
        for run in completed
        if isinstance((run.get("run_costs") or {}).get("total_usd"), (int, float))
    ]
    stage_latencies: dict[str, list[float]] = {}
    for run in completed:
        for row in run.get("model_executions") or []:
            if row.get("status") != "done" or row.get("service") not in {"pipeline_stage", "llm"}:
                continue
            latency = row.get("latency_ms")
            if isinstance(latency, (int, float)):
                stage_latencies.setdefault(str(row.get("stage") or "unknown"), []).append(float(latency))
    return {
        "profile_id": profile_id,
        "run_count": len(runs),
        "completed_runs": len(completed),
        "incomplete_runs": len(runs) - len(completed),
        "structured_success_rate": round(
            sum(1 for run in runs if run.get("structured_success") is True) / max(1, len(runs)),
            6,
        ),
        "cost_per_company_usd": round(statistics.mean(costs), 8) if costs else None,
        "total_cost_usd": round(sum(costs), 8) if costs else None,
        "median_wall_clock_seconds": _percentile(wall_times, 0.50),
        "p95_wall_clock_seconds": _percentile(wall_times, 0.95),
        "retry_count": sum(int(run.get("retry_count") or 0) for run in runs),
        "output_tokens": sum(int(run.get("output_tokens") or 0) for run in runs),
        "stage_latency_ms": {
            stage: {
                "median": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for stage, values in sorted(stage_latencies.items())
        },
    }


def _write_summary_csv(path: Path, profiles: list[dict[str, Any]]) -> None:
    fields = (
        "profile_id",
        "run_count",
        "completed_runs",
        "incomplete_runs",
        "structured_success_rate",
        "cost_per_company_usd",
        "total_cost_usd",
        "median_wall_clock_seconds",
        "p95_wall_clock_seconds",
        "retry_count",
        "output_tokens",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            writer.writerow({field: profile.get(field) for field in fields})


def _write_blinded_review_bundle(
    root: Path,
    runs: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    codes = ["Variant Alder", "Variant Birch", "Variant Cedar"]
    rng = random.Random(seed)
    rng.shuffle(codes)
    blind_map = dict(zip(PRIMARY_PROFILE_IDS, codes, strict=True))
    _write_json(root / "blinding-key.json", {"profile_to_variant": blind_map})
    bundle_dir = root / "review-bundle"
    output_dir = bundle_dir / "outputs"
    review_rows: list[dict[str, Any]] = []
    for run in runs:
        variant = blind_map[run["profile_id"]]
        output_payload = {
            "company_id": run["company_id"],
            "repeat": run["repeat"],
            "variant": variant,
            "final_decision": run.get("final_decision"),
            "ranking_result": run.get("ranking_result"),
            "questions_answers": run.get("questions_answers"),
            "final_arguments": run.get("final_arguments"),
            "top_final_arguments": run.get("top_final_arguments"),
        }
        file_name = (
            f"{_safe_component(run['company_id'])}__r{run['repeat']}__"
            f"{_safe_component(variant)}.json"
        )
        _write_json(output_dir / file_name, output_payload)
        review_rows.append(
            {
                "company_id": run["company_id"],
                "repeat": run["repeat"],
                "variant": variant,
                "reviewer": "",
                "factual_support": "",
                "material_completeness": "",
                "investment_insight": "",
                "pro_con_balance": "",
                "ranking_calibration": "",
                "company_rank": "",
                "critical_unsupported_claim": "",
                "systemic_omission_class": "",
                "notes": "",
            }
        )
    fields = list(review_rows[0]) if review_rows else []
    for reviewer_index in (1, 2):
        path = bundle_dir / f"reviewer-{reviewer_index}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(review_rows)
    return blind_map


async def run_campaign(
    campaign_dir: str | Path,
    *,
    evaluate_run: Any | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Run the frozen schedule sequentially without application persistence."""
    root = Path(campaign_dir)
    manifest = verify_manifest(root, require_approval=True)
    campaign = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    if profile_id is not None and profile_id not in PRIMARY_PROFILE_IDS:
        raise ValueError("Unknown primary benchmark profile.")
    preflight_path = root / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("OpenRouter preflight must complete before the campaign.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("eligible") is not True:
        raise RuntimeError(
            "Every staging OpenRouter model must pass the strict schema and privacy preflight."
        )
    provider_pins = dict(preflight.get("provider_pins") or {})
    required_openrouter_models = {
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v4-flash",
    }
    if not required_openrouter_models.issubset(provider_pins):
        raise RuntimeError("A primary experiment arm lacks a compliant provider pin.")

    companies: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("companies") or []:
        company_id = str(entry["company_id"])
        companies[company_id] = {
            "company": json.loads((root / entry["company_path"]).read_text(encoding="utf-8")),
            "chunks": json.loads((root / entry["chunks_path"]).read_text(encoding="utf-8")),
            "corpus_dir": (root / entry["company_path"]).parent,
        }

    evaluator = evaluate_run or _evaluate_frozen_run
    run_dir = root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule = list(campaign.get("schedule") or [])
    selected_schedule = [
        item for item in schedule
        if profile_id is None or item.get("profile_id") == profile_id
    ]
    invocation_started_at = datetime.now(timezone.utc).isoformat()
    invocation_started = time.monotonic()
    executed_sequences: list[int] = []
    for item in selected_schedule:
        company_id = str(item["company_id"])
        item_profile_id = str(item["profile_id"])
        run_path = run_dir / f"{int(item['sequence']):03d}-{_safe_component(company_id)}-{item_profile_id}.json"
        if run_path.is_file():
            continue
        policy = _pinned_profile_policy(item_profile_id, provider_pins)
        frozen = companies[company_id]
        try:
            result = evaluator(
                company_id=company_id,
                company=frozen["company"],
                chunks=frozen["chunks"],
                policy=policy,
                corpus_dir=frozen["corpus_dir"],
                schedule_item=dict(item),
            )
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise RuntimeError("Benchmark evaluator returned an invalid result.")
            report = {**dict(item), **_jsonable(result)}
        except Exception as exc:
            report = {
                **dict(item),
                "status": "error",
                "structured_success": False,
                "error": str(exc),
                "wall_clock_seconds": None,
                "run_costs": None,
                "model_executions": [],
                "retry_count": 0,
                "output_tokens": 0,
            }
        _write_json(run_path, report)
        executed_sequences.append(int(item["sequence"]))

    invocation_wall_clock_seconds = round(time.monotonic() - invocation_started, 6)
    invocation_finished_at = datetime.now(timezone.utc).isoformat()
    invocation_label = profile_id or "all-profiles"
    profile_batch_path = root / "profile-batches" / f"{_safe_component(invocation_label)}.json"
    if executed_sequences or not profile_batch_path.is_file():
        _write_json(
            profile_batch_path,
            {
                "profile_id": profile_id,
                "started_at": invocation_started_at,
                "finished_at": invocation_finished_at,
                "wall_clock_seconds": invocation_wall_clock_seconds,
                "executed_sequences": executed_sequences,
                "executed_run_count": len(executed_sequences),
            },
        )

    run_reports: list[dict[str, Any]] = []
    for item in schedule:
        run_path = run_dir / (
            f"{int(item['sequence']):03d}-{_safe_component(str(item['company_id']))}-"
            f"{item['profile_id']}.json"
        )
        if run_path.is_file():
            run_reports.append(json.loads(run_path.read_text(encoding="utf-8")))

    profile_summaries = [
        _profile_run_summary(
            profile_id,
            [run for run in run_reports if run.get("profile_id") == profile_id],
        )
        for profile_id in PRIMARY_PROFILE_IDS
    ]
    complete_campaign = len(run_reports) == len(schedule)
    blind_map = (
        _write_blinded_review_bundle(
            root,
            run_reports,
            seed=int(campaign.get("seed") or DEFAULT_CAMPAIGN_SEED),
        )
        if complete_campaign
        else {}
    )
    profile_batch_times: dict[str, float] = {}
    for candidate_profile in PRIMARY_PROFILE_IDS:
        path = root / "profile-batches" / f"{candidate_profile}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("wall_clock_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            profile_batch_times[candidate_profile] = float(value)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_run_count": len(schedule),
        "run_count": len(run_reports),
        "completed_runs": sum(1 for run in run_reports if run.get("status") == "done"),
        "sequential": True,
        "live_web_search": False,
        "live_specter_mcp": False,
        "profiles": profile_summaries,
        "blinded_variants": sorted(blind_map.values()),
        "profile_batch_wall_clock_seconds": profile_batch_times,
        "invocation_profile_id": profile_id,
        "invocation_wall_clock_seconds": invocation_wall_clock_seconds,
        "invocation_executed_runs": len(executed_sequences),
    }
    _write_json(root / "summary.json", summary)
    _write_summary_csv(root / "summary.csv", profile_summaries)
    return summary


def _parse_score(value: Any, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Review field {field} must be scored from 1 to 5.") from exc
    if not 1.0 <= score <= 5.0:
        raise ValueError(f"Review field {field} must be scored from 1 to 5.")
    return score


def _parse_bool(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError("Boolean review fields must use yes/no or true/false.")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paired_cluster_bootstrap_lower(
    baseline: dict[tuple[str, int], float],
    candidate: dict[tuple[str, int], float],
    *,
    seed: int = DEFAULT_CAMPAIGN_SEED,
    samples: int = 5000,
) -> float:
    common = sorted(set(baseline) & set(candidate))
    by_company: dict[str, list[float]] = {}
    for key in common:
        by_company.setdefault(key[0], []).append(candidate[key] - baseline[key])
    cluster_diffs = [_mean(values) for values in by_company.values()]
    if not cluster_diffs:
        raise ValueError("No paired company review scores are available.")
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        sample = [rng.choice(cluster_diffs) for _ in cluster_diffs]
        bootstrapped.append(_mean(sample))
    return round(sorted(bootstrapped)[max(0, int(samples * 0.025) - 1)], 6)


def _rank_values(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for cursor in range(index, end):
            ranks[ordered[cursor][0]] = average_rank
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _rank_values(left)
    right_ranks = _rank_values(right)
    left_mean = _mean(left_ranks)
    right_mean = _mean(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left_ranks)
        * sum((b - right_mean) ** 2 for b in right_ranks)
    )
    return round(numerator / denominator, 6) if denominator else 0.0


def select_recommendation(profile_metrics: dict[str, dict[str, Any]]) -> str:
    """Apply the predeclared preference rule after candidate hard gates pass."""
    kimi = profile_metrics.get("kimi_k26") or {}
    hybrid = profile_metrics.get("glm_deepseek_flash") or {}
    if not kimi.get("passed") and not hybrid.get("passed"):
        return "gpt_current"
    if kimi.get("passed") and not hybrid.get("passed"):
        return "kimi_k26"
    if hybrid.get("passed") and not kimi.get("passed"):
        return "glm_deepseek_flash"
    kimi_quality = float(kimi.get("overall_quality_mean") or 0.0)
    hybrid_quality = float(hybrid.get("overall_quality_mean") or 0.0)
    kimi_cost = float(kimi.get("cost_per_company_usd") or 0.0)
    hybrid_cost = float(hybrid.get("cost_per_company_usd") or 0.0)
    relative_cost_difference = (
        abs(kimi_cost - hybrid_cost) / max(kimi_cost, hybrid_cost)
        if max(kimi_cost, hybrid_cost) > 0
        else 0.0
    )
    if abs(kimi_quality - hybrid_quality) <= 0.25 and relative_cost_difference <= 0.10:
        return "kimi_k26"
    if hybrid_quality > kimi_quality + 0.25 or hybrid_cost < kimi_cost * 0.90:
        return "glm_deepseek_flash"
    return "gpt_current"


def evaluate_reviews(
    campaign_dir: str | Path,
    *,
    adjudication_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate blinded reviews, adjudicate material conflicts, then apply gates."""
    root = Path(campaign_dir)
    verify_manifest(root, require_approval=True)
    score_fields = (
        "factual_support",
        "material_completeness",
        "investment_insight",
        "pro_con_balance",
        "ranking_calibration",
    )

    reviewer_rows: list[dict[tuple[str, int, str], dict[str, Any]]] = []
    for reviewer_index in (1, 2):
        path = root / "review-bundle" / f"reviewer-{reviewer_index}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        parsed: dict[tuple[str, int, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["company_id"]), int(row["repeat"]), str(row["variant"]))
            parsed[key] = {
                **{field: _parse_score(row.get(field), field) for field in score_fields},
                "company_rank": float(row.get("company_rank") or 0),
                "critical_unsupported_claim": _parse_bool(row.get("critical_unsupported_claim")),
                "systemic_omission_class": str(row.get("systemic_omission_class") or "").strip(),
            }
            if not 1 <= parsed[key]["company_rank"] <= 12:
                raise ValueError("company_rank must be between 1 and 12.")
        reviewer_rows.append(parsed)
    if set(reviewer_rows[0]) != set(reviewer_rows[1]):
        raise ValueError("Reviewer files do not contain the same blinded outputs.")

    disputes: list[tuple[str, int, str]] = []
    for key in reviewer_rows[0]:
        left, right = reviewer_rows[0][key], reviewer_rows[1][key]
        if (
            any(abs(left[field] - right[field]) > 1.0 for field in score_fields)
            or left["critical_unsupported_claim"] != right["critical_unsupported_claim"]
        ):
            disputes.append(key)

    adjudicated: dict[tuple[str, int, str], dict[str, Any]] = {}
    if disputes and adjudication_path:
        with Path(adjudication_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["company_id"]), int(row["repeat"]), str(row["variant"]))
                adjudicated[key] = {
                    **{field: _parse_score(row.get(field), field) for field in score_fields},
                    "company_rank": float(row.get("company_rank") or 0),
                    "critical_unsupported_claim": _parse_bool(row.get("critical_unsupported_claim")),
                    "systemic_omission_class": str(row.get("systemic_omission_class") or "").strip(),
                }
    unresolved = [key for key in disputes if key not in adjudicated]
    if unresolved:
        report = {
            "status": "adjudication_required",
            "unresolved_count": len(unresolved),
            "unresolved": [list(key) for key in unresolved],
        }
        _write_json(root / "review-status.json", report)
        return report

    consensus: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key in reviewer_rows[0]:
        if key in adjudicated:
            consensus[key] = adjudicated[key]
            continue
        left, right = reviewer_rows[0][key], reviewer_rows[1][key]
        consensus[key] = {
            **{field: _mean([left[field], right[field]]) for field in score_fields},
            "company_rank": _mean([left["company_rank"], right["company_rank"]]),
            "critical_unsupported_claim": left["critical_unsupported_claim"] or right["critical_unsupported_claim"],
            "systemic_omission_class": left["systemic_omission_class"] or right["systemic_omission_class"],
        }

    blind_key = json.loads((root / "blinding-key.json").read_text(encoding="utf-8"))
    variant_to_profile = {
        variant: profile
        for profile, variant in (blind_key.get("profile_to_variant") or {}).items()
    }
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "runs").glob("*.json"))
    ]
    run_by_key = {
        (str(run["company_id"]), int(run["repeat"]), str(run["profile_id"])): run
        for run in runs
    }
    review_by_profile: dict[str, list[tuple[tuple[str, int], dict[str, Any]]]] = {
        profile: [] for profile in PRIMARY_PROFILE_IDS
    }
    for (company_id, repeat, variant), scores in consensus.items():
        profile = variant_to_profile[variant]
        scores = dict(scores)
        scores["overall_quality"] = _mean([scores[field] for field in score_fields])
        review_by_profile[profile].append(((company_id, repeat), scores))

    automated_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    automated_by_profile = {
        item["profile_id"]: item for item in automated_summary.get("profiles") or []
    }
    profile_metrics: dict[str, dict[str, Any]] = {}
    for profile_id, entries in review_by_profile.items():
        reviews = {key: scores for key, scores in entries}
        ranking_left: list[float] = []
        ranking_right: list[float] = []
        for (company_id, repeat), scores in reviews.items():
            ranking = (run_by_key.get((company_id, repeat, profile_id)) or {}).get("ranking_result") or {}
            composite = ranking.get("composite_score")
            if isinstance(composite, (int, float)):
                ranking_left.append(float(composite))
                ranking_right.append(-float(scores["company_rank"]))
        decisions_by_company: dict[str, list[str]] = {}
        scores_by_company: dict[str, list[float]] = {}
        for run in runs:
            if run.get("profile_id") != profile_id:
                continue
            company_id = str(run["company_id"])
            decisions_by_company.setdefault(company_id, []).append(str(run.get("final_decision") or "unknown"))
            composite = (run.get("ranking_result") or {}).get("composite_score")
            if isinstance(composite, (int, float)):
                scores_by_company.setdefault(company_id, []).append(float(composite))
        profile_metrics[profile_id] = {
            **automated_by_profile.get(profile_id, {}),
            "overall_quality_mean": round(_mean([score["overall_quality"] for score in reviews.values()]), 6),
            "factual_support_mean": round(_mean([score["factual_support"] for score in reviews.values()]), 6),
            "completeness_mean": round(_mean([score["material_completeness"] for score in reviews.values()]), 6),
            "critical_unsupported_claims": sum(1 for score in reviews.values() if score["critical_unsupported_claim"]),
            "systemic_omission_classes": sorted({score["systemic_omission_class"] for score in reviews.values() if score["systemic_omission_class"]}),
            "ranking_spearman": _spearman(ranking_left, ranking_right),
            "repeat_decision_agreement": round(
                _mean([1.0 if len(set(values)) == 1 else 0.0 for values in decisions_by_company.values()]),
                6,
            ),
            "score_stddev": round(
                _mean([statistics.pstdev(values) for values in scores_by_company.values() if len(values) > 1]),
                6,
            ),
        }

    baseline_scores = {key: score for key, score in review_by_profile["gpt_current"]}
    baseline = profile_metrics["gpt_current"]
    for candidate_id in ("kimi_k26", "glm_deepseek_flash"):
        candidate_scores = {key: score for key, score in review_by_profile[candidate_id]}
        candidate = profile_metrics[candidate_id]
        candidate.update(
            {
                "quality_ci_lower_delta": _paired_cluster_bootstrap_lower(
                    {key: value["overall_quality"] for key, value in baseline_scores.items()},
                    {key: value["overall_quality"] for key, value in candidate_scores.items()},
                ),
                "factual_support_ci_lower_delta": _paired_cluster_bootstrap_lower(
                    {key: value["factual_support"] for key, value in baseline_scores.items()},
                    {key: value["factual_support"] for key, value in candidate_scores.items()},
                ),
                "completeness_ci_lower_delta": _paired_cluster_bootstrap_lower(
                    {key: value["material_completeness"] for key, value in baseline_scores.items()},
                    {key: value["material_completeness"] for key, value in candidate_scores.items()},
                ),
                "new_systemic_omission_class": bool(
                    set(candidate["systemic_omission_classes"])
                    - set(baseline["systemic_omission_classes"])
                ),
            }
        )
        gate_report = evaluate_candidate_gates(baseline, candidate)
        candidate["passed"] = gate_report["passed"]
        candidate["gates"] = gate_report["gates"]

    decision = {
        "status": "complete",
        "profiles": profile_metrics,
        "recommendation": select_recommendation(profile_metrics),
        "production_promoted": False,
        "requires_explicit_approval": True,
    }
    _write_json(root / "decision.json", decision)
    return decision


def _require_staging() -> None:
    if os.getenv("APP_ENV", "").strip().lower() != "staging":
        raise RuntimeError("Model experiment commands must run with APP_ENV=staging.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark lifecycle commands."""
    parser = argparse.ArgumentParser(
        description="Prepare and run the staging OpenRouter A/B/C campaign.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze the read-only staging corpus.")
    prepare.add_argument("--campaign-dir", required=True)
    prepare.add_argument("--base-job-id", default="f20aa510")
    prepare.add_argument("--seed", type=int, default=DEFAULT_CAMPAIGN_SEED)

    approve = subparsers.add_parser("approve", help="Approve a verified frozen manifest.")
    approve.add_argument("--campaign-dir", required=True)
    approve.add_argument("--approved-by", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run synthetic OpenRouter checks with the final B/C request settings.",
    )
    smoke.add_argument("--output-dir", required=True)

    preflight = subparsers.add_parser("preflight", help="Verify strict ZDR-capable provider routes.")
    preflight.add_argument("--campaign-dir", required=True)
    preflight.add_argument("--conformance-calls", type=int, default=2)

    run = subparsers.add_parser("run", help="Run or resume the 18-run sequential campaign.")
    run.add_argument("--campaign-dir", required=True)
    run.add_argument("--profile", choices=PRIMARY_PROFILE_IDS)

    evaluate = subparsers.add_parser("evaluate", help="Adjudicate reviews and apply promotion gates.")
    evaluate.add_argument("--campaign-dir", required=True)
    evaluate.add_argument("--adjudication")
    return parser.parse_args(list(argv) if argv is not None else None)


async def async_main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Execute one benchmark lifecycle command."""
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    campaign_dir = Path(getattr(args, "campaign_dir", "."))
    if args.command == "prepare":
        _require_staging()
        import web.db as db

        return prepare_staging_corpus(
            campaign_dir,
            db_module=db,
            base_job_id=args.base_job_id,
            seed=args.seed,
        )
    if args.command == "approve":
        return approve_manifest(campaign_dir, approved_by=args.approved_by)
    if args.command == "smoke":
        _require_staging()
        return await run_openrouter_smoke(Path(args.output_dir))
    if args.command == "preflight":
        _require_staging()
        return await run_openrouter_preflight(
            campaign_dir,
            conformance_calls=args.conformance_calls,
        )
    if args.command == "run":
        _require_staging()
        return await run_campaign(campaign_dir, profile_id=args.profile)
    if args.command == "evaluate":
        return evaluate_reviews(
            campaign_dir,
            adjudication_path=args.adjudication,
        )
    raise RuntimeError("Unknown benchmark command.")


def main() -> None:
    """Run the benchmark command-line entry point."""
    result = asyncio.run(async_main())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))  # noqa: T201


if __name__ == "__main__":
    main()
