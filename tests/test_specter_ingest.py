import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import Response

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.evidence_answering import answer_all_trees_from_evidence
from agent.ingest.store import Chunk, EvidenceStore
from agent.ingest.specter_ingest import (
    ingest_specter,
    ingest_specter_company,
    list_specter_companies,
)
from agent.run_context import RunTelemetryCollector
from web import app as web_app


def test_ingest_specter_companies_only_returns_all_rows(tmp_path: Path) -> None:
    companies_path = tmp_path / "specter-export.csv"
    pd.DataFrame(
        [
            {"Company Name": "Alpha", "Industry": "SaaS", "Description": "A"},
            {"Company Name": "Beta", "Industry": "Fintech", "Description": "B"},
            {"Company Name": "Gamma", "Industry": "Health", "Description": "C"},
        ]
    ).to_csv(companies_path, index=False)

    results = ingest_specter(companies_path)

    assert [company.name for company, _ in results] == ["Alpha", "Beta", "Gamma"]


def test_list_specter_companies_returns_lightweight_descriptors(tmp_path: Path) -> None:
    companies_path = tmp_path / "specter-export.csv"
    pd.DataFrame(
        [
            {"Company Name": "Alpha Health", "Industry": "Health", "Domain": "alpha.example"},
            {"Company Name": "Beta AI", "Industry": "AI", "Domain": "beta.example"},
        ]
    ).to_csv(companies_path, index=False)

    descriptors = list_specter_companies(companies_path)

    assert descriptors == [
        {
            "index": 0,
            "name": "Alpha Health",
            "slug": "alpha-health",
            "industry": "Health",
            "domain": "alpha.example",
        },
        {
            "index": 1,
            "name": "Beta AI",
            "slug": "beta-ai",
            "industry": "AI",
            "domain": "beta.example",
        },
    ]


def test_ingest_specter_company_builds_only_selected_company(tmp_path: Path) -> None:
    companies_path = tmp_path / "companies.csv"
    people_path = tmp_path / "people.csv"
    pd.DataFrame(
        [
            {
                "Company Name": "Alpha",
                "Industry": "SaaS",
                "Description": "Alpha company",
                "Domain": "alpha.com",
                "Founders": '[{"specter_person_id":"p-1"}]',
            },
            {
                "Company Name": "Beta",
                "Industry": "Fintech",
                "Description": "Beta company",
                "Domain": "beta.com",
                "Founders": '[{"specter_person_id":"p-2"}]',
            },
        ]
    ).to_csv(companies_path, index=False)
    pd.DataFrame(
        [
            {
                "Specter - Person ID": "p-1",
                "Full Name": "Alice Founder",
                "Current Position Title": "CEO",
                "Current Position Company Name": "Alpha",
            },
            {
                "Specter - Person ID": "p-2",
                "Full Name": "Bob Founder",
                "Current Position Title": "CEO",
                "Current Position Company Name": "Beta",
            },
        ]
    ).to_csv(people_path, index=False)

    company, store = ingest_specter_company(
        companies_path,
        people_path,
        company_index=1,
    )

    assert company.name == "Beta"
    assert store.startup_slug == "beta"
    assert company.team and [member.name for member in company.team] == ["Bob Founder"]
    assert any("Beta" in chunk.text for chunk in store.chunks)
    assert all("Alpha company" not in chunk.text for chunk in store.chunks)


def test_detect_specter_csvs_by_headers_without_people_file(tmp_path: Path) -> None:
    companies_path = tmp_path / "generic-upload.csv"
    pd.DataFrame(
        [
            {
                "Company Name": "Alpha",
                "Founders": "[]",
                "Industry": "SaaS",
                "Domain": "alpha.com",
            }
        ]
    ).to_csv(companies_path, index=False)

    detected = web_app._detect_specter_csvs(tmp_path, [companies_path.name])

    assert detected == {"companies": str(companies_path)}


def test_build_single_company_specter_overlay_merges_documents_and_structured_chunks(
    tmp_path: Path,
) -> None:
    companies_path = tmp_path / "companies.csv"
    people_path = tmp_path / "people.csv"
    notes_path = tmp_path / "notes.txt"

    pd.DataFrame(
        [
            {
                "Company Name": "Alpha",
                "Industry": "SaaS",
                "Description": "Alpha sells workflow software.",
                "Domain": "alpha.com",
                "Founders": '[{"specter_person_id":"p-1"}]',
            }
        ]
    ).to_csv(companies_path, index=False)
    pd.DataFrame(
        [
            {
                "Specter - Person ID": "p-1",
                "Full Name": "Alice Founder",
                "Current Position Title": "CEO",
                "Current Position Company Name": "Alpha",
            }
        ]
    ).to_csv(people_path, index=False)
    notes_path.write_text("Alpha also shared a customer memo and extra diligence notes.")

    company, store, parsed_count = web_app._build_single_company_specter_overlay(
        tmp_path,
        {"companies": str(companies_path), "people": str(people_path)},
    )

    assert parsed_count == 1
    assert company is not None
    assert company.name == "Alpha"
    assert company.team and company.team[0].name == "Alice Founder"
    assert store is not None
    assert any(chunk.source_file == "notes.txt" for chunk in store.chunks)
    assert any(chunk.source_file == "specter-company" for chunk in store.chunks)
    assert any(chunk.source_file == "specter-people" for chunk in store.chunks)
    assert [chunk.chunk_id for chunk in store.chunks] == [
        f"chunk_{idx}" for idx in range(len(store.chunks))
    ]


def test_build_results_payload_keeps_batch_mode_when_only_one_company_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(web_app, "rank_batch_companies", lambda results: results)
    monkeypatch.setattr(
        web_app,
        "build_summary_rows",
        lambda results: [
            {
                "startup_slug": "alpha",
                "company_name": "Alpha",
                "decision": "invest",
                "total_score": 1.5,
                "avg_pro": 3.0,
                "avg_contra": 1.5,
            }
        ],
    )
    monkeypatch.setattr(web_app, "build_argument_rows", lambda results: [])
    monkeypatch.setattr(web_app, "build_qa_provenance_rows", lambda results: [])
    monkeypatch.setattr(web_app, "build_failed_rows", lambda results: results[1:])
    job_id = "job-batch-mode"
    web_app._results_cache[job_id] = {}

    results = [
        {
            "slug": "alpha",
            "company": Company(name="Alpha"),
            "evidence_store": type("Store", (), {"chunks": []})(),
            "final_state": {"final_arguments": [], "final_decision": "invest", "ranking_result": None},
            "skipped": False,
        },
        {"slug": "beta", "company_name": "Beta", "error": "Failed beta", "skipped": True},
        {"slug": "gamma", "company_name": "Gamma", "error": "Failed gamma", "skipped": True},
    ]

    web_app._build_results_payload(results, job_id, tmp_path)

    payload = web_app._results_cache[job_id]["results"]
    assert payload["mode"] == "batch"
    assert payload["num_companies"] == 1
    assert payload["num_skipped"] == 2


def test_stop_control_sets_terminal_progress_message(monkeypatch) -> None:
    job_id = "job-stop-message"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Working...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)

    try:
        asyncio.run(
            web_app.control_analysis_job(
                job_id,
                web_app.JobControlRequest(action="stop"),
                session_id="session",
            )
        )
        job = web_app._jobs[job_id]
        assert job.status == "stopped"
        assert job.progress == "Stopped by user."
        assert job.progress_log[-1] == "Stopped by user."
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)


def test_answer_all_trees_honors_async_control_checkpoints(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_answer_question_from_evidence(*args, **kwargs):
        question = args[0]
        calls.append(f"answer:{question}")
        return (f"Answer for {question}", {"chunk_ids": []})

    checkpoints = {"count": 0}

    async def on_cooperate() -> None:
        checkpoints["count"] += 1

    monkeypatch.setattr(
        "agent.evidence_answering.answer_question_from_evidence",
        fake_answer_question_from_evidence,
    )

    tree = QuestionTree(
        aspect="team",
        root_node=QuestionNode(
            question="Root question?",
            sub_nodes=[QuestionNode(question="Child question?")],
        ),
    )
    store = EvidenceStore(startup_slug="alpha", chunks=[])
    company = Company(name="Alpha")

    qa_pairs = asyncio.run(
        answer_all_trees_from_evidence(
            {"team": tree},
            company,
            store,
            on_cooperate=on_cooperate,
        )
    )

    assert [pair["question"] for pair in qa_pairs] == ["Root question?", "Child question?"]
    assert calls == ["answer:Child question?", "answer:Root question?"]
    assert checkpoints["count"] >= 4


def test_run_specter_analysis_persists_partial_results_on_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-stop-partial"
    company_a = Company(name="Alpha")
    company_b = Company(name="Beta")
    store_a = type("Store", (), {"startup_slug": "alpha"})()
    store_b = type("Store", (), {"startup_slug": "beta"})()

    completed_result = {
        "slug": "alpha",
        "company": company_a,
        "evidence_store": type("Evidence", (), {"chunks": []})(),
        "final_state": {"final_arguments": [], "final_decision": "invest", "ranking_result": None},
        "skipped": False,
    }

    state = {"calls": 0}

    async def fake_evaluate_from_specter(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return completed_result
        raise web_app._JobStoppedError("Job stopped by user")

    def fake_build_results_payload(results_list, current_job_id, upload_dir):
        web_app._results_cache[current_job_id]["results"] = {
            "mode": "batch",
            "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
        }

    monkeypatch.setattr(
        web_app,
        "list_specter_companies",
        lambda companies: [
            {"index": 0, "name": "Alpha", "slug": "alpha"},
            {"index": 1, "name": "Beta", "slug": "beta"},
        ],
    )
    monkeypatch.setattr(web_app, "ingest_specter", lambda companies, people=None: [(company_a, store_a), (company_b, store_b)])
    monkeypatch.setattr(web_app, "evaluate_from_specter", fake_evaluate_from_specter)
    monkeypatch.setattr(web_app, "_build_results_payload", fake_build_results_payload)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda job_id, results_list: None)
    monkeypatch.setattr(web_app, "_llm_telemetry_base", lambda: {})

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {}

    try:
        asyncio.run(
            web_app._run_specter_analysis(
                job_id,
                tmp_path,
                {"companies": "companies.csv", "people": "people.csv"},
                use_web_search=False,
            )
        )

        job = web_app._jobs[job_id]
        assert job.status == "stopped"
        assert "1/2 companies ranked" in job.progress
        assert job.results == web_app._results_cache[job_id]["results"]
        assert job.results["job_status"] == "stopped"
        assert job.results["summary_rows"] == [{"startup_slug": "alpha", "company_name": "Alpha"}]
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_run_specter_analysis_persists_first_company_before_starting_second(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-partial-persist-order"
    company_a = Company(name="Alpha")
    company_b = Company(name="Beta")
    store_a = type("Store", (), {"startup_slug": "alpha"})()
    store_b = type("Store", (), {"startup_slug": "beta"})()

    results = {
        "alpha": {
            "slug": "alpha",
            "company": company_a,
            "evidence_store": EvidenceStore(
                startup_slug="alpha",
                chunks=[Chunk(chunk_id="chunk_0", text="A" * 900, source_file="deck.pdf", page_or_slide=1)],
            ),
            "final_state": {"final_arguments": [], "final_decision": "invest", "ranking_result": None},
            "skipped": False,
        },
        "beta": {
            "slug": "beta",
            "company": company_b,
            "evidence_store": EvidenceStore(
                startup_slug="beta",
                chunks=[Chunk(chunk_id="chunk_0", text="B" * 300, source_file="deck.pdf", page_or_slide=1)],
            ),
            "final_state": {"final_arguments": [], "final_decision": "invest", "ranking_result": None},
            "skipped": False,
        },
    }

    persisted: list[str] = []
    saw_partial_status = {"value": False}
    state = {"calls": 0}

    async def fake_evaluate_from_specter(company, store, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            assert persisted == ["alpha"]
            current = web_app._results_cache[job_id]["results"]
            assert current["_memory_compact"] is True
            assert current["summary_rows_count"] == 1
            saw_partial_status["value"] = True
        return results[store.startup_slug]

    def fake_build_results_payload(results_list, current_job_id, upload_dir, write_excel=True):
        summary_rows = [
            {
                "startup_slug": item["slug"],
                "company_name": item.get("company_name") or item["company"].name,
            }
            for item in results_list
            if not item.get("skipped")
        ]
        web_app._results_cache[current_job_id]["results"] = {
            "mode": "batch",
            "summary_rows": summary_rows,
        }

    monkeypatch.setattr(
        web_app,
        "list_specter_companies",
        lambda companies: [
            {"index": 0, "name": "Alpha", "slug": "alpha"},
            {"index": 1, "name": "Beta", "slug": "beta"},
        ],
    )
    monkeypatch.setattr(web_app, "ingest_specter", lambda companies, people=None: [(company_a, store_a), (company_b, store_b)])
    monkeypatch.setattr(web_app, "evaluate_from_specter", fake_evaluate_from_specter)
    monkeypatch.setattr(web_app, "_build_results_payload", fake_build_results_payload)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda job_id, results_list: None)
    monkeypatch.setattr(
        web_app,
        "_persist_company_result_to_db",
        lambda current_job_id, result: persisted.append(result["slug"]) or True,
    )
    monkeypatch.setattr(web_app, "_llm_telemetry_base", lambda: {})

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {}

    try:
        asyncio.run(
            web_app._run_specter_analysis(
                job_id,
                tmp_path,
                {"companies": "companies.csv", "people": "people.csv"},
                use_web_search=False,
            )
        )

        assert persisted == ["alpha", "beta"]
        assert saw_partial_status["value"] is True
        assert results["alpha"] == {
            "slug": "alpha",
            "company_name": "Alpha",
            "skipped": False,
            "persisted": True,
            "decision": "invest",
            "total_score": None,
            "composite_score": None,
            "persisted_at": results["alpha"]["persisted_at"],
        }
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_run_specter_analysis_subprocess_mode_avoids_parent_full_ingest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-subprocess-company-mode"
    descriptors = [
        {"index": 0, "name": "Alpha", "slug": "alpha", "industry": "SaaS", "domain": "alpha.com"},
        {"index": 1, "name": "Beta", "slug": "beta", "industry": "AI", "domain": "beta.com"},
        {"index": 2, "name": "Gamma", "slug": "gamma", "industry": "Health", "domain": "gamma.com"},
    ]
    subprocess_calls: list[tuple[int, int]] = []

    async def fake_run_subprocess(current_job_id, *, company_index, absolute_index, **kwargs):
        assert current_job_id == job_id
        subprocess_calls.append((company_index, absolute_index))

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def load_job_results(job_id_legacy, preferred_mode=None):
            assert job_id_legacy == job_id
            return {
                "results": {
                    "summary_rows": [
                        {"startup_slug": "alpha"},
                        {"startup_slug": "beta"},
                        {"startup_slug": "gamma"},
                    ],
                    "failed_rows": [],
                }
            }

    original_db = web_app.db
    original_chunked = web_app.ENABLE_CHUNKED_SPECTER_PERSISTENCE
    original_subprocess = web_app.ENABLE_SPECTER_SUBPROCESS_CHUNKS
    web_app.db = FakeDb()
    web_app.ENABLE_CHUNKED_SPECTER_PERSISTENCE = True
    web_app.ENABLE_SPECTER_SUBPROCESS_CHUNKS = True

    monkeypatch.setattr(web_app, "list_specter_companies", lambda companies_csv: descriptors)
    monkeypatch.setattr(
        web_app,
        "ingest_specter",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parent should not ingest full specter payload")),
    )
    monkeypatch.setattr(web_app, "_batch_chunking_config", lambda *args, **kwargs: {
        "enabled": True,
        "label": "specter batch",
        "total_chunks": 2,
        "chunk_size": 2,
        "cooldown_seconds": 0,
    })
    monkeypatch.setattr(web_app, "_run_specter_company_subprocess", fake_run_subprocess)
    monkeypatch.setattr(web_app, "_load_persisted_company_keys", lambda current_job_id: set())
    def fake_refresh(current_job_id, *, progress_message=None, full=False):
        web_app._results_cache[current_job_id]["results"] = {
            "mode": "batch",
            "summary_rows_count": 3,
        }
        return True

    monkeypatch.setattr(web_app, "_refresh_persisted_batch_results", fake_refresh)
    monkeypatch.setattr(web_app, "_flush_chunk_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "_mark_terminal_persistence_complete", lambda *args, **kwargs: None)

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {"input_mode": "specter"}

    try:
        asyncio.run(
            web_app._run_specter_analysis(
                job_id,
                tmp_path,
                {"companies": "companies.csv", "people": "people.csv"},
                use_web_search=False,
            )
        )

        assert subprocess_calls == [(0, 1), (1, 2), (2, 3)]
        assert web_app._jobs[job_id].status == "done"
    finally:
        web_app.db = original_db
        web_app.ENABLE_CHUNKED_SPECTER_PERSISTENCE = original_chunked
        web_app.ENABLE_SPECTER_SUBPROCESS_CHUNKS = original_subprocess
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_merge_run_cost_summaries_prefers_available_status() -> None:
    merged = web_app._merge_run_cost_summaries(
        web_app._empty_run_costs_summary(),
        {
            "currency": "USD",
            "status": "complete",
            "total_usd": 0.001,
            "llm_usd": 0.001,
            "perplexity_usd": 0.0,
            "llm_tokens": {"prompt": 100, "completion": 50, "total": 150},
            "perplexity_search": {"requests": 0, "total_usd": 0.0},
            "by_model": [
                {
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite-preview",
                    "label": "Gemini 3.1 Flash Lite",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "usd": 0.001,
                    "pricing_available": True,
                    "partial": False,
                }
            ],
        },
    )

    assert merged["status"] == "complete"
    assert merged["llm_usd"] == 0.001
    assert merged["llm_tokens"] == {"prompt": 100, "completion": 50, "total": 150}
    assert merged["by_model"][0]["provider"] == "gemini"


def test_append_progress_updates_cached_results_metadata() -> None:
    job_id = "job-progress-sync"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._results_cache[job_id] = {
        "results": {
            "mode": "batch",
            "job_status": "running",
            "job_message": "Starting...",
        }
    }

    try:
        web_app._append_progress(job_id, "Chunk 2/2 — Evaluating Joe AI (6/10) — [2/3] Decomposing...")

        assert web_app._jobs[job_id].progress.startswith("Chunk 2/2")
        assert web_app._results_cache[job_id]["results"]["job_message"].startswith("Chunk 2/2")
        assert web_app._jobs[job_id].results == web_app._results_cache[job_id]["results"]
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_flush_chunk_telemetry_persists_incremental_rows_and_updates_aggregate() -> None:
    job_id = "job-flush-telemetry"
    collector = RunTelemetryCollector()
    collector.record_llm_usage(
        provider="gemini",
        model="gemini-3.1-flash-lite-preview",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    persisted_rows: list[dict] = []

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def persist_model_executions(job_id_legacy, rows, *, run_config=None, versions=None) -> bool:
            assert job_id_legacy == job_id
            assert run_config == {"input_mode": "specter"}
            assert versions == {"app_version": "test"}
            persisted_rows.extend(rows)
            return True

    original_db = web_app.db
    web_app.db = FakeDb()
    web_app._results_cache[job_id] = {
        "telemetry_collector": collector,
        "run_costs_aggregate": web_app._empty_run_costs_summary(),
        "model_executions": [],
        "run_config": {"input_mode": "specter"},
        "versions": {"app_version": "test"},
    }

    try:
        web_app._flush_chunk_telemetry(job_id)

        assert len(persisted_rows) == 1
        assert persisted_rows[0]["service"] == "llm"
        assert collector.snapshot_model_executions() == []
        assert web_app._results_cache[job_id]["model_executions"] == []
        assert web_app._results_cache[job_id]["run_costs_aggregate"]["status"] == "complete"
        assert web_app._results_cache[job_id]["run_costs_aggregate"]["llm_tokens"] == {
            "prompt": 100,
            "completion": 50,
            "total": 150,
        }
    finally:
        web_app.db = original_db
        web_app._results_cache.pop(job_id, None)


def test_flush_chunk_telemetry_retains_rows_for_final_persist_when_incremental_write_fails() -> None:
    job_id = "job-flush-telemetry-fallback"
    collector = RunTelemetryCollector()
    collector.record_llm_usage(
        provider="gemini",
        model="gemini-3.1-flash-lite-preview",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def persist_model_executions(job_id_legacy, rows, *, run_config=None, versions=None) -> bool:
            return False

    original_db = web_app.db
    web_app.db = FakeDb()
    web_app._results_cache[job_id] = {
        "telemetry_collector": collector,
        "run_costs_aggregate": web_app._empty_run_costs_summary(),
        "model_executions": [],
        "run_config": {"input_mode": "specter"},
        "versions": {"app_version": "test"},
    }

    try:
        web_app._flush_chunk_telemetry(job_id)

        assert collector.snapshot_model_executions() == []
        assert len(web_app._results_cache[job_id]["model_executions"]) == 1
        assert web_app._results_cache[job_id]["run_costs_aggregate"]["status"] == "complete"
    finally:
        web_app.db = original_db
        web_app._results_cache.pop(job_id, None)


def test_run_specter_analysis_persists_timeout_company_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-timeout-persist"
    company = Company(name="Alpha")
    store = EvidenceStore(startup_slug="alpha", chunks=[])
    persisted_failures: list[dict[str, str]] = []

    async def fake_evaluate_from_specter(*args, **kwargs):
        raise TimeoutError("final evaluation timed out")

    monkeypatch.setattr(
        web_app,
        "ingest_specter",
        lambda companies, people=None: [(company, store)],
    )
    monkeypatch.setattr(
        web_app,
        "list_specter_companies",
        lambda companies: [{"index": 0, "name": "Alpha", "slug": "alpha"}],
    )
    monkeypatch.setattr(web_app, "evaluate_from_specter", fake_evaluate_from_specter)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda current_job_id, results_list: None)
    monkeypatch.setattr(
        web_app,
        "_persist_failed_company_result_to_db",
        lambda current_job_id, **kwargs: persisted_failures.append(
            {
                "job_id": current_job_id,
                "slug": kwargs["slug"],
                "status": kwargs["status"],
            }
        ) or True,
    )
    monkeypatch.setattr(web_app, "_llm_telemetry_base", lambda: {})
    monkeypatch.setattr(
        web_app,
        "db",
        type(
            "DB",
            (),
            {
                "is_configured": staticmethod(lambda: True),
                "insert_analysis_event": staticmethod(lambda *args, **kwargs: None),
                "insert_analysis_error": staticmethod(lambda *args, **kwargs: None),
                "insert_job_status_history": staticmethod(lambda *args, **kwargs: None),
            },
        )(),
    )

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {}

    try:
        asyncio.run(
            web_app._run_specter_analysis(
                job_id,
                tmp_path,
                {"companies": "companies.csv", "people": "people.csv"},
                use_web_search=False,
            )
        )

        assert persisted_failures == [{"job_id": job_id, "slug": "alpha", "status": "timeout"}]
        job = web_app._jobs[job_id]
        assert job.status == "error"
        assert "No startups were successfully evaluated." in job.progress
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_set_job_status_tolerates_db_history_write_failure(monkeypatch) -> None:
    job_id = "job-status-write-failure"
    failing_db = type(
        "DB",
        (),
        {
            "is_configured": staticmethod(lambda: True),
            "insert_job_status_history": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down"))),
        },
    )()
    monkeypatch.setattr(web_app, "db", failing_db)
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Working...",
        progress_log=[],
    )

    try:
        web_app._set_job_status(job_id, "done", "Analysis complete", source="test")
        assert web_app._jobs[job_id].status == "done"
        assert web_app._jobs[job_id].progress == "Analysis complete"
    finally:
        web_app._jobs.pop(job_id, None)


def test_status_endpoint_does_not_return_partial_results_for_running_job(monkeypatch) -> None:
    job_id = "job-running-status"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Alpha complete",
        progress_log=[],
    )
    web_app._results_cache[job_id] = {
        "results": {
            "mode": "batch",
            "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
            "job_status": "running",
            "job_message": "Alpha complete",
        }
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)

    try:
        payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))
        assert payload["status"] == "running"
        assert payload["results"] is None
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_job_log_endpoint_returns_live_progress_for_active_job(monkeypatch) -> None:
    job_id = "job-live-log"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Working",
        progress_log=["one", "two"],
    )
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)

    try:
        payload = asyncio.run(web_app.get_job_log(job_id, response=Response(), session_id="session"))
        assert payload == {
            "job_id": job_id,
            "status": "running",
            "progress": "Working",
            "progress_log": ["one", "two"],
        }
    finally:
        web_app._jobs.pop(job_id, None)


def test_job_log_endpoint_rejects_interrupted_run(monkeypatch) -> None:
    job_id = "job-interrupted-log"
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Chunk 2/2 — Evaluating Delta",
            }
            if current_job_id == job_id
            else None,
        ),
    )

    with pytest.raises(web_app.HTTPException) as exc_info:
        asyncio.run(web_app.get_job_log(job_id, response=Response(), session_id="session"))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Run is no longer active."


def test_job_log_endpoint_returns_db_progress_for_worker_backed_job(monkeypatch) -> None:
    job_id = "job-worker-log"
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Worker running — alpha (2/10)",
            }
            if current_job_id == job_id
            else None,
            load_analysis_events=lambda current_job_id, limit=200: ["Queued for worker...", "Worker running — alpha (2/10)"]
            if current_job_id == job_id
            else [],
        ),
    )

    payload = asyncio.run(web_app.get_job_log(job_id, response=Response(), session_id="session"))

    assert payload == {
        "job_id": job_id,
        "status": "running",
        "progress": "Worker running — alpha (2/10)",
        "progress_log": ["Queued for worker...", "Worker running — alpha (2/10)"],
    }


def test_status_endpoint_promotes_running_job_when_persisted_report_is_terminal(monkeypatch) -> None:
    job_id = "job-running-promoted"
    results = {
        "mode": "batch",
        "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
        "job_status": "done",
        "job_message": "Analysis complete",
    }
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Chunk 1/2 — Evaluating Alpha",
        progress_log=[],
    )
    web_app._results_cache[job_id] = {"input_mode": "batch"}

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "done",
                "progress": "Analysis complete",
            }
            if current_job_id == job_id
            else None,
        ),
    )
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": results} if current_job_id == job_id else None,
    )
    monkeypatch.setattr(web_app.time, "monotonic", lambda: 100.0)

    try:
        payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))
        assert payload["status"] == "done"
        assert payload["progress"] == "Analysis complete"
        assert payload["results"]["summary_rows"] == [{"startup_slug": "alpha", "company_name": "Alpha"}]
        assert web_app._jobs[job_id].status == "done"
        assert web_app._jobs[job_id].progress == "Analysis complete"
        assert web_app._jobs[job_id].terminal_results_served is True
        assert web_app._results_cache[job_id]["results"]["job_status"] == "done"
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_status_endpoint_marks_persisted_running_job_as_interrupted_when_not_live(monkeypatch) -> None:
    job_id = "job-interrupted-status"
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Chunk 1/2 — Evaluating TopK",
                "worker_active": False,
            }
            if current_job_id == job_id
            else None,
        ),
    )
    monkeypatch.setattr(web_app, "_load_persisted_job_results", lambda current_job_id, preferred_mode=None: None)

    payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))

    assert payload["status"] == "stopped"
    assert payload["progress"] == "Run interrupted before completion."
    assert payload["results"] is None
    assert payload["progress_log"] == []


def test_status_endpoint_keeps_worker_job_running_without_progress_log_when_heartbeat_is_live(monkeypatch) -> None:
    job_id = "job-live-heartbeat-no-log"

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Worker evaluating beta (2/5)",
                "worker_active": True,
            }
            if current_job_id == job_id
            else None,
        ),
    )
    monkeypatch.setattr(web_app, "_load_worker_progress_log", lambda current_job_id: [])
    monkeypatch.setattr(web_app, "_load_persisted_job_results", lambda current_job_id, preferred_mode=None: None)

    payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))

    assert payload["status"] == "running"
    assert payload["progress"] == "Worker evaluating beta (2/5)"
    assert payload["results"] is None
    assert payload["progress_log"] == []


def test_status_endpoint_does_not_return_partial_persisted_results_for_active_worker_job(monkeypatch) -> None:
    job_id = "job-worker-partial-status"
    partial_results = {
        "mode": "batch",
        "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
        "job_status": "running",
        "job_message": "Worker evaluating beta (2/6)",
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Worker evaluating beta (2/6)",
                "worker_active": True,
            }
            if current_job_id == job_id
            else None,
            load_saved_job=lambda current_job_id: None,
            load_analysis_events=lambda current_job_id, limit=200: [
                "Queued for worker...",
                "Worker evaluating beta (2/6)",
            ]
            if current_job_id == job_id
            else [],
        ),
    )
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": partial_results} if current_job_id == job_id else None,
    )

    payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))

    assert payload["status"] == "running"
    assert payload["progress"] == "Worker evaluating beta (2/6)"
    assert payload["results"] is None
    assert payload["progress_log"] == ["Queued for worker...", "Worker evaluating beta (2/6)"]


def test_status_endpoint_strips_compact_cached_results_for_active_job(monkeypatch) -> None:
    job_id = "job-active-compact-cache"
    compact_results = {
        "mode": "batch",
        "num_companies": 2,
        "summary_rows_count": 2,
        "job_status": "running",
        "job_message": "Partial results updated — 2/5 companies completed.",
    }
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Worker evaluating gamma (3/5)",
        progress_log=["Queued for worker...", "Worker evaluating gamma (3/5)"],
        results=compact_results,
    )
    web_app._results_cache[job_id] = {
        "worker_backed": True,
        "results": compact_results,
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Worker evaluating gamma (3/5)",
            }
            if current_job_id == job_id
            else None,
            load_saved_job=lambda current_job_id: None,
            load_analysis_events=lambda current_job_id, limit=200: [
                "Queued for worker...",
                "Worker evaluating gamma (3/5)",
            ]
            if current_job_id == job_id
            else [],
        ),
    )

    try:
        payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))
        assert payload["status"] == "running"
        assert payload["progress"] == "Worker evaluating gamma (3/5)"
        assert payload["results"] is None
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_status_endpoint_marks_saved_job_without_status_history_as_interrupted(monkeypatch) -> None:
    job_id = "job-saved-only"
    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: None,
            load_saved_job=lambda current_job_id: {
                "job_id": job_id,
                "status": "interrupted",
                "progress": "Run interrupted before completion.",
                "llm": "Gemini 3.1 Flash Lite",
            }
            if current_job_id == job_id
            else None,
        ),
    )
    monkeypatch.setattr(web_app, "_load_persisted_job_results", lambda current_job_id, preferred_mode=None: None)

    payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))

    assert payload["status"] == "stopped"
    assert payload["progress"] == "Run interrupted before completion."
    assert payload["results"] is None
    assert payload["llm"] == "Gemini 3.1 Flash Lite"


def test_get_analysis_rejects_partial_persisted_results_for_active_job(monkeypatch) -> None:
    job_id = "job-worker-partial-analysis"
    partial_results = {
        "mode": "batch",
        "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
        "job_status": "running",
        "job_message": "Worker evaluating beta (2/6)",
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_job_status=lambda current_job_id: {
                "status": "running",
                "progress": "Worker evaluating beta (2/6)",
            }
            if current_job_id == job_id
            else None,
        ),
    )
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": partial_results} if current_job_id == job_id else None,
    )

    with pytest.raises(web_app.HTTPException) as exc_info:
        asyncio.run(web_app.get_analysis(job_id, response=Response(), session_id="session"))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Analysis is still in progress."


def test_get_analysis_prefers_terminal_saved_job_over_stale_in_memory_status(monkeypatch) -> None:
    job_id = "job-worker-terminal-saved"
    final_results = {
        "mode": "batch",
        "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
        "job_status": "done",
        "job_message": "Analysis complete — 1/1 companies ranked",
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Worker evaluating alpha (1/1)",
        progress_log=[],
    )
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_saved_job=lambda current_job_id: {
                "job_id": job_id,
                "status": "done",
                "progress": "Analysis complete — 1/1 companies ranked",
                "has_results": True,
            }
            if current_job_id == job_id
            else None,
        ),
    )
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": final_results} if current_job_id == job_id else None,
    )

    payload = asyncio.run(web_app.get_analysis(job_id, response=Response(), session_id="session"))

    assert payload == {"job_id": job_id, "results": final_results}
    assert web_app._jobs[job_id].status == "done"


def test_get_analysis_uses_saved_job_as_only_db_gate(monkeypatch) -> None:
    job_id = "job-saved-terminal-no-secondary-gate"
    final_results = {
        "mode": "batch",
        "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha"}],
        "job_status": "done",
        "job_message": "Analysis complete — 1/1 companies ranked",
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            load_saved_job=lambda current_job_id: {
                "job_id": job_id,
                "status": "done",
                "progress": "Analysis complete — 1/1 companies ranked",
                "has_results": True,
            }
            if current_job_id == job_id
            else None,
            load_job_status=lambda _current_job_id: (_ for _ in ()).throw(
                AssertionError("load_job_status should not be consulted when load_saved_job is available")
            ),
        ),
    )
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": final_results} if current_job_id == job_id else None,
    )

    payload = asyncio.run(web_app.get_analysis(job_id, response=Response(), session_id="session"))

    assert payload == {"job_id": job_id, "results": final_results}


def test_start_analysis_queues_worker_backed_specter_job_without_starting_thread(tmp_path: Path, monkeypatch) -> None:
    job_id = "job-worker-start"
    original_flag = web_app.ENABLE_SPECTER_WORKER_SERVICE
    web_app.ENABLE_SPECTER_WORKER_SERVICE = True
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending", progress="Queued", progress_log=[])
    web_app._results_cache[job_id] = {
        "files": [
            {
                "name": "companies.csv",
                "local_path": str(tmp_path / "companies.csv"),
                "mime_type": "text/csv",
                "size": 123,
            }
        ],
        "specter": {"companies": str(tmp_path / "companies.csv")},
    }
    (tmp_path / "companies.csv").write_text("Company Name\nAlpha\n", encoding="utf-8")

    queued_calls: list[dict[str, object]] = []
    thread_started = {"value": False}

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            thread_started["value"] = True

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "validate_requested_selection",
        lambda provider, model: SimpleNamespace(provider="openai", model="gpt-5"),
    )
    monkeypatch.setattr(web_app, "_runtime_versions", lambda: {"app_version": "test"})
    monkeypatch.setattr(web_app, "list_specter_companies", lambda path: [{"index": 0, "name": "Alpha", "slug": "alpha"}])
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            upsert_job=lambda *args, **kwargs: "job-uuid",
            upsert_job_control=lambda *args, **kwargs: None,
            upload_source_file=lambda *args, **kwargs: "jobs/job-worker-start/inputs/companies.csv",
            persist_source_files=lambda job_id_legacy, source_files, **kwargs: True,
            queue_specter_worker_job=lambda job_id_legacy, **kwargs: queued_calls.append({"job_id": job_id_legacy, **kwargs}) or True,
        ),
    )

    try:
        payload = asyncio.run(
            web_app.start_analysis(
                job_id,
                web_app.AnalyzeRequest(input_mode="specter"),
                session_id="session",
            )
        )

        assert payload["status"] == "running"
        assert thread_started["value"] is False
        assert queued_calls and queued_calls[0]["job_id"] == job_id
        assert web_app._results_cache[job_id]["worker_backed"] is True
        assert "Queued for worker" in web_app._jobs[job_id].progress
    finally:
        web_app.ENABLE_SPECTER_WORKER_SERVICE = original_flag
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_start_analysis_worker_queue_failure_does_not_fallback_to_web(tmp_path: Path, monkeypatch) -> None:
    job_id = "job-worker-fail-closed"
    original_flag = web_app.ENABLE_SPECTER_WORKER_SERVICE
    web_app.ENABLE_SPECTER_WORKER_SERVICE = True
    web_app._jobs[job_id] = web_app.AnalysisStatus(job_id=job_id, status="pending", progress="Queued", progress_log=[])
    web_app._results_cache[job_id] = {
        "files": [
            {
                "name": "companies.csv",
                "local_path": str(tmp_path / "companies.csv"),
                "mime_type": "text/csv",
                "size": 123,
            }
        ],
        "specter": {"companies": str(tmp_path / "companies.csv")},
    }
    (tmp_path / "companies.csv").write_text("Company Name\nAlpha\n", encoding="utf-8")

    thread_started = {"value": False}

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            thread_started["value"] = True

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "validate_requested_selection",
        lambda provider, model: SimpleNamespace(provider="openai", model="gpt-5"),
    )
    monkeypatch.setattr(web_app, "_runtime_versions", lambda: {"app_version": "test"})
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        web_app,
        "_queue_worker_backed_specter_job",
        lambda current_job_id: (False, "Could not persist Specter source file metadata.")
        if current_job_id == job_id
        else (False, None),
    )

    try:
        with pytest.raises(web_app.HTTPException) as exc_info:
            asyncio.run(
                web_app.start_analysis(
                    job_id,
                    web_app.AnalyzeRequest(input_mode="specter"),
                    session_id="session",
                )
            )

        assert exc_info.value.status_code == 503
        assert "Specter worker queue failed" in exc_info.value.detail
        assert thread_started["value"] is False
        assert web_app._jobs[job_id].status == "error"
        assert "Worker queue failed" in web_app._jobs[job_id].progress
    finally:
        web_app.ENABLE_SPECTER_WORKER_SERVICE = original_flag
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_append_progress_caps_in_memory_log(monkeypatch) -> None:
    job_id = "job-progress-cap"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="",
        progress_log=[],
    )
    original_cap = web_app.MAX_PROGRESS_LOG_ENTRIES
    monkeypatch.setattr(web_app, "MAX_PROGRESS_LOG_ENTRIES", 3)
    monkeypatch.setattr(web_app, "db", None)

    try:
        for msg in ("one", "two", "three", "four", "five"):
            web_app._append_progress(job_id, msg)
        assert web_app._jobs[job_id].progress_log == ["three", "four", "five"]
    finally:
        monkeypatch.setattr(web_app, "MAX_PROGRESS_LOG_ENTRIES", original_cap)
        web_app._jobs.pop(job_id, None)


def test_release_job_runtime_resources_drops_heavy_cache(tmp_path: Path) -> None:
    job_id = "job-release-runtime"
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    results = {
        "mode": "single",
        "company_name": "Alpha",
        "llm_selection": {
            "provider": "openai",
            "model": "gpt-5",
            "label": "GPT-5",
        },
    }
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        progress_log=["a", "b", "c"],
        results=results,
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {
        "upload_dir": str(upload_dir),
        "files": [{"name": "deck.pdf"}],
        "specter": {"companies": "companies.csv"},
        "telemetry_collector": object(),
        "model_executions": [{"provider": "openai"}],
        "run_costs_aggregate": {"status": "available"},
        "versions": {"app_version": "test"},
        "results": results,
    }

    try:
        web_app._release_job_runtime_resources(job_id, drop_results=True)
        assert not upload_dir.exists()
        assert web_app._jobs[job_id].results is None
        assert "results" not in web_app._results_cache[job_id]
        assert "files" not in web_app._results_cache[job_id]
        assert "run_costs_aggregate" not in web_app._results_cache[job_id]
        assert "versions" not in web_app._results_cache[job_id]
        assert job_id not in web_app._job_controls
        assert web_app._results_cache[job_id]["input_mode"] == "single"
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)


def test_refresh_persisted_batch_results_stores_compact_runtime_payload(monkeypatch) -> None:
    job_id = "job-compact-refresh"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Working",
        progress_log=[],
    )
    web_app._results_cache[job_id] = {
        "input_mode": "specter",
        "llm_selection": {
            "provider": "google",
            "model": "gemini-3.1-flash-lite-preview",
            "label": "Gemini 3.1 Flash Lite",
        },
    }
    fake_db = SimpleNamespace(
        is_configured=lambda: True,
        load_job_progress_snapshot=lambda current_job_id, preferred_mode=None: {
            "results": {
                "mode": "batch",
                "num_companies": 3,
                "summary_rows_count": 3,
                "failed_rows_count": 1,
                "summary_rows": [
                    {"company_name": "A"},
                    {"company_name": "B"},
                    {"company_name": "C"},
                ],
                "argument_rows": [{"company_name": "A"}],
                "qa_provenance_rows": [{"company_name": "A"}],
            }
        } if current_job_id == job_id else None,
        load_job_results=lambda current_job_id, preferred_mode=None: None,
    )
    monkeypatch.setattr(web_app, "db", fake_db)

    try:
        refreshed = web_app._refresh_persisted_batch_results(job_id, progress_message="Partial results updated")
        assert refreshed is True
        payload = web_app._results_cache[job_id]["results"]
        assert payload["_memory_compact"] is True
        assert payload["summary_rows_count"] == 3
        assert payload["argument_rows_count"] == 1
        assert payload["qa_provenance_rows_count"] == 1
        assert "summary_rows" not in payload
        assert web_app._jobs[job_id].results == payload
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_status_terminal_compact_results_loads_full_persisted_report(monkeypatch) -> None:
    job_id = "job-compact-terminal"
    compact_results = {
        "_memory_compact": True,
        "mode": "batch",
        "summary_rows_count": 2,
        "job_status": "done",
        "job_message": "Analysis complete",
    }
    full_results = {
        "mode": "batch",
        "summary_rows": [{"company_name": "Alpha"}, {"company_name": "Beta"}],
        "job_status": "done",
        "job_message": "Analysis complete",
    }
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        progress_log=[],
        results=compact_results,
    )
    web_app._results_cache[job_id] = {
        "input_mode": "specter",
        "results": compact_results,
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": full_results} if current_job_id == job_id else None,
    )

    try:
        payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))
        assert payload["status"] == "done"
        assert payload["results"]["summary_rows"] == full_results["summary_rows"]
        assert payload["results"].get("_memory_compact") is None
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_status_lazy_loads_terminal_results_after_memory_cleanup(monkeypatch) -> None:
    job_id = "job-lazy-status"
    results = {
        "mode": "single",
        "company_name": "Alpha",
        "job_status": "done",
        "job_message": "Analysis complete",
        "llm_selection": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "label": "Claude Haiku 4.5",
        },
    }
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        progress_log=[],
        results=None,
    )
    web_app._results_cache[job_id] = {"input_mode": "single"}

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": results} if current_job_id == job_id else None,
    )

    try:
        payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))
        assert payload["status"] == "done"
        assert payload["results"]["company_name"] == "Alpha"
        assert payload["llm"] == "Claude Haiku 4.5"
        assert web_app._jobs[job_id].results is None
        assert "results" not in web_app._results_cache[job_id]
        assert web_app._jobs[job_id].terminal_results_served is True
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_schedule_idle_restart_when_enabled(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeTimer:
        def __init__(self, interval, func):
            created["interval"] = interval
            created["func"] = func
            self.started = False
            self.cancelled = False
            self.daemon = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started and not self.cancelled

        def cancel(self):
            self.cancelled = True

    original_timer = web_app._restart_timer
    monkeypatch.setattr(web_app, "RESTART_ON_IDLE_AFTER_ANALYSIS", True)
    monkeypatch.setattr(web_app, "RESTART_ON_IDLE_DELAY_SECONDS", 15)
    monkeypatch.setattr(
        web_app,
        "db",
        type("DbStub", (), {"is_configured": staticmethod(lambda: True)})(),
    )
    monkeypatch.setattr(web_app, "_has_active_analysis_jobs", lambda: False)
    monkeypatch.setattr(web_app, "_has_active_person_jobs", lambda: False)
    monkeypatch.setattr(web_app.threading, "Timer", FakeTimer)
    web_app._restart_timer = None

    web_app._jobs["job-done"] = web_app.AnalysisStatus(
        job_id="job-done",
        status="done",
        progress="Analysis complete",
        progress_log=[],
        terminal_results_served=True,
        restart_pending=True,
        persistence_complete=True,
    )

    try:
        web_app._schedule_idle_restart_if_enabled("job-done")
        assert created["interval"] == 15
        assert isinstance(web_app._restart_timer, FakeTimer)
        assert web_app._restart_timer.started is True
    finally:
        web_app._jobs.pop("job-done", None)
        web_app._restart_timer = original_timer


def test_schedule_idle_restart_skips_when_another_job_is_active(monkeypatch) -> None:
    timer_calls = {"count": 0}

    class FakeTimer:
        def __init__(self, interval, func):
            timer_calls["count"] += 1

        def start(self):
            pass

        def is_alive(self):
            return False

        def cancel(self):
            pass

    original_timer = web_app._restart_timer
    monkeypatch.setattr(web_app, "RESTART_ON_IDLE_AFTER_ANALYSIS", True)
    monkeypatch.setattr(
        web_app,
        "db",
        type("DbStub", (), {"is_configured": staticmethod(lambda: True)})(),
    )
    monkeypatch.setattr(web_app.threading, "Timer", FakeTimer)
    web_app._restart_timer = None
    web_app._jobs["job-done"] = web_app.AnalysisStatus(
        job_id="job-done",
        status="done",
        progress="Analysis complete",
        progress_log=[],
        terminal_results_served=True,
        restart_pending=True,
        persistence_complete=True,
    )
    web_app._jobs["job-active"] = web_app.AnalysisStatus(
        job_id="job-active",
        status="running",
        progress="Working",
        progress_log=[],
    )

    try:
        web_app._schedule_idle_restart_if_enabled("job-done")
        assert timer_calls["count"] == 0
    finally:
        web_app._jobs.pop("job-done", None)
        web_app._jobs.pop("job-active", None)
        web_app._restart_timer = original_timer


def test_mark_terminal_results_served_triggers_restart_only_after_persistence(monkeypatch) -> None:
    calls: list[str] = []
    job_id = "job-terminal-served"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        progress_log=[],
        restart_pending=True,
        persistence_complete=True,
    )
    monkeypatch.setattr(web_app, "_schedule_idle_restart_if_enabled", lambda current_job_id: calls.append(current_job_id))

    try:
        web_app._mark_terminal_results_served(job_id)
        assert web_app._jobs[job_id].terminal_results_served is True
        assert calls == [job_id]
    finally:
        web_app._jobs.pop(job_id, None)


def test_mark_terminal_results_served_does_not_trigger_restart_before_persistence(monkeypatch) -> None:
    timer_calls = {"count": 0}
    job_id = "job-terminal-served-not-persisted"

    class FakeTimer:
        def __init__(self, interval, func):
            timer_calls["count"] += 1

        def start(self):
            pass

        def is_alive(self):
            return False

        def cancel(self):
            pass

    original_timer = web_app._restart_timer
    monkeypatch.setattr(web_app, "RESTART_ON_IDLE_AFTER_ANALYSIS", True)
    monkeypatch.setattr(
        web_app,
        "db",
        type("DbStub", (), {"is_configured": staticmethod(lambda: True)})(),
    )
    monkeypatch.setattr(web_app.threading, "Timer", FakeTimer)
    web_app._restart_timer = None

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        progress_log=[],
        restart_pending=True,
        persistence_complete=False,
    )

    try:
        web_app._mark_terminal_results_served(job_id)
        assert web_app._jobs[job_id].terminal_results_served is True
        assert timer_calls["count"] == 0
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._restart_timer = original_timer


def test_mark_terminal_persistence_complete_triggers_restart_after_results_served(monkeypatch) -> None:
    calls: list[str] = []
    job_id = "job-terminal-persisted"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        progress_log=[],
        restart_pending=True,
        terminal_results_served=True,
    )
    monkeypatch.setattr(web_app, "_schedule_idle_restart_if_enabled", lambda current_job_id: calls.append(current_job_id))

    try:
        web_app._mark_terminal_persistence_complete(job_id)
        assert web_app._jobs[job_id].persistence_complete is True
        assert calls == [job_id]
    finally:
        web_app._jobs.pop(job_id, None)


def test_loaded_persisted_terminal_results_do_not_arm_restart(monkeypatch) -> None:
    job_id = "job-persisted-only"
    results = {
        "mode": "single",
        "company_name": "Alpha",
        "job_status": "done",
        "job_message": "Analysis complete",
    }

    monkeypatch.setattr(web_app, "_check_session", lambda session_id: True)
    monkeypatch.setattr(
        web_app,
        "_load_persisted_job_results",
        lambda current_job_id, preferred_mode=None: {"results": results} if current_job_id == job_id else None,
    )

    try:
        payload = asyncio.run(web_app.get_status(job_id, response=Response(), session_id="session"))
        assert payload["status"] == "done"
        assert payload["results"]["company_name"] == "Alpha"
        assert web_app._jobs[job_id].terminal_results_served is True
        assert web_app._jobs[job_id].restart_pending is False
        assert web_app._jobs[job_id].persistence_complete is True
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_list_jobs_for_ui_prefers_persisted_terminal_state_over_live_running(monkeypatch) -> None:
    job_id = "job-overview-terminal"
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Chunk 1/2 — Evaluating TopK",
        progress_log=[],
    )

    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            list_saved_jobs=lambda: [
                {
                    "job_id": job_id,
                    "status": "done",
                    "progress": "Analysis complete",
                    "created_at": "2026-03-13T10:05:00Z",
                    "input_mode": "specter",
                    "use_web_search": True,
                    "results": None,
                    "has_results": True,
                }
            ],
        ),
    )

    try:
        rows = web_app._list_jobs_for_ui()
        row = next(item for item in rows if item["job_id"] == job_id)
        assert row == {
            "job_id": job_id,
            "status": "done",
            "progress": "Analysis complete",
            "created_at": "2026-03-13T10:05:00Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "results": None,
            "has_results": True,
            "can_open_results": True,
            "can_view_log": False,
            "llm": web_app._get_llm_display(),
        }
    finally:
        web_app._jobs.pop(job_id, None)


def test_list_jobs_for_ui_marks_persisted_running_without_live_job_as_interrupted(monkeypatch) -> None:
    job_id = "job-overview-interrupted"
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            list_saved_jobs=lambda: [
                {
                    "job_id": job_id,
                    "status": "running",
                    "progress": "Chunk 1/2 — Evaluating TopK",
                    "created_at": "2026-03-13T10:05:00Z",
                    "input_mode": "specter",
                    "use_web_search": True,
                    "results": None,
                    "has_results": False,
                }
            ],
        ),
    )

    rows = web_app._list_jobs_for_ui()
    row = next(item for item in rows if item["job_id"] == job_id)
    assert row == {
        "job_id": job_id,
        "status": "interrupted",
        "progress": "Run interrupted before completion.",
        "created_at": "2026-03-13T10:05:00Z",
        "input_mode": "specter",
        "use_web_search": True,
        "run_name": None,
        "started_by_user_id": None,
        "started_by_email": None,
        "started_by_display_name": None,
        "started_by_label": None,
        "results": None,
        "has_results": False,
        "can_open_results": False,
        "can_view_log": False,
        "llm": web_app._get_llm_display(),
    }


def test_list_jobs_for_ui_keeps_partial_worker_backed_results_non_terminal(monkeypatch) -> None:
    job_id = "job-overview-partial-active"
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            list_saved_jobs=lambda: [
                {
                    "job_id": job_id,
                    "status": "running",
                    "progress": "Worker running — alpha (3/6)",
                    "created_at": "2026-03-13T10:05:00Z",
                    "input_mode": "specter",
                    "use_web_search": True,
                    "results": None,
                    "has_results": True,
                    "worker_active": True,
                }
            ],
        ),
    )

    rows = web_app._list_jobs_for_ui()
    row = next(item for item in rows if item["job_id"] == job_id)
    assert row == {
        "job_id": job_id,
        "status": "running",
        "progress": "Worker running — alpha (3/6)",
        "created_at": "2026-03-13T10:05:00Z",
        "input_mode": "specter",
        "use_web_search": True,
        "run_name": None,
        "started_by_user_id": None,
        "started_by_email": None,
        "started_by_display_name": None,
        "started_by_label": None,
        "results": None,
        "has_results": True,
        "can_open_results": False,
        "can_view_log": True,
        "llm": web_app._get_llm_display(),
    }


def test_list_jobs_for_ui_carries_optional_run_name(monkeypatch) -> None:
    job_id = "job-overview-named"
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            list_saved_jobs=lambda: [
                {
                    "job_id": job_id,
                    "status": "running",
                    "progress": "Queued for worker...",
                    "created_at": "2026-03-14T10:05:00Z",
                    "input_mode": "specter",
                    "use_web_search": True,
                    "run_name": "Germany shortlist",
                    "results": None,
                    "has_results": False,
                    "worker_active": True,
                }
            ],
        ),
    )

    rows = web_app._list_jobs_for_ui()
    row = next(item for item in rows if item["job_id"] == job_id)
    assert row["run_name"] == "Germany shortlist"


def test_list_jobs_for_ui_uses_persisted_run_config_for_active_llm_label(monkeypatch) -> None:
    job_id = "job-overview-phase-llm"
    monkeypatch.setattr(
        web_app,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            list_saved_jobs=lambda: [
                {
                    "job_id": job_id,
                    "status": "running",
                    "progress": "Worker evaluating Apaleo (1/1)",
                    "created_at": "2026-03-15T09:01:16Z",
                    "input_mode": "specter",
                    "use_web_search": True,
                    "run_name": "Apaleo",
                    "results": None,
                    "has_results": False,
                    "worker_active": True,
                    "run_config": {
                        "phase_models": {
                            "decomposition": {"provider": "openai", "model": "gpt-5-mini"},
                            "answering": {"provider": "openai", "model": "gpt-5-mini"},
                            "generation": {"provider": "openai", "model": "gpt-5-mini"},
                            "evaluation": {"provider": "openai", "model": "gpt-5-mini"},
                            "ranking": {"provider": "openai", "model": "gpt-5-mini"},
                        },
                        "effective_phase_models": {
                            "decomposition": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                            "answering": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                            "generation": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                            "critique": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                            "evaluation": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                            "refinement": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                            "ranking": {"provider": "openai", "model": "gpt-5-mini", "label": "GPT-5 mini"},
                        },
                    },
                }
            ],
        ),
    )

    rows = web_app._list_jobs_for_ui()
    row = next(item for item in rows if item["job_id"] == job_id)
    assert row["llm"] == "Per-phase · D GPT-5 mini · A GPT-5 mini · G GPT-5 mini · E GPT-5 mini · R GPT-5 mini"


def test_batch_chunking_config_enables_for_large_anthropic_batch() -> None:
    job_id = "job-chunking-config"
    web_app._results_cache[job_id] = {
        "llm_selection": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "label": "Claude Haiku 4.5",
        }
    }

    try:
        config = web_app._batch_chunking_config(job_id, total_items=5, mode="specter")
        assert config["enabled"] is True
        assert config["chunk_size"] == 2
        assert config["total_chunks"] == 3
        assert config["cooldown_seconds"] == 20
    finally:
        web_app._results_cache.pop(job_id, None)


def test_batch_chunking_config_uses_claude_phase_policy_even_when_answering_is_gemini() -> None:
    job_id = "job-phase-chunking-config"
    web_app._results_cache[job_id] = {
        "llm_selection": {
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "label": "Gemini 3.1 Flash Lite",
        },
        "run_config": {
            "phase_models": {
                "decomposition": {"provider": "openai", "model": "gpt-5"},
                "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
                "generation": {"provider": "openai", "model": "gpt-5"},
                "evaluation": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
                "ranking": {"provider": "openai", "model": "gpt-5"},
            },
            "effective_phase_models": {
                "decomposition": {"provider": "openai", "model": "gpt-5", "label": "GPT-5"},
                "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash Lite"},
                "generation": {"provider": "openai", "model": "gpt-5", "label": "GPT-5"},
                "critique": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash Lite"},
                "evaluation": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
                "refinement": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash Lite"},
                "ranking": {"provider": "openai", "model": "gpt-5", "label": "GPT-5"},
            },
        },
    }

    try:
        config = web_app._batch_chunking_config(job_id, total_items=5, mode="specter")
        assert config["enabled"] is True
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-haiku-4-5-20251001"
        assert config["chunk_size"] == 2
        assert config["cooldown_seconds"] == 20
    finally:
        web_app._results_cache.pop(job_id, None)


def test_run_specter_analysis_chunks_large_anthropic_batch_with_cooldown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-anthropic-chunked"
    companies = [Company(name=name) for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")]
    stores = [type("Store", (), {"startup_slug": company.name.lower()})() for company in companies]
    evaluate_order: list[str] = []
    sleep_calls: list[int] = []
    printed: list[str] = []

    async def fake_evaluate_from_specter(company, store, *args, **kwargs):
        evaluate_order.append(company.name)
        return {
            "slug": store.startup_slug,
            "company": company,
            "evidence_store": EvidenceStore(startup_slug=store.startup_slug, chunks=[]),
            "final_state": {"final_arguments": [], "final_decision": "invest", "ranking_result": None},
            "skipped": False,
        }

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        web_app,
        "list_specter_companies",
        lambda companies_path: [
            {"index": idx, "name": company.name, "slug": store.startup_slug}
            for idx, (company, store) in enumerate(zip(companies, stores))
        ],
    )
    monkeypatch.setattr(
        web_app,
        "ingest_specter",
        lambda companies_path, people=None: list(zip(companies, stores)),
    )
    monkeypatch.setattr(web_app, "evaluate_from_specter", fake_evaluate_from_specter)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda job_id, results_list: None)
    monkeypatch.setattr(web_app, "_persist_company_result_to_db", lambda current_job_id, result: None)
    monkeypatch.setattr(web_app.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)))

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {
        "llm_selection": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "label": "Claude Haiku 4.5",
        }
    }

    try:
        asyncio.run(
            web_app._run_specter_analysis(
                job_id,
                tmp_path,
                {"companies": "companies.csv", "people": "people.csv"},
                use_web_search=False,
            )
        )

        assert evaluate_order == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        assert sleep_calls == [20, 20]
        progress_log = web_app._jobs[job_id].progress_log
        assert any("Large batch chunking enabled" in entry for entry in progress_log)
        assert any("Starting chunk 1/3" in entry for entry in progress_log)
        assert any("Starting chunk 2/3" in entry for entry in progress_log)
        assert any("Starting chunk 3/3" in entry for entry in progress_log)
        assert any("Large batch chunking enabled" in entry for entry in printed)
        assert any("Starting chunk 1/3" in entry for entry in printed)
        assert any("Chunk 1/3 complete" in entry for entry in printed)
        assert any("Chunked batch processing complete" in entry for entry in printed)
        assert web_app._results_cache[job_id]["batch_chunking"]["enabled"] is True
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_run_specter_analysis_uses_subprocess_chunk_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-subprocess-chunked"
    companies = [Company(name=f"Company {idx}") for idx in range(1, 11)]
    stores = [type("Store", (), {"startup_slug": f"company-{idx}"})() for idx in range(1, 11)]
    chunk_calls: list[tuple[int, int, int]] = []
    sleep_calls: list[int] = []
    progress_counts = {"completed": 0}

    async def fake_company_subprocess(
        current_job_id: str,
        *,
        company_index: int,
        absolute_index: int,
        chunk_idx: int,
        **kwargs,
    ) -> None:
        assert current_job_id == job_id
        chunk_calls.append((chunk_idx, company_index, absolute_index))
        progress_counts["completed"] += 1

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    def fake_refresh(
        current_job_id: str,
        *,
        progress_message: str | None = None,
        full: bool = False,
    ) -> bool:
        web_app._results_cache[current_job_id]["results"] = {
            "mode": "batch",
            "summary_rows_count": progress_counts["completed"],
            "run_costs": {
                "currency": "USD",
                "status": "complete",
                "total_usd": 0.1,
                "llm_usd": 0.1,
                "perplexity_usd": 0.0,
                "llm_tokens": {"prompt": 1, "completion": 1, "total": 2},
                "perplexity_search": {"requests": 0, "total_usd": 0.0},
                "by_model": [],
            },
        }
        if progress_message:
            web_app._results_cache[current_job_id]["results"]["job_message"] = progress_message
        web_app._jobs[current_job_id].results = web_app._results_cache[current_job_id]["results"]
        return True

    class FakeDb:
        @staticmethod
        def is_configured() -> bool:
            return True

        @staticmethod
        def load_job_results(job_id_legacy, preferred_mode=None):
            return {
                "results": {
                    "mode": "batch",
                    "summary_rows": [{"startup_slug": f"company-{idx}"} for idx in range(progress_counts["completed"])],
                    "failed_rows": [],
                    "run_costs": {
                        "currency": "USD",
                        "status": "complete",
                        "total_usd": 0.1,
                        "llm_usd": 0.1,
                        "perplexity_usd": 0.0,
                        "llm_tokens": {"prompt": 1, "completion": 1, "total": 2},
                        "perplexity_search": {"requests": 0, "total_usd": 0.0},
                        "by_model": [],
                    },
                }
            }

    monkeypatch.setattr(web_app, "ENABLE_CHUNKED_SPECTER_PERSISTENCE", True)
    monkeypatch.setattr(web_app, "ENABLE_SPECTER_SUBPROCESS_CHUNKS", True)
    monkeypatch.setenv("BATCH_CHUNKING_THRESHOLD", "5")
    monkeypatch.setenv("BATCH_CHUNKING_SIZE", "5")
    monkeypatch.setenv("BATCH_CHUNKING_COOLDOWN_SECONDS", "15")
    monkeypatch.setattr(
        web_app,
        "list_specter_companies",
        lambda companies_path: [
            {"index": idx, "name": company.name, "slug": store.startup_slug}
            for idx, (company, store) in enumerate(zip(companies, stores))
        ],
    )
    monkeypatch.setattr(
        web_app,
        "ingest_specter",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parent should not ingest full specter payload")),
    )
    monkeypatch.setattr(web_app, "_run_specter_company_subprocess", fake_company_subprocess)
    monkeypatch.setattr(web_app, "_refresh_persisted_batch_results", fake_refresh)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda current_job_id, results_list: None)
    monkeypatch.setattr(web_app.asyncio, "sleep", fake_sleep)

    original_db = web_app.db
    web_app.db = FakeDb()
    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {
        "llm_selection": {
            "provider": "openai",
            "model": "gpt-5",
            "label": "GPT-5",
        },
        "run_config": {
            "input_mode": "specter",
            "llm_provider": "openai",
            "llm_model": "gpt-5",
        },
        "versions": {"app_version": "test"},
    }

    try:
        asyncio.run(
            web_app._run_specter_analysis(
                job_id,
                tmp_path,
                {"companies": "companies.csv", "people": "people.csv"},
                use_web_search=False,
            )
        )

        assert chunk_calls == [
            (1, 0, 1),
            (1, 1, 2),
            (1, 2, 3),
            (1, 3, 4),
            (1, 4, 5),
            (2, 5, 6),
            (2, 6, 7),
            (2, 7, 8),
            (2, 8, 9),
            (2, 9, 10),
        ]
        assert sleep_calls == [15]
        assert web_app._jobs[job_id].status == "done"
        assert "10/10 companies ranked" in web_app._jobs[job_id].progress
    finally:
        web_app.db = original_db
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_run_document_analysis_passes_specter_overlay_in_multi_file_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_id = "job-original-specter"
    companies_path = tmp_path / "companies.csv"
    people_path = tmp_path / "people.csv"
    notes_path = tmp_path / "notes.txt"

    pd.DataFrame(
        [
            {
                "Company Name": "Alpha",
                "Industry": "SaaS",
                "Description": "Alpha sells workflow software.",
                "Domain": "alpha.com",
                "Founders": '[{"specter_person_id":"p-1"}]',
            }
        ]
    ).to_csv(companies_path, index=False)
    pd.DataFrame(
        [
            {
                "Specter - Person ID": "p-1",
                "Full Name": "Alice Founder",
                "Current Position Title": "CEO",
                "Current Position Company Name": "Alpha",
            }
        ]
    ).to_csv(people_path, index=False)
    notes_path.write_text("Alpha also shared a customer memo and extra diligence notes.")

    captured: dict[str, object] = {}

    async def fake_evaluate_startup(*args, **kwargs):
        captured["initial_company"] = kwargs.get("initial_company")
        captured["initial_store"] = kwargs.get("initial_store")
        return {
            "slug": "alpha",
            "company": kwargs.get("initial_company"),
            "evidence_store": kwargs.get("initial_store"),
            "final_state": {"final_arguments": [], "final_decision": "invest", "ranking_result": None},
            "skipped": False,
        }

    def fake_build_results_payload(results_list, current_job_id, upload_dir):
        web_app._results_cache[current_job_id]["results"] = {
            "mode": "single",
            "company_name": "Alpha",
        }

    monkeypatch.setattr(web_app, "evaluate_startup", fake_evaluate_startup)
    monkeypatch.setattr(web_app, "_build_results_payload", fake_build_results_payload)
    monkeypatch.setattr(web_app, "_persist_jobs", lambda: None)
    monkeypatch.setattr(web_app, "_persist_results_to_db", lambda job_id, results_list: None)
    monkeypatch.setattr(web_app, "_llm_telemetry_base", lambda: {})

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="running",
        progress="Starting...",
        progress_log=[],
    )
    web_app._job_controls[job_id] = {"pause_requested": False, "stop_requested": False}
    web_app._results_cache[job_id] = {
        "files": [
            {"name": companies_path.name},
            {"name": people_path.name},
            {"name": notes_path.name},
        ],
        "specter": {"companies": str(companies_path), "people": str(people_path)},
    }

    try:
        asyncio.run(
            web_app._run_document_analysis(
                job_id,
                tmp_path,
                use_web_search=False,
                one_company=True,
            )
        )

        initial_company = captured["initial_company"]
        initial_store = captured["initial_store"]
        assert initial_company is not None
        assert initial_company.name == "Alpha"
        assert initial_company.team and initial_company.team[0].name == "Alice Founder"
        assert initial_store is not None
        assert any(chunk.source_file == "notes.txt" for chunk in initial_store.chunks)
        assert any(chunk.source_file == "specter-company" for chunk in initial_store.chunks)
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._job_controls.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)


def test_merge_runtime_payload_metadata_keeps_persisted_batch_run_costs_for_terminal_job() -> None:
    job_id = "job-terminal-costs"
    persisted_run_costs = {
        "currency": "USD",
        "status": "complete",
        "total_usd": 0.1694,
        "llm_usd": 0.0444,
        "perplexity_usd": 0.125,
        "llm_tokens": {"prompt": 105030, "completion": 12098, "total": 117128},
        "perplexity_search": {"requests": 25, "total_usd": 0.125},
        "by_model": [{"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "usd": 0.0444}],
    }
    stale_runtime_run_costs = {
        "currency": "USD",
        "status": "complete",
        "total_usd": 0.0169,
        "llm_usd": 0.0044,
        "perplexity_usd": 0.0125,
        "llm_tokens": {"prompt": 10030, "completion": 1098, "total": 11128},
        "perplexity_search": {"requests": 2, "total_usd": 0.0125},
        "by_model": [{"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "usd": 0.0044}],
    }

    web_app._jobs[job_id] = web_app.AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        persistence_complete=True,
    )
    web_app._results_cache[job_id] = {
        "run_costs_aggregate": stale_runtime_run_costs,
        "results": {
            "mode": "batch",
            "job_status": "done",
            "job_message": "Analysis complete",
        },
    }

    try:
        merged = web_app._merge_runtime_payload_metadata(
            job_id,
            {
                "mode": "batch",
                "job_status": "done",
                "job_message": "Analysis complete",
                "run_costs": persisted_run_costs,
            },
        )
        assert merged["run_costs"] == persisted_run_costs
    finally:
        web_app._jobs.pop(job_id, None)
        web_app._results_cache.pop(job_id, None)
