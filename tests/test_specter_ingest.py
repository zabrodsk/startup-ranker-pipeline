import asyncio
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.evidence_answering import answer_all_trees_from_evidence
from agent.ingest.store import EvidenceStore
from agent.ingest.specter_ingest import ingest_specter
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
    monkeypatch.setattr(web_app, "export_excel", lambda results, output_path: Path(output_path).write_text("ok"))

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
