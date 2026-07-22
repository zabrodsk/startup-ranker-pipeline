from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.model_benchmark import (  # noqa: E402
    PRIMARY_PROFILE_IDS,
    approve_manifest,
    build_run_schedule,
    evaluate_candidate_gates,
    freeze_corpus,
    prepare_staging_corpus,
    run_campaign,
    run_openrouter_preflight,
    verify_manifest,
)


@pytest.fixture(autouse=True)
def _staging_environment(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")


def test_campaign_schedule_is_deterministic_balanced_and_has_72_runs() -> None:
    company_ids = [f"company-{index:02d}" for index in range(1, 13)]

    first = build_run_schedule(company_ids, repeats=2, seed=20260722)
    second = build_run_schedule(company_ids, repeats=2, seed=20260722)

    assert first == second
    assert len(first) == 72
    assert [item["sequence"] for item in first] == list(range(1, 73))
    counts = Counter((item["company_id"], item["profile_id"]) for item in first)
    assert set(item["profile_id"] for item in first) == set(PRIMARY_PROFILE_IDS)
    assert all(count == 2 for count in counts.values())


def test_freeze_corpus_hashes_every_artifact_and_requires_approval(tmp_path: Path) -> None:
    companies = []
    for index in range(1, 13):
        companies.append(
            {
                "company_id": f"company-{index:02d}",
                "source_job_id": "f20aa510" if index <= 10 else f"extra-{index}",
                "input_mode": "specter" if index != 11 else "pitchdeck",
                "company": {"name": f"Company {index}", "industry": "SaaS"},
                "chunks": [
                    {
                        "chunk_id": f"chunk-{index}",
                        "text": f"Evidence for company {index}",
                        "source_file": "source.txt",
                        "page_or_slide": "1",
                    }
                ],
            }
        )

    manifest = freeze_corpus(tmp_path, companies, seed=20260722)

    assert manifest["approval"]["status"] == "pending"
    assert manifest["company_count"] == 12
    assert manifest["run_count"] == 72
    assert manifest["live_web_search"] is False
    assert manifest["live_specter_mcp"] is False
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    assert campaign["pricing_snapshot"]["openai:gpt-5.4-mini"] is not None
    assert campaign["pricing_snapshot"]["openrouter:z-ai/glm-5.2"] == {
        "input_per_million_tokens_usd": 0.8106,
        "output_per_million_tokens_usd": 2.548,
    }
    assert len(manifest["files"]) == 25  # campaign config + 12 company + 12 chunk files
    for item in manifest["files"]:
        path = tmp_path / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    stored = json.loads((tmp_path / "manifest.json").read_text())
    assert stored == manifest


def test_prepare_staging_corpus_selects_ten_base_plus_pitchdeck_and_specter(tmp_path: Path) -> None:
    rows = [
        {
            "company_id": f"base-{index}",
            "company_name": f"Base {index}",
            "job_id_legacy": "f20aa510",
            "status": "done",
            "created_at": f"2026-07-{index:02d}T10:00:00Z",
            "run_config": {"input_mode": "specter"},
        }
        for index in range(1, 11)
    ]
    rows = [
        {
            "company_id": "fixture-alpha",
            "company_name": "Alpha",
            "job_id_legacy": "job-original-specter",
            "status": "done",
            "created_at": "2026-07-22T13:00:00Z",
            "run_config": {"input_mode": "pitchdeck"},
        },
        {
            "company_id": "pitch-1",
            "company_name": "Pitch One",
            "job_id_legacy": "pitch-job",
            "status": "done",
            "created_at": "2026-07-22T12:00:00Z",
            "run_config": {"input_mode": "pitchdeck"},
        },
        {
            "company_id": "specter-1",
            "company_name": "Specter One",
            "job_id_legacy": "specter-job",
            "status": "done",
            "created_at": "2026-07-22T11:00:00Z",
            "run_config": {"input_mode": "specter"},
        },
        *rows,
    ]

    class FakeDb:
        @staticmethod
        def is_configured():
            return True

        @staticmethod
        def admin_get_recent_analyses(_limit):
            return rows

        @staticmethod
        def get_company_by_id(company_id):
            return {"name": company_id.replace("-", " ").title(), "industry": "SaaS"}

        @staticmethod
        def get_all_company_chunks(company_id):
            return [{"chunk_id": f"{company_id}-1", "text": "Evidence", "source_file": "source.txt", "page_or_slide": "1"}]

        @staticmethod
        def load_source_files(job_id):
            return [{"name": f"{job_id}.txt", "storage_path": f"jobs/{job_id}/input.txt"}]

        @staticmethod
        def download_source_file_to_path(storage_path, destination):
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_text(storage_path)
            return True

    manifest = prepare_staging_corpus(tmp_path, db_module=FakeDb())

    assert manifest["company_count"] == 12
    assert manifest["selection"]["base_job_id"] == "f20aa510"
    assert manifest["selection"]["extra_jobs"] == {
        "pitchdeck": "pitch-job",
        "specter": "specter-job",
    }
    assert len(manifest["source_files"]) == 3
    assert all((tmp_path / item["path"]).is_file() for item in manifest["source_files"])


def test_manifest_must_be_approved_and_fails_closed_on_tampering(tmp_path: Path) -> None:
    companies = [
        {
            "company_id": f"company-{index}",
            "source_job_id": "job",
            "input_mode": "specter",
            "company": {"name": f"Company {index}"},
            "chunks": [{"chunk_id": "1", "text": "Evidence", "source_file": "x", "page_or_slide": "1"}],
        }
        for index in range(12)
    ]
    freeze_corpus(tmp_path, companies)

    approved = approve_manifest(tmp_path, approved_by="Dusan")
    assert approved["approval"]["status"] == "approved"
    assert verify_manifest(tmp_path, require_approval=True)["approval"]["approved_by"] == "Dusan"

    (tmp_path / "corpus" / "company-0" / "chunks.json").write_text("tampered")
    try:
        verify_manifest(tmp_path, require_approval=True)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Tampered benchmark corpus must fail closed")


def test_candidate_gates_treat_quality_as_a_hard_gate() -> None:
    baseline = {
        "critical_unsupported_claims": 1,
        "ranking_spearman": 0.82,
        "repeat_decision_agreement": 0.90,
        "score_stddev": 0.40,
        "cost_per_company_usd": 1.00,
        "p95_wall_clock_seconds": 100.0,
    }
    candidate = {
        "quality_ci_lower_delta": -0.10,
        "factual_support_ci_lower_delta": -0.20,
        "completeness_ci_lower_delta": -0.15,
        "critical_unsupported_claims": 1,
        "new_systemic_omission_class": False,
        "ranking_spearman": 0.79,
        "repeat_decision_agreement": 0.84,
        "score_stddev": 0.45,
        "structured_success_rate": 0.995,
        "incomplete_runs": 0,
        "cost_per_company_usd": 0.70,
        "p95_wall_clock_seconds": 115.0,
    }

    passing = evaluate_candidate_gates(baseline, candidate)
    assert passing["passed"] is True
    assert all(item["passed"] for item in passing["gates"])

    candidate["quality_ci_lower_delta"] = -0.26
    failing = evaluate_candidate_gates(baseline, candidate)
    assert failing["passed"] is False
    assert next(item for item in failing["gates"] if item["id"] == "quality")["passed"] is False


def test_preflight_requires_consistent_zdr_provider_pin(tmp_path: Path) -> None:
    companies = [
        {
            "company_id": f"company-{index}",
            "source_job_id": "job",
            "input_mode": "specter",
            "company": {"name": f"Company {index}"},
            "chunks": [{"chunk_id": "1", "text": "Evidence", "source_file": "x", "page_or_slide": "1"}],
        }
        for index in range(12)
    ]
    freeze_corpus(tmp_path, companies)
    approve_manifest(tmp_path, approved_by="Dusan")

    async def fake_invoke(model, attempt):
        return {
            "structured_ok": True,
            "selected_provider": f"provider-for-{model.split('/')[-1]}",
            "generation_id": f"{model}-{attempt}",
        }

    report = asyncio.run(
        run_openrouter_preflight(
            tmp_path,
            invoke_model=fake_invoke,
            conformance_calls=2,
        )
    )

    assert report["eligible"] is True
    assert len(report["models"]) == 4
    assert set(report["provider_pins"]) == {
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    }
    assert all(model["routing_policy"]["zdr"] is True for model in report["models"])


def test_campaign_runner_writes_blinded_non_persisted_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    companies = [
        {
            "company_id": f"company-{index}",
            "source_job_id": "job",
            "input_mode": "specter",
            "company": {"name": f"Company {index}"},
            "chunks": [{"chunk_id": "1", "text": "Evidence", "source_file": "x", "page_or_slide": "1"}],
        }
        for index in range(12)
    ]
    freeze_corpus(tmp_path, companies)
    approve_manifest(tmp_path, approved_by="Dusan")

    async def fake_preflight(model, attempt):
        return {"structured_ok": True, "selected_provider": f"pin-{model.split('/')[-1]}"}

    asyncio.run(run_openrouter_preflight(tmp_path, invoke_model=fake_preflight))
    seen_pinned = []

    async def fake_evaluate(**kwargs):
        policy = kwargs["policy"]
        for selection in policy.as_dict().values():
            if selection["provider"] == "openrouter":
                seen_pinned.append(selection["openrouter_routing"])
        return {
            "status": "done",
            "structured_success": True,
            "wall_clock_seconds": 10.0,
            "final_decision": "invest",
            "ranking_result": {"composite_score": 75.0},
            "top_final_arguments": [],
            "run_costs": {"total_usd": 0.10, "llm_usd": 0.10},
            "model_executions": [],
            "retry_count": 0,
            "output_tokens": 100,
        }

    summary = asyncio.run(run_campaign(tmp_path, evaluate_run=fake_evaluate))

    assert summary["run_count"] == 72
    assert summary["completed_runs"] == 72
    assert len(list((tmp_path / "runs").glob("*.json"))) == 72
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "blinding-key.json").is_file()
    assert (tmp_path / "review-bundle" / "reviewer-1.csv").is_file()
    assert (tmp_path / "review-bundle" / "reviewer-2.csv").is_file()
    assert seen_pinned
    assert all(item["allow_fallbacks"] is False and len(item["only"]) == 1 for item in seen_pinned)
