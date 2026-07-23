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
    capture_specter_corpus,
    prepare_staging_corpus,
    run_campaign,
    run_openrouter_smoke,
    run_openrouter_preflight,
    verify_manifest,
)


@pytest.fixture(autouse=True)
def _staging_environment(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")


def test_campaign_schedule_is_deterministic_balanced_and_has_18_runs() -> None:
    company_ids = [f"company-{index:02d}" for index in range(1, 7)]

    first = build_run_schedule(company_ids, repeats=1, seed=20260722)
    second = build_run_schedule(company_ids, repeats=1, seed=20260722)

    assert first == second
    assert len(first) == 18
    assert [item["sequence"] for item in first] == list(range(1, 19))
    counts = Counter((item["company_id"], item["profile_id"]) for item in first)
    assert set(item["profile_id"] for item in first) == set(PRIMARY_PROFILE_IDS)
    assert all(count == 1 for count in counts.values())


def test_freeze_corpus_hashes_every_artifact_and_requires_approval(tmp_path: Path) -> None:
    companies = []
    for index in range(1, 7):
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
    assert manifest["company_count"] == 6
    assert manifest["run_count"] == 18
    assert manifest["live_web_search"] is False
    assert manifest["live_specter_mcp"] is False
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    assert campaign["pricing_snapshot"]["openai:gpt-5.4-mini"] is not None
    assert campaign["pricing_snapshot"]["openrouter:z-ai/glm-5.2"] == {
        "input_per_million_tokens_usd": 0.8106,
        "output_per_million_tokens_usd": 2.548,
    }
    assert len(manifest["files"]) == 13  # campaign config + 6 company + 6 chunk files
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
        for index in range(6)
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
        for index in range(6)
    ]
    freeze_corpus(tmp_path, companies)
    approve_manifest(tmp_path, approved_by="Dusan")

    observed_routing = []

    async def fake_invoke(model, attempt, routing):
        observed_routing.append(dict(routing))
        return {
            "structured_ok": True,
            "selected_provider": "DeepInfra",
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
    assert observed_routing
    assert all(item["only"] == ["deepinfra"] for item in observed_routing)
    assert all(item["allow_fallbacks"] is False for item in observed_routing)


def test_smoke_covers_every_distinct_profile_reasoning_and_sampling_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    observed = []

    async def fake_invoke(case, routing):
        observed.append((dict(case), dict(routing)))
        return {
            "structured_ok": True,
            "selected_provider": "DeepInfra",
            "generation_id": f"gen-{case['id']}",
        }

    report = asyncio.run(run_openrouter_smoke(tmp_path, invoke_case=fake_invoke))

    assert report["eligible"] is True
    assert {case["id"] for case, _ in observed} == {
        "kimi-thinking-on",
        "kimi-thinking-off",
        "glm-high-evaluation",
        "glm-thinking-off-upside",
        "deepseek-thinking-off-answering",
        "deepseek-high-refinement",
        "deepseek-pro-high-admin",
    }
    assert all(routing["only"] == ["deepinfra"] for _, routing in observed)
    assert all(routing["allow_fallbacks"] is False for _, routing in observed)
    assert (tmp_path / "smoke.json").is_file()


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
        for index in range(6)
    ]
    freeze_corpus(tmp_path, companies)
    approve_manifest(tmp_path, approved_by="Dusan")

    async def fake_preflight(model, attempt, routing):
        return {"structured_ok": True, "selected_provider": "DeepInfra"}

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
            "questions_answers": [{"question": "Q", "answer": "A"}],
            "final_arguments": [{"content": "Argument", "score": 80}],
            "top_final_arguments": [{"content": "Argument", "score": 80}],
            "run_costs": {"total_usd": 0.10, "llm_usd": 0.10},
            "model_executions": [],
            "retry_count": 0,
            "output_tokens": 100,
        }

    first = asyncio.run(
        run_campaign(tmp_path, evaluate_run=fake_evaluate, profile_id="gpt_current")
    )
    assert first["run_count"] == 6
    assert first["invocation_profile_id"] == "gpt_current"
    assert first["planned_run_count"] == 18
    assert (tmp_path / "profile-batches" / "gpt_current.json").is_file()

    second = asyncio.run(
        run_campaign(tmp_path, evaluate_run=fake_evaluate, profile_id="kimi_k26")
    )
    assert second["run_count"] == 12

    summary = asyncio.run(
        run_campaign(tmp_path, evaluate_run=fake_evaluate, profile_id="glm_deepseek_flash")
    )

    assert summary["run_count"] == 18
    assert summary["completed_runs"] == 18
    assert set(summary["profile_batch_wall_clock_seconds"]) == set(PRIMARY_PROFILE_IDS)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 18
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "blinding-key.json").is_file()
    assert (tmp_path / "review-bundle" / "reviewer-1.csv").is_file()
    assert (tmp_path / "review-bundle" / "reviewer-2.csv").is_file()
    review_output = json.loads(next((tmp_path / "review-bundle" / "outputs").glob("*.json")).read_text())
    assert review_output["questions_answers"] == [{"question": "Q", "answer": "A"}]
    assert review_output["final_arguments"] == [{"content": "Argument", "score": 80}]
    assert seen_pinned
    assert all(item["allow_fallbacks"] is False and len(item["only"]) == 1 for item in seen_pinned)


def test_capture_specter_corpus_fetches_deep_team_once_and_freezes_raw_responses(tmp_path: Path) -> None:
    calls = Counter()

    class FakeSpecterClient:
        def find_company(self, identifier):
            calls[("find_company", identifier)] += 1
            stem = identifier.split(".")[0]
            return {
                "external_company_id": f"specter-{stem}",
                "name": stem.title(),
                "domain": identifier,
            }

        def get_company_profile(self, company_id):
            calls[("get_company_profile", company_id)] += 1
            stem = company_id.removeprefix("specter-")
            return {
                "name": stem.title(),
                "domain": f"{stem}.example",
                "industry": ["Software"],
                "short_description": "Frozen company evidence.",
            }

        def get_company_intelligence(self, company_id):
            calls[("get_company_intelligence", company_id)] += 1
            return {
                "founders": [{
                    "external_person_id": f"person-{company_id}",
                    "full_name": "Founder Name",
                    "title": "Founder",
                    "linkedin_url": "https://linkedin.com/in/founder",
                }]
            }

        def get_company_financials(self, company_id):
            calls[("get_company_financials", company_id)] += 1
            return {}

        def get_person_profile(self, person_id):
            calls[("get_person_profile", person_id)] += 1
            return {
                "tagline": "Founder",
                "about": "Deep founder profile.",
                "linkedin_url": "https://linkedin.com/in/founder",
                "positions": [],
                "education": [],
            }

        def search_people(self, query, *, limit=20):
            calls[("search_people", query)] += 1
            return {"product": "people", "items": []}

    selected = [
        {"company_id": f"company-{index}", "identifier": f"company{index}.example"}
        for index in range(1, 7)
    ]
    manifest = capture_specter_corpus(
        tmp_path,
        selected,
        client=FakeSpecterClient(),
    )

    assert manifest["company_count"] == 6
    assert manifest["run_count"] == 18
    assert manifest["specter_capture_once"] is True
    assert manifest["fetch_full_team"] is True
    assert manifest["deep_team_complete"] is True
    assert manifest["leadership_search_complete"] is True
    assert len(manifest["raw_specter_responses"]) == 6
    assert all((tmp_path / item["path"]).is_file() for item in manifest["raw_specter_responses"])
    assert all(item["person_profile_errors"] == 0 for item in manifest["raw_specter_responses"])
    assert all(item["profile_coverage_complete"] is True for item in manifest["raw_specter_responses"])
    assert all(item["leadership_member_count"] == 1 for item in manifest["raw_specter_responses"])
    assert sum(count for (method, _), count in calls.items() if method == "find_company") == 6
    assert sum(count for (method, _), count in calls.items() if method == "search_people") == 6
    assert sum(count for (method, _), count in calls.items() if method == "get_person_profile") == 6

    capture_specter_corpus(
        tmp_path,
        selected,
        client=FakeSpecterClient(),
    )
    assert sum(count for (method, _), count in calls.items() if method == "find_company") == 6
    assert sum(count for (method, _), count in calls.items() if method == "search_people") == 6
    assert sum(count for (method, _), count in calls.items() if method == "get_person_profile") == 6

    # A legacy founders-only cache is upgraded using its frozen company calls;
    # only the missing leadership search is sent to Specter.
    legacy_path = tmp_path / "raw-specter" / "company-1.json"
    legacy_capture = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_capture.pop("team_capture_version")
    legacy_capture.pop("leadership_search_performed")
    legacy_capture["calls"] = [
        call
        for call in legacy_capture["calls"]
        if call.get("method") != "search_people"
    ]
    legacy_path.write_text(json.dumps(legacy_capture), encoding="utf-8")

    upgraded = capture_specter_corpus(
        tmp_path,
        selected,
        client=FakeSpecterClient(),
    )
    assert upgraded["leadership_search_complete"] is True
    assert sum(count for (method, _), count in calls.items() if method == "find_company") == 6
    assert sum(count for (method, _), count in calls.items() if method == "get_company_profile") == 6
    assert sum(count for (method, _), count in calls.items() if method == "search_people") == 7
    assert sum(count for (method, _), count in calls.items() if method == "get_person_profile") == 6
