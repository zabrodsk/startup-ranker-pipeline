import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_web_db_reads_supabase_config_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")

    import web.db as web_db

    reloaded = importlib.reload(web_db)

    assert reloaded.is_configured() is True


def test_company_history_group_key_collapses_legacy_and_new_keys() -> None:
    import web.db as web_db

    slug_row = {
        "company_key": "slug:apify",
        "company_name": "Apify",
        "startup_slug": "apify",
        "result_payload": {"company_name": "Apify", "startup_slug": "apify"},
    }
    legacy_name_row = {
        "company_key": "name:apify--legacy-2",
        "company_name": "Apify",
        "startup_slug": None,
        "result_payload": {"company_name": "Apify"},
    }
    domain_row = {
        "company_key": "domain:apify.com",
        "company_name": "Apify",
        "startup_slug": None,
        "result_payload": {"company_name": "Apify", "summary_rows": [{"company_name": "Apify"}]},
    }

    slug_key = web_db._company_history_group_key(slug_row)
    legacy_key = web_db._company_history_group_key(legacy_name_row)
    domain_key = web_db._company_history_group_key(domain_row)

    assert slug_key == "name:apify"
    assert legacy_key == "name:apify"
    assert domain_key == "name:apify"


def test_list_company_histories_reconciles_missing_company_runs(monkeypatch) -> None:
    import web.db as web_db

    expected_rows = [
        {
            "company_key": "slug:apify",
            "company_name": "Apify",
            "startup_slug": "apify",
            "job_id_legacy": "job-123",
            "decision": "invest",
            "total_score": 73.7,
            "composite_score": 73.7,
            "bucket": "watchlist",
            "mode": "specter",
            "input_order": 1,
            "run_created_at": "2026-03-07T10:00:00Z",
            "created_at": "2026-03-07T10:00:00Z",
            "result_payload": {"company_name": "Apify", "startup_slug": "apify"},
        }
    ]

    monkeypatch.setattr(web_db, "_get_client", lambda: object())
    fetch_calls = {"count": 0}

    def fake_fetch(_client, limit_runs):
        fetch_calls["count"] += 1
        return [] if fetch_calls["count"] == 1 else expected_rows

    monkeypatch.setattr(web_db, "_fetch_company_run_rows", fake_fetch)
    monkeypatch.setattr(web_db, "backfill_company_runs_from_analyses", lambda limit_jobs=500: 1)
    monkeypatch.setattr(web_db, "_reconcile_missing_company_runs", lambda client, existing_rows, limit_jobs: 0)

    histories = web_db.list_company_histories(limit_runs=50)

    assert len(histories) == 1
    assert histories[0]["company_name"] == "Apify"
    assert histories[0]["runs"][0]["job_id"] == "job-123"
    assert fetch_calls["count"] == 2


def test_list_company_histories_requeries_after_recent_reconcile(monkeypatch) -> None:
    import web.db as web_db

    rows_before = [
        {
            "company_key": "slug:apify",
            "company_name": "Apify",
            "startup_slug": "apify",
            "job_id_legacy": "job-old",
            "decision": "invest",
            "total_score": 70.0,
            "composite_score": 70.0,
            "bucket": "watchlist",
            "mode": "specter",
            "input_order": 1,
            "run_created_at": "2026-03-07T10:00:00Z",
            "created_at": "2026-03-07T10:00:00Z",
            "result_payload": {"company_name": "Apify", "startup_slug": "apify"},
        }
    ]
    rows_after = rows_before + [
        {
            "company_key": "slug:apaleo",
            "company_name": "Apaleo",
            "startup_slug": "apaleo",
            "job_id_legacy": "job-new",
            "decision": "invest",
            "total_score": 75.0,
            "composite_score": 75.0,
            "bucket": "priority_review",
            "mode": "specter",
            "input_order": 2,
            "run_created_at": "2026-03-07T11:00:00Z",
            "created_at": "2026-03-07T11:00:00Z",
            "result_payload": {"company_name": "Apaleo", "startup_slug": "apaleo"},
        }
    ]

    monkeypatch.setattr(web_db, "_get_client", lambda: object())
    fetch_calls = {"count": 0}

    def fake_fetch(_client, limit_runs):
        fetch_calls["count"] += 1
        return rows_before if fetch_calls["count"] == 1 else rows_after

    monkeypatch.setattr(web_db, "_fetch_company_run_rows", fake_fetch)
    monkeypatch.setattr(web_db, "backfill_company_runs_from_analyses", lambda limit_jobs=500: 0)
    monkeypatch.setattr(web_db, "_reconcile_missing_company_runs", lambda client, existing_rows, limit_jobs: 1)

    histories = web_db.list_company_histories(limit_runs=50)

    assert len(histories) == 2
    assert {item["company_name"] for item in histories} == {"Apify", "Apaleo"}
    assert fetch_calls["count"] == 2


def test_compose_results_payload_from_company_runs_rebuilds_batch_ranking() -> None:
    import web.db as web_db

    rows = [
        {
            "startup_slug": "alpha",
            "company_name": "Alpha",
            "input_order": 2,
            "run_created_at": "2026-03-09T09:00:00Z",
            "created_at": "2026-03-09T09:00:00Z",
            "result_payload": {
                "mode": "single",
                "startup_slug": "alpha",
                "company_name": "Alpha",
                "decision": "invest",
                "total_score": 8.0,
                "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha", "decision": "invest", "total_score": 8.0}],
                "argument_rows": [{"startup_slug": "alpha", "argument_text": "Alpha arg"}],
                "qa_provenance_rows": [{"startup_slug": "alpha", "question": "Why alpha?"}],
                "founders": [{"full_name": "Alice"}],
                "team_members": [{"full_name": "Alice"}],
                "ranking_result": {
                    "composite_score": 82.0,
                    "strategy_fit_score": 80.0,
                    "team_score": 79.0,
                    "upside_score": 87.0,
                    "bucket": "priority_review",
                    "dimension_scores": [
                        {"dimension": "strategy_fit", "adjusted_score": 80.0, "confidence": 0.8, "critical_gaps": []},
                        {"dimension": "team", "adjusted_score": 79.0, "confidence": 0.7, "critical_gaps": []},
                    ],
                },
            },
        },
        {
            "startup_slug": "beta",
            "company_name": "Beta",
            "input_order": 1,
            "run_created_at": "2026-03-09T09:05:00Z",
            "created_at": "2026-03-09T09:05:00Z",
            "result_payload": {
                "mode": "single",
                "startup_slug": "beta",
                "company_name": "Beta",
                "decision": "watch",
                "total_score": 6.0,
                "summary_rows": [{"startup_slug": "beta", "company_name": "Beta", "decision": "watch", "total_score": 6.0}],
                "argument_rows": [{"startup_slug": "beta", "argument_text": "Beta arg"}],
                "qa_provenance_rows": [{"startup_slug": "beta", "question": "Why beta?"}],
                "founders": [{"full_name": "Bob"}],
                "team_members": [{"full_name": "Bob"}],
                "ranking_result": {
                    "composite_score": 70.0,
                    "strategy_fit_score": 69.0,
                    "team_score": 68.0,
                    "upside_score": 73.0,
                    "bucket": "watchlist",
                    "dimension_scores": [
                        {"dimension": "strategy_fit", "adjusted_score": 69.0, "confidence": 0.6, "critical_gaps": []},
                    ],
                },
            },
        },
    ]

    payload = web_db._compose_results_payload_from_company_runs(
        rows,
        preferred_mode="batch",
        snapshot_payload={"llm": "Claude", "job_status": "done"},
    )

    assert payload["mode"] == "batch"
    assert [row["startup_slug"] for row in payload["summary_rows"]] == ["alpha", "beta"]
    assert [row["rank"] for row in payload["summary_rows"]] == [1, 2]
    assert payload["founders_by_slug"]["alpha"] == [{"full_name": "Alice"}]
    assert payload["argument_rows"] == [
        {"startup_slug": "alpha", "argument_text": "Alpha arg"},
        {"startup_slug": "beta", "argument_text": "Beta arg"},
    ]
    assert payload["llm"] == "Claude"
    assert payload["job_status"] == "done"


def test_load_job_results_reconstructs_from_company_runs(monkeypatch) -> None:
    import web.db as web_db

    monkeypatch.setattr(web_db, "_get_client", lambda: object())
    monkeypatch.setattr(
        web_db,
        "_load_latest_analysis_snapshot",
        lambda client, job_id_legacy: {"results_payload": {"mode": "batch", "job_status": "done"}},
    )
    monkeypatch.setattr(
        web_db,
        "_load_company_run_rows_for_job",
        lambda client, job_id_legacy: [
            {
                "startup_slug": "alpha",
                "company_name": "Alpha",
                "input_order": 1,
                "run_created_at": "2026-03-09T09:00:00Z",
                "created_at": "2026-03-09T09:00:00Z",
                "result_payload": {
                    "mode": "single",
                    "startup_slug": "alpha",
                    "company_name": "Alpha",
                    "decision": "invest",
                    "total_score": 8.0,
                    "summary_rows": [{"startup_slug": "alpha", "company_name": "Alpha", "decision": "invest", "total_score": 8.0}],
                    "argument_rows": [],
                    "qa_provenance_rows": [],
                    "founders": [],
                    "team_members": [],
                    "ranking_result": {"composite_score": 82.0, "dimension_scores": []},
                },
            }
        ],
    )

    loaded = web_db.load_job_results("job-123", preferred_mode="batch")

    assert loaded is not None
    assert loaded["results"]["mode"] == "batch"
    assert loaded["results"]["summary_rows"][0]["startup_slug"] == "alpha"
    assert loaded["results"]["job_status"] == "done"
