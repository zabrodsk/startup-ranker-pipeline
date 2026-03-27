import importlib
import logging
import sys
from datetime import datetime, timezone
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
    assert histories[0]["company_lookup_key"] == "name:apify"
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
    assert {item["company_lookup_key"] for item in histories} == {"name:apify", "name:apaleo"}
    assert fetch_calls["count"] == 2

def test_list_company_histories_keeps_audit_rows_for_company_detail(monkeypatch) -> None:
    import web.db as web_db

    rows = [
        {
            "company_key": "slug:apaleo",
            "company_name": "Apaleo",
            "startup_slug": "apaleo",
            "job_id_legacy": "job-28",
            "decision": "invest",
            "total_score": 60.1,
            "composite_score": 60.1,
            "bucket": "watchlist",
            "mode": "specter",
            "input_order": 1,
            "run_created_at": "2026-03-15T22:58:47Z",
            "created_at": "2026-03-15T22:58:47Z",
            "result_payload": {
                "mode": "single",
                "startup_slug": "apaleo",
                "company_name": "Apaleo",
                "decision": "invest",
                "total_score": 60.1,
                "avg_pro": 81,
                "avg_contra": 43,
                "summary_rows": [{
                    "startup_slug": "apaleo",
                    "company_name": "Apaleo",
                    "decision": "invest",
                    "total_score": 60.1,
                }],
                "qa_provenance_rows": [{
                    "startup_slug": "apaleo",
                    "question": "Why now?",
                    "answer": "Momentum and integrations.",
                }],
                "argument_rows": [{
                    "startup_slug": "apaleo",
                    "type": "pro",
                    "score": 9,
                    "argument_text": "Strong API-first product wedge.",
                }],
            },
        }
    ]

    monkeypatch.setattr(web_db, "_get_client", lambda: object())
    monkeypatch.setattr(web_db, "_fetch_company_run_rows", lambda _client, limit_runs: rows)
    monkeypatch.setattr(web_db, "backfill_company_runs_from_analyses", lambda limit_jobs=500: 0)
    monkeypatch.setattr(web_db, "_reconcile_missing_company_runs", lambda client, existing_rows, limit_jobs: 0)

    histories = web_db.list_company_histories(limit_runs=50)

    run_results = histories[0]["runs"][0]["results"]
    assert run_results["qa_provenance_rows"] == [{
        "startup_slug": "apaleo",
        "question": "Why now?",
        "answer": "Momentum and integrations.",
    }]
    assert run_results["argument_rows"] == [{
        "startup_slug": "apaleo",
        "type": "pro",
        "score": 9,
        "argument_text": "Strong API-first product wedge.",
    }]


def test_load_company_chat_context_collects_runs_and_chunks(monkeypatch) -> None:
    import web.db as web_db

    monkeypatch.setattr(web_db, "_get_client", lambda: object())
    monkeypatch.setattr(
        web_db,
        "_fetch_chat_company_run_rows",
        lambda client, limit_runs=2000: [
            {
                "company_id": "comp-1",
                "company_key": "slug:apify",
                "company_name": "Apify",
                "startup_slug": "apify",
                "job_id_legacy": "job-new",
                "decision": "invest",
                "run_created_at": "2026-03-10T10:00:00Z",
                "created_at": "2026-03-10T10:00:00Z",
                "result_payload": {
                    "company_name": "Apify",
                    "qa_provenance_rows": [{"question": "Why now?", "answer": "Growth inflected"}],
                    "argument_rows": [{"type": "pro", "argument_text": "Strong wedge"}],
                },
            },
            {
                "company_id": "comp-legacy",
                "company_key": "name:apify--legacy-2",
                "company_name": "Apify",
                "startup_slug": None,
                "job_id_legacy": "job-old",
                "decision": "watch",
                "run_created_at": "2026-03-01T10:00:00Z",
                "created_at": "2026-03-01T10:00:00Z",
                "result_payload": {"company_name": "Apify"},
            },
        ],
    )
    monkeypatch.setattr(
        web_db,
        "_fetch_analyses_for_company_ids",
        lambda client, company_ids: [
            {"company_id": "comp-1", "pitch_deck_id": "deck-1", "job_id_legacy": "job-new", "created_at": "2026-03-10T10:00:00Z"},
            {"company_id": "comp-legacy", "pitch_deck_id": "deck-2", "job_id_legacy": "job-old", "created_at": "2026-03-01T10:00:00Z"},
        ],
    )
    monkeypatch.setattr(
        web_db,
        "_fetch_chunks_for_pitch_deck_ids",
        lambda client, pitch_deck_ids: [
            {"pitch_deck_id": "deck-1", "chunk_id": "chunk_1", "text": "New deck evidence", "source_file": "deck.pdf", "page_or_slide": "3"},
            {"pitch_deck_id": "deck-2", "chunk_id": "chunk_1", "text": "Old deck evidence", "source_file": "deck-old.pdf", "page_or_slide": "4"},
        ],
    )

    context = web_db.load_company_chat_context("name:apify")

    assert context is not None
    assert context["company_lookup_key"] == "name:apify"
    assert [run["job_id"] for run in context["runs"]] == ["job-new", "job-old"]
    assert context["runs"][0]["chunks"][0]["text"] == "New deck evidence"
    assert context["runs"][1]["chunks"][0]["source_file"] == "deck-old.pdf"


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
    monkeypatch.setattr(
        web_db,
        "load_run_costs",
        lambda job_id_legacy: {
            "currency": "USD",
            "status": "complete",
            "total_usd": 0.0123,
            "llm_usd": 0.0123,
            "perplexity_usd": 0.0,
            "llm_tokens": {"prompt": 123, "completion": 45, "total": 168},
            "perplexity_search": {"requests": 0, "total_usd": 0.0},
            "by_model": [],
        },
    )

    loaded = web_db.load_job_results("job-123", preferred_mode="batch")

    assert loaded is not None
    assert loaded["results"]["mode"] == "batch"
    assert loaded["results"]["summary_rows"][0]["startup_slug"] == "alpha"
    assert loaded["results"]["job_status"] == "done"
    assert loaded["results"]["run_costs"]["total_usd"] == 0.0123


def test_load_latest_analysis_snapshot_ignores_company_rows(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self) -> None:
            self.filters: list[tuple[str, str, object]] = []

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, column, value):
            self.filters.append(("eq", column, value))
            return self

        def is_(self, column, value):
            self.filters.append(("is", column, value))
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            assert ("eq", "job_id_legacy", "job-123") in self.filters
            assert ("is", "company_id", "null") in self.filters
            return FakeResponse(
                [
                    {
                        "status": "done",
                        "created_at": "2026-03-14T18:33:36Z",
                        "results_payload": {
                            "mode": "batch",
                            "job_status": "done",
                            "summary_rows": [{"startup_slug": "alpha"}],
                        },
                    }
                ]
            )

    class FakeClient:
        def table(self, table_name: str):
            assert table_name == "analyses"
            return FakeQuery()

    snapshot = web_db._load_latest_analysis_snapshot(FakeClient(), "job-123")

    assert snapshot == {
        "status": "done",
        "created_at": "2026-03-14T18:33:36Z",
        "results_payload": {
            "mode": "batch",
            "job_status": "done",
            "summary_rows": [{"startup_slug": "alpha"}],
        },
    }


def test_load_job_progress_snapshot_returns_lightweight_counts(monkeypatch) -> None:
    import web.db as web_db

    monkeypatch.setattr(web_db, "_get_client", lambda: object())
    monkeypatch.setattr(
        web_db,
        "_load_latest_analysis_snapshot",
        lambda client, job_id_legacy: {
            "results_payload": {
                "mode": "batch",
                "job_status": "running",
                "job_message": "Partial results updated",
            }
        },
    )
    monkeypatch.setattr(
        web_db,
        "_load_company_progress_rows_for_job",
        lambda client, job_id_legacy: [
            {"decision": "invest"},
            {"decision": "watch"},
            {"decision": "timeout"},
        ],
    )

    payload = web_db.load_job_progress_snapshot("job-123", preferred_mode="batch")

    assert payload == {
        "results": {
            "mode": "batch",
            "num_companies": 2,
            "num_skipped": 1,
            "summary_rows_count": 2,
            "failed_rows_count": 1,
            "job_status": "running",
            "job_message": "Partial results updated",
        }
    }


def test_load_job_status_prefers_terminal_analysis_over_stale_running_status(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "status": "running",
                            "progress": "Chunk 1/2 - Evaluating company 5",
                            "created_at": "2026-03-13T10:00:00Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(
        web_db,
        "_load_latest_analysis_snapshot",
        lambda client, job_id_legacy: {
            "status": "done",
            "created_at": "2026-03-13T10:05:00Z",
            "results_payload": {"job_status": "done", "job_message": "Analysis complete"},
        },
    )

    status = web_db.load_job_status("job-123")

    assert status == {"status": "done", "progress": "Analysis complete", "worker_active": False}


def test_load_job_status_prefers_active_worker_state_over_stale_status_history(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "run_config": {
                                "worker_state": {
                                    "status": "running",
                                    "progress": "Worker running — alpha (2/10)",
                                }
                            }
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "status": "running",
                            "progress": "Chunk 1/2 - Evaluating company 1",
                            "created_at": "2026-03-13T10:00:00Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(web_db, "_load_latest_analysis_snapshot", lambda client, job_id_legacy: {})

    status = web_db.load_job_status("job-123")

    assert status == {
        "status": "running",
        "progress": "Worker running — alpha (2/10)",
        "worker_active": True,
    }


def test_load_job_status_marks_stale_worker_execution_interrupted(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "run_config": {
                                "worker_state": {
                                    "status": "running",
                                    "progress": "Worker running — alpha (2/10)",
                                    "last_heartbeat_at": "2026-03-14T18:27:35+00:00",
                                }
                            }
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "status": "running",
                            "progress": "Worker running — alpha (2/10)",
                            "created_at": "2026-03-14T18:27:35Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(web_db, "_load_latest_analysis_snapshot", lambda client, job_id_legacy: {})
    monkeypatch.setattr(
        web_db,
        "_utcnow",
        lambda: datetime(2026, 3, 14, 18, 35, 0, tzinfo=timezone.utc),
    )

    status = web_db.load_job_status("job-123")

    assert status == {
        "status": "interrupted",
        "progress": "Worker interrupted before completion.",
        "worker_active": False,
    }


def test_load_job_status_promotes_terminal_worker_state_over_stale_running_status(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "run_config": {
                                "worker_state": {
                                    "status": "done",
                                    "progress": "Analysis complete — 1/1 companies ranked",
                                }
                            }
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "status": "running",
                            "progress": "Worker running — alpha (1/1)",
                            "created_at": "2026-03-15T17:00:00Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(web_db, "_load_latest_analysis_snapshot", lambda client, job_id_legacy: {})

    status = web_db.load_job_status("job-123")

    assert status == {
        "status": "done",
        "progress": "Analysis complete — 1/1 companies ranked",
        "worker_active": False,
    }


def test_load_job_status_prefers_newer_done_worker_state_over_stale_stopped_status(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "run_config": {
                                "worker_state": {
                                    "status": "done",
                                    "progress": "Analysis complete — 1/1 companies ranked",
                                    "run_finished_at": "2026-03-15T17:05:00Z",
                                }
                            }
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "status": "stopped",
                            "progress": "Worker interrupted before completion.",
                            "created_at": "2026-03-15T17:00:00Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(web_db, "_load_latest_analysis_snapshot", lambda client, job_id_legacy: {})

    status = web_db.load_job_status("job-123")

    assert status == {
        "status": "done",
        "progress": "Analysis complete — 1/1 companies ranked",
        "worker_active": False,
    }


def test_list_saved_jobs_prefers_terminal_analysis_status_over_stale_running_status(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name
            self._filters: dict[str, object] = {}

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, key, values):
            self._filters[key] = values
            return self

        def is_(self, key, value):
            self._filters[key] = value
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-13T10:00:00Z",
                            "run_config": {},
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "status": "running",
                            "progress": "Chunk 1/2 - Evaluating company 5",
                            "created_at": "2026-03-13T10:01:00Z",
                        }
                    ]
                )
            if self.table_name == "analyses":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "status": "done",
                            "results_payload": {"job_status": "done", "job_message": "Analysis complete"},
                            "created_at": "2026-03-13T10:05:00Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-123",
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
            "run_config": {},
            "results": None,
            "has_results": True,
            "worker_active": False,
        }
    ]


def test_list_saved_jobs_ignores_company_level_analysis_rows(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name
            self.filters: list[tuple[str, str, object]] = []

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, key, values):
            self.filters.append(("in", key, tuple(values)))
            return self

        def is_(self, key, value):
            self.filters.append(("is", key, value))
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-13T10:00:00Z",
                            "run_config": {
                                "worker_state": {
                                    "status": "running",
                                    "progress": "Worker running — alpha (2/6)",
                                }
                            },
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse([])
            if self.table_name == "analyses":
                assert ("is", "company_id", "null") in self.filters
                return FakeResponse([])
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-123",
            "status": "running",
            "progress": "Worker running — alpha (2/6)",
            "created_at": "2026-03-13T10:00:00Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {
                "worker_state": {
                    "status": "running",
                    "progress": "Worker running — alpha (2/6)",
                }
            },
            "results": None,
            "has_results": False,
            "worker_active": True,
        }
    ]


def test_list_saved_jobs_marks_worker_backed_runs_active(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-13T10:00:00Z",
                            "run_config": {
                                "worker_state": {
                                    "status": "running",
                                    "progress": "Worker running — alpha (2/10)",
                                }
                            },
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse([])
            if self.table_name == "analyses":
                return FakeResponse([])
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-123",
            "status": "running",
            "progress": "Worker running — alpha (2/10)",
            "created_at": "2026-03-13T10:00:00Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {
                "worker_state": {
                    "status": "running",
                    "progress": "Worker running — alpha (2/10)",
                }
            },
            "results": None,
            "has_results": False,
            "worker_active": True,
        }
    ]


def test_list_saved_jobs_marks_stale_worker_execution_interrupted(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-14T18:27:34Z",
                            "run_config": {
                                "worker_state": {
                                    "status": "running",
                                    "progress": "Worker running — alpha (2/10)",
                                    "last_heartbeat_at": "2026-03-14T18:27:35+00:00",
                                }
                            },
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse([])
            if self.table_name == "analyses":
                return FakeResponse([])
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(
        web_db,
        "_utcnow",
        lambda: datetime(2026, 3, 14, 18, 35, 0, tzinfo=timezone.utc),
    )

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-123",
            "status": "interrupted",
            "progress": "Worker interrupted before completion.",
            "created_at": "2026-03-14T18:27:34Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {
                "worker_state": {
                    "status": "running",
                    "progress": "Worker running — alpha (2/10)",
                    "last_heartbeat_at": "2026-03-14T18:27:35+00:00",
                }
            },
            "results": None,
            "has_results": False,
            "worker_active": False,
        }
    ]


def test_list_saved_jobs_promotes_terminal_worker_state_over_stale_running_status(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-15T17:00:00Z",
                            "run_config": {
                                "worker_state": {
                                    "status": "done",
                                    "progress": "Analysis complete — 1/1 companies ranked",
                                }
                            },
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "status": "running",
                            "progress": "Worker running — alpha (1/1)",
                            "created_at": "2026-03-15T17:00:00Z",
                        }
                    ]
                )
            if self.table_name == "analyses":
                return FakeResponse([])
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-123",
            "status": "done",
            "progress": "Analysis complete — 1/1 companies ranked",
            "created_at": "2026-03-15T17:00:00Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {
                "worker_state": {
                    "status": "done",
                    "progress": "Analysis complete — 1/1 companies ranked",
                }
            },
            "results": None,
            "has_results": False,
            "worker_active": False,
        }
    ]


def test_list_saved_jobs_prefers_newer_done_analysis_over_stale_stopped_status(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-15T17:00:00Z",
                            "run_config": {
                                "worker_state": {
                                    "status": "done",
                                    "progress": "Analysis complete — 1/1 companies ranked",
                                    "run_finished_at": "2026-03-15T17:05:00Z",
                                }
                            },
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "status": "stopped",
                            "progress": "Worker interrupted before completion.",
                            "created_at": "2026-03-15T17:00:00Z",
                        }
                    ]
                )
            if self.table_name == "analyses":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-123",
                            "status": "done",
                            "results_payload": {"job_status": "done", "job_message": "Analysis complete"},
                            "created_at": "2026-03-15T17:05:01Z",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-123",
            "status": "done",
            "progress": "Analysis complete",
            "created_at": "2026-03-15T17:05:01Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {
                "worker_state": {
                    "status": "done",
                    "progress": "Analysis complete — 1/1 companies ranked",
                    "run_finished_at": "2026-03-15T17:05:00Z",
                }
            },
            "results": None,
            "has_results": True,
            "worker_active": False,
        }
    ]


def test_list_saved_jobs_keeps_recent_queued_worker_job_active(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-queued",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-14T18:27:34Z",
                            "run_config": {
                                "worker_state": {
                                    "status": "queued",
                                    "progress": "Queued for worker...",
                                    "last_heartbeat_at": "2026-03-14T18:27:34+00:00",
                                }
                            },
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse([])
            if self.table_name == "analyses":
                return FakeResponse([])
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(
        web_db,
        "_utcnow",
        lambda: datetime(2026, 3, 14, 18, 30, 0, tzinfo=timezone.utc),
    )

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-queued",
            "status": "queued",
            "progress": "Queued for worker...",
            "created_at": "2026-03-14T18:27:34Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {
                "worker_state": {
                    "status": "queued",
                    "progress": "Queued for worker...",
                    "last_heartbeat_at": "2026-03-14T18:27:34+00:00",
                }
            },
            "results": None,
            "has_results": False,
            "worker_active": True,
        }
    ]


def test_list_claimable_specter_worker_jobs_prefers_newest_rows(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self):
            self.order_kwargs: dict[str, object] = {}

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **kwargs):
            self.order_kwargs = kwargs
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            assert self.order_kwargs == {"desc": True}
            return FakeResponse(
                [
                    {
                        "job_id_legacy": "job-new",
                        "input_mode": "specter",
                        "use_web_search": False,
                        "created_at": "2026-03-14T10:00:00Z",
                        "run_config": {"worker_state": {"status": "queued", "progress": "Queued for worker..."}},
                    },
                    {
                        "job_id_legacy": "job-old",
                        "input_mode": "specter",
                        "use_web_search": False,
                        "created_at": "2026-03-01T10:00:00Z",
                        "run_config": {"worker_state": {"status": "done", "progress": "Analysis complete"}},
                    },
                ]
            )

    class FakeClient:
        def table(self, table_name: str):
            assert table_name == "jobs"
            return FakeQuery()

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_claimable_specter_worker_jobs(limit=5)

    assert rows == [
        {
            "job_id": "job-new",
            "input_mode": "specter",
            "use_web_search": False,
            "created_at": "2026-03-14T10:00:00Z",
            "run_config": {"worker_state": {"status": "queued", "progress": "Queued for worker..."}},
            "worker_state": {"status": "queued", "progress": "Queued for worker..."},
        }
    ]


def test_list_saved_jobs_exposes_optional_run_name(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-named",
                            "input_mode": "specter",
                            "use_web_search": False,
                            "created_at": "2026-03-14T20:00:00Z",
                            "started_by_label": "Jane Doe",
                            "run_config": {"run_name": "Germany shortlist"},
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse([])
            if self.table_name == "analyses":
                return FakeResponse([])
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows[0]["run_name"] == "Germany shortlist"
    assert rows[0]["started_by_label"] == "Jane Doe"


def test_list_saved_jobs_allows_reconstructable_specter_results_without_snapshot(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name
            self.filters: list[tuple[str, str, object]] = []

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, key, values):
            self.filters.append(("in", key, tuple(values)))
            return self

        def is_(self, key, value):
            self.filters.append(("is", key, value))
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-no-snapshot",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-15T09:49:19Z",
                            "run_config": {},
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-no-snapshot",
                            "status": "done",
                            "progress": "Analysis complete",
                            "created_at": "2026-03-15T09:55:00Z",
                        }
                    ]
                )
            if self.table_name == "analyses":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-no-snapshot",
                            "status": "done",
                            "results_payload": {},
                            "created_at": "2026-03-15T09:55:00Z",
                        }
                    ]
                )
            if self.table_name == "company_runs":
                assert ("in", "job_id_legacy", ("job-terminal-no-snapshot",)) in self.filters
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-no-snapshot",
                        }
                    ]
                )
            raise AssertionError(f"Unexpected table lookup: {self.table_name}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows == [
        {
            "job_id": "job-terminal-no-snapshot",
            "status": "done",
            "progress": "Analysis complete",
            "created_at": "2026-03-15T09:55:00Z",
            "input_mode": "specter",
            "use_web_search": True,
            "run_name": None,
            "started_by_user_id": None,
            "started_by_email": None,
            "started_by_display_name": None,
            "started_by_label": None,
            "run_config": {},
            "results": None,
            "has_results": True,
            "worker_active": False,
        }
    ]


def test_list_saved_jobs_probes_terminal_rows_when_batch_lookup_misses(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def __init__(self, table_name: str):
            self.table_name = table_name
            self.filters: list[tuple[str, str, object]] = []

        def select(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def in_(self, key, values):
            self.filters.append(("in", key, tuple(values)))
            return self

        def is_(self, key, value):
            self.filters.append(("is", key, value))
            return self

        def eq(self, key, value):
            self.filters.append(("eq", key, value))
            return self

        def execute(self):
            if self.table_name == "jobs":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-probe",
                            "input_mode": "specter",
                            "use_web_search": True,
                            "created_at": "2026-03-15T10:24:01Z",
                            "run_config": {},
                        }
                    ]
                )
            if self.table_name == "job_status_history":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-probe",
                            "status": "done",
                            "progress": "Analysis complete",
                            "created_at": "2026-03-15T10:29:00Z",
                        }
                    ]
                )
            if self.table_name == "analyses":
                return FakeResponse(
                    [
                        {
                            "job_id_legacy": "job-terminal-probe",
                            "status": "done",
                            "results_payload": {},
                            "created_at": "2026-03-15T10:29:00Z",
                        }
                    ]
                )
            if self.table_name == "company_runs":
                if ("in", "job_id_legacy", ("job-terminal-probe",)) in self.filters:
                    return FakeResponse([])
                if ("eq", "job_id_legacy", "job-terminal-probe") in self.filters:
                    return FakeResponse(
                        [
                            {
                                "company_key": "slug:apify",
                                "company_name": "Apify",
                                "startup_slug": "apify",
                                "job_id_legacy": "job-terminal-probe",
                                "decision": "invest",
                                "total_score": 8.0,
                                "composite_score": 82.0,
                                "bucket": "priority_review",
                                "mode": "specter",
                                "input_order": 1,
                                "run_created_at": "2026-03-15T10:28:00Z",
                                "created_at": "2026-03-15T10:28:00Z",
                                "result_payload": {
                                    "mode": "single",
                                    "startup_slug": "apify",
                                    "company_name": "Apify",
                                    "decision": "invest",
                                    "total_score": 8.0,
                                    "summary_rows": [
                                        {
                                            "startup_slug": "apify",
                                            "company_name": "Apify",
                                            "decision": "invest",
                                            "total_score": 8.0,
                                        }
                                    ],
                                },
                            }
                        ]
                    )
            raise AssertionError(f"Unexpected table lookup: {self.table_name} with filters {self.filters}")

    class FakeClient:
        def table(self, table_name: str):
            return FakeQuery(table_name)

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    rows = web_db.list_saved_jobs(limit=10)

    assert rows[0]["job_id"] == "job-terminal-probe"
    assert rows[0]["has_results"] is True


def test_ensure_source_files_bucket_accepts_dict_bucket_rows(monkeypatch) -> None:
    import web.db as web_db

    calls: list[str] = []

    class FakeStorage:
        def list_buckets(self):
            return [{"name": web_db.SOURCE_FILES_BUCKET}]

        def create_bucket(self, *_args, **_kwargs):
            calls.append("create_bucket")

    class FakeClient:
        storage = FakeStorage()

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    web_db.ensure_source_files_bucket()

    assert calls == []


def test_ensure_source_files_bucket_creates_private_bucket_with_options(monkeypatch) -> None:
    import web.db as web_db

    calls: list[tuple[str, object, object]] = []

    class FakeStorage:
        def list_buckets(self):
            return []

        def create_bucket(self, bucket_id, name=None, options=None):
            calls.append((bucket_id, name, options))

    class FakeClient:
        storage = FakeStorage()

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    web_db.ensure_source_files_bucket()

    assert calls == [(web_db.SOURCE_FILES_BUCKET, None, {"public": False})]


def test_upload_source_file_uses_string_upsert(monkeypatch, tmp_path: Path) -> None:
    import web.db as web_db

    uploads: list[tuple[str, bytes, dict[str, object]]] = []

    class FakeBucket:
        def upload(self, storage_path, payload, options):
            uploads.append((storage_path, payload, options))
            return {"Key": storage_path}

    class FakeStorage:
        def list_buckets(self):
            return [{"name": web_db.SOURCE_FILES_BUCKET}]

        def from_(self, bucket_name):
            assert bucket_name == web_db.SOURCE_FILES_BUCKET
            return FakeBucket()

    class FakeClient:
        storage = FakeStorage()

    file_path = tmp_path / "companies.csv"
    file_path.write_text("Company Name\nAlpha\n", encoding="utf-8")

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    storage_path = web_db.upload_source_file(
        "job-123",
        file_path,
        mime_type="text/csv",
    )

    assert storage_path == "jobs/job-123/inputs/companies.csv"
    assert uploads == [
        (
            "jobs/job-123/inputs/companies.csv",
            b"Company Name\nAlpha\n",
            {"upsert": "true", "content-type": "text/csv"},
        )
    ]


def test_load_run_costs_rehydrates_service_and_cost_fields_from_metadata(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def range(self, *_args, **_kwargs):
            return self

        def execute(self):
            return FakeResponse(
                [
                    {
                        "provider": "gemini",
                        "model": "gemini-3.1-flash-lite-preview",
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "total_tokens": 1500,
                        "metadata": {
                            "service": "llm",
                            "estimated_cost_usd": 0.001,
                            "request_count": 1,
                        },
                    },
                    {
                        "provider": "perplexity",
                        "model": "search_api",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "metadata": {
                            "service": "perplexity_search",
                            "estimated_cost_usd": 0.005,
                            "request_count": 1,
                        },
                    },
                ]
            )

    class FakeClient:
        def table(self, table_name: str):
            assert table_name == "model_executions"
            return FakeQuery()

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    costs = web_db.load_run_costs("job-123")

    assert costs is not None
    assert costs["llm_tokens"] == {"prompt": 1000, "completion": 500, "total": 1500}
    assert costs["llm_usd"] == 0.001
    assert costs["perplexity_usd"] == 0.005
    assert costs["total_usd"] == 0.006


def test_load_run_costs_ignores_retrying_llm_rows(monkeypatch) -> None:
    import web.db as web_db

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    class FakeQuery:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def range(self, *_args, **_kwargs):
            return self

        def execute(self):
            return FakeResponse(
                [
                    {
                        "provider": "openai",
                        "model": "gpt-5-nano",
                        "status": "done",
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "total_tokens": 1500,
                        "metadata": {
                            "service": "llm",
                            "estimated_cost_usd": 0.00025,
                            "request_count": 1,
                        },
                    },
                    {
                        "provider": "openai",
                        "model": "gpt-5-nano",
                        "status": "retrying",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "metadata": {
                            "service": "llm",
                            "request_count": 1,
                        },
                    },
                ]
            )

    class FakeClient:
        def table(self, table_name: str):
            assert table_name == "model_executions"
            return FakeQuery()

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())

    costs = web_db.load_run_costs("job-123")

    assert costs is not None
    assert costs["status"] == "complete"
    assert costs["llm_usd"] == 0.00025
    assert costs["total_usd"] == 0.00025
    assert costs["by_model"][0]["partial"] is False


def test_upsert_job_logs_supabase_failures(monkeypatch, caplog) -> None:
    import web.db as web_db

    class FakeQuery:
        def upsert(self, *_args, **_kwargs):
            raise RuntimeError("permission denied for table jobs")

    class FakeClient:
        def table(self, table_name: str):
            assert table_name == "jobs"
            return FakeQuery()

    monkeypatch.setattr(web_db, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(web_db, "_get_job_uuid", lambda client, job_id_legacy: None)

    with caplog.at_level(logging.WARNING):
        result = web_db.upsert_job("job-123")

    assert result is None
    assert "operation=upsert_job" in caplog.text
    assert "table=jobs" in caplog.text
    assert "job_id_legacy=job-123" in caplog.text
