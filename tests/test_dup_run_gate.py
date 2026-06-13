"""Sprint 3: duplicate-run gate behind RDI_DUP_RUN_GATE.

Companies whose evidence fingerprint (src/agent/dup_fingerprint.py) matches a
recent non-failed company_runs row reuse the prior result instead of
re-running the pipeline. Off by default; force_reanalyze=true bypasses;
always-visible reuse markers flow through the compose path.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.dup_fingerprint import (
    doc_fingerprint,
    identity_fingerprint,
    specter_csv_fingerprint,
)

GATE_ENV = "RDI_DUP_RUN_GATE"
WINDOW_ENV = "RDI_DUP_RUN_GATE_WINDOW_DAYS"


# --- fingerprint determinism ----------------------------------------------------


def test_doc_fingerprint_is_order_insensitive_and_content_sensitive():
    base = doc_fingerprint(["sha-a", "sha-b"])
    assert doc_fingerprint(["sha-b", "sha-a"]) == base
    assert doc_fingerprint(["sha-a", "sha-c"]) != base
    assert doc_fingerprint(["sha-a"]) != base
    assert doc_fingerprint([]) is None
    assert doc_fingerprint(["", ""]) is None


def test_specter_csv_fingerprint_is_key_order_insensitive_and_value_sensitive():
    base = specter_csv_fingerprint({"name": "Acme", "domain": "acme.com", "index": 0})
    assert (
        specter_csv_fingerprint({"index": 0, "domain": "acme.com", "name": "Acme"})
        == base
    )
    assert (
        specter_csv_fingerprint({"name": "Acme", "domain": "acme.io", "index": 0})
        != base
    )
    assert specter_csv_fingerprint({}) is None


def test_identity_fingerprint_requires_lookup_key_and_separates_modes():
    base = identity_fingerprint("name:acme", "specter")
    assert identity_fingerprint("name:acme", "specter") == base
    assert identity_fingerprint("name:other", "specter") != base
    assert identity_fingerprint("name:acme", "leadgen") != base
    assert identity_fingerprint("", "specter") is None


# --- db helpers ------------------------------------------------------------------


class _FakeQuery:
    """Records filter calls; returns canned rows on execute()."""

    def __init__(self, rows: list[dict], calls: list[tuple]):
        self._rows = rows
        self.calls = calls

    def select(self, *args):
        self.calls.append(("select", args))
        return self

    def eq(self, column, value):
        self.calls.append(("eq", column, value))
        return self

    def gte(self, column, value):
        self.calls.append(("gte", column, value))
        return self

    def order(self, column, desc=False):
        self.calls.append(("order", column, desc))
        return self

    def limit(self, n):
        self.calls.append(("limit", n))
        return self

    def upsert(self, payload, on_conflict=None):
        self.calls.append(("upsert", payload, on_conflict))
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeClient:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls: list[tuple] = []

    def table(self, name):
        self.calls.append(("table", name))
        return _FakeQuery(self.rows, self.calls)


def _prior_row(**overrides) -> dict[str, Any]:
    row = {
        "job_id_legacy": "old-job",
        "company_id": "cid-1",
        "company_key": "slug:acme",
        "company_lookup_key": "name:acme",
        "company_name": "Acme",
        "startup_slug": "acme",
        "input_order": 2,
        "decision": "invest",
        "total_score": 81.5,
        "composite_score": 74.0,
        "strategy_fit_score": 70.0,
        "team_score": 65.0,
        "upside_score": 60.0,
        "bucket": "watchlist",
        "mode": "specter",
        "evidence_fingerprint": "fp-1",
        "run_created_at": "2026-06-01T10:00:00+00:00",
        "result_payload": {
            "company_name": "Acme",
            "decision": "invest",
            "summary_rows": [{"company_name": "Acme", "decision": "invest"}],
        },
    }
    row.update(overrides)
    return row


def test_find_recent_company_run_filters_and_excludes_failures(monkeypatch):
    import web.db as web_db

    failed = _prior_row(decision="error", run_created_at="2026-06-10T10:00:00+00:00")
    good = _prior_row()
    client = _FakeClient([failed, good])
    monkeypatch.setattr(web_db, "_get_client", lambda: client)

    row = web_db.find_recent_company_run(
        evidence_fingerprint="fp-1", window_days=30, company_lookup_key="name:acme"
    )

    assert row is good  # failed row skipped, next non-failed wins
    eq_calls = [c for c in client.calls if c[0] == "eq"]
    assert ("eq", "evidence_fingerprint", "fp-1") in eq_calls
    assert ("eq", "company_lookup_key", "name:acme") in eq_calls
    assert any(c[0] == "gte" and c[1] == "run_created_at" for c in client.calls)
    assert ("order", "run_created_at", True) in client.calls


def test_find_recent_company_run_requires_fingerprint(monkeypatch):
    import web.db as web_db

    monkeypatch.setattr(
        web_db, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("no db"))
    )
    assert web_db.find_recent_company_run(evidence_fingerprint="", window_days=30) is None


def test_persist_reused_company_run_copies_columns_and_injects_markers(monkeypatch):
    import web.db as web_db

    client = _FakeClient([])
    monkeypatch.setattr(web_db, "_get_client", lambda: client)
    monkeypatch.setattr(web_db, "upsert_job", lambda *a, **k: "job-uuid-new")

    payload = web_db.persist_reused_company_run(
        "new-job", _prior_row(), run_config={"input_mode": "specter"}
    )

    assert payload["reused"] is True
    assert payload["reused_from_job_id_legacy"] == "old-job"
    assert payload["reused_run_created_at"] == "2026-06-01T10:00:00+00:00"
    assert payload["summary_rows"][0]["reused"] is True
    assert payload["summary_rows"][0]["reused_from_job"] == "old-job"

    upserts = [c for c in client.calls if c[0] == "upsert"]
    assert len(upserts) == 1
    row, on_conflict = upserts[0][1], upserts[0][2]
    assert on_conflict == "job_id_legacy,company_key"
    assert row["job_id_legacy"] == "new-job"
    assert row["job_id"] == "job-uuid-new"
    assert row["company_key"] == "slug:acme"
    assert row["company_lookup_key"] == "name:acme"
    assert row["evidence_fingerprint"] == "fp-1"
    assert row["decision"] == "invest"
    assert row["total_score"] == 81.5
    assert row["run_created_at"] != "2026-06-01T10:00:00+00:00"
    assert row["result_payload"]["reused"] is True


def test_persist_reused_company_run_returns_none_without_payload(monkeypatch):
    import web.db as web_db

    client = _FakeClient([])
    monkeypatch.setattr(web_db, "_get_client", lambda: client)
    monkeypatch.setattr(web_db, "upsert_job", lambda *a, **k: "job-uuid-new")

    assert (
        web_db.persist_reused_company_run(
            "new-job", _prior_row(result_payload=None), run_config={}
        )
        is None
    )


def test_company_lookup_key_for_name_matches_internal_derivation():
    import web.db as web_db

    assert web_db.company_lookup_key_for_name("Acme") == (
        web_db._company_lookup_key_from_values("Acme", None, None)
    )
    assert web_db.company_lookup_key_for_name("") is None


# --- compose path surfaces reuse markers ----------------------------------------


def test_reused_row_flows_through_compose_with_badge_fields():
    import web.db as web_db

    reused_payload = {
        "company_name": "Acme",
        "startup_slug": "acme",
        "decision": "invest",
        "total_score": 81.5,
        "reused": True,
        "reused_from_job_id_legacy": "old-job",
        "summary_rows": [
            {
                "company_name": "Acme",
                "startup_slug": "acme",
                "decision": "invest",
                "total_score": 81.5,
                "reused": True,
                "reused_run_created_at": "2026-06-01T10:00:00+00:00",
            }
        ],
    }
    fresh_payload = {
        "company_name": "Other",
        "startup_slug": "other",
        "decision": "invest",
        "total_score": 50.0,
        "summary_rows": [
            {"company_name": "Other", "startup_slug": "other", "decision": "invest"}
        ],
    }
    rows = [
        {"company_name": "Acme", "decision": "invest", "result_payload": reused_payload,
         "run_created_at": "2026-06-12T08:00:00+00:00"},
        {"company_name": "Other", "decision": "invest", "result_payload": fresh_payload,
         "run_created_at": "2026-06-12T08:00:00+00:00"},
    ]
    composed = web_db._compose_results_payload_from_company_runs(
        rows, preferred_mode="batch"
    )

    by_name = {r["company_name"]: r for r in composed["summary_rows"]}
    assert by_name["Acme"]["reused"] is True
    assert by_name["Acme"]["reused_run_created_at"] == "2026-06-01T10:00:00+00:00"
    assert "reused" not in by_name["Other"]


# --- app-level gate behavior ------------------------------------------------------


def _gate_env(monkeypatch, *, on: bool):
    if on:
        monkeypatch.setenv(GATE_ENV, "on")
    else:
        monkeypatch.delenv(GATE_ENV, raising=False)
    monkeypatch.delenv(WINDOW_ENV, raising=False)


def test_analyze_request_has_force_reanalyze_defaulting_false():
    import web.app as app

    req = app.AnalyzeRequest()
    assert req.force_reanalyze is False
    assert app.AnalyzeRequest(force_reanalyze=True).force_reanalyze is True


def test_gate_flag_defaults_off_and_window_falls_back(monkeypatch, caplog):
    import web.app as app

    _gate_env(monkeypatch, on=False)
    assert app._dup_run_gate_enabled() is False

    monkeypatch.setenv(WINDOW_ENV, "not-a-number")
    monkeypatch.setattr(app, "_WARNED_INVALID_DUP_GATE", set())
    with caplog.at_level(logging.WARNING, logger="web.app"):
        assert app._dup_gate_window_days() == 30
        assert app._dup_gate_window_days() == 30
    warnings = [r for r in caplog.records if WINDOW_ENV in r.getMessage()]
    assert len(warnings) == 1


def test_gate_off_never_queries_db(monkeypatch):
    import web.app as app
    import web.db as web_db

    _gate_env(monkeypatch, on=False)

    def _boom(**_kwargs):
        raise AssertionError("gate must not query when off")

    monkeypatch.setattr(web_db, "find_recent_company_run", _boom)
    result = app._maybe_reuse_company_run(
        "job-x", evidence_fingerprint="fp-1", run_config={}
    )
    assert result is None


def test_force_reanalyze_bypasses_gate(monkeypatch):
    import web.app as app
    import web.db as web_db

    _gate_env(monkeypatch, on=True)
    monkeypatch.setattr(web_db, "is_configured", lambda: True)

    def _boom(**_kwargs):
        raise AssertionError("gate must not query when force_reanalyze is set")

    monkeypatch.setattr(web_db, "find_recent_company_run", _boom)
    app._results_cache["job-force"] = {"force_reanalyze": True}
    try:
        result = app._maybe_reuse_company_run(
            "job-force", evidence_fingerprint="fp-1", run_config={}
        )
    finally:
        app._results_cache.pop("job-force", None)
    assert result is None


def test_gate_on_reuses_and_reports_progress(monkeypatch):
    import web.app as app
    import web.db as web_db

    _gate_env(monkeypatch, on=True)
    monkeypatch.setattr(web_db, "is_configured", lambda: True)
    prior = _prior_row()
    monkeypatch.setattr(
        web_db, "find_recent_company_run", lambda **_k: prior
    )
    persisted = []

    def _fake_persist(job_id, row, *, run_config, versions=None):
        persisted.append((job_id, row))
        return {"reused": True, "summary_rows": []}

    monkeypatch.setattr(web_db, "persist_reused_company_run", _fake_persist)
    progress = []
    monkeypatch.setattr(app, "_append_progress", lambda job_id, msg, **k: progress.append(msg))

    app._results_cache["job-hit"] = {}
    try:
        payload = app._maybe_reuse_company_run(
            "job-hit", evidence_fingerprint="fp-1", run_config={}, label="Acme"
        )
    finally:
        app._results_cache.pop("job-hit", None)

    assert payload == {"reused": True, "summary_rows": []}
    assert persisted == [("job-hit", prior)]
    assert any("reusing that result" in m for m in progress)


def test_identity_fingerprint_for_url_item_requires_resolved_name():
    import web.app as app
    import web.db as web_db

    fingerprint, lookup_key = app._identity_fingerprint_for_url_item(
        {"url": "https://acme.com", "resolved_name": "Acme"}
    )
    assert lookup_key == web_db.company_lookup_key_for_name("Acme")
    assert fingerprint == identity_fingerprint(lookup_key, "specter")

    assert app._identity_fingerprint_for_url_item({"url": "https://acme.com"}) == (
        None,
        None,
    )
    assert app._identity_fingerprint_for_url_item("https://acme.com") == (None, None)


def test_merge_reused_payloads_appends_rows_and_counts(monkeypatch):
    import web.app as app

    job_id = "job-merge"
    app._results_cache[job_id] = {
        "results": {
            "mode": "batch",
            "num_companies": 1,
            "summary_rows": [{"company_name": "Fresh"}],
            "argument_rows": [],
            "qa_provenance_rows": [],
        }
    }
    try:
        app._merge_reused_payloads_into_results(
            job_id,
            [
                {
                    "summary_rows": [{"company_name": "Acme", "reused": True}],
                    "argument_rows": [{"a": 1}],
                    "reused_run_created_at": "2026-06-01T10:00:00+00:00",
                }
            ],
        )
        results = app._results_cache[job_id]["results"]
        assert results["num_companies"] == 2
        assert results["summary_rows"][1]["company_name"] == "Acme"
        assert results["summary_rows"][1]["reused"] is True
        assert results["argument_rows"] == [{"a": 1}]
    finally:
        app._results_cache.pop(job_id, None)


# --- worker task plumbing ----------------------------------------------------------


def test_url_tasks_carry_identity_fingerprint_when_resolved(monkeypatch):
    from agent import specter_batch_worker as sbw

    run_config = {
        "input_mode": "specter",
        "specter_urls": [
            {"url": "https://acme.com", "resolved_name": "Acme"},
            {"url": "https://other.com"},
        ],
    }
    tasks = sbw._normalize_specter_urls(run_config)
    import web.db as web_db

    expected = identity_fingerprint(
        web_db.company_lookup_key_for_name("Acme"), "specter"
    )
    assert tasks[0]["evidence_fingerprint"] == expected
    assert tasks[1]["evidence_fingerprint"] == ""


def test_csv_tasks_carry_descriptor_fingerprint(tmp_path, monkeypatch):
    from agent import specter_batch_worker as sbw

    descriptor = {"index": 0, "name": "Acme", "domain": "acme.com"}
    monkeypatch.setattr(sbw, "list_specter_companies", lambda _p: [dict(descriptor)])
    tasks = sbw._build_company_tasks({}, companies_csv=Path("fake.csv"))

    assert tasks[0]["mode"] == "csv"
    assert tasks[0]["evidence_fingerprint"] == specter_csv_fingerprint(descriptor)
