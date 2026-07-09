"""PR-A2: batch-worker stop semantics.

The parent worker polls the existing ``job_controls`` table once per company and,
on a stop request, finalizes the partial results with status ``stopped`` instead
of running the remaining companies. With no stop requested the run is
byte-identical to today (same companies, same ``_final_job_outcome`` status).

The ``get_job_control`` reader fails open (any read failure -> not-stopped) so a
flaky DB read can never spuriously kill a running batch.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import asyncio

import agent.specter_batch_worker as scw
import web.db as web_db


# ---------------------------------------------------------------------------
# get_job_control reader (fail-open)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, resp=None, error=False):
        self._resp = resp
        self._error = error

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self._error:
            raise RuntimeError("transient supabase error")
        return self._resp


class _FakeClient:
    def __init__(self, resp=None, error=False):
        self._resp = resp
        self._error = error

    def table(self, _name):
        return _FakeQuery(self._resp, self._error)


def test_get_job_control_parses_row(monkeypatch):
    client = _FakeClient(resp=_FakeResp([{"stop_requested": True, "pause_requested": False, "last_action": "stop"}]))
    monkeypatch.setattr(web_db, "_get_client", lambda: client)
    out = web_db.get_job_control("job-x")
    assert out["stop_requested"] is True
    assert out["last_action"] == "stop"


def test_get_job_control_no_client_returns_false(monkeypatch):
    monkeypatch.setattr(web_db, "_get_client", lambda: None)
    assert web_db.get_job_control("job-x")["stop_requested"] is False


def test_get_job_control_read_error_fails_open(monkeypatch):
    monkeypatch.setattr(web_db, "_get_client", lambda: _FakeClient(error=True))
    assert web_db.get_job_control("job-x")["stop_requested"] is False


def test_get_job_control_missing_row_returns_false(monkeypatch):
    monkeypatch.setattr(web_db, "_get_client", lambda: _FakeClient(resp=_FakeResp([])))
    assert web_db.get_job_control("job-x")["stop_requested"] is False


# ---------------------------------------------------------------------------
# _process_job stop loop
# ---------------------------------------------------------------------------
class _FakeWorkerDb:
    def __init__(self, stop_after=None, load_job_results_payload=None):
        self.stop_after = stop_after
        self.load_job_results_payload = load_job_results_payload
        self.poll_count = 0
        self.events = []
        self.snapshots = []
        self.finished = None
        self.load_job_results_calls = 0

    def get_job_control(self, _job_id):
        self.poll_count += 1
        stop = self.stop_after is not None and self.poll_count > self.stop_after
        return {"stop_requested": stop, "pause_requested": False, "last_action": None}

    def insert_analysis_event(self, _job_id, **kw):
        self.events.append(kw)

    def insert_analysis_error(self, _job_id, **kw):
        self.events.append({"is_error": True, **kw})

    def heartbeat_specter_worker_job(self, _job_id, **kw):
        pass

    def load_job_results(self, _job_id, preferred_mode=None):
        self.load_job_results_calls += 1
        return self.load_job_results_payload

    def load_run_costs(self, _job_id):
        return {}

    def persist_analysis_snapshot(self, _job_id, **kw):
        self.snapshots.append(kw)
        return True

    def finish_specter_worker_job(self, _job_id, **kw):
        self.finished = kw


def _drive_process_job(
    monkeypatch,
    *,
    n_companies,
    stop_after=None,
    subprocess_status="done",
    load_job_results_payload=None,
):
    if load_job_results_payload is None and subprocess_status == "done":
        load_job_results_payload = {"results": {"companies": []}}
    fake_db = _FakeWorkerDb(
        stop_after=stop_after,
        load_job_results_payload=load_job_results_payload,
    )
    monkeypatch.setattr(scw, "db", fake_db)
    monkeypatch.setattr(scw, "_download_worker_inputs", lambda _job_id, _rc: (None, None, None))
    tasks = [{"name": f"co{i}", "slug": f"co{i}", "mode": "url"} for i in range(1, n_companies + 1)]
    monkeypatch.setattr(scw, "_build_company_tasks", lambda _rc, _csv: tasks)
    monkeypatch.setattr(scw, "_load_completed_company_keys", lambda _job_id: (set(), 0, 0))
    monkeypatch.setattr(scw.web_app, "_parse_max_startups_from_instructions", lambda _instr: None)

    processed = []

    async def _fake_subprocess(*, company_descriptor, completed_companies, failed_companies, **_kw):
        processed.append(company_descriptor["name"])
        if subprocess_status == "quota":
            raise scw._SpecterWorkerQuotaExhausted(
                attempted_company_index=len(processed),
                total_companies=n_companies,
                completed_companies=completed_companies,
                failed_companies=failed_companies + 1,
                reset_hint="00:00 UTC",
                error_message="Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC.",
            )
        if subprocess_status == "error":
            return (completed_companies, failed_companies + 1)
        return (completed_companies + 1, failed_companies)

    monkeypatch.setattr(scw, "_run_company_subprocess", _fake_subprocess)

    asyncio.run(scw._process_job({"job_id": "job-x", "run_config": {}}, "worker-1"))
    return fake_db, processed


def test_stop_after_k_persists_partials_and_marks_stopped(monkeypatch):
    fake_db, processed = _drive_process_job(monkeypatch, n_companies=5, stop_after=3)
    assert processed == ["co1", "co2", "co3"]  # exactly k, remaining skipped
    assert fake_db.finished is not None
    assert fake_db.finished["status"] == "stopped"
    assert fake_db.finished["completed_companies"] == 3
    assert fake_db.load_job_results_calls == 1
    assert any(e.get("event_type") == "worker_stopped" for e in fake_db.events)


def test_no_stop_processes_all_and_marks_done(monkeypatch):
    fake_db, processed = _drive_process_job(monkeypatch, n_companies=4, stop_after=None)
    assert processed == ["co1", "co2", "co3", "co4"]
    assert fake_db.finished["status"] == "done"
    assert not any(e.get("event_type") == "worker_stopped" for e in fake_db.events)


def test_all_failed_without_company_runs_persists_clear_error_snapshot(monkeypatch):
    fake_db, processed = _drive_process_job(
        monkeypatch,
        n_companies=2,
        subprocess_status="error",
        load_job_results_payload=None,
    )

    assert processed == ["co1", "co2"]
    assert fake_db.finished["status"] == "error"
    assert fake_db.finished["completed_companies"] == 0
    assert fake_db.finished["failed_companies"] == 2
    assert fake_db.snapshots
    payload = fake_db.snapshots[-1]["results_payload"]
    assert payload["job_status"] == "error"
    assert payload["job_message"] == "No companies were successfully evaluated. 2/2 failed."
    assert any(e.get("event_type") == "worker_error" for e in fake_db.events)


def test_quota_exhaustion_stops_remaining_companies_and_persists_error_code(monkeypatch):
    fake_db, processed = _drive_process_job(
        monkeypatch,
        n_companies=3,
        subprocess_status="quota",
        load_job_results_payload=None,
    )

    assert processed == ["co1"]
    assert fake_db.finished["status"] == "error"
    assert fake_db.finished["completed_companies"] == 0
    assert fake_db.finished["failed_companies"] == 1
    payload = fake_db.snapshots[-1]["results_payload"]
    assert payload["error_code"] == "specter_mcp_quota_exhausted"
    assert payload["quota_remaining"] == "unknown"
    assert payload["reset_hint"] == "00:00 UTC"
    assert payload["job_message"] == (
        "Specter MCP quota exhausted after 1/3 attempts; 2 companies were not started. Reset: 00:00 UTC."
    )
    worker_state = fake_db.snapshots[-1]["worker_state"]
    assert worker_state["error_code"] == "specter_mcp_quota_exhausted"
