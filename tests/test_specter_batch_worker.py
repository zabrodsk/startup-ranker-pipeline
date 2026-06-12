from agent.specter_batch_worker import _final_job_outcome


def test_final_job_outcome_marks_zero_success_worker_runs_as_error() -> None:
    status, message = _final_job_outcome(
        completed_companies=0,
        failed_companies=1,
        total_companies=1,
    )

    assert status == "error"
    assert message == "No companies were successfully evaluated. 1/1 failed."


def test_final_job_outcome_keeps_partial_success_runs_done() -> None:
    status, message = _final_job_outcome(
        completed_companies=3,
        failed_companies=2,
        total_companies=5,
    )

    assert status == "done"
    assert message == "Analysis complete — 3/5 companies ranked"


# ---------------------------------------------------------------------------
# Child fetch/ingest failures surface as structured per-company errors (S3)
# ---------------------------------------------------------------------------

import asyncio
import json
from argparse import Namespace

from agent import specter_company_worker as scw
from agent.ingest.specter_mcp_client import SpecterMCPError


def _child_args(tmp_path, **overrides) -> Namespace:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_config": {}, "versions": {}}))
    base = dict(
        job_id="job-1",
        specter_url="acme.com",
        expected_name="Acme",
        specter_companies=None,
        specter_people=None,
        company_index=None,
        absolute_index=3,
        config_path=str(config_path),
        use_web_search=False,
        vc_investment_strategy=None,
        fetch_full_team=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _capture_failure_persistence(monkeypatch):
    calls: dict = {"errors": [], "failures": []}
    # Policy building inspects configured models; CI has no API keys, so keep
    # the tests hermetic by skipping it (it runs before the fetch guard).
    monkeypatch.setattr(scw, "_pipeline_policy_from_run_config", lambda rc: None)
    monkeypatch.setattr(
        scw.db, "insert_analysis_error",
        lambda job_id, **kw: calls["errors"].append({"job_id": job_id, **kw}),
    )
    monkeypatch.setattr(
        scw.db, "persist_company_failure_result",
        lambda **kw: calls["failures"].append(kw) or True,
    )
    monkeypatch.setattr(
        scw.web_app, "_failure_result_payload",
        lambda job_id, **kw: {"job_id": job_id, "status": kw.get("status")},
    )
    return calls


def _last_event(capsys) -> dict:
    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith(scw.EVENT_PREFIX)
    ]
    assert lines, "expected a structured worker event on stdout"
    return json.loads(lines[-1][len(scw.EVENT_PREFIX):])


def test_url_fetch_failure_emits_structured_event_and_persists(tmp_path, monkeypatch, capsys):
    """A Specter outage during fetch must produce a company_complete{error}
    event + persisted failure row instead of a bare non-zero exit."""
    calls = _capture_failure_persistence(monkeypatch)

    def _boom(*args, **kwargs):
        raise SpecterMCPError("Specter token refresh failed (400): invalid_grant")

    monkeypatch.setattr(scw, "fetch_specter_company", _boom)

    rc = asyncio.run(scw._process_company(_child_args(tmp_path)))

    assert rc == 0
    event = _last_event(capsys)
    assert event["type"] == "company_complete"
    assert event["status"] == "error"
    assert event["company_name"] == "Acme"
    assert event["absolute_index"] == 3
    assert "invalid_grant" in event["error"]

    assert len(calls["errors"]) == 1
    assert calls["errors"][0]["stage"] == "specter_company_worker.fetch"
    assert calls["errors"][0]["company_slug"] == "acme"
    assert "invalid_grant" in calls["errors"][0]["message"]

    assert len(calls["failures"]) == 1
    row = calls["failures"][0]["result_row"]
    assert row["analysis_status"] == "error"
    assert row["company_name"] == "Acme"
    assert row["final_state"]["all_qa_pairs"] == []


def test_csv_ingest_failure_synthesizes_identity_from_index(tmp_path, monkeypatch, capsys):
    calls = _capture_failure_persistence(monkeypatch)
    monkeypatch.setattr(
        scw, "ingest_specter_company",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad CSV row")),
    )

    rc = asyncio.run(
        scw._process_company(
            _child_args(
                tmp_path,
                specter_url=None,
                expected_name=None,
                specter_companies="companies.csv",
                company_index=2,
            )
        )
    )

    assert rc == 0
    event = _last_event(capsys)
    assert event["status"] == "error"
    assert event["company_name"] == "company #2"
    assert "bad CSV row" in event["error"]
    assert calls["failures"][0]["result_row"]["slug"] == "company-2"


def test_fetch_failure_event_still_emitted_when_persistence_fails(tmp_path, monkeypatch, capsys):
    """DB unavailability must not suppress the structured event."""
    monkeypatch.setattr(scw, "_pipeline_policy_from_run_config", lambda rc: None)
    monkeypatch.setattr(
        scw.db, "insert_analysis_error",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    monkeypatch.setattr(scw, "fetch_specter_company",
                        lambda *a, **k: (_ for _ in ()).throw(SpecterMCPError("boom")))

    rc = asyncio.run(scw._process_company(_child_args(tmp_path)))

    assert rc == 0
    event = _last_event(capsys)
    assert event["status"] == "error"
    assert "boom" in event["error"]
