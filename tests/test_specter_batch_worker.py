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
