from web.app import _safe_evidence_log


def test_evidence_log_marks_score_bucket_as_authoritative() -> None:
    payload = _safe_evidence_log({
        "decision": "invest",
        "ranking_result": {
            "composite_score": 59.65,
            "bucket": "low_priority",
        },
    })

    assert payload["decision"] == "invest"
    assert payload["decision_authority"] == "advisory"
    assert payload["authoritative_outcome"] == "score_and_bucket"
    assert payload["scores"] == {
        "composite_score": 59.65,
        "strategy_fit_score": None,
        "team_score": None,
        "upside_score": None,
        "bucket": "low_priority",
    }
