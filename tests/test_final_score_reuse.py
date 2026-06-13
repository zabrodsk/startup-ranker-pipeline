"""Sprint 3 (W9): PIPELINE_FINAL_SCORE_MODE controls final-argument scoring.

"rescore" (default) keeps the legacy behavior byte-identical: exactly one
positional score_arguments_in_parallel call over state.final_arguments and the
same-shaped arguments_history entry. "reuse" trusts the carried selection-time
scores; "rescore_changed" re-scores only arguments whose text changed in the
last refinement.
"""

import asyncio
import logging

from agent.dataclasses.argument import Argument
from agent.pipeline.stages import decision
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState


def _argument(
    *,
    content: str = "Strong repeat-founder team",
    argument_type: str = "pro",
    score: int = 0,
    feedback: str | None = None,
    tracking_id: str = "",
    refined_content: str | None = None,
) -> Argument:
    return Argument(
        content=content,
        argument_type=argument_type,
        qa_indices=[0],
        score=score,
        argument_feedback=feedback,
        tracking_id=tracking_id,
        refined_content=refined_content,
    )


def _state(arguments: list[Argument], history: list[dict] | None = None):
    state = IterativeInvestmentStoryState()
    state.current_arguments = arguments
    if history:
        state.arguments_history = history
    return state


class _ScoringRecorder:
    """Records score_arguments_in_parallel calls and marks args as scored."""

    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        arguments = args[0] if args else kwargs["arguments"]
        for arg in arguments:
            arg.score = 42
            arg.argument_feedback = "scored-by-recorder"
        return arguments


def _run_prepare(monkeypatch, state, mode: str | None):
    recorder = _ScoringRecorder()
    monkeypatch.setattr(decision, "score_arguments_in_parallel", recorder)
    if mode is None:
        monkeypatch.delenv(decision.PIPELINE_FINAL_SCORE_MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(decision.PIPELINE_FINAL_SCORE_MODE_ENV, mode)
    result = asyncio.run(decision.prepare_final_arguments(state))
    return result, recorder


def test_default_mode_is_rescore(monkeypatch):
    monkeypatch.delenv(decision.PIPELINE_FINAL_SCORE_MODE_ENV, raising=False)
    assert decision._final_score_mode() == "rescore"


def test_rescore_calls_scoring_once_positionally_and_pins_history_shape(monkeypatch):
    arguments = [_argument(feedback="prior-feedback"), _argument(argument_type="contra")]
    state = _state(arguments)

    result, recorder = _run_prepare(monkeypatch, state, None)

    # Exactly one call, positional, with the full final_arguments list —
    # the byte-identical legacy contract.
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["args"] == (state.final_arguments,)
    assert recorder.calls[0]["kwargs"] == {}
    assert result.final_arguments is arguments

    entry = result.arguments_history[-1]
    assert set(entry.keys()) == {"iteration", "selected_arguments", "refined_arguments"}
    assert entry["selected_arguments"] == arguments
    assert entry["refined_arguments"] == arguments


def test_reuse_skips_scoring_when_all_arguments_carry_scores(monkeypatch):
    arguments = [
        _argument(score=70, feedback="fb-1"),
        _argument(argument_type="contra", score=55, feedback="fb-2"),
    ]
    state = _state(arguments)

    result, recorder = _run_prepare(monkeypatch, state, "reuse")

    assert recorder.calls == []
    assert [arg.score for arg in result.final_arguments] == [70, 55]
    entry = result.arguments_history[-1]
    assert set(entry.keys()) == {"iteration", "selected_arguments", "refined_arguments"}
    assert entry["selected_arguments"] is state.final_arguments
    assert entry["refined_arguments"] is state.final_arguments


def test_reuse_scores_arguments_that_were_never_scored(monkeypatch):
    scored = _argument(score=70, feedback="fb-1")
    unscored = _argument(argument_type="contra")
    state = _state([scored, unscored])

    result, recorder = _run_prepare(monkeypatch, state, "reuse")

    assert len(recorder.calls) == 1
    assert recorder.calls[0]["args"] == ([unscored],)
    assert unscored.score == 42
    assert scored.score == 70  # untouched
    assert result.final_arguments == [scored, unscored]


def test_rescore_changed_rescores_only_changed_tracking_ids(monkeypatch):
    # Prior iteration: refinement changed t-1's text but left t-2 unchanged.
    prior_changed = _argument(
        content="old text", tracking_id="t-1", refined_content="new text"
    )
    prior_unchanged = _argument(
        content="same text",
        argument_type="contra",
        tracking_id="t-2",
        refined_content="same text",
    )
    history = [
        {
            "iteration": 1,
            "refined_arguments": [prior_changed, prior_unchanged],
            "selected_arguments": [],
        }
    ]
    final_changed = _argument(
        content="new text", tracking_id="t-1", score=60, feedback="fb-1"
    )
    final_unchanged = _argument(
        content="same text",
        argument_type="contra",
        tracking_id="t-2",
        score=50,
        feedback="fb-2",
    )
    state = _state([final_changed, final_unchanged], history=history)

    _, recorder = _run_prepare(monkeypatch, state, "rescore_changed")

    assert len(recorder.calls) == 1
    assert recorder.calls[0]["args"] == ([final_changed],)
    assert final_changed.score == 42
    assert final_unchanged.score == 50


def test_rescore_changed_rescores_arguments_without_a_history_match(monkeypatch):
    no_tracking = _argument(score=60, feedback="fb-1")
    unknown_tracking = _argument(
        argument_type="contra", tracking_id="t-unknown", score=50, feedback="fb-2"
    )
    state = _state([no_tracking, unknown_tracking], history=[
        {"iteration": 1, "refined_arguments": [], "selected_arguments": []}
    ])

    _, recorder = _run_prepare(monkeypatch, state, "rescore_changed")

    assert len(recorder.calls) == 1
    assert recorder.calls[0]["args"] == ([no_tracking, unknown_tracking],)


def test_invalid_mode_warns_once_and_falls_back_to_rescore(monkeypatch, caplog):
    monkeypatch.setenv(decision.PIPELINE_FINAL_SCORE_MODE_ENV, "bogus-mode")
    monkeypatch.setattr(decision, "_WARNED_INVALID_FINAL_SCORE_MODE", set())

    with caplog.at_level(logging.WARNING, logger="agent.pipeline.stages.decision"):
        assert decision._final_score_mode() == "rescore"
        assert decision._final_score_mode() == "rescore"

    warnings = [r for r in caplog.records if "PIPELINE_FINAL_SCORE_MODE" in r.getMessage()]
    assert len(warnings) == 1


def test_decision_uses_carried_scores_in_reuse_mode(monkeypatch):
    arguments = [
        _argument(score=80, feedback="fb-1"),
        _argument(argument_type="contra", score=20, feedback="fb-2"),
    ]
    state = _state(arguments)

    result, recorder = _run_prepare(monkeypatch, state, "reuse")
    result = decision.decide_final_investment_decision(result)

    assert recorder.calls == []
    assert result.final_decision == "invest"
