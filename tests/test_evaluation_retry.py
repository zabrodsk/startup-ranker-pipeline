"""Sprint 1 (W8): SingleArgumentScore enforces exactly 14 scores; manual retry is 1."""

import asyncio

import pytest
from pydantic import ValidationError

from agent.dataclasses.argument import Argument
from agent.pipeline.stages import evaluation
from agent.pipeline.state.schemas import CriterionScore, SingleArgumentScore


def _fourteen_scores() -> list[CriterionScore]:
    return [
        CriterionScore(criterion=f"criterion-{i}", score=5, reasoning="ok")
        for i in range(14)
    ]


def test_schema_accepts_exactly_14_scores():
    result = SingleArgumentScore(scores=_fourteen_scores())
    assert len(result.scores) == 14


@pytest.mark.parametrize("count", [0, 13, 15])
def test_schema_rejects_wrong_score_counts(count):
    with pytest.raises(ValidationError):
        SingleArgumentScore(
            scores=[CriterionScore(score=5, reasoning="ok") for _ in range(count)]
        )


def _make_validation_error() -> ValidationError:
    try:
        SingleArgumentScore(scores=[])
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


class _FlakyStructuredLLM:
    """Raises a schema validation error for the first `failures` calls, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise _make_validation_error()
        return SingleArgumentScore(scores=_fourteen_scores())


def _argument() -> Argument:
    return Argument(content="Strong repeat-founder team", argument_type="pro", qa_indices=[0])


def test_malformed_first_attempt_recovers_on_single_retry(monkeypatch):
    fake = _FlakyStructuredLLM(failures=1)
    monkeypatch.setattr(evaluation, "get_llm", lambda **_kwargs: fake)

    scored = asyncio.run(evaluation.score_single_argument(_argument()))

    assert fake.calls == 2
    assert scored.score == 5 * 14
    assert scored.argument_feedback


def test_persistent_malformed_scores_raise_after_exactly_two_attempts(monkeypatch):
    fake = _FlakyStructuredLLM(failures=99)
    monkeypatch.setattr(evaluation, "get_llm", lambda **_kwargs: fake)

    with pytest.raises(ValidationError):
        asyncio.run(evaluation.score_single_argument(_argument()))

    assert fake.calls == 2
