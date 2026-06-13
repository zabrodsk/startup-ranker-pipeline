"""Sprint 3 (W3): PIPELINE_EVIDENCE_DIGEST gates the evidence-corpus digest.

Off (default): critique/refinement receive the exact legacy full-corpus
strings. On: each per-argument call receives the digest plus only the evidence
cited by that argument; refinement keeps GLOBAL indices because
refined_qa_indices index into state.all_qa_pairs. Digest generation is
fail-open: any error yields None and the stages fall back to the full corpus.
"""

import asyncio
import logging

from agent.common.utils import (
    format_qa_pairs_with_index,
    format_qa_pairs_without_index,
)
from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.pipeline.stages import critique, evidence_digest, refinement
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState

QA_PAIRS = [
    {"question": "Q0: Team?", "answer": "A0: Two repeat founders."},
    {"question": "Q1: Market?", "answer": "A1: $5B SAM."},
    {"question": "Q2: Traction?", "answer": "A2: 40% MoM growth."},
]


def _company() -> Company:
    return Company(name="Acme Robotics", industry="Robotics")


def _argument(qa_indices: list[int], argument_type: str = "pro") -> Argument:
    return Argument(
        content="Strong repeat-founder team",
        argument_type=argument_type,
        qa_indices=qa_indices,
        qa_pairs=[QA_PAIRS[i] for i in qa_indices],
        argument_feedback="feedback",
    )


def _state(arguments: list[Argument], digest: str | None) -> IterativeInvestmentStoryState:
    state = IterativeInvestmentStoryState()
    state.all_qa_pairs = QA_PAIRS
    state.current_arguments = arguments
    state.selected_arguments = arguments
    state.evidence_digest = digest
    return state


class _Recorder:
    """Async recorder standing in for the per-argument critique/refine calls."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result

    async def __call__(self, argument, qa_pairs_formatted, *args, **kwargs):
        self.calls.append(qa_pairs_formatted)
        return self._result if self._result is not None else argument


# --- flag getter -----------------------------------------------------------


def test_digest_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, raising=False)
    assert evidence_digest.is_evidence_digest_enabled() is False


def test_invalid_digest_mode_warns_once_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "bogus")
    monkeypatch.setattr(evidence_digest, "_WARNED_INVALID_DIGEST_MODE", set())

    with caplog.at_level(logging.WARNING, logger=evidence_digest.__name__):
        assert evidence_digest.is_evidence_digest_enabled() is False
        assert evidence_digest.is_evidence_digest_enabled() is False

    warnings = [
        r for r in caplog.records if "PIPELINE_EVIDENCE_DIGEST" in r.getMessage()
    ]
    assert len(warnings) == 1


# --- maybe_build_evidence_digest -------------------------------------------


def test_digest_is_none_when_flag_off(monkeypatch):
    monkeypatch.delenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, raising=False)

    async def _boom(*_args, **_kwargs):
        raise AssertionError("must not invoke any LLM when off")

    monkeypatch.setattr(evidence_digest, "ainvoke_with_phase_fallback", _boom)
    result = asyncio.run(
        evidence_digest.maybe_build_evidence_digest(QA_PAIRS, _company())
    )
    assert result is None


def test_digest_is_none_for_empty_corpus(monkeypatch):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "on")
    result = asyncio.run(evidence_digest.maybe_build_evidence_digest([], _company()))
    assert result is None


def test_digest_failure_is_fail_open(monkeypatch, caplog):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "on")

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(evidence_digest, "ainvoke_with_phase_fallback", _boom)
    with caplog.at_level(logging.WARNING, logger=evidence_digest.__name__):
        result = asyncio.run(
            evidence_digest.maybe_build_evidence_digest(QA_PAIRS, _company())
        )
    assert result is None
    assert any("fall back" in r.getMessage() for r in caplog.records)


def test_digest_success_returns_stripped_text(monkeypatch):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "on")

    async def _fake(_selection, _invoke):
        return "  My digest  "

    monkeypatch.setattr(evidence_digest, "ainvoke_with_phase_fallback", _fake)
    result = asyncio.run(
        evidence_digest.maybe_build_evidence_digest(QA_PAIRS, _company())
    )
    assert result == "My digest"


# --- critique routing -------------------------------------------------------


def test_critique_off_mode_passes_exact_full_corpus(monkeypatch):
    monkeypatch.delenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(critique, "_apply_devils_advocate_to_pro_argument", recorder)

    state = _state([_argument([0])], digest="should-not-be-used")
    asyncio.run(critique.apply_devils_advocate_to_pro_arguments(state))

    assert recorder.calls == [format_qa_pairs_without_index(QA_PAIRS)]


def test_critique_on_mode_injects_digest_plus_own_pairs(monkeypatch):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "on")
    recorder = _Recorder()
    monkeypatch.setattr(critique, "_apply_devils_advocate_to_pro_argument", recorder)

    state = _state([_argument([0, 2])], digest="THE-DIGEST")
    asyncio.run(critique.apply_devils_advocate_to_pro_arguments(state))

    (evidence_text,) = recorder.calls
    assert "=== Evidence corpus digest ===" in evidence_text
    assert "THE-DIGEST" in evidence_text
    assert QA_PAIRS[0]["question"] in evidence_text
    assert QA_PAIRS[2]["question"] in evidence_text
    assert QA_PAIRS[1]["question"] not in evidence_text  # not cited by the argument


def test_critique_on_mode_without_digest_falls_back_to_full_corpus(monkeypatch):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "on")
    recorder = _Recorder()
    monkeypatch.setattr(
        critique, "_apply_devils_advocate_to_contra_argument", recorder
    )

    state = _state([_argument([0], argument_type="contra")], digest=None)
    asyncio.run(critique.apply_devils_advocate_to_contra_arguments(state))

    assert recorder.calls == [format_qa_pairs_without_index(QA_PAIRS)]


# --- refinement routing ------------------------------------------------------


class _RefinedResult:
    content = "refined"
    qa_indices = [0]


def test_refinement_off_mode_passes_exact_indexed_corpus(monkeypatch):
    monkeypatch.delenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, raising=False)
    recorder = _Recorder(result=_RefinedResult())
    monkeypatch.setattr(refinement, "_refine_individual_pro_argument", recorder)

    state = _state([_argument([0])], digest="should-not-be-used")
    asyncio.run(refinement.refine_pro_arguments(state))

    assert recorder.calls == [format_qa_pairs_with_index(QA_PAIRS)]


def test_refinement_on_mode_uses_global_indices(monkeypatch):
    monkeypatch.setenv(evidence_digest.PIPELINE_EVIDENCE_DIGEST_ENV, "on")
    recorder = _Recorder(result=_RefinedResult())
    monkeypatch.setattr(refinement, "_refine_individual_pro_argument", recorder)

    state = _state([_argument([2])], digest="THE-DIGEST")
    asyncio.run(refinement.refine_pro_arguments(state))

    (evidence_text,) = recorder.calls
    assert "THE-DIGEST" in evidence_text
    # Cited pair keeps its corpus-global index (2), not a local index (0).
    assert f"2: {QA_PAIRS[2]['question']}" in evidence_text
    assert QA_PAIRS[0]["question"] not in evidence_text
    assert QA_PAIRS[1]["question"] not in evidence_text


def test_refinement_composite_skips_out_of_bounds_indices():
    argument = _argument([0])
    argument.qa_indices = [0, 99]
    text = evidence_digest.compose_refinement_evidence("D", argument, QA_PAIRS)
    assert f"0: {QA_PAIRS[0]['question']}" in text
    assert "99:" not in text


# --- prompt registry ---------------------------------------------------------


def test_evidence_digest_prompts_are_registered():
    from agent.prompt_library.manager import get_prompt

    system_prompt = get_prompt("evidence_digest.system")
    user_prompt = get_prompt("evidence_digest.user")
    assert "digest" in system_prompt.lower()
    assert "{company_name}" in user_prompt
    assert "{qa_corpus}" in user_prompt
