"""Stage 7: Make final investment decision.

After all iterations complete:
1. Score final arguments
2. Compare average pro vs contra scores
3. Determine invest/not_invest recommendation

This stage also handles iteration management - tracking history,
resetting state for the next iteration, and deciding when to finalize.
"""

import logging
import os
from typing import Literal

from agent.dataclasses.argument import Argument
from agent.pipeline.stages.evaluation import score_arguments_in_parallel
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState

logger = logging.getLogger(__name__)

# Final-scoring mode for prepare_final_arguments (Sprint 3 W9):
#   "rescore"         = legacy behavior: one full re-score of the final
#                       arguments before the decision (default)
#   "reuse"           = trust the selection-time scores already carried on the
#                       arguments; only score arguments that were never scored
#   "rescore_changed" = re-score only arguments whose text changed in the last
#                       refinement (their carried score was computed on the
#                       pre-refinement text)
# Read at call time (not import time) so tests and rollout can flip it via env.
PIPELINE_FINAL_SCORE_MODE_ENV = "PIPELINE_FINAL_SCORE_MODE"
_FINAL_SCORE_MODES = ("rescore", "reuse", "rescore_changed")
_DEFAULT_FINAL_SCORE_MODE = "rescore"
_WARNED_INVALID_FINAL_SCORE_MODE: set[str] = set()


def _final_score_mode() -> str:
    """Return the active final-scoring mode."""
    raw = os.getenv(PIPELINE_FINAL_SCORE_MODE_ENV, _DEFAULT_FINAL_SCORE_MODE)
    mode = raw.strip().lower()
    if mode not in _FINAL_SCORE_MODES:
        if mode not in _WARNED_INVALID_FINAL_SCORE_MODE:
            _WARNED_INVALID_FINAL_SCORE_MODE.add(mode)
            logger.warning(
                "Invalid %s=%r; falling back to %r",
                PIPELINE_FINAL_SCORE_MODE_ENV, raw, _DEFAULT_FINAL_SCORE_MODE,
            )
        return _DEFAULT_FINAL_SCORE_MODE
    return mode


def add_arguments_to_history(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Track iteration state in history.

    Saves a snapshot of the current iteration's arguments for
    later analysis and debugging.
    """
    state.arguments_history.append(
        {
            "iteration": state.current_iteration,
            "refined_arguments": state.refined_arguments,
            "selected_arguments": state.selected_arguments,
            "pro_arguments": state.pro_arguments,
            "contra_arguments": state.contra_arguments,
            "devils_advocate_pro_arguments": state.devils_advocate_pro_arguments,
            "devils_advocate_contra_arguments": state.devils_advocate_contra_arguments,
            "refined_pro_arguments": state.refined_pro_arguments,
            "refined_contra_arguments": state.refined_contra_arguments,
        }
    )

    return state


def reset_arguments_and_increment_iteration(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Prepare for next iteration cycle.

    Converts refined arguments to new current arguments,
    preserving tracking IDs and critique history.
    """
    new_current_arguments = []
    for refined_arg in state.refined_arguments:
        new_arg = Argument(
            # refined_content becomes content
            content=refined_arg.refined_content,
            qa_indices=refined_arg.refined_qa_indices,
            argument_type=refined_arg.argument_type,
            score=refined_arg.score,
            qa_pairs=refined_arg.qa_pairs,
            # Add former critique to the argument
            former_critique=refined_arg.critique,
            # Preserve tracking_id across iterations
            tracking_id=refined_arg.tracking_id,
        )
        new_current_arguments.append(new_arg)

    state.current_arguments = new_current_arguments

    # Reset other arguments to empty lists
    state.refined_arguments = []
    state.selected_arguments = []
    state.pro_arguments = []
    state.contra_arguments = []
    state.devils_advocate_pro_arguments = []
    state.devils_advocate_contra_arguments = []
    state.refined_pro_arguments = []
    state.refined_contra_arguments = []

    # Increment iteration
    state.current_iteration += 1

    return state


def check_continue(
    state: IterativeInvestmentStoryState,
) -> Literal["apply_devils_advocate", "prepare_final_arguments"]:
    """Router: continue iterating or finalize.

    Checks if we've reached max_iterations. If so, proceeds to
    final argument preparation. Otherwise, continues the loop.
    """
    if not state.should_continue_iterations:
        return "prepare_final_arguments"
    return "apply_devils_advocate"


def _last_refined_arguments_by_tracking_id(
    state: IterativeInvestmentStoryState,
) -> dict[str, Argument]:
    """Index the previous iteration's refined arguments by tracking_id.

    Arguments without a tracking_id are skipped: they cannot be matched
    reliably, so callers treat them as changed (conservative re-score).
    """
    if not state.arguments_history:
        return {}
    refined = state.arguments_history[-1].get("refined_arguments") or []
    return {arg.tracking_id: arg for arg in refined if arg.tracking_id}


def _arguments_needing_final_score(
    state: IterativeInvestmentStoryState, mode: str
) -> list[Argument]:
    """Select which final arguments still need a scoring call.

    An argument with argument_feedback is None was never scored (feedback is
    set together with the score in score_single_argument), so it is always
    scored regardless of mode. "rescore_changed" additionally re-scores
    arguments whose text changed in the last refinement — their carried score
    was computed on the pre-refinement text — and arguments it cannot match
    against the previous iteration.
    """
    never_scored = [
        arg for arg in state.final_arguments if arg.argument_feedback is None
    ]
    if mode == "reuse":
        return never_scored

    previous = _last_refined_arguments_by_tracking_id(state)
    changed = []
    for arg in state.final_arguments:
        if arg.argument_feedback is None:
            continue  # already collected above
        prior = previous.get(arg.tracking_id) if arg.tracking_id else None
        if prior is None:
            changed.append(arg)
        elif prior.refined_content is not None and prior.refined_content != prior.content:
            changed.append(arg)
    return never_scored + changed


async def prepare_final_arguments(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Score the final set of arguments.

    In the default "rescore" mode this performs one final scoring of all
    remaining arguments before the investment decision. The W9 modes
    ("reuse"/"rescore_changed") skip scoring for arguments whose carried
    selection-time score is still trusted; see _final_score_mode.
    """
    state.final_arguments = state.current_arguments

    mode = _final_score_mode()
    if mode == "rescore":
        # Score the final arguments
        scored_arguments = await score_arguments_in_parallel(state.final_arguments)
    else:
        arguments_to_score = _arguments_needing_final_score(state, mode)
        if arguments_to_score:
            # Scoring mutates the Argument objects in place, so the full
            # final_arguments list ends up consistent.
            await score_arguments_in_parallel(arguments_to_score)
        scored_arguments = state.final_arguments

    state.arguments_history.append(
        {
            "iteration": state.current_iteration,
            "selected_arguments": scored_arguments,
            "refined_arguments": scored_arguments,
        }
    )

    return state


def decide_final_investment_decision(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Compare pro/contra scores, make decision.

    Calculates average scores for pro and contra arguments.
    If pro average > contra average, recommend invest.
    Otherwise, recommend not_invest.
    """
    pro_final_arguments = [
        arg for arg in state.final_arguments if arg.argument_type == "pro"
    ]
    contra_final_arguments = [
        arg for arg in state.final_arguments if arg.argument_type == "contra"
    ]

    pro_final_arguments_score = (
        sum(arg.score for arg in pro_final_arguments) / len(pro_final_arguments)
        if pro_final_arguments
        else 0
    )
    contra_final_arguments_score = (
        sum(arg.score for arg in contra_final_arguments) / len(contra_final_arguments)
        if contra_final_arguments
        else 0
    )

    if pro_final_arguments_score > contra_final_arguments_score:
        state.final_decision = "invest"
    else:
        state.final_decision = "not_invest"

    return state


def create_final_investment_story(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Finalize the investment story state.

    Currently a no-op since we don't generate investment proposals.
    The state already contains all necessary final data.
    """
    return state
