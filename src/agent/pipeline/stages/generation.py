"""Stage 3: Generate pro and contra investment arguments.

Takes the answered question trees (via all_qa_pairs) and generates initial
arguments for and against investing in the company based on the Q&A pairs.

This is the first stage of the argument refinement loop.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agent.common.llm_config import get_llm
from agent.common.utils import format_qa_pairs_with_index
from agent.prompt_library.manager import get_prompt
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.pipeline.state.schemas import ArgumentsOutput
from agent.pipeline.utils.helpers import convert_llm_arguments_to_objects
from agent.pipeline.utils.phase_llm import invoke_with_phase_fallback
from agent.run_context import get_current_pipeline_policy, use_stage_context

def check_if_final(
    state: IterativeInvestmentStoryState,
) -> Literal["score_and_select_best_k", "generate_pro_and_contra_arguments"]:
    """Router: determines entry point based on is_final flag.

    If is_final is True, skip generation and go straight to scoring.
    This is used when we already have arguments and just want to
    run the final evaluation.
    """
    if state.is_final:
        if len(state.current_arguments) == 0:
            raise ValueError("No current arguments to prepare final arguments")
        return "score_and_select_best_k"
    return "generate_pro_and_contra_arguments"


def generate_pro_and_contra_arguments(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Helper node that routes to parallel pro/contra generation."""
    return state


def generate_pro_arguments(
    state: IterativeInvestmentStoryState,
) -> dict:
    """Generate N pro arguments from Q&A pairs.

    Only runs in iteration 0 - subsequent iterations use refined
    arguments from previous iterations.

    Returns dict with pro_arguments to update state.
    """
    if state.current_iteration > 0:
        return state

    # Use all_qa_pairs from the answering stage
    qa_pairs = state.all_qa_pairs
    formatted_qa_pairs = format_qa_pairs_with_index(qa_pairs)
    system_prompt = get_prompt("generation.system", state.prompt_overrides)
    pro_user_prompt = get_prompt("generation.pro_user", state.prompt_overrides)

    policy = get_current_pipeline_policy()

    def _invoke() -> ArgumentsOutput:
        with use_stage_context("generation_pro"):
            llm = get_llm(temperature=0.5)
            llm_with_structured_output = llm.with_structured_output(ArgumentsOutput)
            return llm_with_structured_output.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=pro_user_prompt.format(
                            n_pro_arguments=state.config.n_pro_arguments,
                            questions_and_answers=formatted_qa_pairs,
                        )
                    ),
                ]
            )

    arguments = invoke_with_phase_fallback(
        policy.generation if policy else None,
        _invoke,
    )

    pro_argument_objects, _ = convert_llm_arguments_to_objects(
        arguments.arguments, "pro", tracking_id_counter=1
    )

    # Attach Q&A pairs to each argument using indices
    for arg in pro_argument_objects:
        arg.qa_pairs = [
            qa_pairs[index]
            for index in arg.qa_indices
            if index < len(qa_pairs)
        ]

    return {"pro_arguments": pro_argument_objects}


def generate_contra_arguments(
    state: IterativeInvestmentStoryState,
) -> dict:
    """Generate N contra arguments from Q&A pairs.

    Only runs in iteration 0 - subsequent iterations use refined
    arguments from previous iterations.

    Returns dict with contra_arguments to update state.
    """
    if state.current_iteration > 0:
        return state

    # Use all_qa_pairs from the answering stage
    qa_pairs = state.all_qa_pairs
    formatted_qa_pairs = format_qa_pairs_with_index(qa_pairs)
    system_prompt = get_prompt("generation.system", state.prompt_overrides)
    contra_user_prompt = get_prompt("generation.contra_user", state.prompt_overrides)

    policy = get_current_pipeline_policy()

    def _invoke() -> ArgumentsOutput:
        with use_stage_context("generation_contra"):
            llm = get_llm(temperature=0.5)
            llm_with_structured_output = llm.with_structured_output(ArgumentsOutput)
            return llm_with_structured_output.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=contra_user_prompt.format(
                            n_contra_arguments=state.config.n_contra_arguments,
                            questions_and_answers=formatted_qa_pairs,
                        )
                    ),
                ]
            )

    arguments = invoke_with_phase_fallback(
        policy.generation if policy else None,
        _invoke,
    )

    # Start counter after pro arguments
    pro_args_count = len(state.pro_arguments) if state.pro_arguments else 0
    contra_argument_objects, _ = convert_llm_arguments_to_objects(
        arguments.arguments, "contra", tracking_id_counter=pro_args_count + 1
    )

    # Attach Q&A pairs to each argument using indices
    for arg in contra_argument_objects:
        arg.qa_pairs = [
            qa_pairs[index]
            for index in arg.qa_indices
            if index < len(qa_pairs)
        ]

    return {"contra_arguments": contra_argument_objects}


def merge_arguments(
    state: IterativeInvestmentStoryState,
) -> dict:
    """Combine pro and contra into current_arguments.

    This merges the separately generated arguments into a single
    list for the next stage of processing.
    """
    return {"current_arguments": state.pro_arguments + state.contra_arguments}
