"""Stage 4: Apply devil's advocate critiques to arguments.

Each argument is challenged from the opposing perspective:
- Pro arguments are critiqued by someone against investing
- Contra arguments are critiqued by someone in favor of investing

This helps identify weaknesses and areas for improvement,
making the final arguments more robust and well-reasoned.
"""

import asyncio

import backoff
from langchain_core.messages import HumanMessage, SystemMessage
from openai import RateLimitError

from agent.common.llm_config import get_llm
from agent.common.utils import format_qa_pairs_without_index
from agent.dataclasses.argument import Argument
from agent.prompt_library.manager import get_prompt
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.pipeline.state.schemas import ArgumentCritique
from agent.rate_limit import gather_with_concurrency
from agent.run_context import get_current_pipeline_policy, use_phase_llm, use_stage_context

@backoff.on_exception(
    backoff.expo, RateLimitError, max_tries=5, max_time=60, jitter=backoff.full_jitter
)
async def _apply_devils_advocate_to_pro_argument(
    argument: Argument,
    qa_pairs_formatted: str,
    former_critique: str | None = None,
    prompt_overrides: dict | None = None,
) -> Argument:
    """Critique a single pro argument from the opposing perspective.

    The critique comes from someone against investing, challenging
    the reasons why this would be a good investment.
    """
    pro_user_prompt = get_prompt("critique.pro_user", prompt_overrides)
    pro_system_prompt = get_prompt("critique.pro_system", prompt_overrides)
    policy = get_current_pipeline_policy()
    with use_phase_llm(policy.critique if policy else None):
        with use_stage_context("critique"):
            llm = get_llm(temperature=0.5)
            llm_with_structured_output = llm.with_structured_output(ArgumentCritique)

    user_prompt = pro_user_prompt.format(
        questions_and_answers=qa_pairs_formatted, argument=argument.content
    )
    if former_critique:
        user_prompt += f"\nHere is your past critique - do not repeat the same critique but find a new one:\n{former_critique}"
    critique: ArgumentCritique = await llm_with_structured_output.ainvoke(
        [
            SystemMessage(content=pro_system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    argument.critique = critique.critique
    return argument


@backoff.on_exception(
    backoff.expo, RateLimitError, max_tries=5, max_time=60, jitter=backoff.full_jitter
)
async def _apply_devils_advocate_to_contra_argument(
    argument: Argument,
    qa_pairs_formatted: str,
    former_critique: str | None = None,
    prompt_overrides: dict | None = None,
) -> Argument:
    """Critique a single contra argument from the opposing perspective.

    The critique comes from someone in favor of investing, challenging
    the reasons why this would be a bad investment.
    """
    contra_user_prompt = get_prompt("critique.contra_user", prompt_overrides)
    contra_system_prompt = get_prompt("critique.contra_system", prompt_overrides)
    policy = get_current_pipeline_policy()
    with use_phase_llm(policy.critique if policy else None):
        with use_stage_context("critique"):
            llm = get_llm(temperature=0.5)
            llm_with_structured_output = llm.with_structured_output(ArgumentCritique)

    user_prompt = contra_user_prompt.format(
        questions_and_answers=qa_pairs_formatted, argument=argument.content
    )
    if former_critique:
        user_prompt += f"\nHere is your past critique - do not repeat the same critique but find a new one:\n{former_critique}"
    critique: ArgumentCritique = await llm_with_structured_output.ainvoke(
        [
            SystemMessage(content=contra_system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    argument.critique = critique.critique
    return argument


def apply_devils_advocate(
    state: IterativeInvestmentStoryState,
) -> IterativeInvestmentStoryState:
    """Helper node that routes to parallel pro/contra critique."""
    return state


async def apply_devils_advocate_to_pro_arguments(
    state: IterativeInvestmentStoryState,
) -> dict:
    """Apply critiques to all pro arguments in parallel.

    Uses asyncio.gather for concurrent execution, significantly
    speeding up the critique phase.
    """
    if not state.current_arguments:
        raise ValueError("No current arguments to apply devil's advocate to")

    # Use all_qa_pairs from the answering stage
    formatted_qa_pairs = format_qa_pairs_without_index(state.all_qa_pairs)

    pro_arguments = [
        arg for arg in state.current_arguments if arg.argument_type == "pro"
    ]
    if len(pro_arguments) == 0:
        return {"devils_advocate_pro_arguments": []}

    critique_tasks = [
        _apply_devils_advocate_to_pro_argument(
            arg,
            formatted_qa_pairs,
            arg.former_critique,
            state.prompt_overrides,
        )
        for arg in pro_arguments
    ]

    critiqued_arguments = await gather_with_concurrency(critique_tasks)

    return {"devils_advocate_pro_arguments": critiqued_arguments}


async def apply_devils_advocate_to_contra_arguments(
    state: IterativeInvestmentStoryState,
) -> dict:
    """Apply critiques to all contra arguments in parallel.

    Uses asyncio.gather for concurrent execution, significantly
    speeding up the critique phase.
    """
    if not state.current_arguments:
        raise ValueError("No current arguments to apply devil's advocate to")

    # Use all_qa_pairs from the answering stage
    formatted_qa_pairs = format_qa_pairs_without_index(state.all_qa_pairs)

    contra_arguments = [
        arg for arg in state.current_arguments if arg.argument_type == "contra"
    ]
    if len(contra_arguments) == 0:
        return {"devils_advocate_contra_arguments": []}

    critique_tasks = [
        _apply_devils_advocate_to_contra_argument(
            arg,
            formatted_qa_pairs,
            arg.former_critique,
            state.prompt_overrides,
        )
        for arg in contra_arguments
    ]

    critiqued_arguments = await gather_with_concurrency(critique_tasks)

    return {"devils_advocate_contra_arguments": critiqued_arguments}
