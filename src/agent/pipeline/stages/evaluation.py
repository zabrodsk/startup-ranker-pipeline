"""Stage 5: Score arguments on 14 quality criteria.

Each argument is evaluated on dimensions like:
- Local Acceptability, Relevance, Sufficiency
- Cogency, Credibility, Clarity
- Global Acceptability, Relevance, Sufficiency
- Reasonableness, etc.

Scores determine which arguments advance to refinement.
The top K arguments (based on config) are selected for the next stage.
"""

import asyncio

import backoff
from langchain_core.messages import HumanMessage, SystemMessage
from openai import RateLimitError

from agent.common.llm_config import get_llm
from agent.dataclasses.argument import Argument
from agent.prompt_library.manager import get_criteria_mapping, get_prompt
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.pipeline.state.schemas import SingleArgumentScore
from agent.rate_limit import gather_with_concurrency
from agent.pipeline.utils.helpers import format_argument_feedback
from agent.pipeline.utils.phase_llm import ainvoke_with_phase_fallback
from agent.run_context import get_current_pipeline_policy, use_stage_context


@backoff.on_exception(
    backoff.expo, RateLimitError, max_tries=5, max_time=60, jitter=backoff.full_jitter
)
async def score_single_argument(
    argument: Argument, prompt_overrides: dict | None = None
) -> Argument:
    """Score one argument against all 14 criteria.

    Uses temperature=0.0 for consistent, reproducible scoring.
    Includes retry logic to ensure we get exactly 14 scores.
    """
    system_prompt = get_prompt("evaluation.system", prompt_overrides)
    user_prompt = get_prompt("evaluation.user", prompt_overrides)
    criteria_mapping = get_criteria_mapping(prompt_overrides)
    critique = (
        "Critique of the argument: " + argument.critique if argument.critique else ""
    )
    policy = get_current_pipeline_policy()

    # Retry logic for getting correct number of scores
    max_retries = 5
    score = None
    async def _invoke() -> SingleArgumentScore:
        with use_stage_context("evaluation"):
            llm = get_llm(temperature=0.0)
            llm_with_structured_output = llm.with_structured_output(SingleArgumentScore)
            for attempt in range(max_retries):
                result = await llm_with_structured_output.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=user_prompt.format(
                                argument=argument.content, critique=critique
                            )
                        ),
                    ]
                )

                if len(result.scores) == len(criteria_mapping):
                    return result
                if attempt < max_retries - 1:
                    print(
                        f"Attempt {attempt + 1}: Got {len(result.scores)} scores instead of {len(criteria_mapping)}, retrying..."
                    )
                    continue
                raise ValueError(
                    f"After {max_retries} attempts, still got {len(result.scores)} scores instead of {len(criteria_mapping)}"
                )

    score = await ainvoke_with_phase_fallback(
        policy.evaluation if policy else None,
        _invoke,
    )

    argument.score = sum(criterion.score for criterion in score.scores)
    argument.argument_feedback = format_argument_feedback(
        score.scores, criteria_mapping=criteria_mapping
    )
    return argument


async def score_arguments_in_parallel(
    arguments: list[Argument], prompt_overrides: dict | None = None
) -> list[Argument]:
    """Score multiple arguments concurrently.

    Uses asyncio.gather for parallel execution, significantly
    speeding up the evaluation phase.
    """
    if not arguments:
        return []

    tasks = [
        score_single_argument(argument, prompt_overrides=prompt_overrides)
        for argument in arguments
    ]
    scored_arguments = await gather_with_concurrency(tasks)

    return scored_arguments


async def score_and_select_best_k(
    state: IterativeInvestmentStoryState,
) -> dict:
    """Score all arguments, select top K for refinement.

    Scores all current arguments individually, then selects the
    top K based on the current iteration's k_best setting.

    Returns both the full scored list and the selected subset.
    """
    arguments_to_score = state.current_arguments
    scored_arguments = await score_arguments_in_parallel(
        arguments_to_score,
        prompt_overrides=state.prompt_overrides,
    )

    # Sort and select top K for current iteration
    k_best = state.get_current_k_best()
    top_arguments = sorted(scored_arguments, key=lambda x: x.score, reverse=True)[
        :k_best
    ]

    return {
        "current_arguments": scored_arguments,
        "selected_arguments": top_arguments,
    }
