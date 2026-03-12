"""Parallel decomposition of all 4 investment questions.

This stage takes a company as input and decomposes all 4 predefined
investment questions in parallel, checking cache first to avoid
redundant LLM calls.

The 4 questions cover:
- General company alignment
- Market size and growth
- Product features and technology
- Team experience and track record
"""

import hashlib
from typing import Dict

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionTree
from agent.prompt_library.manager import get_prompt, get_questions
from agent.pipeline.stages.cache import cache_question_tree, get_cached_question_tree
from agent.pipeline.stages.constants import QuestionAspect
from agent.pipeline.stages.decomposition import graph as decomposition_graph
from agent.pipeline.state.decomposition import DecompositionInput
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.rate_limit import gather_with_concurrency


async def _decompose_single_question(
    question: str,
    industry: str,
    aspect: QuestionAspect,
    prompt_overrides: dict | None = None,
) -> Dict[str, QuestionTree | str]:
    """Decompose a single question using the decomposition graph.

    Args:
        question: The question to decompose
        industry: The company's industry for customization
        aspect: The question aspect (general_company, market, product, team)

    Returns:
        Dict with aspect key and decomposed QuestionTree
    """
    result = await decomposition_graph.ainvoke(
        DecompositionInput(
            question=question,
            industry=industry,
            aspect=aspect,
            prompt_overrides=prompt_overrides or {},
        )
    )
    return {"aspect": aspect, "tree": result["question_tree"]}


async def _get_or_decompose_question(
    question: str,
    industry: str,
    aspect: QuestionAspect,
    company_name: str,
    prompt_overrides: dict | None = None,
) -> Dict[str, QuestionTree | str]:
    """Get cached question tree or decompose if not cached.

    Args:
        question: The question to decompose
        industry: The company's industry for customization
        aspect: The question aspect
        company_name: The company name for cache key

    Returns:
        Dict with aspect key and QuestionTree (cached or newly decomposed)
    """
    # Keep decomposition caches company-scoped to preserve company-specific
    # question shaping even if prompts or upstream context evolve.
    cache_company = Company(name=company_name, industry=industry)
    decomposition_signature = hashlib.sha256(
        (
            str(get_prompt("decomposition.system", prompt_overrides))
            + "\n||\n"
            + str(get_prompt("decomposition.user", prompt_overrides))
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_question = f"{question} [decompose:{decomposition_signature}]"

    # Check cache first
    cached_tree = get_cached_question_tree(cache_question, cache_company, aspect)
    if cached_tree is not None:
        return {"aspect": aspect, "tree": cached_tree}

    # Not cached - decompose and cache
    result = await _decompose_single_question(
        question,
        industry,
        aspect,
        prompt_overrides=prompt_overrides,
    )
    cache_question_tree(cache_question, cache_company, result["tree"])
    return result


async def decompose_all_questions(
    state: IterativeInvestmentStoryState,
) -> Dict[str, Dict[str, QuestionTree]]:
    """Decompose all 4 investment questions in parallel.

    Takes the company from state and decomposes all 4 predefined questions,
    checking cache first for each. Returns dict of question_trees keyed by aspect.

    Args:
        state: The pipeline state containing the company

    Returns:
        Dict with question_trees mapping aspect to QuestionTree
    """
    if state.company is None:
        raise ValueError("Company is required for decomposition")

    questions = get_questions(state.prompt_overrides)

    # Create tasks for all 4 questions
    tasks = []
    for aspect, question in questions.items():
        tasks.append(
            _get_or_decompose_question(
                question=question,
                industry=state.company.industry,
                aspect=aspect,
                company_name=state.company.name,
                prompt_overrides=state.prompt_overrides,
            )
        )

    results = await gather_with_concurrency(tasks)

    # Build the question_trees dict
    question_trees: Dict[str, QuestionTree] = {}
    for result in results:
        question_trees[result["aspect"]] = result["tree"]

    return {"question_trees": question_trees}
