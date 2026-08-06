"""Stage 1: Decompose complex questions into hierarchical question trees.

This stage takes a high-level investment question and breaks it down into
a tree of sub-questions that can be answered individually.

Example:
    "What is the market opportunity?" ->
        - "What is the TAM?"
        - "What is the SAM?"
        - "What is the SOM?"

The decomposition uses an LLM to generate a hierarchical question tree (HQDT)
that captures all the sub-questions needed to fully answer the main question.
"""

import asyncio
import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.common.llm_config import get_llm
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.pipeline.stages.question_portfolio import (
    QuestionPortfolioValidationError,
    build_portfolio_instruction,
    validate_question_portfolio,
)
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionOutput,
    DecompositionTree,
)
from agent.pipeline.utils.phase_llm import ainvoke_with_phase_fallback
from agent.prompt_library.manager import get_prompt
from agent.run_context import get_current_pipeline_policy, use_stage_context
from agent.web_search.planner import ROUTE_TAGGING_INSTRUCTION, normalize_route_tag

logger = logging.getLogger(__name__)

QUESTION_PORTFOLIO_MAX_ATTEMPTS = 3


def _normalized_route_value(tag: str | None) -> str | None:
    """Normalize an LLM-emitted route tag to a QuestionRoute value or None."""
    normalized = normalize_route_tag(tag)
    return normalized.value if normalized else None


def _build_question_tree_from_decomposition_tree(
    decomposition_tree: DecompositionTree,
    aspect: Literal["general_company", "market", "product", "team"] | None = "general_company",
) -> QuestionTree:
    """Build a hierarchical QuestionTree from the flat DecompositionTree.

    The algorithm works in two passes:
    1. Create a QuestionNode for every unique question
    2. Wire every node to its direct children, building the tree

    The first node in decomposition_tree.nodes is assumed to be the root.
    """
    # 1. Create mapping from question text to QuestionNode
    # Route tags are normalized here (unknown/missing -> None) so downstream
    # consumers only ever see valid QuestionRoute values or None.
    node_map: dict[str, QuestionNode] = {
        node.question: QuestionNode(
            question=node.question,
            sub_nodes=[],
            aspect=aspect,
            route=_normalized_route_value(node.route),
        )
        for node in decomposition_tree.nodes
    }

    # 2. Populate parent-child relationships
    for node in decomposition_tree.nodes:
        parent = node_map[node.question]
        for child_q in node.sub_questions:
            child_node = node_map.get(child_q)
            if child_node is None:
                # LLM returned a child we didn't see as standalone - create it
                child_node = QuestionNode(question=child_q, sub_nodes=[], aspect=aspect)
                node_map[child_q] = child_node
            parent.sub_nodes.append(child_node)

    # 3. Root is the first element
    root_question = decomposition_tree.nodes[0].question
    root_node = node_map[root_question]

    return QuestionTree(root_node=root_node, aspect=aspect)


async def decompose_question_async(state: DecompositionInput) -> DecompositionOutput:
    """Decompose a complex question into a hierarchical question tree.

    Takes a high-level investment question and uses an LLM to break it
    down into a tree of sub-questions customized for the given industry.
    """
    decompose_system_prompt = get_prompt("decomposition.system", state.prompt_overrides)
    decompose_user_prompt = get_prompt("decomposition.user", state.prompt_overrides)
    portfolio_instruction = ""
    if state.question_budget is not None:
        if state.aspect is None:
            raise ValueError("A question aspect is required when using a portfolio budget")
        portfolio_instruction = build_portfolio_instruction(
            state.aspect, state.question_budget
        )
    # Route tagging rides the same decomposition call (zero extra LLM calls).
    # Appended in code, not in the editable catalog, so stale persisted
    # library.json overlays cannot silently drop the instruction.
    messages = [
        SystemMessage(
            content=(
                decompose_system_prompt
                + "\n\n"
                + ROUTE_TAGGING_INSTRUCTION
                + ("\n\n" + portfolio_instruction if portfolio_instruction else "")
            )
        ),
        HumanMessage(
            content=decompose_user_prompt.format(
                question=state.question, industry=state.industry
            )
        ),
    ]

    policy = get_current_pipeline_policy()

    async def _invoke() -> DecompositionTree:
        with use_stage_context("decomposition"):
            llm = get_llm(temperature=0.5)
            llm_with_structured_output = llm.with_structured_output(DecompositionTree)
            attempt_messages = list(messages)
            attempts = (
                QUESTION_PORTFOLIO_MAX_ATTEMPTS
                if state.question_budget is not None
                else 1
            )
            for attempt in range(1, attempts + 1):
                decomposition_tree = await llm_with_structured_output.ainvoke(
                    attempt_messages
                )
                if state.question_budget is None:
                    return decomposition_tree
                try:
                    validate_question_portfolio(
                        decomposition_tree,
                        aspect=state.aspect,
                        root_question=state.question or "",
                        budget=state.question_budget,
                    )
                    return decomposition_tree
                except QuestionPortfolioValidationError as exc:
                    if attempt >= attempts:
                        raise
                    logger.warning(
                        "Question portfolio validation failed for %s (attempt %s/%s): %s",
                        state.aspect,
                        attempt,
                        attempts,
                        exc,
                    )
                    attempt_messages = messages + [
                        HumanMessage(
                            content=(
                                "The generated portfolio violated the contract: "
                                f"{exc}. Regenerate the complete category from scratch. "
                                "Do not truncate or patch the previous tree. Return "
                                f"exactly {state.question_budget} nodes and satisfy every "
                                "structural, coverage, rationale, and priority requirement."
                            )
                        )
                    ]
            raise RuntimeError("Question portfolio generation ended unexpectedly")

    decomposition_tree = await ainvoke_with_phase_fallback(
        policy.decomposition if policy else None,
        _invoke,
    )

    question_tree: QuestionTree = _build_question_tree_from_decomposition_tree(
        decomposition_tree, state.aspect
    )

    return {
        "question_tree": question_tree,
        "original_question": state.question,
    }


def decompose_question(state: DecompositionInput) -> DecompositionOutput:
    """Synchronous compatibility wrapper for tests and legacy call sites."""
    return asyncio.run(decompose_question_async(state))


# Build the graph
builder = StateGraph(DecompositionInput, output=DecompositionOutput)

builder.add_node("decompose", decompose_question_async)

builder.add_edge(START, "decompose")
builder.add_edge("decompose", END)

graph = builder.compile()


if __name__ == "__main__":
    decompose_user_prompt = get_prompt("decomposition.user")
    messages = [
        HumanMessage(
            content=decompose_user_prompt
            + "Q: Who are the key members of the founding team, and what relevant experience and track record do they have?\nA:"
        ),
    ]

    llm = get_llm(temperature=0.5)
    llm_output = llm.invoke(messages)
    print(json.dumps(llm_output.content, indent=4))
