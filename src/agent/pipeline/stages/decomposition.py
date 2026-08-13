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
import re
import unicodedata
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.common.llm_config import get_llm
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.pipeline.stages.question_portfolio import build_portfolio_instruction
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionOutput,
    DecompositionTree,
)
from agent.pipeline.utils.phase_llm import ainvoke_with_phase_fallback
from agent.prompt_library.manager import get_prompt
from agent.run_context import get_current_pipeline_policy, use_stage_context
from agent.web_search.planner import ROUTE_TAGGING_INSTRUCTION, normalize_route_tag


def _normalized_route_value(tag: str | None) -> str | None:
    """Normalize an LLM-emitted route tag to a QuestionRoute value or None."""
    normalized = normalize_route_tag(tag)
    return normalized.value if normalized else None


def _normalized_question_key(value: str) -> str:
    """Normalize superficial model variations for portfolio deduplication."""
    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"[^\w]+", " ", normalized).strip()


def _build_question_tree_from_decomposition_tree(
    decomposition_tree: DecompositionTree,
    aspect: Literal["general_company", "market", "product", "team"]
    | None = "general_company",
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


def _build_bounded_question_tree(
    decomposition_tree: DecompositionTree,
    *,
    root_question: str,
    aspect: Literal["general_company", "market", "product", "team"],
    max_nodes: int,
) -> QuestionTree:
    """Keep usable questions up to a soft maximum without rejecting LLM output."""
    root_text = (root_question or "").strip()
    if not root_text:
        root_text = next(
            (
                node.question.strip()
                for node in decomposition_tree.nodes
                if (node.question or "").strip()
            ),
            "What is the investment case for this company?",
        )

    root_key = _normalized_question_key(root_text)
    root_route = next(
        (
            _normalized_route_value(node.route)
            for node in decomposition_tree.nodes
            if _normalized_question_key(node.question) == root_key
        ),
        None,
    )

    root_node = QuestionNode(
        question=root_text,
        sub_nodes=[],
        aspect=aspect,
        route=root_route,
    )
    seen = {root_key}
    candidates = [(node.question, node.route) for node in decomposition_tree.nodes] + [
        (child_question, None)
        for node in decomposition_tree.nodes
        for child_question in node.sub_questions
    ]
    child_limit = max(max_nodes - 1, 0)

    for question, route in candidates:
        if len(root_node.sub_nodes) >= child_limit:
            break
        question_text = (question or "").strip()
        question_key = _normalized_question_key(question_text)
        if not question_key or question_key in seen:
            continue
        seen.add(question_key)
        root_node.sub_nodes.append(
            QuestionNode(
                question=question_text,
                sub_nodes=[],
                aspect=aspect,
                route=_normalized_route_value(route),
            )
        )

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
            raise ValueError(
                "A question aspect is required when using a portfolio budget"
            )
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
            return await llm_with_structured_output.ainvoke(messages)

    decomposition_tree = await ainvoke_with_phase_fallback(
        policy.decomposition if policy else None,
        _invoke,
    )

    if state.question_budget is not None and state.aspect is not None:
        question_tree = _build_bounded_question_tree(
            decomposition_tree,
            root_question=state.question or "",
            aspect=state.aspect,
            max_nodes=state.question_budget,
        )
    else:
        question_tree = _build_question_tree_from_decomposition_tree(
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
