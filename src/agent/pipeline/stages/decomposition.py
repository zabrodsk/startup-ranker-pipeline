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
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.common.llm_config import get_llm
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.prompt_library.manager import get_prompt
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionNode,
    DecompositionOutput,
    DecompositionTree,
)
from agent.pipeline.utils.phase_llm import ainvoke_with_phase_fallback, invoke_with_phase_fallback
from agent.run_context import get_current_pipeline_policy, use_stage_context


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
    node_map: dict[str, QuestionNode] = {
        node.question: QuestionNode(question=node.question, sub_nodes=[], aspect=aspect)
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
    messages = [
        SystemMessage(content=decompose_system_prompt),
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
