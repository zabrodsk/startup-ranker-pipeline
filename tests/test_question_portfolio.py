import asyncio

import pytest

from agent.pipeline.stages import decomposition
from agent.pipeline.stages.question_portfolio import (
    QUESTION_PORTFOLIO_ALLOCATION,
    QUESTION_PORTFOLIO_TOTAL,
    build_portfolio_instruction,
)
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionNode,
    DecompositionTree,
)


def test_portfolio_instruction_sets_a_soft_maximum() -> None:
    assert QUESTION_PORTFOLIO_ALLOCATION == {
        "general_company": 14,
        "market": 24,
        "product": 20,
        "team": 18,
    }
    assert sum(QUESTION_PORTFOLIO_ALLOCATION.values()) == QUESTION_PORTFOLIO_TOTAL == 76

    instruction = build_portfolio_instruction("market", 24)

    assert "up to 24 questions" in instruction
    assert "including the root question" in instruction
    assert "Fewer questions are acceptable" in instruction
    assert "exactly 24 nodes" not in instruction
    assert "coverage_tags" not in instruction


def _general_company_tree(*, count: int = 14) -> DecompositionTree:
    tags = [
        "sector_fit",
        "stage_fit",
        "geography_fit",
        "check_size_and_ownership",
        "business_model_fit",
        "thesis_exceptions",
    ]
    children = [f"Material diligence question {index}?" for index in range(1, count)]
    nodes = [
        DecompositionNode(
            question="Does the company fit the investment strategy?",
            sub_questions=children,
            coverage_tags=tags,
            priority="core",
        )
    ]
    nodes.extend(
        DecompositionNode(
            question=question,
            sub_questions=[],
            coverage_tags=[tags[index % len(tags)]],
            decision_rationale=f"Tests material investment issue {index}.",
            priority="core" if index < 7 else "supporting",
        )
        for index, question in enumerate(children)
    )
    return DecompositionTree(nodes=nodes)


@pytest.mark.parametrize(
    ("generated_count", "question_budget", "expected_children"),
    [
        (9, 14, 8),
        (23, 24, 22),
        (19, 20, 18),
        (26, 24, 23),
    ],
)
def test_decomposition_accepts_observed_count_mismatches_without_retry(
    monkeypatch, generated_count: int, question_budget: int, expected_children: int
) -> None:
    invocations = []
    output = _general_company_tree(count=generated_count)

    class SequenceRunnable:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            invocations.append(messages)
            return output

    monkeypatch.setattr(
        decomposition, "get_llm", lambda temperature=0.0: SequenceRunnable()
    )

    result = asyncio.run(
        decomposition.decompose_question_async(
            DecompositionInput(
                question="Does the company fit the investment strategy?",
                industry="Fintech",
                aspect="general_company",
                question_budget=question_budget,
            )
        )
    )

    assert len(invocations) == 1
    assert len(result["question_tree"].root_node.sub_nodes) == expected_children


def test_decomposition_ignores_unknown_coverage_tags(monkeypatch) -> None:
    output = _general_company_tree()
    output.nodes[-1].coverage_tags = ["geography"]

    class StaticRunnable:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return output

    monkeypatch.setattr(
        decomposition, "get_llm", lambda temperature=0.0: StaticRunnable()
    )

    result = asyncio.run(
        decomposition.decompose_question_async(
            DecompositionInput(
                question="Does the company fit the investment strategy?",
                industry="Fintech",
                aspect="general_company",
                question_budget=14,
            )
        )
    )

    assert len(result["question_tree"].root_node.sub_nodes) == 13


def test_decomposition_uses_the_known_root_when_generation_is_empty(
    monkeypatch,
) -> None:
    class StaticRunnable:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return DecompositionTree(nodes=[])

    monkeypatch.setattr(
        decomposition, "get_llm", lambda temperature=0.0: StaticRunnable()
    )

    result = asyncio.run(
        decomposition.decompose_question_async(
            DecompositionInput(
                question="Does the company fit the investment strategy?",
                industry="Fintech",
                aspect="general_company",
                question_budget=14,
            )
        )
    )

    assert result["question_tree"].root_node.question == (
        "Does the company fit the investment strategy?"
    )
    assert result["question_tree"].root_node.sub_nodes == []


def test_bounded_tree_preserves_the_generated_root_route() -> None:
    output = DecompositionTree(
        nodes=[
            DecompositionNode(
                question="What is the TAM",
                sub_questions=[],
                route="Sector-Market",
            )
        ]
    )

    tree = decomposition._build_bounded_question_tree(
        output,
        root_question="What is the TAM?",
        aspect="market",
        max_nodes=24,
    )

    assert tree.root_node.route == "sector_market"


def test_bounded_tree_deduplicates_punctuation_and_unicode_variants() -> None:
    output = DecompositionTree(
        nodes=[
            DecompositionNode(question="What is the TAM", sub_questions=[]),
            DecompositionNode(question="What is the TＡM?", sub_questions=[]),
            DecompositionNode(question="What is the SAM?", sub_questions=[]),
        ]
    )

    tree = decomposition._build_bounded_question_tree(
        output,
        root_question="What is the TAM?",
        aspect="market",
        max_nodes=24,
    )

    assert [node.question for node in tree.root_node.sub_nodes] == ["What is the SAM?"]
