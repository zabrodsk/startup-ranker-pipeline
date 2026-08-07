import asyncio

import pytest

from agent.pipeline.stages import decomposition
from agent.pipeline.stages.question_portfolio import (
    QUESTION_PORTFOLIO_ALLOWANCES,
    QUESTION_PORTFOLIO_MAX_TOTAL,
    QuestionPortfolioValidationError,
    build_portfolio_instruction,
    validate_question_portfolio,
)
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionNode,
    DecompositionTree,
)


def test_portfolio_instruction_gives_market_a_flexible_upfront_allowance() -> None:
    assert QUESTION_PORTFOLIO_ALLOWANCES == {
        "general_company": 16,
        "market": 28,
        "product": 24,
        "team": 20,
    }
    assert sum(QUESTION_PORTFOLIO_ALLOWANCES.values()) == QUESTION_PORTFOLIO_MAX_TOTAL == 88

    instruction = build_portfolio_instruction("market", 28)

    assert "maximum total allowance is 88" in instruction
    assert "no more than 28 nodes" in instruction
    assert "including the root question" in instruction
    assert "not required to use the full allowance" in instruction
    assert "Never add filler" in instruction
    assert "rank the candidates by decision value" in instruction.lower()
    assert "coverage_tags" not in instruction
    assert "decision_rationale" not in instruction
    assert "priority" not in instruction


def _general_company_tree(*, count: int = 14) -> DecompositionTree:
    children = [f"Material diligence question {index}?" for index in range(1, count)]
    nodes = [
        DecompositionNode(
            question="Does the company fit the investment strategy?",
            sub_questions=children,
        )
    ]
    nodes.extend(
        DecompositionNode(
            question=question,
            sub_questions=[],
        )
        for question in children
    )
    return DecompositionTree(nodes=nodes)


def test_valid_portfolio_allows_fewer_than_the_category_allowance() -> None:
    tree = _general_company_tree(count=11)

    validate_question_portfolio(
        tree,
        aspect="general_company",
        root_question="Does the company fit the investment strategy?",
        budget=16,
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda tree: tree.nodes.extend(
                [
                    DecompositionNode(
                        question=f"Extra material question {index}?",
                        sub_questions=[],
                    )
                    for index in range(3)
                ]
            ),
            "no more than 16 nodes",
        ),
        (
            lambda tree: setattr(tree.nodes[-1], "question", tree.nodes[-2].question.lower()),
            "duplicate question",
        ),
        (
            lambda tree: tree.nodes[0].sub_questions.append("Unlisted question?"),
            "not present as a node",
        ),
    ],
)
def test_invalid_structure_or_excess_is_rejected_instead_of_truncated(mutate, expected: str) -> None:
    tree = _general_company_tree()
    mutate(tree)

    with pytest.raises(QuestionPortfolioValidationError, match=expected):
        validate_question_portfolio(
            tree,
            aspect="general_company",
            root_question="Does the company fit the investment strategy?",
            budget=16,
        )


def test_legacy_taxonomy_metadata_is_ignored_and_cannot_break_validation() -> None:
    payload = _general_company_tree(count=12).model_dump()
    payload["nodes"][4]["coverage_tags"] = ["technology_validation"]
    payload["nodes"][4]["decision_rationale"] = "Legacy metadata."
    payload["nodes"][4]["priority"] = "core"
    tree = DecompositionTree.model_validate(payload)

    assert "coverage_tags" not in tree.nodes[4].model_dump()
    validate_question_portfolio(
        tree,
        aspect="general_company",
        root_question="Does the company fit the investment strategy?",
        budget=16,
    )


def test_decomposition_regenerates_only_when_allowance_is_exceeded(monkeypatch) -> None:
    invocations = []
    outputs = [_general_company_tree(count=17), _general_company_tree(count=13)]

    class SequenceRunnable:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            invocations.append(messages)
            return outputs.pop(0)

    monkeypatch.setattr(decomposition, "get_llm", lambda temperature=0.0: SequenceRunnable())

    result = asyncio.run(
        decomposition.decompose_question_async(
            DecompositionInput(
                question="Does the company fit the investment strategy?",
                industry="Fintech",
                aspect="general_company",
                question_budget=16,
            )
        )
    )

    assert len(invocations) == 2
    assert "no more than 16 nodes" in invocations[0][0].content
    assert "no more than 16 nodes" in invocations[1][-1].content
    assert "got 17" in invocations[1][-1].content
    assert "technology_validation" not in invocations[0][0].content
    assert "coverage_tags" not in invocations[0][0].content
    assert len(result["question_tree"].root_node.sub_nodes) == 12
