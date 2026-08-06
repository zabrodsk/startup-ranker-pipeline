import asyncio

import pytest

from agent.pipeline.stages import decomposition
from agent.pipeline.stages.question_portfolio import (
    QUESTION_PORTFOLIO_ALLOCATION,
    QUESTION_PORTFOLIO_TOTAL,
    QuestionPortfolioValidationError,
    build_portfolio_instruction,
    validate_question_portfolio,
)
from agent.pipeline.state.decomposition import (
    DecompositionInput,
    DecompositionNode,
    DecompositionTree,
)


def test_portfolio_instruction_gives_market_an_exact_upfront_share() -> None:
    assert QUESTION_PORTFOLIO_ALLOCATION == {
        "general_company": 14,
        "market": 24,
        "product": 20,
        "team": 18,
    }
    assert sum(QUESTION_PORTFOLIO_ALLOCATION.values()) == QUESTION_PORTFOLIO_TOTAL == 76

    instruction = build_portfolio_instruction("market", 24)

    assert "76-question investment-diligence portfolio" in instruction
    assert "exactly 24 nodes" in instruction
    assert "including the root question" in instruction
    assert "Do not generate extra questions" in instruction
    assert "rank the candidates by decision value" in instruction.lower()


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


def test_valid_portfolio_requires_exact_connected_nonduplicative_tree() -> None:
    tree = _general_company_tree()

    validate_question_portfolio(
        tree,
        aspect="general_company",
        root_question="Does the company fit the investment strategy?",
        budget=14,
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda tree: tree.nodes.pop(), "exactly 14 nodes"),
        (
            lambda tree: setattr(tree.nodes[-1], "question", tree.nodes[-2].question.lower()),
            "duplicate question",
        ),
        (
            lambda tree: tree.nodes[0].sub_questions.append("Unlisted question?"),
            "not present as a node",
        ),
        (
            lambda tree: tree.nodes[-1].coverage_tags.clear(),
            "coverage_tags",
        ),
        (
            lambda tree: setattr(tree.nodes[-1], "decision_rationale", ""),
            "decision_rationale",
        ),
    ],
)
def test_invalid_portfolio_is_rejected_instead_of_truncated(mutate, expected: str) -> None:
    tree = _general_company_tree()
    mutate(tree)

    with pytest.raises(QuestionPortfolioValidationError, match=expected):
        validate_question_portfolio(
            tree,
            aspect="general_company",
            root_question="Does the company fit the investment strategy?",
            budget=14,
        )


def test_decomposition_regenerates_invalid_portfolio_without_truncating(monkeypatch) -> None:
    invocations = []
    outputs = [_general_company_tree(count=13), _general_company_tree(count=14)]

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
                question_budget=14,
            )
        )
    )

    assert len(invocations) == 2
    assert "exactly 14 nodes" in invocations[0][0].content
    assert "exactly 14 nodes" in invocations[1][-1].content
    assert "got 13" in invocations[1][-1].content
    assert len(result["question_tree"].root_node.sub_nodes) == 13
