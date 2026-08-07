"""Flexible allowance and quality contract for investment-question decomposition.

The allowance is disclosed before generation, never applied as post-generation
truncation. The model may stop below the allowance whenever additional
questions would add little decision value.
"""

from __future__ import annotations

import re
import unicodedata

from agent.pipeline.stages.constants import QuestionAspect
from agent.pipeline.state.decomposition import DecompositionTree

QUESTION_PORTFOLIO_MAX_TOTAL = 88
QUESTION_PORTFOLIO_ALLOWANCES: dict[QuestionAspect, int] = {
    "general_company": 16,
    "market": 28,
    "product": 24,
    "team": 20,
}


class QuestionPortfolioValidationError(ValueError):
    """Raised when generated questions violate the upfront portfolio contract."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("; ".join(violations))


def _normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"[^\w]+", " ", normalized).strip()


def validate_question_portfolio(
    tree: DecompositionTree,
    *,
    aspect: QuestionAspect,
    root_question: str,
    budget: int,
) -> None:
    """Validate only the allowance and essential tree invariants."""
    violations: list[str] = []
    nodes = tree.nodes

    if len(nodes) > budget:
        violations.append(
            f"portfolio must contain no more than {budget} nodes including the root; got {len(nodes)}"
        )
    if not nodes:
        raise QuestionPortfolioValidationError(violations or ["portfolio has no root node"])

    root_key = _normalize_question(root_question)
    first_key = _normalize_question(nodes[0].question)
    if first_key != root_key:
        violations.append("the first node must be the supplied root question")

    node_by_key: dict[str, object] = {}
    for index, node in enumerate(nodes):
        key = _normalize_question(node.question)
        if not key:
            violations.append(f"node {index} has an empty question")
            continue
        if key in node_by_key:
            violations.append(f"duplicate question: {node.question!r}")
        else:
            node_by_key[key] = node

    parent_counts = {key: 0 for key in node_by_key}
    edges: dict[str, list[str]] = {key: [] for key in node_by_key}
    for node in nodes:
        parent_key = _normalize_question(node.question)
        for child_question in node.sub_questions:
            child_key = _normalize_question(child_question)
            if child_key not in node_by_key:
                violations.append(
                    f"sub-question {child_question!r} is not present as a node"
                )
                continue
            if child_key == parent_key:
                violations.append(f"question {node.question!r} cannot be its own child")
                continue
            edges.setdefault(parent_key, []).append(child_key)
            parent_counts[child_key] += 1

    if root_key in parent_counts and parent_counts[root_key] != 0:
        violations.append("root question must not have a parent")
    for key, count in parent_counts.items():
        if key != root_key and count != 1:
            violations.append(
                f"every non-root node must have exactly one parent; {key!r} has {count}"
            )

    reachable: set[str] = set()
    pending = [root_key]
    while pending:
        key = pending.pop()
        if key in reachable or key not in node_by_key:
            continue
        reachable.add(key)
        pending.extend(edges.get(key, []))
    disconnected = set(node_by_key) - reachable
    if disconnected:
        violations.append(
            f"tree must be connected; {len(disconnected)} node(s) are unreachable from the root"
        )

    if violations:
        raise QuestionPortfolioValidationError(violations)


def build_portfolio_instruction(aspect: QuestionAspect, budget: int) -> str:
    """Return the upfront quality and allowance contract for one category."""
    allowances = ", ".join(
        f"{name}<= {count}" for name, count in QUESTION_PORTFOLIO_ALLOWANCES.items()
    )
    return f"""
QUESTION PORTFOLIO CONTRACT
You are selecting the {aspect} category of an investment-diligence portfolio.
The maximum total allowance is {QUESTION_PORTFOLIO_MAX_TOTAL} nodes, allocated as: {allowances}.

Return no more than {budget} nodes for this category, including the root question.
You are not required to use the full allowance. Use fewer nodes whenever additional
questions would add little decision value. Never add filler to reach a count.

Before producing the structured output, privately:
1. Generate a broader candidate pool.
2. Remove overlapping, cosmetic, and low-materiality candidates.
3. Rank the candidates by decision value: how much a credible answer could change
   the investment assessment, expose a material risk, or test a key hypothesis.
4. Select only high-quality, non-duplicative questions and stop once the remaining
   candidates would not materially improve the assessment.

Return one connected, acyclic tree. Every sub-question must also appear exactly
once as a node; every non-root node must have exactly one parent. Questions must
be specific, independently answerable from company documents or credible external
evidence, belong to the {aspect} category, and collectively cover its material
opportunities, risks, and uncertainties.
""".strip()
