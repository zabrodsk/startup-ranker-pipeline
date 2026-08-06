"""Budget and quality contract for investment-question decomposition.

The budget is a generation constraint, not a post-generation cutoff. Each
decomposition call knows its exact share of the full portfolio before it
creates any questions.
"""

from __future__ import annotations

import re
import unicodedata

from agent.pipeline.stages.constants import QuestionAspect
from agent.pipeline.state.decomposition import DecompositionTree

QUESTION_PORTFOLIO_TOTAL = 76
QUESTION_PORTFOLIO_ALLOCATION: dict[QuestionAspect, int] = {
    "general_company": 14,
    "market": 24,
    "product": 20,
    "team": 18,
}

QUESTION_PORTFOLIO_COVERAGE: dict[QuestionAspect, tuple[str, ...]] = {
    "general_company": (
        "sector_fit",
        "stage_fit",
        "geography_fit",
        "check_size_and_ownership",
        "business_model_fit",
        "thesis_exceptions",
    ),
    "market": (
        "customer_problem",
        "target_segments",
        "tam_sam_som",
        "growth_and_timing",
        "competition_and_substitutes",
        "go_to_market_economics",
        "regulation_and_market_risk",
    ),
    "product": (
        "value_proposition",
        "core_workflow",
        "differentiation",
        "technical_architecture",
        "defensibility_ip_and_data",
        "product_maturity",
        "customer_validation",
        "scalability_security_and_compliance",
    ),
    "team": (
        "founder_identity_and_roles",
        "domain_expertise",
        "execution_track_record",
        "technical_commercial_balance",
        "commitment_and_incentives",
        "hiring_gaps",
        "governance_and_reputation",
        "network_and_access",
    ),
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
    """Validate a generated category as a portfolio; never mutate or truncate it."""
    violations: list[str] = []
    nodes = tree.nodes

    if len(nodes) != budget:
        violations.append(
            f"portfolio must contain exactly {budget} nodes including the root; got {len(nodes)}"
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

    allowed_tags = set(QUESTION_PORTFOLIO_COVERAGE[aspect])
    covered_tags: set[str] = set()
    for index, node in enumerate(nodes):
        tags = {tag.strip() for tag in node.coverage_tags if tag and tag.strip()}
        if not tags:
            violations.append(f"node {index} must include coverage_tags")
        unknown_tags = tags - allowed_tags
        if unknown_tags:
            violations.append(
                f"node {index} uses unknown coverage_tags: {', '.join(sorted(unknown_tags))}"
            )
        covered_tags.update(tags & allowed_tags)
        if index > 0:
            if not (node.decision_rationale or "").strip():
                violations.append(f"node {index} must include decision_rationale")
            if node.priority not in {"core", "supporting"}:
                violations.append(f"node {index} must set priority to core or supporting")

    missing_tags = allowed_tags - covered_tags
    if missing_tags:
        violations.append(
            "portfolio is missing required coverage_tags: "
            + ", ".join(sorted(missing_tags))
        )

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
    """Return the upfront selection contract for one portfolio category."""
    allocation = ", ".join(
        f"{name}={count}" for name, count in QUESTION_PORTFOLIO_ALLOCATION.items()
    )
    coverage = ", ".join(QUESTION_PORTFOLIO_COVERAGE[aspect])
    return f"""
QUESTION PORTFOLIO CONTRACT
You are selecting the {aspect} category of a {QUESTION_PORTFOLIO_TOTAL}-question investment-diligence portfolio.
The full allocation is fixed before generation: {allocation}.

Return exactly {budget} nodes for this category, including the root question as one node.
Do not generate extra questions. Do not draft a longer list for later truncation.

Before producing the structured output, privately:
1. Generate a broader candidate pool.
2. Remove overlapping, cosmetic, and low-materiality candidates.
3. Rank the candidates by decision value: how much a credible answer could change
   the investment assessment, expose a material risk, or test a key hypothesis.
4. Select only the strongest candidates that collectively satisfy the exact budget.

The selected questions must collectively cover these topic tags: {coverage}.
Tag every node with one or more applicable coverage_tags. Every non-root node must
include a concise decision_rationale and a priority of core or supporting.

Return one connected, acyclic tree. Every sub-question must also appear exactly
once as a node; every non-root node must have exactly one parent. Questions must
be specific, independently answerable from company documents or credible external
evidence, and non-duplicative. Never add filler merely to reach the budget.
""".strip()
