"""Company-wide web-evidence planning with an explicit search budget.

The per-question planner remains responsible for query quality. This module
groups those queries into evidence objectives for the whole analysis and
allocates the core budget round-robin, so every represented category gets a
query before any category gets a second one.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Iterable

from .planner import (
    QuestionRoute,
    RelevancePolicy,
    SearchQuerySpec,
    build_web_search_plan,
)


@dataclass(frozen=True)
class PortfolioQuestion:
    """One question considered by the company-wide evidence allocator."""

    question: str
    route_tag: str | None
    aspect: str
    is_root: bool = False


@dataclass(frozen=True)
class PortfolioObjective:
    """One core search objective selected for the company portfolio."""

    key: str
    bucket_key: str
    route: QuestionRoute
    relevance_policy: RelevancePolicy
    representative_question: str
    query: SearchQuerySpec


@dataclass(frozen=True)
class CompanyWebEvidencePlan:
    """Bounded core objectives plus a material-gap reserve budget."""

    objectives: tuple[PortfolioObjective, ...]
    core_budget: int
    reserve_budget: int


def evidence_bucket_key(route: QuestionRoute, aspect: str) -> str:
    """Return the shared-evidence bucket used by planning and answering."""
    if route in {QuestionRoute.COMPANY_SPECIFIC, QuestionRoute.INTERNAL_FIT}:
        return f"company_specific:{(aspect or 'general_company').strip().lower()}"
    return route.value


def build_company_web_evidence_plan(
    *,
    questions: Iterable[PortfolioQuestion],
    company_name: str,
    company_domain: str | None,
    industry_hint: str | None,
    geo_hint: str | None,
    current_year: int,
    core_budget: int = 10,
    reserve_budget: int = 2,
) -> CompanyWebEvidencePlan:
    """Create a deterministic, category-balanced portfolio of search objectives."""
    normalized_core_budget = max(0, core_budget)
    normalized_reserve_budget = max(0, reserve_budget)
    buckets: OrderedDict[
        str,
        deque[tuple[QuestionRoute, RelevancePolicy, str, SearchQuerySpec]],
    ] = OrderedDict()
    seen_queries: set[str] = set()

    # Roots are deliberately considered before descendants within the caller's
    # stable aspect order: they tend to express the broadest evidence objective.
    ordered_questions = sorted(
        list(questions),
        key=lambda item: (not item.is_root,),
    )
    for item in ordered_questions:
        plan = build_web_search_plan(
            question=item.question,
            company_name=company_name,
            company_domain=company_domain,
            industry_hint=industry_hint,
            geo_hint=geo_hint,
            current_year=current_year,
            route_tag=item.route_tag,
            aspect=item.aspect,
            is_root=item.is_root,
        )
        if plan.is_skip:
            continue
        bucket_key = evidence_bucket_key(plan.route, item.aspect)
        bucket = buckets.setdefault(bucket_key, deque())
        for query in plan.queries:
            normalized_query = " ".join(query.query.lower().split())
            if not normalized_query or normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            bucket.append(
                (plan.route, plan.relevance_policy, item.question, query)
            )

    objectives: list[PortfolioObjective] = []
    active = list(buckets.items())
    while active and len(objectives) < normalized_core_budget:
        next_active: list[
            tuple[
                str,
                deque[tuple[QuestionRoute, RelevancePolicy, str, SearchQuerySpec]],
            ]
        ] = []
        for bucket_key, queue in active:
            if len(objectives) >= normalized_core_budget:
                break
            if not queue:
                continue
            route, policy, representative_question, query = queue.popleft()
            objectives.append(
                PortfolioObjective(
                    key=f"core-{len(objectives) + 1:02d}",
                    bucket_key=bucket_key,
                    route=route,
                    relevance_policy=policy,
                    representative_question=representative_question,
                    query=query,
                )
            )
            if queue:
                next_active.append((bucket_key, queue))
        active = next_active

    return CompanyWebEvidencePlan(
        objectives=tuple(objectives),
        core_budget=normalized_core_budget,
        reserve_budget=normalized_reserve_budget,
    )
