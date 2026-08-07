from __future__ import annotations

from agent.web_search.planner import QuestionRoute
from agent.web_search.portfolio import (
    PortfolioQuestion,
    build_company_web_evidence_plan,
    evidence_bucket_key,
)


QUESTIONS = [
    PortfolioQuestion(
        question="What is the company and its investment fit?",
        route_tag="internal_fit",
        aspect="general_company",
        is_root=True,
    ),
    PortfolioQuestion(
        question="What funding has Apaleo raised?",
        route_tag="company_specific",
        aspect="general_company",
    ),
    PortfolioQuestion(
        question="How large is the hotel software market?",
        route_tag="sector_market",
        aspect="market",
        is_root=True,
    ),
    PortfolioQuestion(
        question="Who are the main competitors?",
        route_tag="competitors",
        aspect="market",
    ),
    PortfolioQuestion(
        question="What customer demand supports adoption?",
        route_tag="customer_need",
        aspect="market",
    ),
    PortfolioQuestion(
        question="How is the product technically validated?",
        route_tag="technology_validation",
        aspect="product",
        is_root=True,
    ),
    PortfolioQuestion(
        question="Which regulations affect hotel software?",
        route_tag="regulation",
        aspect="product",
    ),
    PortfolioQuestion(
        question="Who are the founders and executives?",
        route_tag="company_specific",
        aspect="team",
        is_root=True,
    ),
    PortfolioQuestion(
        question="What is the company's exact burn rate?",
        route_tag="skip_public_web",
        aspect="general_company",
    ),
]


def test_company_plan_allocates_core_budget_across_evidence_categories() -> None:
    plan = build_company_web_evidence_plan(
        questions=QUESTIONS,
        company_name="Apaleo",
        company_domain="apaleo.com",
        industry_hint="hospitality software",
        geo_hint="Germany",
        current_year=2026,
        core_budget=10,
        reserve_budget=2,
    )

    assert len(plan.objectives) <= 10
    assert plan.core_budget == 10
    assert plan.reserve_budget == 2
    buckets = {objective.bucket_key for objective in plan.objectives}
    assert evidence_bucket_key(QuestionRoute.SECTOR_MARKET, "market") in buckets
    assert evidence_bucket_key(QuestionRoute.COMPETITORS, "market") in buckets
    assert evidence_bucket_key(QuestionRoute.CUSTOMER_NEED, "market") in buckets
    assert evidence_bucket_key(QuestionRoute.TECHNOLOGY_VALIDATION, "product") in buckets
    assert evidence_bucket_key(QuestionRoute.REGULATION, "product") in buckets
    assert evidence_bucket_key(QuestionRoute.COMPANY_SPECIFIC, "team") in buckets
    assert all("burn rate" not in objective.query.query for objective in plan.objectives)


def test_company_plan_round_robins_before_allocating_second_query() -> None:
    plan = build_company_web_evidence_plan(
        questions=QUESTIONS,
        company_name="Apaleo",
        company_domain="apaleo.com",
        industry_hint="hospitality software",
        geo_hint="Germany",
        current_year=2026,
        core_budget=6,
        reserve_budget=2,
    )

    assert len(plan.objectives) == 6
    assert len({objective.bucket_key for objective in plan.objectives}) == 6


def test_company_plan_is_deterministic_and_deduplicates_queries() -> None:
    kwargs = dict(
        questions=QUESTIONS + QUESTIONS,
        company_name="Apaleo",
        company_domain="apaleo.com",
        industry_hint="hospitality software",
        geo_hint="Germany",
        current_year=2026,
        core_budget=10,
        reserve_budget=2,
    )

    first = build_company_web_evidence_plan(**kwargs)
    second = build_company_web_evidence_plan(**kwargs)

    assert first == second
    queries = [objective.query.query for objective in first.objectives]
    assert len(queries) == len(set(queries))
