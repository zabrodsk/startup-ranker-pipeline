from __future__ import annotations

from agent.coverage_planner import (
    build_coverage_search_plan,
    resolve_coverage_search_mode,
)

PACKET_ENTRIES = [
    {
        "objective": "stage_and_funding",
        "status": "supports",
        "confidence": "medium",
        "stale_objective": False,
    },
    {
        "objective": "product_software_usp",
        "status": "supports",
        "confidence": "medium",
        "stale_objective": False,
    },
]


def test_invalid_mode_defaults_off() -> None:
    assert resolve_coverage_search_mode("banana") == "off"


def test_shadow_plan_skips_supported_company_specific_objective() -> None:
    plan = build_coverage_search_plan(
        question="Has the company announced funding?",
        company_name="Acme",
        route="company_specific",
        aspect="general_company",
        legacy_query_count=1,
        packet_entries=PACKET_ENTRIES,
        mode="shadow",
    )

    assert plan is not None
    assert plan.external_route is False
    assert plan.mapped_objectives == ("stage_and_funding",)
    assert plan.objective_states["stage_and_funding"] == "supported"
    assert plan.proposed_search_count == 0
    assert plan.estimated_search_cost == 0


def test_shadow_plan_focuses_only_missing_objectives() -> None:
    plan = build_coverage_search_plan(
        question="What moat or defensibility does the company have?",
        company_name="Acme",
        route="company_specific",
        aspect="product",
        legacy_query_count=2,
        packet_entries=PACKET_ENTRIES,
        mode="shadow",
    )

    assert plan is not None
    assert plan.objective_states["moat_or_defensibility"] == "missing"
    assert plan.proposed_search_count == 1
    assert plan.proposed_queries[0].objective == "moat_or_defensibility"
    assert '"Acme"' in plan.proposed_queries[0].query


def test_external_route_retains_broad_search_budget() -> None:
    plan = build_coverage_search_plan(
        question="Who are the main competitors and alternatives?",
        company_name="Acme",
        route="competitors",
        aspect="market",
        legacy_query_count=3,
        packet_entries=PACKET_ENTRIES,
        mode="shadow",
    )

    assert plan is not None
    assert plan.external_route is True
    assert plan.proposed_search_count == 3
    assert plan.estimated_search_cost == 3
    assert all(query.objective == "external_route" for query in plan.proposed_queries)
