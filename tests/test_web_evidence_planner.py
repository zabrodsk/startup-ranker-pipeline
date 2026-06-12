"""Unit tests for the route-aware WebEvidencePlanner (pure logic, no network)."""

import pytest

from agent.web_search.planner import (
    MAX_QUERIES_PER_QUESTION,
    WEB_SEARCH_QUERY_MAX_LEN,
    QuestionRoute,
    RelevancePolicy,
    build_web_search_plan,
    evaluate_web_result_relevance,
    normalize_route_tag,
)

MANTIC = dict(
    company_name="Mantic",
    company_domain="mantic.ai",
    industry_hint="wildfire detection sensors",
    geo_hint="Europe",
    current_year=2026,
)


def _all_query_text(plan) -> str:
    return " ".join(q.query for q in plan.queries)


# --- Handoff §18 minimal cases -------------------------------------------------


def test_sector_market_question_routes_without_company_name():
    plan = build_web_search_plan(
        question="Is the sector attractive given market size, growth and regulation?",
        **{**MANTIC, "company_domain": None},
    )
    assert plan.route == QuestionRoute.SECTOR_MARKET
    assert plan.relevance_policy == RelevancePolicy.TOPIC_KEYWORDS
    assert plan.queries, "sector route must produce queries"
    for spec in plan.queries:
        assert "mantic" not in spec.query.lower()
        assert spec.domain_filter is None


def test_funding_question_routes_company_specific_with_restrictive_domains():
    plan = build_web_search_plan(
        question="Has the company announced funding?", **MANTIC
    )
    assert plan.route == QuestionRoute.COMPANY_SPECIFIC
    assert plan.relevance_policy == RelevancePolicy.COMPANY_MENTION
    assert len(plan.queries) == 1
    spec = plan.queries[0]
    assert "mantic" in spec.query.lower()
    assert spec.domain_filter == ("mantic.ai", "crunchbase.com", "linkedin.com")


def test_arr_question_routes_to_skip_with_reason():
    plan = build_web_search_plan(
        question="What is the company's current ARR?", **MANTIC
    )
    assert plan.route == QuestionRoute.SKIP_PUBLIC_WEB
    assert plan.queries == ()
    assert plan.is_skip
    assert plan.skip_reason
    assert plan.relevance_policy == RelevancePolicy.NONE


def test_competitor_question_routes_broad_web_max_three_queries():
    plan = build_web_search_plan(
        question="Who are the main competitors and alternatives?", **MANTIC
    )
    assert plan.route == QuestionRoute.COMPETITORS
    assert 1 <= len(plan.queries) <= MAX_QUERIES_PER_QUESTION
    for spec in plan.queries:
        assert "mantic" not in spec.query.lower()
        assert spec.domain_filter is None


# --- Routing details -----------------------------------------------------------


def test_regulation_question_gets_broad_web():
    plan = build_web_search_plan(
        question="What regulations affect deployment in the sector?", **MANTIC
    )
    assert plan.route == QuestionRoute.REGULATION
    assert len(plan.queries) <= MAX_QUERIES_PER_QUESTION
    assert all(spec.domain_filter is None for spec in plan.queries)


def test_internal_fit_route_runs_zero_queries():
    plan = build_web_search_plan(
        question="Does the company's stage align with the VC's investment thesis?",
        **MANTIC,
    )
    assert plan.route == QuestionRoute.INTERNAL_FIT
    assert plan.queries == ()
    assert plan.skip_reason


@pytest.mark.parametrize(
    "question",
    [
        # The benchmark "growth" collision: a stage question must NOT go to sector_market.
        "Is the company pre-seed, seed, Series A, or growth stage?",
        "What traction has the company demonstrated so far?",
        "What are the product's core features and underlying technology?",
        "Who are the key members of the founding team?",
        "What has been the company's user growth and revenue growth?",
        "What partnerships has the company established?",
    ],
)
def test_fallback_router_defaults_ambiguous_questions_to_company_specific(question):
    plan = build_web_search_plan(question=question, **MANTIC)
    assert plan.route == QuestionRoute.COMPANY_SPECIFIC
    assert plan.rationale == "keyword_fallback:default"


def test_market_revenue_growth_question_is_not_skipped():
    # Dry-run false positive: "revenue" must not send market questions to skip.
    plan = build_web_search_plan(
        question="What has been the market's revenue growth over recent years?",
        **MANTIC,
    )
    assert plan.route != QuestionRoute.SKIP_PUBLIC_WEB


def test_route_tag_takes_priority_over_keywords():
    plan = build_web_search_plan(
        question="Has the company announced funding?",
        route_tag="sector_market",
        **MANTIC,
    )
    assert plan.route == QuestionRoute.SECTOR_MARKET
    assert plan.rationale == "tag:sector_market"


def test_unknown_route_tag_falls_back_to_keywords():
    plan = build_web_search_plan(
        question="Has the company announced funding?",
        route_tag="banana",
        **MANTIC,
    )
    assert plan.route == QuestionRoute.COMPANY_SPECIFIC
    assert plan.rationale == "keyword_fallback:default"


@pytest.mark.parametrize(
    "aspect,expected",
    [
        ("general_company", QuestionRoute.INTERNAL_FIT),
        ("market", QuestionRoute.SECTOR_MARKET),
        ("product", QuestionRoute.COMPANY_SPECIFIC),
        ("team", QuestionRoute.COMPANY_SPECIFIC),
    ],
)
def test_static_root_routes(aspect, expected):
    plan = build_web_search_plan(
        question="Templated root question text irrelevant here.",
        aspect=aspect,
        is_root=True,
        **MANTIC,
    )
    assert plan.route == expected
    assert plan.rationale == f"static_root:{aspect}"


def test_static_root_map_ignored_for_non_root_nodes():
    plan = build_web_search_plan(
        question="Has the company announced funding?",
        aspect="market",
        is_root=False,
        **MANTIC,
    )
    assert plan.route == QuestionRoute.COMPANY_SPECIFIC


@pytest.mark.parametrize("tag,expected", [
    ("sector_market", QuestionRoute.SECTOR_MARKET),
    ("Sector-Market", QuestionRoute.SECTOR_MARKET),
    ("  INTERNAL_FIT  ", QuestionRoute.INTERNAL_FIT),
    ("nonsense", None),
    (None, None),
    ("", None),
])
def test_normalize_route_tag(tag, expected):
    assert normalize_route_tag(tag) == expected


# --- Query construction --------------------------------------------------------


def test_public_description_never_leaks_into_queries():
    sentinel = "confidentialsentinelxyz"
    for question in (
        "Is the sector attractive given market size and growth?",
        "Has the company announced funding?",
        "Who are the main competitors?",
    ):
        plan = build_web_search_plan(
            question=question,
            public_description=f"secret deck text {sentinel}",
            **MANTIC,
        )
        assert sentinel not in _all_query_text(plan).lower()


def test_queries_respect_length_cap():
    long_question = "What is the market size for " + " ".join(
        f"verylongkeyword{i}" for i in range(60)
    )
    plan = build_web_search_plan(question=long_question, **MANTIC)
    for spec in plan.queries:
        assert len(spec.query) <= WEB_SEARCH_QUERY_MAX_LEN


def test_geo_hint_terms_appear_when_set_and_absent_when_none():
    question = "Is the sector attractive given market size and growth?"
    with_geo = build_web_search_plan(question=question, **MANTIC)
    assert "europe" in _all_query_text(with_geo).lower()

    no_geo = build_web_search_plan(question=question, **{**MANTIC, "geo_hint": None})
    assert "europe" not in _all_query_text(no_geo).lower()


def test_company_domain_normalization_in_filter():
    plan = build_web_search_plan(
        question="Has the company announced funding?",
        **{**MANTIC, "company_domain": "https://www.mantic.ai/about"},
    )
    assert plan.queries[0].domain_filter == ("mantic.ai", "crunchbase.com", "linkedin.com")


def test_company_specific_filter_without_domain():
    plan = build_web_search_plan(
        question="Has the company announced funding?",
        **{**MANTIC, "company_domain": None},
    )
    assert plan.queries[0].domain_filter == ("crunchbase.com", "linkedin.com")


def test_current_year_only_added_for_current_intent():
    current = build_web_search_plan(
        question="What is the current market size and forecast?", **MANTIC
    )
    assert "2026" in _all_query_text(current)

    historical = build_web_search_plan(
        question="What is the market size in this segment (TAM)?", **MANTIC
    )
    assert "2026" not in _all_query_text(historical)


def test_topic_queries_fall_back_to_question_keywords_without_industry_hint():
    plan = build_web_search_plan(
        question="Is the wildfire detection sector attractive given market size?",
        **{**MANTIC, "industry_hint": None},
    )
    assert plan.route == QuestionRoute.SECTOR_MARKET
    assert plan.queries
    text = _all_query_text(plan).lower()
    assert "wildfire" in text
    assert "mantic" not in text


def test_telemetry_dict_is_json_shaped():
    plan = build_web_search_plan(
        question="Is the sector attractive given market size and growth?", **MANTIC
    )
    payload = plan.to_telemetry_dict()
    assert payload["route"] == "sector_market"
    assert payload["relevance_policy"] == "topic_keywords"
    assert isinstance(payload["queries"], list)
    assert all(set(q) == {"query", "domain_filter", "purpose"} for q in payload["queries"])


# --- Relevance gates -----------------------------------------------------------

SECTOR_PARAGRAPH = (
    "The European market for wildfire detection sensors is projected to grow "
    "rapidly, with adoption by utilities and forest agencies driven by new "
    "monitoring requirements and increasing demand for early-warning systems. "
    "Analysts estimate the market size in the hundreds of millions of euros."
)

GENERIC_FUNDING_PARAGRAPH = (
    "Venture funding for wildfire technology startups has accelerated, with "
    "several early-stage companies raising seed rounds to develop detection "
    "sensors and monitoring platforms across Europe and North America in 2025."
)


def test_sector_evidence_accepted_without_company_mention():
    ok, reason = evaluate_web_result_relevance(
        policy=RelevancePolicy.TOPIC_KEYWORDS,
        route=QuestionRoute.SECTOR_MARKET,
        question="Is the sector attractive given market size, growth and regulation?",
        company_name="Mantic",
        web_results=SECTOR_PARAGRAPH,
        industry_hint="wildfire detection sensors",
    )
    assert ok is True
    assert "company mention not required" in reason


def test_generic_funding_paragraph_rejected_for_company_specific_facts():
    ok, reason = evaluate_web_result_relevance(
        policy=RelevancePolicy.COMPANY_MENTION,
        route=QuestionRoute.COMPANY_SPECIFIC,
        question="Has Mantic announced funding?",
        company_name="Mantic",
        web_results=GENERIC_FUNDING_PARAGRAPH,
    )
    assert ok is False
    assert reason == "web results do not mention the company"


def test_company_mention_gate_accepts_relevant_company_results():
    ok, reason = evaluate_web_result_relevance(
        policy=RelevancePolicy.COMPANY_MENTION,
        route=QuestionRoute.COMPANY_SPECIFIC,
        question="Has Mantic announced funding?",
        company_name="Mantic",
        web_results=(
            "Mantic, a Greek wildfire detection startup, announced a seed funding "
            "round to expand its sensor network across southern Europe."
        ),
    )
    assert ok is True
    assert reason == "web results relevant to company/question"


@pytest.mark.parametrize("policy", [RelevancePolicy.COMPANY_MENTION, RelevancePolicy.TOPIC_KEYWORDS])
def test_gates_reject_failure_noise(policy):
    ok, reason = evaluate_web_result_relevance(
        policy=policy,
        route=QuestionRoute.SECTOR_MARKET,
        question="Is the sector attractive?",
        company_name="Mantic",
        web_results="Web search failed: rate limit exceeded",
    )
    assert ok is False
    assert reason == "web results indicate failure/noise"


@pytest.mark.parametrize("policy", [RelevancePolicy.COMPANY_MENTION, RelevancePolicy.TOPIC_KEYWORDS])
def test_gates_reject_short_results(policy):
    ok, reason = evaluate_web_result_relevance(
        policy=policy,
        route=QuestionRoute.SECTOR_MARKET,
        question="Is the sector attractive?",
        company_name="Mantic",
        web_results="Too short.",
    )
    assert ok is False
    assert reason == "web results too short"


def test_topic_gate_rejects_off_topic_results():
    ok, reason = evaluate_web_result_relevance(
        policy=RelevancePolicy.TOPIC_KEYWORDS,
        route=QuestionRoute.SECTOR_MARKET,
        question="Is the wildfire detection sensor market attractive?",
        company_name="Mantic",
        web_results=(
            "The history of medieval shipbuilding shows that oak planking and "
            "iron rivets dominated construction techniques for several centuries "
            "throughout the Baltic region and beyond."
        ),
        industry_hint="wildfire detection sensors",
    )
    assert ok is False
    assert "do not match" in reason


def test_none_policy_rejects_with_route_reason():
    ok, reason = evaluate_web_result_relevance(
        policy=RelevancePolicy.NONE,
        route=QuestionRoute.INTERNAL_FIT,
        question="Does the company fit the VC's thesis?",
        company_name="Mantic",
        web_results="anything",
    )
    assert ok is False
    assert "internal_fit" in reason
