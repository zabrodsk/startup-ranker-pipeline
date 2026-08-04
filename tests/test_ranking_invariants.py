from agent.dataclasses.ranking import CompanyRankingResult
from agent.pipeline.stages.ranking import compute_composite_rank
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState


def _rank(strategy: float, team: float, upside: float) -> CompanyRankingResult:
    state = IterativeInvestmentStoryState(
        ranking_result=CompanyRankingResult(
            strategy_fit_score=strategy,
            team_score=team,
            upside_score=upside,
        )
    )
    result = compute_composite_rank(state)["ranking_result"]
    assert isinstance(result, CompanyRankingResult)
    return result


def test_composite_is_equal_weight_and_rounded_to_two_decimals() -> None:
    result = _rank(74.0, 75.0, 76.0)
    assert result.composite_score == 75.0

    rounded = _rank(74.0, 74.0, 75.0)
    assert rounded.composite_score == 74.33


def test_operational_74_threshold_does_not_change_rdi_priority_bucket() -> None:
    result = _rank(74.0, 74.0, 74.0)
    assert result.composite_score == 74.0
    assert result.bucket == "watchlist"


def test_priority_review_requires_75_composite_and_55_minimum_dimension() -> None:
    qualified = _rank(75.0, 75.0, 75.0)
    assert qualified.bucket == "priority_review"
    assert qualified.min_dimension_score == 75.0

    low_dimension = _rank(55.0, 85.0, 85.0)
    assert low_dimension.composite_score == 75.0
    assert low_dimension.min_dimension_score == 55.0
    assert low_dimension.bucket == "priority_review"

    below_floor = _rank(54.99, 85.0, 85.01)
    assert below_floor.composite_score == 75.0
    assert below_floor.min_dimension_score == 54.99
    assert below_floor.bucket == "watchlist"
