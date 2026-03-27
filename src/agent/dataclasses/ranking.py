"""Data models for the ranking decision layer.

Defines DimensionScore and CompanyRankingResult used by the composite
ranking stage to score companies on Strategy Fit, Team Quality, and
Problem/Upside, with evidence-backed confidence adjustment.
"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class DimensionScore(BaseModel):
    """Score for a single ranking dimension (strategy_fit, team, upside)."""

    dimension: Literal["strategy_fit", "team", "upside"]
    raw_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0)
    top_qa_indices: list[int] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    critical_gaps: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def adjusted_score(self) -> float:
        """Confidence-adjusted score: raw * (0.7 + 0.3 * confidence)."""
        return round(self.raw_score * (0.7 + 0.3 * self.confidence), 2)


class CompanyRankingResult(BaseModel):
    """Per-company ranking result with composite score and triage bucket."""

    company_name: str = ""
    slug: str = ""

    strategy_fit_score: float = 0.0
    team_score: float = 0.0
    upside_score: float = 0.0
    composite_score: float = 0.0

    bucket: Literal["priority_review", "watchlist", "low_priority"] = "low_priority"

    dimension_scores: list[DimensionScore] = Field(default_factory=list)

    rank: int = 0
    percentile: float = 0.0

    min_dimension_score: float = 0.0
    avg_confidence: float = 0.0
    critical_gaps_count: int = 0

    # Executive summary (human-readable)
    strategy_fit_summary: str = ""
    team_summary: str = ""
    potential_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
