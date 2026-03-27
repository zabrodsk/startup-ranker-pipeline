"""Ranking decision layer stages.

Scores companies on Strategy Fit, Team Quality, and Problem/Upside (0-100 each),
applies confidence adjustment, and computes composite rank with triage buckets.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.common.llm_config import get_llm
from agent.dataclasses.ranking import CompanyRankingResult, DimensionScore
from agent.pipeline.state.investment_story import IterativeInvestmentStoryState
from agent.pipeline.state.schemas import DimensionScoreOutput, ExecutiveSummaryOutput
from agent.pipeline.utils.phase_llm import invoke_with_phase_fallback
from agent.prompt_library.manager import get_prompt
from agent.run_context import get_current_pipeline_policy, use_stage_context


# Aspect-to-dimension mapping: general_company -> strategy_fit, team -> team, market+product -> upside
DIMENSION_ASPECTS = {
    "strategy_fit": ["general_company"],
    "team": ["team"],
    "upside": ["market", "product"],
}


def _normalize_text(value: Any) -> str:
    """Coerce arbitrary payloads to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None).strip()
    return str(value)


def _group_qa_by_dimension(
    all_qa_pairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group Q&A pairs by ranking dimension."""
    grouped: dict[str, list[dict[str, Any]]] = {
        "strategy_fit": [],
        "team": [],
        "upside": [],
    }
    for index, qa in enumerate(all_qa_pairs):
        aspect = qa.get("aspect") or ""
        qa_with_index = dict(qa)
        qa_with_index.setdefault("qa_index", index)
        if aspect in DIMENSION_ASPECTS["strategy_fit"]:
            grouped["strategy_fit"].append(qa_with_index)
        elif aspect in DIMENSION_ASPECTS["team"]:
            grouped["team"].append(qa_with_index)
        elif aspect in DIMENSION_ASPECTS["upside"]:
            grouped["upside"].append(qa_with_index)
    return grouped


def _format_qa_block(qa_pairs: list[dict[str, Any]]) -> str:
    """Format Q&A pairs for the prompt."""
    if not qa_pairs:
        return "No relevant Q&A pairs available."
    lines = []
    for i, qa in enumerate(qa_pairs):
        q = qa.get("question", "")
        a = qa.get("answer", "")
        qa_index = qa.get("qa_index", i)
        lines.append(f"Q{qa_index}: {q}\nA{qa_index}: {a}")
    return "\n---\n".join(lines)


def _sanitize_top_qa_indices(
    candidate_indices: list[int] | None,
    qa_pairs: list[dict[str, Any]],
) -> list[int]:
    """Keep only valid, unique global Q&A indices for the current dimension."""
    if not candidate_indices:
        return []

    valid_indices = {
        int(qa.get("qa_index"))
        for qa in qa_pairs
        if qa.get("qa_index") is not None
    }
    sanitized: list[int] = []
    for index in candidate_indices:
        try:
            normalized = int(index)
        except (TypeError, ValueError):
            continue
        if normalized not in valid_indices or normalized in sanitized:
            continue
        sanitized.append(normalized)
    return sanitized[:3]


def _score_dimension(
    dimension: str,
    qa_pairs: list[dict[str, Any]],
    company_summary: str,
    vc_context: str,
    prompt_overrides: dict[str, Any] | None,
) -> DimensionScore:
    """Score a single dimension via LLM."""
    qa_block = _format_qa_block(qa_pairs)
    policy = get_current_pipeline_policy()

    if dimension == "strategy_fit":
        system_prompt = get_prompt("ranking.strategy_fit.system", prompt_overrides)
        user_prompt = get_prompt("ranking.strategy_fit.user", prompt_overrides)
        vc_text = _normalize_text(vc_context).strip()
        vc_block = vc_text if vc_text else "Not provided."
        user_content = user_prompt.format(
            company_summary=company_summary,
            vc_context=vc_block,
            qa_block=qa_block,
        )
    elif dimension == "team":
        system_prompt = get_prompt("ranking.team.system", prompt_overrides)
        user_prompt = get_prompt("ranking.team.user", prompt_overrides)
        user_content = user_prompt.format(
            company_summary=company_summary,
            qa_block=qa_block,
        )
    else:  # upside
        system_prompt = get_prompt("ranking.upside.system", prompt_overrides)
        user_prompt = get_prompt("ranking.upside.user", prompt_overrides)
        user_content = user_prompt.format(
            company_summary=company_summary,
            qa_block=qa_block,
        )

    try:
        def _invoke() -> DimensionScoreOutput:
            stage_name = "ranking_upside_score" if dimension == "upside" else "ranking_dimension_score"
            temperature = 0.7 if dimension == "upside" else 0.0
            reasoning_effort = "none" if dimension == "upside" else None
            with use_stage_context(stage_name):
                llm = get_llm(temperature=temperature, reasoning_effort=reasoning_effort)
                llm_structured = llm.with_structured_output(DimensionScoreOutput)
                return llm_structured.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_content),
                    ]
                )
        output = invoke_with_phase_fallback(
            policy.ranking if policy else None,
            _invoke,
        )
    except Exception:
        return DimensionScore(
            dimension=dimension,
            raw_score=0.0,
            confidence=0.0,
            evidence_count=len(qa_pairs),
            top_qa_indices=[],
            evidence_snippets=[],
            critical_gaps=["Scoring failed due to LLM error"],
        )

    return DimensionScore(
        dimension=dimension,
        raw_score=output.raw_score,
        confidence=output.confidence,
        evidence_count=output.evidence_count,
        top_qa_indices=_sanitize_top_qa_indices(output.top_qa_indices, qa_pairs),
        evidence_snippets=output.evidence_snippets[:3],
        critical_gaps=output.critical_gaps,
    )


def _dimension_display_score(score: DimensionScore) -> float:
    """Return the user-facing score for a dimension."""
    if score.dimension == "upside":
        return round(score.raw_score, 2)
    return score.adjusted_score


def score_company_dimensions(
    state: IterativeInvestmentStoryState,
) -> dict[str, Any]:
    """Score the company on Strategy Fit, Team Quality, and Upside.

    Groups Q&A pairs by dimension, calls LLM for each, and builds
    DimensionScore objects. Potential intentionally uses raw best-case upside
    for user-facing scoring while Strategy Fit and Team stay adjusted.
    """
    company_summary = state.company.get_company_summary()
    grouped = _group_qa_by_dimension(state.all_qa_pairs)

    dimension_scores: list[DimensionScore] = []

    for dim in ("strategy_fit", "team", "upside"):
        qa_pairs = grouped.get(dim, [])
        score = _score_dimension(
            dimension=dim,
            qa_pairs=qa_pairs,
            company_summary=company_summary,
            vc_context=state.vc_context or "",
            prompt_overrides=state.prompt_overrides,
        )
        dimension_scores.append(score)

    strategy_adj = next((_dimension_display_score(s) for s in dimension_scores if s.dimension == "strategy_fit"), 0.0)
    team_adj = next((_dimension_display_score(s) for s in dimension_scores if s.dimension == "team"), 0.0)
    upside_adj = next((_dimension_display_score(s) for s in dimension_scores if s.dimension == "upside"), 0.0)

    result = CompanyRankingResult(
        company_name=state.company.name,
        slug=state.slug or state.company.name,
        strategy_fit_score=strategy_adj,
        team_score=team_adj,
        upside_score=upside_adj,
        dimension_scores=dimension_scores,
    )
    return {"ranking_result": result}


def compute_composite_rank(
    state: IterativeInvestmentStoryState,
) -> dict[str, Any]:
    """Compute composite score, bucket, and tie-breakers.

    Uses equal weights (1/3 each). Assigns priority_review, watchlist, or low_priority.
    """
    result = state.ranking_result
    if not result:
        return {}

    strategy_adj = result.strategy_fit_score
    team_adj = result.team_score
    upside_adj = result.upside_score

    composite = (1 / 3) * strategy_adj + (1 / 3) * team_adj + (1 / 3) * upside_adj
    result.composite_score = round(composite, 2)

    scores = [result.strategy_fit_score, result.team_score, result.upside_score]
    result.min_dimension_score = min(scores) if scores else 0.0

    if result.dimension_scores:
        result.avg_confidence = sum(s.confidence for s in result.dimension_scores) / len(
            result.dimension_scores
        )
        result.critical_gaps_count = sum(len(s.critical_gaps) for s in result.dimension_scores)
    else:
        result.avg_confidence = 0.0
        result.critical_gaps_count = 0

    if result.composite_score >= 75 and result.min_dimension_score >= 55:
        result.bucket = "priority_review"
    elif result.composite_score >= 60:
        result.bucket = "watchlist"
    else:
        result.bucket = "low_priority"

    return {"ranking_result": result}


def _format_dimension_block(dimension_scores: list[DimensionScore]) -> str:
    """Format dimension scores and evidence for the executive summary prompt."""
    if not dimension_scores:
        return "No dimension scores available."
    lines = []
    labels = {"strategy_fit": "Strategy Fit", "team": "Team", "upside": "Potential"}
    for d in dimension_scores:
        label = labels.get(d.dimension, d.dimension)
        lines.append(f"{label} (score {_dimension_display_score(d)}):")
        if d.evidence_snippets:
            lines.append("  Evidence: " + " | ".join(d.evidence_snippets[:3]))
        if d.critical_gaps:
            lines.append("  Gaps: " + "; ".join(d.critical_gaps[:3]))
        lines.append("")
    return "\n".join(lines).strip()


def generate_executive_summary(
    state: IterativeInvestmentStoryState,
) -> dict[str, Any]:
    """Generate human-readable dimension summaries, key points, and red flags.

    Uses a single LLM call with structured output. Populates ranking_result
    with strategy_fit_summary, team_summary, potential_summary, key_points, red_flags.
    """
    result = state.ranking_result
    if not result:
        return {}

    company_summary = state.company.get_company_summary()
    vc_context = _normalize_text(state.vc_context).strip()
    vc_context = vc_context or "Not provided."

    dimension_block = _format_dimension_block(result.dimension_scores)

    pro_args = [a for a in (state.final_arguments or []) if a.argument_type == "pro"]
    contra_args = [a for a in (state.final_arguments or []) if a.argument_type == "contra"]
    pro_args = sorted(pro_args, key=lambda a: a.score, reverse=True)[:5]
    contra_args = sorted(contra_args, key=lambda a: a.score, reverse=True)[:5]

    pro_arguments = "\n".join(
        f"- (score {a.score}) {a.refined_content or a.content}" for a in pro_args
    ) or "None"
    contra_arguments = "\n".join(
        f"- (score {a.score}) {a.refined_content or a.content}" for a in contra_args
    ) or "None"

    all_gaps = []
    for d in result.dimension_scores:
        all_gaps.extend(d.critical_gaps)
    critical_gaps = "\n".join(f"- {g}" for g in all_gaps[:10]) if all_gaps else "None identified"

    system_prompt = get_prompt("ranking.executive_summary.system", state.prompt_overrides)
    user_prompt = get_prompt("ranking.executive_summary.user", state.prompt_overrides)
    user_content = user_prompt.format(
        company_summary=company_summary,
        vc_context=vc_context,
        dimension_block=dimension_block,
        pro_arguments=pro_arguments,
        contra_arguments=contra_arguments,
        critical_gaps=critical_gaps,
    )

    try:
        policy = get_current_pipeline_policy()
        def _invoke() -> ExecutiveSummaryOutput:
            with use_stage_context("ranking_executive_summary"):
                llm = get_llm(temperature=0.3)
                llm_structured = llm.with_structured_output(ExecutiveSummaryOutput)
                return llm_structured.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_content),
                    ]
                )
        output = invoke_with_phase_fallback(
            policy.ranking if policy else None,
            _invoke,
        )
        result.strategy_fit_summary = output.strategy_fit_summary or ""
        result.team_summary = output.team_summary or ""
        result.potential_summary = output.potential_summary or ""
        result.key_points = output.key_points[:8] if output.key_points else []
        result.red_flags = output.red_flags[:6] if output.red_flags else []
    except Exception:
        result.strategy_fit_summary = ""
        result.team_summary = ""
        result.potential_summary = ""
        result.key_points = []
        result.red_flags = []

    return {"ranking_result": result}
