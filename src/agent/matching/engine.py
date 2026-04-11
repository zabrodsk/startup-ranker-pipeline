"""VC-Startup matching engine.

Re-runs only stages 3–8 of the investment pipeline against a VC's specific
investment thesis, reusing pre-computed Q&A pairs from the startup's latest
analysis. This reduces LLM cost by ~70% compared to a full re-analysis.

Usage (called from FastAPI background task):
    from agent.matching.engine import trigger_matching_for_company
    await trigger_matching_for_company(company_id, db)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.dataclasses.company import Company
from agent.dataclasses.config import Config
from agent.ingest.store import Chunk, EvidenceStore

if TYPE_CHECKING:
    import types

logger = logging.getLogger(__name__)

# Minimal pipeline config for matching (1 iteration → faster, cheaper).
_MATCH_CONFIG = Config(
    n_pro_arguments=3,
    n_contra_arguments=3,
    k_best_arguments_per_iteration=[3, 1],
    max_iterations=1,
)


def _build_company(company_row: dict[str, Any]) -> Company:
    """Construct a Company dataclass from a DB company row."""
    return Company(
        name=company_row.get("name") or "Unknown",
        industry=company_row.get("industry"),
        tagline=company_row.get("tagline"),
        about=company_row.get("about"),
        domain=company_row.get("domain"),
    )


def _build_evidence_store(company_id: str, chunks: list[dict[str, Any]]) -> EvidenceStore:
    """Construct an EvidenceStore from DB chunk rows."""
    store = EvidenceStore(startup_slug=company_id)
    for c in chunks:
        chunk_id = str(c.get("chunk_id") or c.get("id") or "")
        store.chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=c.get("text") or "",
                source_file=c.get("source_file") or "",
                page_or_slide=c.get("page_or_slide") or 0,
            )
        )
    return store


async def run_matching_for_pair(
    *,
    company_row: dict[str, Any],
    chunks: list[dict[str, Any]],
    all_qa_pairs: list[dict[str, Any]],
    vc_thesis: str,
    final_arguments: list[dict[str, Any]] | None = None,
    final_decision: str | None = None,
) -> dict[str, Any] | None:
    """Score one (company, vc_profile) pair against a VC investment thesis.

    Optimised routing:
    - If final_arguments + final_decision are provided (from a prior full analysis),
      the graph enters at Stage 8 (score_company_dimensions) only — ~4 LLM calls.
    - Otherwise falls back to Stage 3 entry using all_qa_pairs (~20 LLM calls).

    Returns a dict with score fields from CompanyRankingResult, or None on failure.
    """
    # Lazy import to avoid circular dependency at module level.
    from agent.pipeline.graph import graph  # noqa: PLC0415

    company = _build_company(company_row)
    company_id = str(company_row.get("id") or company_row.get("name") or "unknown")

    use_stage8_shortcut = bool(final_arguments and final_decision)

    if use_stage8_shortcut:
        # Stage 8-only path: supply pre-computed arguments + decision.
        # check_start_point() will detect these and route to score_company_dimensions.
        invoke_payload: dict[str, Any] = {
            "company": company,
            "config": _MATCH_CONFIG,
            "final_arguments": final_arguments,
            "final_decision": final_decision,
            "vc_context": (vc_thesis or "").strip(),
            "slug": company_id,
            "prompt_overrides": {},
        }
        logger.debug(
            "Matching company=%s via Stage-8-only path (final_arguments=%d)",
            company_id, len(final_arguments or []),
        )
    elif all_qa_pairs:
        # Stage 3 fallback: enter at argument generation using Q&A pairs.
        invoke_payload = {
            "company": company,
            "config": _MATCH_CONFIG,
            "all_qa_pairs": all_qa_pairs,
            "vc_context": (vc_thesis or "").strip(),
            "slug": company_id,
            "prompt_overrides": {},
        }
        logger.debug("Matching company=%s via Stage-3 path (qa_pairs=%d)", company_id, len(all_qa_pairs))
    else:
        logger.warning(
            "No Q&A pairs or final_arguments for company %s — matching skipped",
            company_row.get("id"),
        )
        return None

    _build_evidence_store(company_id, chunks)  # available for future retrieval if needed

    try:
        result_state = await graph.ainvoke(
            invoke_payload,
            config={"recursion_limit": 100},
        )
    except Exception as exc:
        logger.error(
            "Graph invocation failed for company=%s: %s", company_id, exc
        )
        return None

    ranking = result_state.get("ranking_result")
    if not ranking:
        logger.warning("No ranking_result in final_state for company=%s", company_id)
        return None

    # LangGraph may return a Pydantic model or a plain dict.
    if hasattr(ranking, "model_dump"):
        return ranking.model_dump()
    if isinstance(ranking, dict):
        return ranking
    return dict(ranking)


async def trigger_matching_for_company(
    company_id: str,
    db_module: Any,
    *,
    force_refresh: bool = False,
) -> int:
    """Match a fundraising company against all active VC profiles.

    Called as a FastAPI background task when a startup toggles fundraising ON.
    All synchronous DB calls are dispatched via asyncio.to_thread() so the
    event loop (and thus the web server) is never blocked.
    Returns the number of new match records created (or refreshed, when
    ``force_refresh=True``).

    ``force_refresh``: when True, existing matches are re-scored and their
    rows updated in-place via the upsert in ``db.create_match``. This is what
    the re-evaluation path uses so fresh scores land without deleting the
    match rows (which would cascade-destroy the associated debate). When
    False (default), existing matches are skipped — used by the fundraising
    toggle path where we only want brand-new matches.
    """
    import asyncio  # noqa: PLC0415

    company_row = await asyncio.to_thread(db_module.get_company_by_id, company_id)
    if not company_row:
        logger.warning("Company %s not found — matching aborted", company_id)
        return 0

    chunks, all_qa_pairs = await asyncio.gather(
        asyncio.to_thread(db_module.get_company_chunks, company_id),
        asyncio.to_thread(db_module.get_analysis_qa_pairs, company_id),
    )

    # Attempt Stage-8-only optimisation: load pre-computed final_arguments + decision.
    analysis_final: dict[str, Any] | None = None
    if hasattr(db_module, "get_analysis_final_state"):
        analysis_final = await asyncio.to_thread(db_module.get_analysis_final_state, company_id)

    final_arguments: list[dict[str, Any]] | None = None
    final_decision: str | None = None
    if analysis_final:
        final_arguments = analysis_final.get("final_arguments")
        final_decision = analysis_final.get("final_decision")
        logger.info(
            "Company %s: Stage-8-only matching enabled (final_arguments=%d, decision=%s)",
            company_id, len(final_arguments or []), final_decision,
        )
    elif not all_qa_pairs:
        logger.info(
            "Company %s has no completed analysis Q&A pairs — matching skipped", company_id
        )
        return 0

    vc_profiles = await asyncio.to_thread(db_module.get_active_vc_profiles)
    if not vc_profiles:
        logger.info("No active VC profiles to match against")
        return 0

    logger.info(
        "Starting matching for company=%s against %d VC profiles (stage8=%s, qa_pairs=%d)",
        company_id, len(vc_profiles), bool(final_arguments), len(all_qa_pairs),
    )

    latest_analysis = await asyncio.to_thread(db_module.get_company_latest_analysis, company_id)
    analysis_id: str | None = latest_analysis.get("id") if latest_analysis else None

    created = 0
    for vc_profile in vc_profiles:
        vc_profile_id: str = vc_profile.get("id") or ""
        if not vc_profile_id:
            continue

        match_already_exists = await asyncio.to_thread(
            db_module.match_exists, vc_profile_id, company_id
        )
        if match_already_exists and not force_refresh:
            logger.debug("Match already exists: vc=%s company=%s", vc_profile_id, company_id)
            continue
        if match_already_exists and force_refresh:
            logger.debug(
                "Force-refreshing existing match: vc=%s company=%s", vc_profile_id, company_id
            )

        vc_thesis: str = vc_profile.get("investment_thesis") or ""
        min_strategy: float = float(vc_profile.get("min_strategy_fit") or 0)
        min_team: float = float(vc_profile.get("min_team") or 0)
        min_potential: float = float(vc_profile.get("min_potential") or 0)

        try:
            scores = await run_matching_for_pair(
                company_row=company_row,
                chunks=chunks,
                all_qa_pairs=all_qa_pairs,
                vc_thesis=vc_thesis,
                final_arguments=final_arguments,
                final_decision=final_decision,
            )
        except Exception as exc:
            logger.error(
                "Matching error vc=%s company=%s: %s", vc_profile_id, company_id, exc
            )
            continue

        if not scores:
            logger.warning(
                "run_matching_for_pair returned None for vc=%s company=%s — "
                "graph.ainvoke likely failed or returned no ranking_result",
                vc_profile_id, company_id,
            )
            continue

        strategy_fit: float = float(scores.get("strategy_fit_score") or 0)
        team: float = float(scores.get("team_score") or 0)
        potential: float = float(scores.get("upside_score") or 0)
        logger.info(
            "Scores for vc=%s company=%s: strategy=%.1f team=%.1f potential=%.1f composite=%.1f bucket=%s",
            vc_profile_id, company_id, strategy_fit, team, potential,
            float(scores.get("composite_score") or 0), scores.get("bucket"),
        )

        meets_thresholds = (
            strategy_fit >= min_strategy
            and team >= min_team
            and potential >= min_potential
        )

        if meets_thresholds:
            await asyncio.to_thread(
                db_module.create_match,
                vc_profile_id=vc_profile_id,
                company_id=company_id,
                analysis_id=analysis_id,
                scores={
                    "strategy_fit_score": strategy_fit,
                    "team_score": team,
                    "potential_score": potential,
                    "composite_score": float(scores.get("composite_score") or 0),
                    "bucket": scores.get("bucket"),
                },
            )
            created += 1
            logger.info(
                "Match created: vc=%s company=%s scores=strategy_fit:%.1f team:%.1f potential:%.1f",
                vc_profile_id,
                company_id,
                strategy_fit,
                team,
                potential,
            )
        else:
            logger.info(
                "Below threshold: vc=%s company=%s scores=%.1f/%.1f/%.1f thresholds=%.0f/%.0f/%.0f",
                vc_profile_id,
                company_id,
                strategy_fit,
                team,
                potential,
                min_strategy,
                min_team,
                min_potential,
            )

    return created
