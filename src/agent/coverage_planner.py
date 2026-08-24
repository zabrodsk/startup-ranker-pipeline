"""Pure coverage-aware planning for LeadGen packet objective gaps.

This module is stdlib-only and side-effect free. It derives a proposed
search shape from packet-backed objective coverage while leaving execution
control to the existing web-search planner.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

RDI_COVERAGE_AWARE_SEARCH_ENV = "RDI_COVERAGE_AWARE_SEARCH"
_COVERAGE_MODES = ("off", "shadow", "on")
_DEFAULT_COVERAGE_MODE = "off"
_WARNED_INVALID_MODES: set[str] = set()

_EXTERNAL_ROUTES = {
    "sector_market",
    "regulation",
    "competitors",
    "geography",
    "customer_need",
    "technology_validation",
}
_SEARCHABLE_STATES = {"missing", "stale", "weak", "contradictory"}
_OBJECTIVE_HINT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("european_connection", ("europe", "european", "prague", "czech", "hq", "headquarter", "based", "location")),
    ("stage_and_funding", ("funding", "fundraise", "raised", "round", "pre-seed", "seed", "series", "investor", "stage")),
    ("founder_prior_execution", ("founder", "team", "background", "previous", "built", "experience", "track record")),
    ("founder_market_fit", ("founder", "domain", "market fit", "industry experience", "sector experience")),
    ("product_software_usp", ("product", "software", "platform", "technology", "usp", "differentiation", "features")),
    ("customer_or_deployment", ("customer", "deployment", "user", "users", "case study", "implementation")),
    ("commercial_traction", ("traction", "revenue", "growth", "commercial", "sales", "pipeline", "arr", "mrr")),
    ("moat_or_defensibility", ("moat", "defensible", "defensibility", "advantage", "barrier")),
    ("market_problem_and_buyer", ("buyer", "buyers", "pain", "problem", "customer need", "who buys", "market problem")),
    ("momentum", ("momentum", "hiring", "launch", "expansion", "news", "velocity")),
)
_OBJECTIVE_QUERY_SUFFIX = {
    "european_connection": "hq europe founder location",
    "stage_and_funding": "funding round investors stage",
    "founder_prior_execution": "founder background prior company execution",
    "founder_market_fit": "founder domain experience sector background",
    "product_software_usp": "product software platform differentiation",
    "customer_or_deployment": "customers deployments case study",
    "commercial_traction": "traction revenue growth customers",
    "moat_or_defensibility": "moat defensibility competitive advantage",
    "market_problem_and_buyer": "buyer pain market problem procurement",
    "momentum": "hiring launch expansion momentum",
}


@dataclass(frozen=True)
class ProposedCoverageQuery:
    """One objective-scoped search the coverage planner would like to run."""

    objective: str
    state: str
    query: str
    reason: str


@dataclass(frozen=True)
class CoverageSearchPlan:
    """Telemetry-ready proposed search shape for one question."""

    mode: str
    route: str
    aspect: str
    external_route: bool
    mapped_objectives: tuple[str, ...]
    objective_states: dict[str, str]
    proposed_queries: tuple[ProposedCoverageQuery, ...]
    estimated_search_cost: int
    rationale: str

    @property
    def proposed_search_count(self) -> int:
        """Return the number of provider searches the plan proposes."""
        return len(self.proposed_queries)

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for provenance payloads."""
        return {
            "mode": self.mode,
            "route": self.route,
            "aspect": self.aspect,
            "external_route": self.external_route,
            "mapped_objectives": list(self.mapped_objectives),
            "objective_states": dict(self.objective_states),
            "estimated_search_cost": self.estimated_search_cost,
            "rationale": self.rationale,
            "proposed_queries": [
                {
                    "objective": query.objective,
                    "state": query.state,
                    "query": query.query,
                    "reason": query.reason,
                }
                for query in self.proposed_queries
            ],
        }


def resolve_coverage_search_mode(raw: str | None = None) -> str:
    """Parse the coverage planner mode with strict fallback to `off`."""
    value = raw if raw is not None else os.getenv(
        RDI_COVERAGE_AWARE_SEARCH_ENV,
        _DEFAULT_COVERAGE_MODE,
    )
    mode = str(value or "").strip().lower()
    if mode not in _COVERAGE_MODES:
        if mode not in _WARNED_INVALID_MODES:
            _WARNED_INVALID_MODES.add(mode)
            logging.getLogger(__name__).warning(
                "Invalid %s=%r; falling back to %r",
                RDI_COVERAGE_AWARE_SEARCH_ENV,
                value,
                _DEFAULT_COVERAGE_MODE,
            )
        return _DEFAULT_COVERAGE_MODE
    return mode


def build_coverage_search_plan(
    *,
    question: str,
    company_name: str,
    route: str,
    aspect: str,
    legacy_query_count: int,
    packet_entries: list[Mapping[str, Any]],
    mode: str,
) -> CoverageSearchPlan | None:
    """Derive a proposed search plan from packet-backed objective coverage."""
    if mode == "off" or not packet_entries:
        return None

    route_name = (route or "").strip().lower()
    aspect_name = (aspect or "").strip().lower()
    external_route = route_name in _EXTERNAL_ROUTES
    mapped_objectives = _map_question_to_objectives(
        question=question,
        aspect=aspect_name,
        route=route_name,
    )

    objective_states = _objective_states(mapped_objectives, packet_entries)
    if external_route:
        rationale = "retain broad external-market web searches"
        proposed_queries = tuple(
            ProposedCoverageQuery(
                objective="external_route",
                state="external",
                query="[retain existing planner broad-web queries]",
                reason=rationale,
            )
            for _ in range(max(legacy_query_count, 1))
        )
        return CoverageSearchPlan(
            mode=mode,
            route=route_name,
            aspect=aspect_name,
            external_route=True,
            mapped_objectives=tuple(mapped_objectives),
            objective_states=objective_states,
            proposed_queries=proposed_queries,
            estimated_search_cost=max(legacy_query_count, 1),
            rationale=rationale,
        )

    actionable = [
        objective
        for objective in mapped_objectives
        if objective_states.get(objective, "missing") in _SEARCHABLE_STATES
    ]
    proposed_queries = tuple(
        ProposedCoverageQuery(
            objective=objective,
            state=objective_states[objective],
            query=_objective_query(company_name, objective),
            reason=f"{objective} is {objective_states[objective]}",
        )
        for objective in actionable[: max(legacy_query_count, 1)]
    )
    rationale = (
        "all mapped objectives already supported"
        if mapped_objectives and not actionable
        else "search only mapped missing/stale/weak/contradictory objectives"
    )
    return CoverageSearchPlan(
        mode=mode,
        route=route_name,
        aspect=aspect_name,
        external_route=False,
        mapped_objectives=tuple(mapped_objectives),
        objective_states=objective_states,
        proposed_queries=proposed_queries,
        estimated_search_cost=len(proposed_queries),
        rationale=rationale,
    )


def _map_question_to_objectives(*, question: str, aspect: str, route: str) -> list[str]:
    lowered = question.lower()
    objectives = [
        objective
        for objective, tokens in _OBJECTIVE_HINT_RULES
        if any(token in lowered for token in tokens)
    ]
    if not objectives and aspect == "team":
        objectives.append("founder_prior_execution")
    if not objectives and aspect == "product":
        objectives.append("product_software_usp")
    if not objectives and aspect == "market":
        objectives.append("market_problem_and_buyer")
    if not objectives and route in {"company_specific", "internal_fit"}:
        if "fund" in lowered or "stage" in lowered:
            objectives.append("stage_and_funding")
    return list(dict.fromkeys(objectives))


def _objective_states(
    mapped_objectives: list[str],
    packet_entries: list[Mapping[str, Any]],
) -> dict[str, str]:
    states: dict[str, str] = {}
    for objective in mapped_objectives:
        entries = [
            entry for entry in packet_entries
            if str(entry.get("objective") or "") == objective
        ]
        if not entries:
            states[objective] = "missing"
            continue
        statuses = {str(entry.get("status") or "") for entry in entries}
        confidences = {str(entry.get("confidence") or "") for entry in entries}
        stale = any(bool(entry.get("stale_objective")) for entry in entries)
        if "contradicts" in statuses:
            states[objective] = "contradictory"
        elif stale:
            states[objective] = "stale"
        elif "supports" in statuses and "low" in confidences:
            states[objective] = "weak"
        elif "supports" in statuses:
            states[objective] = "supported"
        else:
            states[objective] = "missing"
    return states


def _objective_query(company_name: str, objective: str) -> str:
    suffix = _OBJECTIVE_QUERY_SUFFIX[objective]
    return f"\"{company_name}\" {suffix}".strip()
