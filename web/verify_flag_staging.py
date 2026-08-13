"""Pure PASS/FAIL assertions for Sprint 3 flag staging verification (Sprint 4 B1).

These operate on the ``get_job_cost_by_stage`` shape::

    {"job_id": str, "stages": [{"stage": str, "llm_calls": int,
        "prompt_tokens": int, ...}], "totals": {"web_search": {"requests": int}}}

Cost-observable features only:
  - w3  (PIPELINE_EVIDENCE_DIGEST): an ``evidence_digest`` stage appears and
        critique+refinement prompt tokens drop vs the off baseline.
  - w9  (PIPELINE_FINAL_SCORE_MODE=reuse): fewer ``evaluation`` LLM calls.
  - w11 (RDI_DECOMP_CACHE_STORE=supabase): fewer ``decomposition`` LLM calls.
  - w13 (RDI_WEB_SEARCH_CACHE): fewer web-search requests on the repeat run.

w7 (Specter identity reuse) and dup_gate are NOT cost-observable — verify them
from MCP logs and the reused badge per docs/handoffs/sprint4-validation-runbook.md.

Pure functions, no I/O — the scripts/ CLI fetches the cost dicts via web.db.
"""

from __future__ import annotations

from typing import Any

COST_OBSERVABLE_FEATURES = ("w3", "w9", "w11", "w13")


def stage_llm_calls(cost: dict[str, Any], stage: str) -> int:
    return sum(
        int(s.get("llm_calls") or 0)
        for s in (cost.get("stages") or [])
        if s.get("stage") == stage
    )


def stage_prompt_tokens(cost: dict[str, Any], stage: str) -> int:
    return sum(
        int(s.get("prompt_tokens") or 0)
        for s in (cost.get("stages") or [])
        if s.get("stage") == stage
    )


def has_stage(cost: dict[str, Any], stage: str) -> bool:
    return any(s.get("stage") == stage for s in (cost.get("stages") or []))


def web_search_requests(cost: dict[str, Any]) -> int:
    totals = cost.get("totals") or {}
    aggregate = totals.get("web_search")
    if isinstance(aggregate, dict):
        return int(aggregate.get("requests") or 0)
    perplexity = totals.get("perplexity_search") or {}
    serper = totals.get("serper_search") or {}
    return int(perplexity.get("requests") or 0) + int(serper.get("requests") or 0)


def verify_feature(
    feature: str,
    feature_cost: dict[str, Any],
    baseline_cost: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return ``(passed, evidence)`` for one cost-observable feature flag."""
    if feature == "w3":
        present = has_stage(feature_cost, "evidence_digest")
        if baseline_cost is None:
            return (present, f"evidence_digest stage present={present}")
        baseline_tokens = stage_prompt_tokens(baseline_cost, "critique") + stage_prompt_tokens(
            baseline_cost, "refinement"
        )
        feature_tokens = stage_prompt_tokens(feature_cost, "critique") + stage_prompt_tokens(
            feature_cost, "refinement"
        )
        delta = baseline_tokens - feature_tokens
        return (
            present and delta > 0,
            f"evidence_digest present={present}; critique+refinement prompt-token delta={delta}",
        )

    if feature == "w9":
        if baseline_cost is None:
            return (False, "w9 needs --baseline-job-id (compares evaluation LLM calls reuse vs rescore)")
        base = stage_llm_calls(baseline_cost, "evaluation")
        feat = stage_llm_calls(feature_cost, "evaluation")
        return (feat < base, f"evaluation llm_calls baseline={base} feature={feat}")

    if feature == "w11":
        if baseline_cost is None:
            return (False, "w11 needs --baseline-job-id (compares decomposition LLM calls)")
        base = stage_llm_calls(baseline_cost, "decomposition")
        feat = stage_llm_calls(feature_cost, "decomposition")
        return (feat < base, f"decomposition llm_calls baseline={base} feature={feat}")

    if feature == "w13":
        if baseline_cost is None:
            return (False, "w13 needs --baseline-job-id (compares web-search requests)")
        base = web_search_requests(baseline_cost)
        feat = web_search_requests(feature_cost)
        return (feat < base, f"web-search requests baseline={base} feature={feat}")

    return (False, f"unknown or non-cost-observable feature {feature!r}; see the runbook")
