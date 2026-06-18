"""Offline replay of the WebEvidencePlanner against a persisted benchmark job.

Stages (driven by scripts/replay_web_planner_benchmark.py, run under
``railway run`` so staging Supabase + provider keys are available):

- ``tag``    one-off LLM route tagging of all benchmark questions with the
             SAME instruction PR4 embeds at decomposition time. Cheap
             (batched), idempotent (keyed by question hash).
- ``replay`` FREE: no provider/LLM calls. Re-routes every persisted question,
             diffs old vs new queries/filters, re-runs BOTH relevance gates
             against the PERSISTED web results, and enforces the hard
             0-false-positive check (no rescued answer may be skip-routed).
- ``live``   sampled: re-executes planned queries for ~80-100 thin questions
             on topic routes and re-answers via the hybrid prompt using the
             persisted chunks_preview as document context.
- ``report`` assembles the gate-review artifact (route distribution, gate
             confusion matrix, skip stats, rescue uplift, 0/35 PASS/FAIL).

Pure logic lives in module-level functions so it is unit-testable without
network; only the stage runners touch Supabase / providers / LLMs.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from agent.evidence_answering import (
    _answer_indicates_no_evidence,
    _run_web_search,
    _web_results_add_value,
)
from agent.pipeline.stages.constants import INVESTMENT_QUESTIONS
from agent.web_search.planner import (
    ROUTE_TAGGING_INSTRUCTION,
    QuestionRoute,
    WebSearchPlan,
    build_web_search_plan,
    evaluate_web_result_relevance,
    normalize_route_tag,
)
from agent.web_search.unanswerable import (
    predict_search_unanswerable,
    predict_targeted_skip,
)

DEFAULT_JOB_ID = "f20aa510"
DEFAULT_OUT_DIR = Path("exports/web_planner_replay")
TOPIC_ROUTES_FOR_LIVE = (
    "sector_market",
    "regulation",
    "competitors",
    "geography",
    "customer_need",
    "technology_validation",
)
TAG_BATCH_SIZE = 25
DEFAULT_LIVE_SAMPLE_SIZE = 100


def question_hash(question: str) -> str:
    """Stable key for a question (route tags survive re-runs and reorderings)."""
    return hashlib.sha1((question or "").strip().lower().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Row extraction from the persisted results payload
# ---------------------------------------------------------------------------


def _company_fields(entry: dict[str, Any]) -> dict[str, Optional[str]]:
    company = entry.get("company") or {}
    if not isinstance(company, dict):
        company = {
            "name": getattr(company, "name", None),
            "industry": getattr(company, "industry", None),
            "domain": getattr(company, "domain", None),
            "geo": getattr(company, "geo", None),
        }
    return {
        "company_name": company.get("name") or entry.get("company_name") or entry.get("slug") or "",
        "industry": company.get("industry"),
        "domain": company.get("domain"),
        "geo": company.get("geo"),
    }


def _find_qa_provenance_rows(obj: Any, depth: int = 0) -> Optional[list]:
    """Locate the flat ``qa_provenance_rows`` list anywhere under the payload.

    ``load_job_results`` returns the worker-composed persisted shape, which
    stores QA as a flat ``qa_provenance_rows`` list (build_qa_provenance_rows),
    not the in-memory company list with ``final_state.all_qa_pairs``.
    """
    if not isinstance(obj, dict) or depth > 4:
        return None
    rows = obj.get("qa_provenance_rows")
    if isinstance(rows, list) and rows:
        return rows
    for value in obj.values():
        if isinstance(value, dict):
            found = _find_qa_provenance_rows(value, depth + 1)
            if found is not None:
                return found
    return None


def _rows_from_qa_provenance(qa_rows: list) -> list[dict[str, Any]]:
    """Build benchmark rows from persisted ``qa_provenance_rows``.

    industry/domain/geo are not persisted at the row level; they don't affect
    the skip decision (routing is by question + tag + aspect), so leaving them
    None keeps the 0/35 gate correct while degrading only the query-text diff.
    """
    rows: list[dict[str, Any]] = []
    for qa in qa_rows:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question") or ""
        rows.append(
            {
                "company_name": qa.get("company_name") or qa.get("startup_slug") or "",
                "industry": None,
                "domain": None,
                "geo": None,
                "aspect": qa.get("aspect"),
                "question": question,
                "question_hash": question_hash(question),
                "answer": qa.get("answer") or "",
                "chunks_preview": qa.get("chunks_preview") or "",
                "web_search_query": qa.get("web_search_query") or "",
                "web_search_results": qa.get("web_search_results") or "",
                "web_search_used": bool(qa.get("web_search_used")),
                "web_search_decision": qa.get("web_search_decision") or "",
            }
        )
    return rows


def extract_qa_rows(results_payload: Any) -> list[dict[str, Any]]:
    """Flatten a persisted job results payload into per-question rows.

    Handles two shapes: the worker-composed persisted shape (flat
    ``qa_provenance_rows``, what ``load_job_results`` returns) and the in-memory
    pipeline shape (a list of company entries with ``final_state.all_qa_pairs``,
    used by the synthetic tests).
    """
    persisted = _find_qa_provenance_rows(results_payload) if isinstance(results_payload, dict) else None
    if persisted is not None:
        return _rows_from_qa_provenance(persisted)

    payload = results_payload
    for _ in range(3):
        if isinstance(payload, dict) and "results" in payload:
            payload = payload["results"]
    if not isinstance(payload, list):
        raise ValueError(
            "Unrecognized results payload shape: expected a company list or qa_provenance_rows"
        )

    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("skipped"):
            continue
        final_state = entry.get("final_state") or {}
        qa_pairs = final_state.get("all_qa_pairs") or []
        fields = _company_fields(entry)
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or ""
            rows.append(
                {
                    **fields,
                    "aspect": qa.get("aspect"),
                    "question": question,
                    "question_hash": question_hash(question),
                    "answer": qa.get("answer") or "",
                    "chunks_preview": qa.get("chunks_preview") or "",
                    "web_search_query": qa.get("web_search_query") or "",
                    "web_search_results": qa.get("web_search_results") or "",
                    "web_search_used": bool(qa.get("web_search_used")),
                    "web_search_decision": qa.get("web_search_decision") or "",
                }
            )
    return rows


def is_thin_answer(answer: str) -> bool:
    return _answer_indicates_no_evidence(answer)


def is_rescued(row: dict[str, Any]) -> bool:
    """A persisted answer the legacy web fallback actually rescued."""
    return bool(row.get("web_search_used")) and not is_thin_answer(row.get("answer", ""))


def was_searched(row: dict[str, Any]) -> bool:
    return bool(row.get("web_search_query"))


# The single source of truth for "web search actually made it into the answer":
# a search ran AND the final answer cites [web] AND the answer is not thin. The
# audit tool (scripts/audit_run_evidence.py) imports this so its WEB_ADDED bucket
# and the replay merge-gate's false-skip denominator can never drift apart.
WEB_CITE = re.compile(r"\[web\]", re.IGNORECASE)


def web_added(row: dict[str, Any]) -> bool:
    """True iff a web search ran and its evidence reached a substantive answer."""
    if not was_searched(row):
        return False
    answer = row.get("answer", "") or ""
    return bool(WEB_CITE.search(answer)) and not is_thin_answer(answer)


# Persisted rows lose tree depth (build_plan_for_row pins is_root=False), so the
# planner's static-root map never fires in replay. Recover it for the targeted
# scorer by matching the question text against the canonical root questions
# (Risk R1) — otherwise the four templated root questions go unmeasured.
_ROOT_QUESTIONS = frozenset(q.strip().lower() for q in INVESTMENT_QUESTIONS.values())


def synthesize_is_root(row: dict[str, Any]) -> bool:
    """True when the persisted question is a canonical root investment question."""
    return (row.get("question") or "").strip().lower() in _ROOT_QUESTIONS


# ---------------------------------------------------------------------------
# Plan building (replay + live share this)
# ---------------------------------------------------------------------------


def build_plan_for_row(
    row: dict[str, Any],
    route_tags: dict[str, str] | None = None,
    current_year: int | None = None,
) -> WebSearchPlan:
    tag = (route_tags or {}).get(row["question_hash"])
    return build_web_search_plan(
        question=row["question"],
        company_name=row["company_name"],
        company_domain=row.get("domain"),
        industry_hint=row.get("industry"),
        geo_hint=row.get("geo"),
        current_year=current_year or datetime.now().year,
        route_tag=tag,
        aspect=row.get("aspect"),
        # Persisted rows do not record tree depth; treat none as root so the
        # static root map never fires and routing relies on tags + fallback.
        is_root=False,
    )


def find_skip_route_false_positives(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """HARD GATE: rescued answers that the planner would route to a skip route.

    Must be empty (the benchmark has 35 rescued answers; target is 0/35).
    """
    offenders = []
    for row in rows:
        if not is_rescued(row):
            continue
        plan = build_plan_for_row(row, route_tags)
        if plan.is_skip:
            offenders.append(
                {
                    "company_name": row["company_name"],
                    "question": row["question"],
                    "route": plan.route.value,
                    "rationale": plan.rationale,
                }
            )
    return offenders


# ---------------------------------------------------------------------------
# Skip-unanswerable gate (Sprint 4 C) — same router, so equivalent to the
# planner skip-route check above; exposed for the `skip` stage / merge gate.
# ---------------------------------------------------------------------------


def predict_skip_for_row(
    row: dict[str, Any],
    route_tags: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Run the skip-unanswerable predictor for one persisted row."""
    tag = (route_tags or {}).get(row["question_hash"])
    return predict_search_unanswerable(
        question=row["question"],
        route_tag=tag,
        aspect=row.get("aspect"),
        company_name=row["company_name"],
        company_domain=row.get("domain"),
        industry=row.get("industry"),
        is_root=False,
    )


def skip_predictor_false_positives(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """HARD GATE: rescued answers the skip-unanswerable gate would suppress.

    Must be empty. The predictor delegates to build_web_search_plan, so this is
    equivalent to find_skip_route_false_positives — proving the gate can never
    regress the planner's already-passing 0/35 check.
    """
    offenders = []
    for row in rows:
        if not is_rescued(row):
            continue
        skip, reason = predict_skip_for_row(row, route_tags)
        if skip:
            offenders.append(
                {
                    "company_name": row["company_name"],
                    "question": row["question"],
                    "reason": reason,
                }
            )
    return offenders


def skip_prediction_stats(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Skip-rate, the 0-false-positive count, and projected per-company savings."""
    rows = list(rows)
    total = len(rows)
    # "documents incomplete" subset: thin answers that actually ran a search.
    thin_searched = [r for r in rows if was_searched(r) and is_thin_answer(r.get("answer", ""))]
    skipped_all = sum(1 for r in rows if predict_skip_for_row(r, route_tags)[0])
    skipped_thin = sum(1 for r in thin_searched if predict_skip_for_row(r, route_tags)[0])
    companies = {r["company_name"] for r in rows}
    offenders = skip_predictor_false_positives(rows, route_tags)
    return {
        "total_rows": total,
        "documents_incomplete_rows": len(thin_searched),
        "skip_rate_all": round(skipped_all / total, 4) if total else 0.0,
        "skip_rate_documents_incomplete": (
            round(skipped_thin / len(thin_searched), 4) if thin_searched else 0.0
        ),
        "rescued_predicted_skip": len(offenders),
        "projected_searches_saved_per_company": round(skipped_thin / max(1, len(companies)), 2),
        "offenders": offenders,
        "pass": not offenders,
    }


# ---------------------------------------------------------------------------
# Targeted ("Off / Targeted / Full") skip set — broader than the shipped
# predictor (re-adds internal_fit + the team aspect band). Opt-in, so the merge
# gate is band-scoped: ZERO false-skips on market/product (a theorem for the
# team arm), while team / general_company drops are the intended savings.
# ---------------------------------------------------------------------------


def predict_targeted_skip_for_row(
    row: dict[str, Any],
    route_tags: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Run the Targeted predictor for one persisted row (root-aware via R1)."""
    tag = (route_tags or {}).get(row["question_hash"])
    return predict_targeted_skip(
        question=row["question"],
        route_tag=tag,
        aspect=row.get("aspect"),
        company_name=row["company_name"],
        company_domain=row.get("domain"),
        industry=row.get("industry"),
        is_root=synthesize_is_root(row),
    )


def targeted_skip_false_positives_by_aspect(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Bucket, by aspect, value-adding rows that Targeted would have skipped.

    Pass rows that genuinely earned their search (``is_rescued`` or ``web_added``);
    every returned row is therefore a false skip. The market and product buckets
    MUST be empty for the gate to pass.
    """
    by_aspect: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        skip, reason = predict_targeted_skip_for_row(row, route_tags)
        if not skip:
            continue
        aspect = str(row.get("aspect") or "?")
        by_aspect.setdefault(aspect, []).append(
            {
                "company_name": row["company_name"],
                "question": row["question"],
                "aspect": row.get("aspect"),
                "reason": reason,
            }
        )
    return by_aspect


def targeted_skip_stats(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Saving + band-scoped false-skip stats for the Targeted skip set.

    ``pass`` == zero false-skips on the market and product aspect bands, measured
    over the conservative ``is_rescued`` denominator (also reported over the
    stricter ``web_added`` denominator). team / general_company drops are the
    intended, opted-into savings and do not fail the gate.
    """
    rows = list(rows)
    total = len(rows)
    companies = {r["company_name"] for r in rows}

    # "documents incomplete" proxy: thin answers that actually ran a search.
    thin_searched = [r for r in rows if was_searched(r) and is_thin_answer(r.get("answer", ""))]
    suppressed = [r for r in thin_searched if predict_targeted_skip_for_row(r, route_tags)[0]]
    suppressed_by_aspect: dict[str, int] = {}
    for r in suppressed:
        key = str(r.get("aspect") or "?")
        suppressed_by_aspect[key] = suppressed_by_aspect.get(key, 0) + 1
    current_suppressed = sum(
        1 for r in thin_searched if predict_skip_for_row(r, route_tags)[0]
    )

    rescued_rows = [r for r in rows if is_rescued(r)]
    fs_by_aspect = targeted_skip_false_positives_by_aspect(rescued_rows, route_tags)
    web_added_rows = [r for r in rows if web_added(r)]
    fs_web_added = targeted_skip_false_positives_by_aspect(web_added_rows, route_tags)

    market_fs = len(fs_by_aspect.get("market", []))
    product_fs = len(fs_by_aspect.get("product", []))
    team_fs = len(fs_by_aspect.get("team", []))
    tagged = sum(1 for r in rows if (route_tags or {}).get(r["question_hash"]))

    return {
        "total_rows": total,
        "documents_incomplete_rows": len(thin_searched),
        "saving": {
            "searches_suppressed_total": len(suppressed),
            "searches_suppressed_by_aspect": suppressed_by_aspect,
            "projected_searches_saved_per_company": round(
                len(suppressed) / max(1, len(companies)), 2
            ),
            "delta_vs_current_skip_set": len(suppressed) - current_suppressed,
        },
        "false_skips": {
            "denominator": "is_rescued",
            "rescued_rows": len(rescued_rows),
            "market": market_fs,
            "product": product_fs,
            "team": team_fs,
            "by_aspect": {a: len(v) for a, v in fs_by_aspect.items()},
            "offenders_market_product": (
                fs_by_aspect.get("market", []) + fs_by_aspect.get("product", [])
            ),
            "offenders_team": fs_by_aspect.get("team", []),
            "web_added_denominator": {
                "web_added_rows": len(web_added_rows),
                "market": len(fs_web_added.get("market", [])),
                "product": len(fs_web_added.get("product", [])),
                "by_aspect": {a: len(v) for a, v in fs_web_added.items()},
            },
        },
        "tag_coverage": {
            "tagged": tagged,
            "total": total,
            "ratio": round(tagged / total, 4) if total else 0.0,
        },
        "pass": market_fs == 0 and product_fs == 0,
    }


def gate_confusion_matrix(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Re-run legacy vs planner gates against PERSISTED web results (free)."""
    matrix = {"both_accept": 0, "both_reject": 0, "newly_accepted": 0, "newly_rejected": 0}
    newly_accepted_samples: list[dict[str, Any]] = []
    for row in rows:
        results_text = row.get("web_search_results") or ""
        if not was_searched(row) or not results_text:
            continue
        legacy_ok, _ = _web_results_add_value(
            row["question"], row["company_name"], results_text
        )
        plan = build_plan_for_row(row, route_tags)
        if plan.is_skip:
            new_ok = False
        else:
            new_ok, _ = evaluate_web_result_relevance(
                policy=plan.relevance_policy,
                route=plan.route,
                question=row["question"],
                company_name=row["company_name"],
                web_results=results_text,
                industry_hint=row.get("industry"),
            )
        if legacy_ok and new_ok:
            matrix["both_accept"] += 1
        elif not legacy_ok and not new_ok:
            matrix["both_reject"] += 1
        elif new_ok:
            matrix["newly_accepted"] += 1
            if len(newly_accepted_samples) < 25:
                newly_accepted_samples.append(
                    {
                        "company_name": row["company_name"],
                        "route": plan.route.value,
                        "question": row["question"],
                        "results_preview": results_text[:300],
                    }
                )
        else:
            matrix["newly_rejected"] += 1
    matrix["newly_accepted_samples"] = newly_accepted_samples
    return matrix


def route_distribution(
    rows: Iterable[dict[str, Any]],
    route_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    by_route: dict[str, int] = {}
    by_source: dict[str, int] = {}
    query_diffs: list[dict[str, Any]] = []
    skip_count = 0
    total = 0
    for row in rows:
        total += 1
        plan = build_plan_for_row(row, route_tags)
        by_route[plan.route.value] = by_route.get(plan.route.value, 0) + 1
        source = plan.rationale.split(":", 1)[0]
        by_source[source] = by_source.get(source, 0) + 1
        if plan.is_skip:
            skip_count += 1
        elif len(query_diffs) < 3 * len(QuestionRoute) and row.get("web_search_query"):
            new_queries = [q.query for q in plan.queries]
            if plan.route.value != "company_specific" and len(
                [d for d in query_diffs if d["route"] == plan.route.value]
            ) < 3:
                query_diffs.append(
                    {
                        "route": plan.route.value,
                        "question": row["question"],
                        "old_query": row["web_search_query"],
                        "new_queries": new_queries,
                    }
                )
    return {
        "total_questions": total,
        "by_route": by_route,
        "by_routing_source": by_source,
        "skip_route_questions": skip_count,
        "old_vs_new_query_examples": query_diffs,
    }


def build_replay_report(
    rows: list[dict[str, Any]],
    route_tags: dict[str, str] | None,
    live_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    searched = [r for r in rows if was_searched(r)]
    rescued = [r for r in rows if is_rescued(r)]
    offenders = find_skip_route_false_positives(rows, route_tags)
    report = {
        "benchmark": {
            "total_questions": len(rows),
            "searched_questions": len(searched),
            "rescued_answers": len(rescued),
            "baseline_rescue_rate": round(len(rescued) / len(searched), 4) if searched else None,
            "thin_answers": sum(1 for r in rows if is_thin_answer(r["answer"])),
        },
        "route_distribution": route_distribution(rows, route_tags),
        "gate_confusion_matrix_on_persisted_results": gate_confusion_matrix(searched, route_tags),
        "false_positive_check": {
            "rescued_answers": len(rescued),
            "rescued_routed_to_skip": len(offenders),
            "offenders": offenders,
            "pass": not offenders,
        },
        "skip_predictor": skip_prediction_stats(rows, route_tags),
        "tagged_questions": len(route_tags or {}),
    }
    if live_results:
        report["live_stage"] = live_results
    return report


# ---------------------------------------------------------------------------
# Stage: tag (one-off LLM tagging of benchmark questions)
# ---------------------------------------------------------------------------

TAG_SYSTEM_PROMPT = (
    "You classify due-diligence questions about early-stage startups.\n\n"
    + ROUTE_TAGGING_INSTRUCTION
    + "\nReturn a route for every numbered question."
)


def merge_route_tags(
    existing: dict[str, str], new_tags: dict[str, str]
) -> dict[str, str]:
    """Idempotent merge: existing tags win (re-runs only fill gaps)."""
    merged = dict(new_tags)
    merged.update(existing)
    return merged


def _untagged_questions(
    rows: list[dict[str, Any]], existing: dict[str, str]
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    pending: list[tuple[str, str]] = []
    for row in rows:
        qh = row["question_hash"]
        if qh in existing or qh in seen:
            continue
        seen.add(qh)
        pending.append((qh, row["question"]))
    return pending


async def _tag_batch_async(questions: list[tuple[str, str]]) -> dict[str, str]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel

    from agent.llm import create_llm

    class QuestionRouteTag(BaseModel):
        index: int
        route: str

    class RouteTagBatch(BaseModel):
        tags: list[QuestionRouteTag]

    numbered = "\n".join(f"{i}. {q}" for i, (_, q) in enumerate(questions))
    llm = create_llm(temperature=0.0)
    structured = llm.with_structured_output(RouteTagBatch)
    result: RouteTagBatch = await structured.ainvoke(
        [
            SystemMessage(content=TAG_SYSTEM_PROMPT),
            HumanMessage(content=f"Questions:\n{numbered}"),
        ]
    )
    tags: dict[str, str] = {}
    for tag in result.tags:
        if 0 <= tag.index < len(questions):
            normalized = normalize_route_tag(tag.route)
            if normalized is not None:
                tags[questions[tag.index][0]] = normalized.value
    return tags


def run_tag_stage(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    tags_path = out_dir / "route_tags.json"
    existing: dict[str, str] = {}
    if tags_path.exists():
        existing = json.loads(tags_path.read_text())
    pending = _untagged_questions(rows, existing)
    print(f"[tag] {len(existing)} tagged, {len(pending)} pending")
    merged = dict(existing)
    for start in range(0, len(pending), TAG_BATCH_SIZE):
        batch = pending[start : start + TAG_BATCH_SIZE]
        batch_tags = asyncio.run(_tag_batch_async(batch))
        merged = merge_route_tags(merged, batch_tags)
        out_dir.mkdir(parents=True, exist_ok=True)
        tags_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
        print(f"[tag] {start + len(batch)}/{len(pending)} tagged this run")
    return merged


# ---------------------------------------------------------------------------
# Stage: live (sampled re-search + re-answer)
# ---------------------------------------------------------------------------


def select_live_sample(
    rows: list[dict[str, Any]],
    route_tags: dict[str, str],
    sample_size: int = DEFAULT_LIVE_SAMPLE_SIZE,
    routes: tuple[str, ...] = TOPIC_ROUTES_FOR_LIVE,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Seeded stratified sample of still-thin questions on topic routes."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not is_thin_answer(row["answer"]) or not was_searched(row):
            continue
        plan = build_plan_for_row(row, route_tags)
        if plan.route.value not in routes:
            continue
        candidates.setdefault(plan.route.value, []).append(row)

    rng = random.Random(seed)
    per_route = max(1, sample_size // max(1, len(candidates)))
    sample: list[dict[str, Any]] = []
    for route in sorted(candidates):
        pool = candidates[route]
        rng.shuffle(pool)
        sample.extend(pool[:per_route])
    return sample[:sample_size]


async def _live_answer_async(row: dict[str, Any], plan: WebSearchPlan) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.llm import create_llm
    from agent.prompt_library.manager import get_prompt

    accepted_blocks: list[str] = []
    executed: list[str] = []
    for position, spec in enumerate(plan.queries, start=1):
        raw = await asyncio.to_thread(
            _run_web_search,
            spec.query,
            list(spec.domain_filter) if spec.domain_filter is not None else None,
            "replay live stage",
            None,
            plan.route.value,
            "replay",
            spec.purpose,
        )
        executed.append(spec.query)
        useful, _ = evaluate_web_result_relevance(
            policy=plan.relevance_policy,
            route=plan.route,
            question=row["question"],
            company_name=row["company_name"],
            web_results=raw,
            industry_hint=row.get("industry"),
        )
        if useful:
            accepted_blocks.append(
                f'=== Web results {position}/{len(plan.queries)} '
                f'(purpose: {spec.purpose}; query: "{spec.query}") ===\n{raw}'
            )

    outcome = {
        "company_name": row["company_name"],
        "route": plan.route.value,
        "question": row["question"],
        "old_answer": row["answer"],
        "queries_executed": executed,
        "results_accepted": len(accepted_blocks),
    }
    if not accepted_blocks:
        outcome.update({"new_answer": row["answer"], "rescued": False})
        return outcome

    hybrid_system = get_prompt("evidence.hybrid.system.web_planner")
    hybrid_user = get_prompt("evidence.hybrid.user")
    company_summary = f"Company: {row['company_name']}"
    if row.get("industry"):
        company_summary += f"\nIndustry: {row['industry']}"
    llm = create_llm(temperature=0.2)
    response = await llm.ainvoke(
        [
            SystemMessage(content=hybrid_system),
            HumanMessage(
                content=hybrid_user.format(
                    company_summary=company_summary,
                    question=row["question"],
                    chunks_text=row.get("chunks_preview") or "No relevant document chunks found.",
                    web_results="\n\n".join(accepted_blocks),
                )
            ),
        ]
    )
    new_answer = str(getattr(response, "content", "")) or row["answer"]
    outcome.update({"new_answer": new_answer, "rescued": not is_thin_answer(new_answer)})
    return outcome


def run_live_stage(
    rows: list[dict[str, Any]],
    route_tags: dict[str, str],
    out_dir: Path,
    sample_size: int,
    routes: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    sample = select_live_sample(rows, route_tags, sample_size, routes, seed)
    print(f"[live] sampled {len(sample)} thin questions on routes {routes}")
    outcomes: list[dict[str, Any]] = []
    for i, row in enumerate(sample, start=1):
        plan = build_plan_for_row(row, route_tags)
        outcome = asyncio.run(_live_answer_async(row, plan))
        outcomes.append(outcome)
        print(f"[live] {i}/{len(sample)} rescued={outcome['rescued']} route={outcome['route']}")

    by_route: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        bucket = by_route.setdefault(outcome["route"], {"sampled": 0, "rescued": 0})
        bucket["sampled"] += 1
        bucket["rescued"] += int(outcome["rescued"])
    rescued_total = sum(int(o["rescued"]) for o in outcomes)
    summary = {
        "sampled": len(outcomes),
        "rescued": rescued_total,
        "sampled_rescue_rate": round(rescued_total / len(outcomes), 4) if outcomes else None,
        "baseline_overall_rescue_rate": 0.074,
        "by_route": by_route,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_outcomes.json").write_text(json.dumps(outcomes, indent=2))
    csv_path = out_dir / "stage2_answers.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "company_name", "route", "question", "rescued",
                "results_accepted", "old_answer", "new_answer",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(outcomes)
    print(f"[live] wrote {csv_path} for manual hallucination review")
    return summary


# ---------------------------------------------------------------------------
# Loading helpers (Supabase access lives at the edge)
# ---------------------------------------------------------------------------


def load_benchmark_rows(job_id: str) -> list[dict[str, Any]]:
    from web.db import load_job_results

    payload = load_job_results(job_id)
    if payload is None:
        raise RuntimeError(
            f"Could not load job {job_id!r} — run under `railway run` so staging "
            "Supabase credentials are available."
        )
    return extract_qa_rows(payload)


def load_route_tags(out_dir: Path) -> dict[str, str]:
    tags_path = out_dir / "route_tags.json"
    if tags_path.exists():
        return json.loads(tags_path.read_text())
    return {}
