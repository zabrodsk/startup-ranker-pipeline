"""Answer question trees using retrieved document chunks and optional web search.

Documents (uploaded spreadsheets, pitch decks, etc.) are the PRIMARY source —
they contain insider information and drive the analysis. Web search (Perplexity)
is a FALLBACK only when the grounded answer indicates lack of evidence.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List

# Timeout per LLM call to avoid indefinite hangs (e.g. API stalls, rate limits)
LLM_ANSWER_TIMEOUT_SEC = 120
# Timeout for web search (Perplexity/Brave can be slow)
WEB_SEARCH_TIMEOUT_SEC = 45
# Max concurrent LLM calls to avoid rate limiting and connection exhaustion
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("LLM_MAX_CONCURRENT", "2"))
# Max web search (Perplexity/Brave) calls per company to control cost
# Default raised from 2 -> 100 to strongly favor coverage over cost.
MAX_PPLX_CALLS_PER_COMPANY = int(os.getenv("MAX_PPLX_CALLS_PER_COMPANY", "100"))
# When to trigger web search:
#   "answer" = only when LLM answer indicates no evidence (e.g. "Unknown from provided documents")
#   "no_chunks" = legacy: only when documents have zero chunks
WEB_SEARCH_TRIGGER = os.getenv("WEB_SEARCH_TRIGGER", "answer").lower()  # "answer" | "no_chunks"

# Gating for the "web-heavy question" override (_question_prefers_web_search),
# which forces a search for market/competitor-style questions even when the
# grounded answer is complete:
#   "always"    = override fires for any question node (current behavior, default)
#   "root_only" = override fires only for tree-root questions
#   "never"     = override disabled; search happens only when the grounded answer
#                 indicates missing evidence (or via the legacy no-chunks trigger)
# Read at call time (not import time) so tests and A/B runs can flip it via env.
WEB_SEARCH_HEAVY_OVERRIDE_ENV = "WEB_SEARCH_HEAVY_OVERRIDE"
_WEB_SEARCH_HEAVY_OVERRIDE_MODES = ("always", "root_only", "never")
_DEFAULT_WEB_SEARCH_HEAVY_OVERRIDE = "always"
_WARNED_INVALID_HEAVY_OVERRIDE: set[str] = set()


def _web_search_heavy_override() -> str:
    """Return the active gating mode for the web-heavy question override."""
    raw = os.getenv(WEB_SEARCH_HEAVY_OVERRIDE_ENV, _DEFAULT_WEB_SEARCH_HEAVY_OVERRIDE)
    mode = raw.strip().lower()
    if mode not in _WEB_SEARCH_HEAVY_OVERRIDE_MODES:
        if mode not in _WARNED_INVALID_HEAVY_OVERRIDE:
            _WARNED_INVALID_HEAVY_OVERRIDE.add(mode)
            logging.getLogger(__name__).warning(
                "Invalid %s=%r; falling back to %r",
                WEB_SEARCH_HEAVY_OVERRIDE_ENV, raw, _DEFAULT_WEB_SEARCH_HEAVY_OVERRIDE,
            )
        return _DEFAULT_WEB_SEARCH_HEAVY_OVERRIDE
    return mode


# Feature flag for the route-aware WebEvidencePlanner (Sprint 2):
#   "off"    = legacy company-anchored behavior (byte-identical, default)
#   "shadow" = legacy behavior executes unchanged; the planner only builds a
#              plan and records it in provenance (zero extra provider calls)
#   "on"     = the planner controls query construction, domain filters,
#              relevance gating, and skip routes
# Read at call time (not import time) so tests and rollout can flip it via env.
RDI_WEB_EVIDENCE_PLANNER_ENV = "RDI_WEB_EVIDENCE_PLANNER"
_WEB_EVIDENCE_PLANNER_MODES = ("off", "shadow", "on")
_DEFAULT_WEB_EVIDENCE_PLANNER_MODE = "off"
_WARNED_INVALID_PLANNER_MODE: set[str] = set()


def _web_evidence_planner_mode() -> str:
    """Return the active WebEvidencePlanner mode."""
    raw = os.getenv(RDI_WEB_EVIDENCE_PLANNER_ENV, _DEFAULT_WEB_EVIDENCE_PLANNER_MODE)
    mode = raw.strip().lower()
    if mode not in _WEB_EVIDENCE_PLANNER_MODES:
        if mode not in _WARNED_INVALID_PLANNER_MODE:
            _WARNED_INVALID_PLANNER_MODE.add(mode)
            logging.getLogger(__name__).warning(
                "Invalid %s=%r; falling back to %r",
                RDI_WEB_EVIDENCE_PLANNER_ENV, raw, _DEFAULT_WEB_EVIDENCE_PLANNER_MODE,
            )
        return _DEFAULT_WEB_EVIDENCE_PLANNER_MODE
    return mode


# Patterns indicating the LLM could not answer from documents
_ANSWER_NO_EVIDENCE_PATTERNS = [
    r"unknown\s+from\s+provided\s+documents",
    r"no\s+information\s+(about|regarding|on)\s+",
    r"there\s+is\s+no\s+information",
    r"does\s+not\s+contain\s+(any\s+)?(information|data|evidence)",
    r"does\s+not\s+contain\s+a\s+specific\s+",
    r"does\s+not\s+name\s+specific",
    r"does\s+not\s+identify\s+.*?\s+by\s+name",
    r"does\s+not\s+provide\s+the\s+necessary\s+data",
    r"absence\s+of\s+(disclosed\s+)?(tam|sam|som|market\s+sizing|market\s+data)",
    r"not\s+named\s+in\s+the\s+(documents|evidence|materials)",
    r"not\s+identified\s+in\s+the\s+(documents|evidence|materials)",
    r"contains\s+no\s+(information|historical|data|evidence)",
    r"however,\s+there\s+is\s+no",
    r"but\s+contains\s+no\s+",
    r"insufficient\s+information\s+available",
    r"difficult\s+to\s+(calculate|estimate|validate)",
    r"no\s+relevant\s+(document\s+)?chunks\s+found",
    r"outlines\s+.*?\s+but\s+contains\s+no\s+",  # "strategy outlines X but contains no Y"
]
_ANSWER_NO_EVIDENCE_RE = re.compile(
    "|".join(f"({p})" for p in _ANSWER_NO_EVIDENCE_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

_SEARCH_BAD_RESULT_PATTERNS = [
    r"no web search api key configured",
    r"web search failed",
    r"\b429\b",
    r"rate limit",
    r"timed out",
    r"timeout",
    r"no relevant (web )?results",
    r"no search results found",
    r"no search results returned",
]
_SEARCH_BAD_RESULT_RE = re.compile(
    "|".join(f"({p})" for p in _SEARCH_BAD_RESULT_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

_TOKEN_STOPWORDS = {
    "what", "which", "when", "where", "why", "how", "does", "do", "is", "are", "the",
    "and", "or", "for", "with", "from", "that", "this", "into", "about", "their",
    "its", "they", "them", "you", "your", "can", "could", "would", "should", "built",
    "support", "supports", "product", "company",
}

_WEB_HEAVY_QUESTION_PATTERNS = [
    r"\btam\b",
    r"\bsam\b",
    r"\bsom\b",
    r"total\s+addressable\s+market",
    r"serviceable\s+available\s+market",
    r"serviceable\s+obtainable\s+market",
    r"market\s+siz(?:e|ing)",
    r"market\s+share",
    r"\bcagr\b",
    r"competitor",
    r"competitive\s+landscape",
    r"\brival",
    r"\balternative",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bmoat",
]
_WEB_HEAVY_QUESTION_RE = re.compile("|".join(f"({p})" for p in _WEB_HEAVY_QUESTION_PATTERNS), re.IGNORECASE)

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dataclasses.company import Company
from agent.dataclasses.question_tree import QuestionNode, QuestionTree
from agent.ingest.store import Chunk, EvidenceStore
from agent.llm import create_llm
from agent.prompt_library.manager import get_prompt
from agent.rate_limit import gather_with_concurrency
from agent.retrieval import retrieve_chunks, retrieve_chunks_with_similarity
from agent.web_search.planner import (
    WebSearchPlan,
    build_web_search_plan,
    evaluate_web_result_relevance,
)
from agent.web_search.portfolio import (
    PortfolioQuestion,
    build_company_web_evidence_plan,
    evidence_bucket_key,
)
from agent.web_search.unanswerable import (
    predict_search_unanswerable,
    predict_targeted_skip,
    resolve_web_search_mode,
    skip_unanswerable_mode,
)
from agent.run_context import (
    get_current_collector,
    get_current_pipeline_policy,
    get_current_web_search_mode,
    use_phase_llm,
    use_stage_context,
)

GROUNDED_SYSTEM_PROMPT = """\
You are an investment analyst answering due-diligence questions about a startup.

Rules:
- Answer ONLY using the evidence chunks provided below.
- Cite chunks by their IDs, e.g. [chunk_12], [chunk_44].
- Keep answers concise (under 80 words) and data-backed.
- If the provided evidence does not contain enough information to answer, \
respond with "Unknown from provided documents."
- Do NOT invent facts or use external knowledge.
"""

HYBRID_SYSTEM_PROMPT = """\
You are an investment analyst answering due-diligence questions about a startup.

You have TWO sources of information:
1. Document evidence chunks from the startup's own materials (cite as [chunk_XX]) — \
this is the PRIMARY source. Insider info from pitch decks, financials, and spreadsheets \
is critical and authoritative.
2. Web search results (cite as [web]) — ONLY a fallback when documents lack the answer.

Rules:
- Prioritize document evidence ALWAYS. It is the main source.
- Use web search results ONLY to fill gaps when documents cannot answer.
- Keep answers concise (under 100 words) and data-backed.
- Always indicate which source you are citing.
- If neither source has enough information, say "Insufficient information available."
"""

GROUNDED_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

Evidence chunks:
{chunks_text}
"""

HYBRID_USER_PROMPT = """\
Company: {company_summary}

Question: {question}

=== Document Evidence ===
{chunks_text}

=== Web Search Results ===
{web_results}
"""


def _coerce_text(value: Any) -> str:
    """Coerce model/tool outputs to plain text.

    Some providers can return list/dict content blocks instead of a raw string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                # Skip OpenAI Responses API reasoning blocks and other
                # non-text metadata items so they never leak into prose.
                if item.get("type") == "reasoning":
                    continue
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val)
                    continue
                content_val = item.get("content")
                if isinstance(content_val, str) and content_val.strip():
                    parts.append(content_val)
                    continue
                # Unknown dict shape — drop silently rather than serialising
                # the Python repr into user-visible text.
                continue
            # Unknown non-dict, non-str item — drop silently.
            continue
        return " ".join(p.strip() for p in parts if p and p.strip()).strip()
    if isinstance(value, dict):
        if value.get("type") == "reasoning":
            return ""
        for key in ("text", "content", "output_text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return ""
    return str(value)


def _answer_indicates_no_evidence(answer: Any) -> bool:
    """Return True if the LLM answer indicates it lacks evidence from documents.

    Matches patterns: Unknown from provided documents; no information about;
    there is no information; contains no / does not contain; however, there is no;
    insufficient information available; no relevant chunks found; outlines...but contains no.
    Empty answers are treated as no evidence.
    """
    text = _coerce_text(answer)
    if not text or not text.strip():
        return True
    return bool(_ANSWER_NO_EVIDENCE_RE.search(text))


def _question_prefers_web_search(question: str) -> bool:
    """Return True for question types that typically benefit from external data."""
    return bool(_WEB_HEAVY_QUESTION_RE.search(question or ""))


def _web_search_domain_filter(company: Company, question: str) -> list[str] | None:
    """Use broad web search for market/competitor questions, tighter domains otherwise."""
    if _question_prefers_web_search(question):
        return None
    if not company.domain:
        return None

    raw = company.domain.strip().lower()
    raw = re.sub(r"^https?://(www\.)?", "", raw).split("/")[0]
    if not raw:
        return None
    return [raw, "crunchbase.com", "linkedin.com"]


def _tokenize_text(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {t for t in tokens if len(t) >= 3 and t not in _TOKEN_STOPWORDS}


def _web_results_add_value(question: str, company_name: str, web_results: str) -> tuple[bool, str]:
    """Heuristic quality gate for fallback search results.

    Returns (is_useful, reason).
    """
    raw = (web_results or "").strip()
    if _SEARCH_BAD_RESULT_RE.search(raw):
        return (False, "web results indicate failure/noise")
    if len(raw) < 80:
        return (False, "web results too short")

    result_tokens = _tokenize_text(raw)
    if not result_tokens:
        return (False, "web results have no usable tokens")

    company_tokens = _tokenize_text(company_name)
    if company_tokens and not (company_tokens & result_tokens):
        return (False, "web results do not mention the company")

    q_tokens = _tokenize_text(question)
    if not q_tokens:
        return (True, "question has no strong tokens; accepting web results")

    overlap = len(q_tokens & result_tokens)
    overlap_ratio = overlap / max(1, len(q_tokens))
    if overlap < 2 and overlap_ratio < 0.20:
        return (False, "web results weakly related to question")

    return (True, "web results relevant to company/question")


_QUESTION_PREFIX_RE = re.compile(
    r"^(what is the?|how (?:does|is|large|many)|which|who are|can you|please|does the)\s+",
    re.IGNORECASE,
)
_WEB_SEARCH_QUERY_MAX_LEN = 180
_WEB_QUERY_DROP_TOKENS = {
    "what", "which", "when", "where", "why", "how", "who", "are", "is", "does", "do",
    "the", "a", "an", "and", "or", "of", "for", "with", "to", "their", "them", "these",
    "those", "main", "direct", "specific",
}


def _ordered_query_terms(text: str, company_name: str) -> list[str]:
    company_tokens = set(re.findall(r"[a-z0-9]+", (company_name or "").lower()))
    terms: list[str] = []

    for raw_token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        token = raw_token
        if token in company_tokens or token in _WEB_QUERY_DROP_TOKENS:
            continue
        if token in {"compare", "compared", "versus", "vs"}:
            token = "comparison"
        if terms and terms[-1] == token:
            continue
        terms.append(token)

    return terms


def _build_web_search_query(company: Company, question: str) -> str:
    """Reformulate question into a company-first, keyword-rich search query.

    Avoids definition-style results (e.g. generic TAM explanations) by stripping
    interrogative scaffolding and leading with the company name.
    """
    q = question.strip()
    q = _QUESTION_PREFIX_RE.sub("", q, count=1)
    q = re.sub(r"\bthe company'?s?\b", company.name, q, flags=re.IGNORECASE)
    q = re.sub(
        rf"\b(?:compare|compared)\s+(?:to|with)\s+{re.escape(company.name)}\b",
        "comparison",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        rf"\b(?:vs\.?|versus)\s+{re.escape(company.name)}\b",
        "comparison",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    ordered_terms = _ordered_query_terms(q, company.name)
    ordered_terms = ordered_terms or ["overview"]

    q_lower = q.lower()
    if any(token in q_lower for token in ("competitor", "rival", "alternative", "compare", "comparison")):
        for token in ("competitors", "alternatives", "rivals"):
            if token not in ordered_terms:
                ordered_terms.append(token)
    if company.industry and ("market" in q_lower or "tam" in q_lower or "segment" in q_lower):
        ordered_terms.extend(_ordered_query_terms(company.industry, company.name))

    deduped_terms: list[str] = []
    for term in ordered_terms:
        if term not in deduped_terms:
            deduped_terms.append(term)

    result = " ".join([company.name, *deduped_terms])
    if len(result) > _WEB_SEARCH_QUERY_MAX_LEN:
        result = result[: _WEB_SEARCH_QUERY_MAX_LEN - 3].rsplit(" ", 1)[0] + "..."
    return result


def _run_web_search(
    search_query: str,
    domain_filter: list[str] | None = None,
    trigger_reason: str | None = None,
    gating_mode: str | None = None,
    route: str | None = None,
    planner_mode: str | None = None,
    query_purpose: str | None = None,
    cache_info: dict | None = None,
    provider_override: str | None = None,
    fallback_from: str | None = None,
) -> str:
    """Run a web search using the configured provider.

    trigger_reason/gating_mode are recorded in telemetry metadata so run costs
    can aggregate WHY searches fired (see build_run_costs_from_model_executions).

    cache_info (W13): when the caller passes a dict, its "hit" key is set so
    the caller can refund the cap slot it acquired before this call. With
    RDI_WEB_SEARCH_CACHE off (default) the cache is never consulted and the
    behavior is byte-identical to the pre-cache version.
    """
    from datetime import datetime

    configured_provider = _resolve_web_search_provider_name()
    if not configured_provider:
        return "No web search API key configured."
    provider_name = (provider_override or configured_provider).strip().lower()

    try:
        from agent.web_search import get_provider, result_cache

        cached_result = result_cache.lookup(provider_name, search_query, domain_filter)
        cache_hit = cached_result is not None
        if cache_info is not None:
            cache_info["hit"] = cache_hit
        if cache_hit:
            result = cached_result
        else:
            search_date = datetime.now().strftime("%Y-%m-%d")
            provider = get_provider(search_end_date=search_date, provider_name=provider_name)
            result = provider.search(search_query, domain_filter=domain_filter)
            metered_provider_name = getattr(provider, "last_provider_name", None) or provider_name
        if cache_hit:
            metered_provider_name = provider_name
        result_is_valid = bool(
            result and not str(result).lower().startswith("web search failed")
        )
        collector = get_current_collector()
        if (
            collector
            and metered_provider_name in {"sonar", "serper"}
            and result_is_valid
            and not cache_hit
        ):
            metadata: dict[str, Any] = {
                "query": search_query,
                "domain_filter": domain_filter or [],
            }
            if trigger_reason:
                metadata["trigger_reason"] = trigger_reason
            if gating_mode:
                metadata["gating_mode"] = gating_mode
            # Planner telemetry: only present in shadow/on modes so legacy
            # (off) metadata stays byte-identical.
            if route:
                metadata["route"] = route
            if planner_mode:
                metadata["planner_mode"] = planner_mode
            if query_purpose:
                metadata["query_purpose"] = query_purpose
            if fallback_from:
                metadata["fallback_from"] = fallback_from
            if metered_provider_name == "sonar":
                collector.record_perplexity_search(metadata=metadata)
            else:
                collector.record_web_search(
                    provider=metered_provider_name,
                    metadata=metadata,
                )
        if not cache_hit and result_is_valid:
            result_cache.store(provider_name, search_query, domain_filter, result)
        return result
    except Exception as exc:
        return f"Web search failed: {exc}"


def _resolve_web_search_provider_name() -> str | None:
    from agent.web_search.providers import resolve_provider_name

    return resolve_provider_name()


def _shared_web_evidence_enabled() -> bool:
    return os.getenv("RDI_SHARED_WEB_EVIDENCE", "off").strip().lower() == "on"


async def _acquire_web_search_slot(web_search_state: dict[str, Any] | None) -> bool:
    """Reserve one real provider call under the company-wide cap."""
    if web_search_state is None:
        return True
    async with web_search_state["lock"]:
        if web_search_state["count"][0] >= web_search_state["max"]:
            return False
        web_search_state["count"][0] += 1
        return True


async def _refund_cached_web_search_slot(web_search_state: dict[str, Any] | None) -> None:
    """Cached evidence does not consume a provider-call slot."""
    if web_search_state is None:
        return
    async with web_search_state["lock"]:
        web_search_state["count"][0] = max(0, web_search_state["count"][0] - 1)


async def _run_quality_routed_search(
    *,
    query: str,
    domain_filter: list[str] | None,
    purpose: str,
    route: Any,
    relevance_policy: Any,
    representative_question: str,
    company: Company,
    trigger_reason: str,
    web_search_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run Serper first and use Perplexity only after a failed quality gate."""
    strategy = _resolve_web_search_provider_name()
    if not strategy:
        return None
    if not await _acquire_web_search_slot(web_search_state):
        return None
    primary_provider = "serper" if strategy == "hybrid" else None
    cache_info: dict[str, Any] = {}
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                _run_web_search,
                query,
                domain_filter,
                trigger_reason,
                "portfolio",
                route.value,
                "portfolio",
                purpose,
                cache_info,
                provider_override=primary_provider,
            ),
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raw = "Web search failed: timeout"
    if cache_info.get("hit"):
        await _refund_cached_web_search_slot(web_search_state)
    useful, reason = evaluate_web_result_relevance(
        policy=relevance_policy,
        route=route,
        question=representative_question,
        company_name=company.name,
        web_results=raw,
        industry_hint=company.industry,
    )
    accepted_provider = primary_provider or strategy

    if (
        not useful
        and strategy == "hybrid"
        and await _acquire_web_search_slot(web_search_state)
    ):
        fallback_cache_info: dict[str, Any] = {}
        try:
            fallback_raw = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_web_search,
                    query,
                    domain_filter,
                    "quality fallback",
                    "portfolio",
                    route.value,
                    "portfolio",
                    purpose,
                    fallback_cache_info,
                    provider_override="sonar",
                    fallback_from="serper",
                ),
                timeout=WEB_SEARCH_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            fallback_raw = "Web search failed: timeout"
        if fallback_cache_info.get("hit"):
            await _refund_cached_web_search_slot(web_search_state)
        fallback_useful, fallback_reason = evaluate_web_result_relevance(
            policy=relevance_policy,
            route=route,
            question=representative_question,
            company_name=company.name,
            web_results=fallback_raw,
            industry_hint=company.industry,
        )
        if fallback_useful:
            raw = fallback_raw
            useful = True
            reason = fallback_reason
            accepted_provider = "perplexity fallback"

    if not useful:
        return None
    return {
        "query": query,
        "purpose": purpose,
        "route": route,
        "relevance_policy": relevance_policy,
        "representative_question": representative_question,
        "provider": accepted_provider,
        "result": raw,
        "quality_reason": reason,
    }


def _portfolio_questions(question_trees: Dict[str, QuestionTree]) -> list[PortfolioQuestion]:
    questions: list[PortfolioQuestion] = []

    def visit(node: QuestionNode, *, aspect: str, is_root: bool) -> None:
        questions.append(
            PortfolioQuestion(
                question=node.question,
                route_tag=getattr(node, "route", None),
                aspect=aspect,
                is_root=is_root,
            )
        )
        for child in node.sub_nodes:
            visit(child, aspect=aspect, is_root=False)

    for aspect, tree in question_trees.items():
        visit(tree.root_node, aspect=aspect, is_root=True)
    return questions


async def _prepare_shared_web_evidence(
    question_trees: Dict[str, QuestionTree],
    company: Company,
    web_search_state: dict[str, Any],
    web_search_mode: str | None = None,
) -> dict[str, Any]:
    """Plan and fetch the bounded company-level evidence portfolio."""
    from datetime import datetime

    core_budget = max(0, int(os.getenv("WEB_SEARCH_CORE_BUDGET", "10")))
    reserve_budget = max(0, int(os.getenv("WEB_SEARCH_RESERVE_BUDGET", "2")))
    questions = _portfolio_questions(question_trees)
    effective_mode = resolve_web_search_mode(
        web_search_mode if web_search_mode is not None else get_current_web_search_mode()
    )
    if effective_mode == "targeted":
        questions = [
            item
            for item in questions
            if not predict_targeted_skip(
                question=item.question,
                route_tag=item.route_tag,
                aspect=item.aspect,
                company_name=company.name,
                company_domain=company.domain,
                industry=company.industry,
                is_root=item.is_root,
            )[0]
        ]
    plan = build_company_web_evidence_plan(
        questions=questions,
        company_name=company.name,
        company_domain=company.domain,
        industry_hint=company.industry,
        geo_hint=getattr(company, "geo", None),
        current_year=datetime.now().year,
        core_budget=core_budget,
        reserve_budget=reserve_budget,
    )
    concurrency = max(1, int(os.getenv("WEB_SEARCH_PREFETCH_CONCURRENCY", "4")))
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(objective: Any) -> tuple[str, dict[str, Any] | None]:
        async with semaphore:
            evidence = await _run_quality_routed_search(
                query=objective.query.query,
                domain_filter=list(objective.query.domain_filter)
                if objective.query.domain_filter is not None
                else None,
                purpose=objective.query.purpose,
                route=objective.route,
                relevance_policy=objective.relevance_policy,
                representative_question=objective.representative_question,
                company=company,
                trigger_reason="portfolio core",
                web_search_state=web_search_state,
            )
            return objective.bucket_key, evidence

    fetched = await asyncio.gather(*(fetch(objective) for objective in plan.objectives))
    evidence_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for bucket_key, evidence in fetched:
        if evidence is not None:
            evidence_by_bucket.setdefault(bucket_key, []).append(evidence)

    return {
        "portfolio_plan": plan,
        "portfolio_evidence": evidence_by_bucket,
        "portfolio_core_queries": {objective.query.query for objective in plan.objectives},
        "portfolio_reserve_used": [0],
        "portfolio_reserve_max": plan.reserve_budget,
        "portfolio_lock": asyncio.Lock(),
    }


CHUNK_PREVIEW_CHARS = 200
WEB_RESULTS_TRUNCATE = 8000

async def answer_question_from_evidence(
    question: str,
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    use_web_search: bool = False,
    semaphore: asyncio.Semaphore | None = None,
    web_search_state: dict | None = None,
    vc_context: str = "",
    aspect: str = "",
    prompt_overrides: dict[str, Any] | None = None,
    is_root: bool = False,
    route: str | None = None,
    web_search_mode: str | None = None,
) -> tuple[str, dict]:
    """Answer a single question using retrieved document chunks and optional web search.

    Perplexity API is used only when ALL of the following hold:
    1. use_web_search=True (user enabled the toggle)
    2. needs_search=True, where:
       - If WEB_SEARCH_TRIGGER=answer: _answer_indicates_no_evidence(grounded_answer)
         (answer matches patterns like "Unknown from provided documents", "contains no",
         "there is no information", etc.)
       - If WEB_SEARCH_TRIGGER=no_chunks: not chunks
    3. Per-company cap not exceeded (web_search_state["count"] < max)

    Flow: (1) Run grounded LLM call first. (2) If answer indicates no evidence and
    above conditions hold, call Perplexity and run hybrid LLM with web results.

    Args:
        question: The due-diligence question to answer.
        company: Company metadata for context.
        store: EvidenceStore to retrieve from.
        k: Number of chunks to retrieve.
        use_web_search: If True, may run a web search when docs have no evidence.
        semaphore: Optional semaphore to limit concurrent LLM calls.
        web_search_state: Optional dict with count, lock, max for per-company search cap.
        is_root: True when this question is a tree root. Gates the web-heavy
            question override when WEB_SEARCH_HEAVY_OVERRIDE=root_only.
        route: Optional decomposition-time route tag for the WebEvidencePlanner
            (RDI_WEB_EVIDENCE_PLANNER=shadow|on). Ignored when the flag is off.

    Returns:
        Tuple of (answer string, provenance dict with chunk_ids, chunks_preview,
        web_search_query, web_search_results).
    """
    chunks = retrieve_chunks(question, store, k=k)

    chunk_ids = [c.chunk_id for c in chunks] if chunks else []
    chunks_preview = "\n---\n".join(
        f"[{c.chunk_id}]: {c.text[:CHUNK_PREVIEW_CHARS]}{'...' if len(c.text) > CHUNK_PREVIEW_CHARS else ''}"
        for c in chunks
    ) if chunks else ""

    chunks_text = "\n---\n".join(
        f"[{c.chunk_id}] (source: {c.source_file}, location: {c.page_or_slide}):\n{c.text}"
        for c in chunks
    ) if chunks else "No relevant document chunks found."

    web_search_query: str | None = None
    web_search_results: str | None = None
    web_search_used = False
    web_search_decision = "not requested"
    web_search_route: str | None = None
    web_search_plan_dict: dict[str, Any] | None = None
    skip_prediction_dict: dict[str, Any] | None = None
    targeted_skip_dict: dict[str, Any] | None = None

    def _provenance() -> dict:
        prov: dict[str, Any] = {
            "chunk_ids": chunk_ids,
            "chunks_preview": chunks_preview,
            "web_search_query": web_search_query,
            "web_search_results": web_search_results,
            "web_search_used": web_search_used,
            "web_search_decision": web_search_decision,
        }
        # Planner keys exist only in shadow/on modes so off stays byte-identical.
        if web_search_plan_dict is not None:
            prov["web_search_route"] = web_search_route
            prov["web_search_plan"] = web_search_plan_dict
        # Skip-gate prediction (Sprint 4 C) — present only in shadow/on modes.
        if skip_prediction_dict is not None:
            prov["skip_unanswerable"] = skip_prediction_dict
        # Targeted intensity prediction — present only when the resolved mode is
        # "targeted"; absent in Full/unset so off-mode provenance is identical.
        if targeted_skip_dict is not None:
            prov["targeted_skip"] = targeted_skip_dict
        return prov


    grounded_system_prompt = get_prompt("evidence.grounded.system", prompt_overrides)
    grounded_user_prompt = get_prompt("evidence.grounded.user", prompt_overrides)
    # Prompt variant selection happens at the call site (the prompt library
    # has no variant mechanism): planner "on" gets the web-as-primary-source
    # variant; off/shadow keep the legacy prompt byte-identical.
    hybrid_system_prompt = get_prompt(
        "evidence.hybrid.system.web_planner"
        if _web_evidence_planner_mode() == "on"
        else "evidence.hybrid.system",
        prompt_overrides,
    )
    hybrid_user_prompt = get_prompt("evidence.hybrid.user", prompt_overrides)

    vc_block = ""
    if aspect == "general_company" and vc_context:
        vc_str = vc_context if isinstance(vc_context, str) else " ".join(str(x) for x in vc_context)
        if vc_str.strip():
            vc_block = f"VC Investment Strategy (use when evaluating alignment):\n{vc_str.strip()}\n\n"

    async def _do_llm_call() -> tuple[str, dict]:
        nonlocal web_search_query, web_search_results, web_search_used, web_search_decision
        nonlocal web_search_route, web_search_plan_dict, skip_prediction_dict, targeted_skip_dict
        policy = get_current_pipeline_policy()

        async def _hybrid_answer(web_results_text: str) -> str:
            with use_phase_llm(policy.answering if policy else None):
                with use_stage_context("answering"):
                    llm = create_llm(temperature=0.2)
                    user_content = hybrid_user_prompt.format(
                        company_summary=company.get_company_summary(),
                        question=question,
                        chunks_text=chunks_text,
                        web_results=web_results_text,
                    )
                    response = await llm.ainvoke([
                        SystemMessage(content=hybrid_system_prompt),
                        HumanMessage(content=vc_block + user_content),
                    ])
            return _coerce_text(response.content) or "Unknown from provided documents."

        async def _run_planned_searches(
            active_plan: WebSearchPlan,
            grounded: str,
            search_reason: str,
            heavy_mode: str,
            planner_mode: str,
        ) -> str:
            """Execute the planner's queries (flag=on): sequential provider calls,
            one cap slot each, sharing a single WEB_SEARCH_TIMEOUT_SEC wall-clock
            budget so multi-query plans cannot blow the LLM_ANSWER_TIMEOUT_SEC
            envelope around the whole answer."""
            nonlocal web_search_query, web_search_results, web_search_used, web_search_decision
            executed: list[str] = []
            accepted_blocks: list[str] = []
            accepted = 0
            accepted_providers: list[str] = []
            cache_hits = 0
            cap_reached = False
            deadline = time.monotonic() + WEB_SEARCH_TIMEOUT_SEC
            total = len(active_plan.queries)
            for position, spec in enumerate(active_plan.queries, start=1):
                if web_search_state is not None:
                    async with web_search_state["lock"]:
                        if web_search_state["count"][0] >= web_search_state["max"]:
                            cap_reached = True
                            break
                        web_search_state["count"][0] += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                strategy = _resolve_web_search_provider_name()
                primary_provider = "serper" if strategy == "hybrid" else None
                cache_info: dict[str, Any] = {}
                try:
                    search_args = (
                        spec.query,
                        list(spec.domain_filter) if spec.domain_filter is not None else None,
                        search_reason,
                        heavy_mode,
                        active_plan.route.value,
                        planner_mode,
                        spec.purpose,
                        cache_info,
                    )
                    search_call = (
                        asyncio.to_thread(
                            _run_web_search,
                            *search_args,
                            provider_override=primary_provider,
                        )
                        if primary_provider
                        else asyncio.to_thread(_run_web_search, *search_args)
                    )
                    raw = await asyncio.wait_for(
                        search_call,
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    break
                # W13: a cache hit consumed no provider call — refund the cap
                # slot acquired above so cached answers don't shrink the budget.
                if cache_info.get("hit"):
                    cache_hits += 1
                    if web_search_state is not None:
                        async with web_search_state["lock"]:
                            web_search_state["count"][0] -= 1
                executed.append(spec.query)
                useful, _reason = evaluate_web_result_relevance(
                    policy=active_plan.relevance_policy,
                    route=active_plan.route,
                    question=question,
                    company_name=company.name,
                    web_results=raw,
                    industry_hint=company.industry,
                )
                accepted_provider = primary_provider or strategy or "configured"

                # Hybrid route: Serper is always attempted first. Perplexity is
                # called only when the same route-aware quality gate rejects the
                # Serper evidence (or Serper failed). The fallback consumes its
                # own provider-call slot and shares the same wall-clock budget.
                if not useful and strategy == "hybrid":
                    fallback_allowed = True
                    if web_search_state is not None:
                        async with web_search_state["lock"]:
                            if web_search_state["count"][0] >= web_search_state["max"]:
                                fallback_allowed = False
                            else:
                                web_search_state["count"][0] += 1
                    remaining = deadline - time.monotonic()
                    if fallback_allowed and remaining > 0:
                        fallback_cache_info: dict[str, Any] = {}
                        try:
                            fallback_raw = await asyncio.wait_for(
                                asyncio.to_thread(
                                    _run_web_search,
                                    spec.query,
                                    list(spec.domain_filter)
                                    if spec.domain_filter is not None
                                    else None,
                                    "quality fallback",
                                    heavy_mode,
                                    active_plan.route.value,
                                    planner_mode,
                                    spec.purpose,
                                    fallback_cache_info,
                                    provider_override="sonar",
                                    fallback_from="serper",
                                ),
                                timeout=remaining,
                            )
                        except asyncio.TimeoutError:
                            fallback_raw = "Web search failed: timeout"
                        if fallback_cache_info.get("hit"):
                            cache_hits += 1
                            if web_search_state is not None:
                                async with web_search_state["lock"]:
                                    web_search_state["count"][0] -= 1
                        fallback_useful, _fallback_reason = evaluate_web_result_relevance(
                            policy=active_plan.relevance_policy,
                            route=active_plan.route,
                            question=question,
                            company_name=company.name,
                            web_results=fallback_raw,
                            industry_hint=company.industry,
                        )
                        if fallback_useful:
                            raw = fallback_raw
                            useful = True
                            accepted_provider = "perplexity fallback"
                if useful:
                    accepted += 1
                    accepted_providers.append(accepted_provider)
                    accepted_blocks.append(
                        f'=== Web results {position}/{total} '
                        f'(provider: {accepted_provider}; purpose: {spec.purpose}; '
                        f'query: "{spec.query}") ===\n{raw}'
                    )

            web_search_query = " | ".join(executed) if executed else None
            gate = active_plan.relevance_policy.value
            # W13: empty suffix when the cache is off/missed keeps the legacy
            # decision strings byte-identical.
            cache_suffix = (
                f" ({cache_hits} cache hits, slots refunded)" if cache_hits else ""
            )
            if not executed:
                web_search_decision = (
                    "skipped: cap reached" if cap_reached else "skipped: no queries executed"
                )
                return grounded
            if not accepted:
                web_search_decision = (
                    f"skipped: {accepted}/{len(executed)} results passed {gate} gate{cache_suffix}"
                )
                return grounded

            combined = "\n\n".join(accepted_blocks)
            web_search_results = combined[:WEB_RESULTS_TRUNCATE]
            if len(combined) > WEB_RESULTS_TRUNCATE:
                web_search_results += "\n...[truncated]"
            web_search_used = True
            provider_suffix = (
                f" via {', '.join(dict.fromkeys(accepted_providers))}"
                if any(
                    provider in {"serper", "perplexity fallback"}
                    for provider in accepted_providers
                )
                else ""
            )
            web_search_decision = (
                f"used: {accepted}/{len(executed)} results passed {gate} gate"
                f"{provider_suffix}{cache_suffix}"
            )
            return await _hybrid_answer(web_search_results)

        async def _answer_from_shared_portfolio(
            active_plan: WebSearchPlan,
            grounded: str,
        ) -> str:
            """Reuse company-level evidence; spend reserve only on a material gap."""
            nonlocal web_search_query, web_search_results, web_search_used, web_search_decision
            assert web_search_state is not None
            bucket_key = evidence_bucket_key(active_plan.route, aspect)
            async with web_search_state["portfolio_lock"]:
                candidates = list(
                    web_search_state["portfolio_evidence"].get(bucket_key, [])
                )

            accepted: list[dict[str, Any]] = []
            for evidence in candidates:
                useful, _reason = evaluate_web_result_relevance(
                    policy=active_plan.relevance_policy,
                    route=active_plan.route,
                    question=question,
                    company_name=company.name,
                    web_results=evidence["result"],
                    industry_hint=company.industry,
                )
                if useful:
                    accepted.append(evidence)

            reserve_used = False
            if not accepted and active_plan.queries:
                reserve_allowed = False
                async with web_search_state["portfolio_lock"]:
                    if (
                        web_search_state["portfolio_reserve_used"][0]
                        < web_search_state["portfolio_reserve_max"]
                    ):
                        web_search_state["portfolio_reserve_used"][0] += 1
                        reserve_allowed = True
                if reserve_allowed:
                    core_queries = web_search_state["portfolio_core_queries"]
                    reserve_query = next(
                        (
                            item
                            for item in active_plan.queries
                            if item.query not in core_queries
                        ),
                        active_plan.queries[0],
                    )
                    reserve_evidence = await _run_quality_routed_search(
                        query=reserve_query.query,
                        domain_filter=list(reserve_query.domain_filter)
                        if reserve_query.domain_filter is not None
                        else None,
                        purpose=reserve_query.purpose,
                        route=active_plan.route,
                        relevance_policy=active_plan.relevance_policy,
                        representative_question=question,
                        company=company,
                        trigger_reason="portfolio reserve",
                        web_search_state=web_search_state,
                    )
                    if reserve_evidence is not None:
                        reserve_used = True
                        accepted.append(reserve_evidence)
                        async with web_search_state["portfolio_lock"]:
                            web_search_state["portfolio_evidence"].setdefault(
                                bucket_key, []
                            ).append(reserve_evidence)

            if not accepted:
                web_search_decision = (
                    "skipped: shared portfolio has no evidence passing the question gate"
                )
                return grounded

            # Prefer a compact bundle: at most three distinct evidence objects,
            # ordered by the deterministic core plan (reserve comes last).
            unique: list[dict[str, Any]] = []
            seen_queries: set[str] = set()
            for evidence in accepted:
                if evidence["query"] in seen_queries:
                    continue
                seen_queries.add(evidence["query"])
                unique.append(evidence)
                if len(unique) >= 3:
                    break
            web_search_query = " | ".join(item["query"] for item in unique)
            blocks = [
                f'=== Shared web evidence {position}/{len(unique)} '
                f'(provider: {item["provider"]}; purpose: {item["purpose"]}; '
                f'query: "{item["query"]}") ===\n{item["result"]}'
                for position, item in enumerate(unique, start=1)
            ]
            combined = "\n\n".join(blocks)
            web_search_results = combined[:WEB_RESULTS_TRUNCATE]
            if len(combined) > WEB_RESULTS_TRUNCATE:
                web_search_results += "\n...[truncated]"
            web_search_used = True
            providers = ", ".join(
                dict.fromkeys(str(item["provider"]) for item in unique)
            )
            source = "reserve" if reserve_used else "core"
            web_search_decision = (
                f"used: shared portfolio {source} evidence via {providers}"
            )
            return await _hybrid_answer(web_search_results)

        # Step 1: Always run the grounded LLM call first (documents only)
        if not chunks:
            grounded_answer = "Unknown from provided documents."
        else:
            with use_phase_llm(policy.answering if policy else None):
                with use_stage_context("answering"):
                    llm = create_llm(temperature=0.2)
                    user_content = grounded_user_prompt.format(
                        company_summary=company.get_company_summary(),
                        question=question,
                        chunks_text=chunks_text,
                    )
                    response = await llm.ainvoke([
                        SystemMessage(content=grounded_system_prompt),
                        HumanMessage(content=vc_block + user_content),
                    ])
            grounded_answer = _coerce_text(response.content) or "Unknown from provided documents."

        # Step 2: Decide whether to run Perplexity based on the answer
        # "answer" trigger: search only when LLM says it lacks evidence
        # "no_chunks" trigger: search only when we had no chunks (legacy)
        # The web-heavy question override is additionally gated by
        # WEB_SEARCH_HEAVY_OVERRIDE (always | root_only | never).
        heavy_mode = _web_search_heavy_override()
        heavy_override_allowed = heavy_mode == "always" or (heavy_mode == "root_only" and is_root)
        search_reason: str | None = None
        if use_web_search:
            if WEB_SEARCH_TRIGGER == "answer" and _answer_indicates_no_evidence(grounded_answer):
                search_reason = "documents incomplete"
            elif heavy_override_allowed and _question_prefers_web_search(question):
                search_reason = "question benefits from external web context"
            elif WEB_SEARCH_TRIGGER != "answer" and not chunks:
                search_reason = "no chunks"
        needs_search = search_reason is not None
        web_search_decision = f"needed: {search_reason}" if search_reason else "not needed"

        # Per-run Targeted intensity (Off/Targeted/Full): gate ONLY the ungated
        # "documents incomplete" reason — the heavy-override (market/product win
        # band) stays ungated in every mode, which is how Targeted keeps searching
        # market/product while suppressing the wasteful documents-incomplete gaps.
        # Mode precedence: explicit arg > per-run run-context > RDI_WEB_SEARCH_MODE
        # > "full". Full/unset resolves to "full" and skips this branch entirely,
        # so the path stays byte-identical.
        web_mode = resolve_web_search_mode(
            web_search_mode if web_search_mode is not None else get_current_web_search_mode()
        )
        if web_mode == "targeted" and search_reason == "documents incomplete":
            predicted_skip, targeted_reason = predict_targeted_skip(
                question=question,
                route_tag=route,
                aspect=aspect,
                company_name=company.name,
                company_domain=company.domain,
                industry=company.industry,
                is_root=is_root,
            )
            targeted_skip_dict = {
                "mode": web_mode,
                "predicted_skip": predicted_skip,
                "reason": targeted_reason,
            }
            if predicted_skip:
                web_search_decision = f"skipped: targeted — {targeted_reason}"
                return (grounded_answer, _provenance())

        # Skip predictably-unanswerable searches (Sprint 4 C): gate ONLY the
        # ungated "documents incomplete" reason. Off-mode never consults the
        # predictor, so the search path stays byte-identical. The predictor
        # delegates to the planner router, so when the planner is also "on" it
        # would skip the same set — this gate just decides it first (no double
        # skip, no cap double-count). Suppressed under "targeted" (the broadened
        # predictor above already decided) so the two never double-skip.
        skip_mode = skip_unanswerable_mode()
        if web_mode != "targeted" and skip_mode != "off" and search_reason == "documents incomplete":
            predicted_skip, skip_reason_text = predict_search_unanswerable(
                question=question,
                route_tag=route,
                aspect=aspect,
                company_name=company.name,
                company_domain=company.domain,
                industry=company.industry,
                is_root=is_root,
            )
            skip_prediction_dict = {
                "mode": skip_mode,
                "predicted_skip": predicted_skip,
                "reason": skip_reason_text,
            }
            if skip_mode == "on" and predicted_skip:
                web_search_decision = f"skipped: predicted unanswerable — {skip_reason_text}"
                return (grounded_answer, _provenance())

        # WebEvidencePlanner (Sprint 2): build a route-aware plan in shadow/on
        # modes. Shadow only records the plan; legacy behavior runs unchanged.
        planner_mode = _web_evidence_planner_mode()
        plan: WebSearchPlan | None = None
        if needs_search and planner_mode in ("shadow", "on"):
            from datetime import datetime

            plan = build_web_search_plan(
                question=question,
                company_name=company.name,
                company_domain=company.domain,
                industry_hint=company.industry,
                geo_hint=getattr(company, "geo", None),
                current_year=datetime.now().year,
                route_tag=route,
                aspect=aspect,
                is_root=is_root,
            )
            web_search_route = plan.route.value
            web_search_plan_dict = {"mode": planner_mode, **plan.to_telemetry_dict()}
        planner_controls = planner_mode == "on" and plan is not None

        # Planner skip routes run zero searches and consume no cap slot.
        if needs_search and planner_controls and plan.is_skip:
            web_search_decision = f"skipped: planner {plan.route.value} — {plan.skip_reason}"
            return (grounded_answer, _provenance())

        if (
            needs_search
            and planner_controls
            and web_search_state is not None
            and "portfolio_plan" in web_search_state
        ):
            answer = await _answer_from_shared_portfolio(plan, grounded_answer)
            return (answer, _provenance())

        # Per-company cap: check and increment under lock
        do_search = False
        if needs_search and planner_controls:
            # Cap slots are acquired per provider call inside _run_planned_searches.
            do_search = True
        elif needs_search and web_search_state is not None:
            async with web_search_state["lock"]:
                if web_search_state["count"][0] < web_search_state["max"]:
                    web_search_state["count"][0] += 1
                    do_search = True
                else:
                    web_search_decision = "skipped: cap reached"
        elif needs_search:
            do_search = True

        if do_search and planner_controls:
            answer = await _run_planned_searches(
                plan, grounded_answer, search_reason or "", heavy_mode, planner_mode
            )
            return (answer, _provenance())

        if do_search:
            web_search_decision = "attempted"
            web_search_query = _build_web_search_query(company, question)
            domain_filter = _web_search_domain_filter(company, question)
            cache_info: dict[str, Any] = {}
            web_results = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_web_search,
                    web_search_query,
                    domain_filter,
                    search_reason,
                    heavy_mode,
                    cache_info=cache_info,
                ),
                timeout=WEB_SEARCH_TIMEOUT_SEC,
            )
            # W13: a cache hit consumed no provider call — refund the cap slot
            # acquired above so cached answers don't shrink the budget.
            legacy_cache_hit = bool(cache_info.get("hit"))
            if legacy_cache_hit and web_search_state is not None:
                async with web_search_state["lock"]:
                    web_search_state["count"][0] -= 1
            web_search_results = web_results[:WEB_RESULTS_TRUNCATE]
            if len(web_results) > WEB_RESULTS_TRUNCATE:
                web_search_results += "\n...[truncated]"

            useful, reason = _web_results_add_value(question, company.name, web_results)
            if not useful:
                web_search_decision = f"skipped: {reason}"
                return (grounded_answer, _provenance())
            web_search_used = True
            web_search_decision = (
                "used: cache hit (no cap slot consumed)"
                if legacy_cache_hit
                else f"used: {reason}"
            )

            answer = await _hybrid_answer(web_results)
            return (answer, _provenance())

        # Return grounded answer (no web search)
        return (grounded_answer, _provenance())

    try:
        if semaphore:
            async with semaphore:
                answer, provenance = await asyncio.wait_for(
                    _do_llm_call(),
                    timeout=LLM_ANSWER_TIMEOUT_SEC,
                )
        else:
            answer, provenance = await asyncio.wait_for(
                _do_llm_call(),
                timeout=LLM_ANSWER_TIMEOUT_SEC,
            )
        return (answer, provenance)
    except asyncio.TimeoutError:
        return ("Answer timed out (API slow or rate limited).", _provenance())


def _count_nodes(node: QuestionNode) -> int:
    """Count total nodes in a question tree (root + all descendants)."""
    return 1 + sum(_count_nodes(c) for c in node.sub_nodes)


def _count_nodes_in_trees(trees: Dict[str, QuestionTree]) -> int:
    """Count total nodes across all question trees."""
    return sum(_count_nodes(t.root_node) for t in trees.values())


async def _answer_node_from_evidence(
    node: QuestionNode,
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    use_web_search: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    on_cooperate: Callable[[], Awaitable[None]] | None = None,
    progress: List[int] | None = None,
    total: int = 0,
    semaphore: asyncio.Semaphore | None = None,
    web_search_state: dict | None = None,
    vc_context: str = "",
    aspect: str = "",
    prompt_overrides: dict[str, Any] | None = None,
    is_root: bool = False,
    web_search_mode: str | None = None,
) -> None:
    """Recursively answer a question node and its children from evidence.

    Leaf nodes retrieve chunks directly. Parent nodes also retrieve chunks
    (rather than synthesizing from children) to maximize grounding.

    is_root marks the tree root; children are always answered with
    is_root=False so WEB_SEARCH_HEAVY_OVERRIDE=root_only can gate the
    web-heavy question override to root questions only.
    """
    if on_cooperate:
        await on_cooperate()

    if node.sub_nodes:
        tasks = [
            _answer_node_from_evidence(
                child, company, store, k, use_web_search,
                on_progress=on_progress,
                on_cooperate=on_cooperate,
                progress=progress,
                total=total,
                semaphore=semaphore,
                web_search_state=web_search_state,
                vc_context=vc_context,
                aspect=aspect,
                prompt_overrides=prompt_overrides,
                is_root=False,
                web_search_mode=web_search_mode,
            )
            for child in node.sub_nodes
        ]
        await gather_with_concurrency(tasks, limit=MAX_CONCURRENT_LLM_CALLS)

    if on_cooperate:
        await on_cooperate()

    answer, provenance = await answer_question_from_evidence(
        node.question, company, store, k, use_web_search,
        semaphore=semaphore,
        web_search_state=web_search_state,
        vc_context=vc_context,
        aspect=aspect,
        prompt_overrides=prompt_overrides,
        is_root=is_root,
        # Decomposition-time route tag (PR4 adds QuestionNode.route; getattr
        # keeps this working against routeless trees and old cached payloads).
        route=getattr(node, "route", None),
        web_search_mode=web_search_mode,
    )
    node.answer = answer
    node.provenance = provenance

    if on_cooperate:
        await on_cooperate()

    if progress is not None and on_progress:
        progress[0] += 1
        on_progress(progress[0], total)


async def answer_all_trees_from_evidence(
    question_trees: Dict[str, QuestionTree],
    company: Company,
    store: EvidenceStore,
    k: int = 8,
    use_web_search: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    on_cooperate: Callable[[], Awaitable[None]] | None = None,
    vc_context: str = "",
    prompt_overrides: dict[str, Any] | None = None,
    web_search_mode: str | None = None,
) -> List[Dict[str, str]]:
    """Answer all question trees using document evidence and optional web search.

    Processes all trees in parallel, then extracts Q&A pairs in the same
    format the downstream pipeline expects.

    Args:
        question_trees: Dict mapping aspect -> QuestionTree.
        company: The Company being analyzed.
        store: EvidenceStore with ingested chunks.
        k: Chunks to retrieve per question.
        use_web_search: If True, supplement document evidence with web search.
        on_progress: Optional callback(current, total) called after each Q&A.

    Returns:
        List of Q&A pair dicts compatible with the argument generation stage.
    """
    total = _count_nodes_in_trees(question_trees)
    progress: List[int] = [0]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    web_search_state: dict | None = None
    if use_web_search:
        web_search_state = {
            "count": [0],
            "lock": asyncio.Lock(),
            "max": MAX_PPLX_CALLS_PER_COMPANY,
        }
        if _shared_web_evidence_enabled() and _web_evidence_planner_mode() == "on":
            web_search_state.update(
                await _prepare_shared_web_evidence(
                    question_trees,
                    company,
                    web_search_state,
                    web_search_mode,
                )
            )

    tasks = [
        _answer_node_from_evidence(
            tree.root_node, company, store, k, use_web_search,
            on_progress=on_progress,
            on_cooperate=on_cooperate,
            progress=progress,
            total=total,
            semaphore=semaphore,
            web_search_state=web_search_state,
            vc_context=vc_context,
            aspect=aspect,
            prompt_overrides=prompt_overrides,
            is_root=True,
            web_search_mode=web_search_mode,
        )
        for aspect, tree in question_trees.items()
    ]
    await gather_with_concurrency(tasks, limit=MAX_CONCURRENT_LLM_CALLS)

    from agent.common.utils import get_qa_pairs_from_question_tree

    all_qa_pairs: List[Dict[str, str]] = []
    for tree in question_trees.values():
        all_qa_pairs.extend(get_qa_pairs_from_question_tree(tree))

    return all_qa_pairs


# ---------------------------------------------------------------------------
# Incremental enrichment — used by re-evaluation to preserve prior answers
# and only rewrite the ones where newly-uploaded evidence is actually
# relevant. Keeps the non-re-eval (fresh analysis) path untouched.
# ---------------------------------------------------------------------------

# Minimum TF-IDF cosine similarity at which a new chunk is considered
# relevant to a question. Below this, the previous answer is preserved
# verbatim. 0.05 is low enough to catch topical overlaps (a single
# shared keyword often hits ~0.08-0.15 on sparse TF-IDF) while avoiding
# noise from stop-word-only matches.
ENRICH_MIN_SIMILARITY = float(os.getenv("REEVAL_ENRICH_MIN_SIMILARITY", "0.05"))
ENRICH_TOP_K = int(os.getenv("REEVAL_ENRICH_TOP_K", "4"))


ENRICH_SYSTEM_PROMPT = """\
You are updating an existing analyst answer with newly uploaded evidence.

Rules:
- Read the PREVIOUS ANSWER and the NEW EVIDENCE CHUNKS carefully.
- If the new evidence adds, corrects, or expands information that is \
genuinely relevant to the question, rewrite the answer so it incorporates \
the new information. Preserve the tone, structure, length, and level of \
detail of the previous answer.
- Cite any new chunks inline using the same [chunk_XX] format as before.
- If the new evidence is NOT actually relevant to this question after you \
read it, return the PREVIOUS ANSWER completely unchanged (verbatim).
- Do NOT invent facts. Do NOT include any JSON, reasoning blocks, \
metadata, or preamble — return ONLY the final answer prose.
"""

ENRICH_USER_PROMPT = """\
QUESTION:
{question}

PREVIOUS ANSWER:
{previous_answer}

NEW EVIDENCE CHUNKS:
{chunks_text}

Return the final answer prose only."""


def _is_question_relevant_to_new_chunks(
    question: str,
    new_store: EvidenceStore,
    k: int = ENRICH_TOP_K,
    min_score: float = ENRICH_MIN_SIMILARITY,
) -> tuple[bool, list[Chunk]]:
    """Return (True, top_chunks) if any new chunk is TF-IDF relevant to the question.

    Uses `retrieve_chunks_with_similarity` (already in `agent.retrieval`) to
    get both the retrieved chunks and the top similarity score in one pass.
    """
    if not new_store.chunks or not question or not question.strip():
        return (False, [])
    try:
        chunks, top_sim = retrieve_chunks_with_similarity(question, new_store, k=k)
    except Exception:
        return (False, [])
    if not chunks or top_sim < min_score:
        return (False, [])
    return (True, chunks)


async def enrich_existing_answer_with_new_chunks(
    question: str,
    previous_answer: str,
    new_chunks: list[Chunk],
    company: Company,
    vc_context: str = "",
) -> str:
    """Call the LLM to update `previous_answer` using the relevant new chunks.

    Fail-safe: returns `previous_answer` verbatim on any error or empty response.
    """
    if not new_chunks:
        return previous_answer

    chunks_text = "\n---\n".join(
        f"[{c.chunk_id}] (source: {c.source_file}, location: {c.page_or_slide}):\n{c.text}"
        for c in new_chunks
    )

    vc_block = ""
    if vc_context:
        vc_str = vc_context if isinstance(vc_context, str) else " ".join(str(x) for x in vc_context)
        if vc_str.strip():
            vc_block = f"VC Investment Strategy (context only):\n{vc_str.strip()}\n\n"

    user_content = ENRICH_USER_PROMPT.format(
        question=question,
        previous_answer=previous_answer or "(no previous answer)",
        chunks_text=chunks_text,
    )

    try:
        policy = get_current_pipeline_policy()
        with use_phase_llm(policy.answering if policy else None):
            with use_stage_context("answering"):
                llm = create_llm(temperature=0.2)
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        SystemMessage(content=ENRICH_SYSTEM_PROMPT),
                        HumanMessage(content=vc_block + user_content),
                    ]),
                    timeout=LLM_ANSWER_TIMEOUT_SEC,
                )
        enriched = _coerce_text(response.content).strip()
        if not enriched:
            return previous_answer
        return enriched
    except Exception:
        return previous_answer


def _collect_nodes_by_question(trees: Dict[str, QuestionTree]) -> Dict[str, QuestionNode]:
    """Flatten all QuestionTree nodes into a {question_text: node} map.

    Used to write enriched answers back into the tree so the state snapshot
    stays consistent with the returned `all_qa_pairs` list.
    """
    out: Dict[str, QuestionNode] = {}

    def walk(node: QuestionNode) -> None:
        if getattr(node, "question", None):
            out[node.question] = node
        for child in getattr(node, "sub_nodes", None) or []:
            walk(child)

    for tree in trees.values():
        walk(tree.root_node)
    return out


async def enrich_qa_pairs_with_new_evidence(
    previous_qa_pairs: List[Dict[str, Any]],
    question_trees: Dict[str, QuestionTree],
    new_chunks: List[Chunk],
    company: Company,
    vc_context: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> List[Dict[str, Any]]:
    """Incrementally update previous Q&A pairs using new evidence chunks.

    For each previous Q&A pair:
    - If no new chunk is TF-IDF relevant to the question, the pair is copied
      through verbatim (answer, citations, provenance all preserved).
    - If any new chunk is relevant, a targeted LLM call rewrites the answer
      while preserving the original shape; the new chunk ids are unioned into
      `chunk_ids` so the Analysis Evidence provenance view surfaces the new
      sources.

    The enriched answers are also written back into the matching nodes of
    `question_trees` so that `final_state["question_trees"]` in the analysis
    snapshot stays consistent with the returned pairs.
    """
    if not previous_qa_pairs:
        return []

    new_store = EvidenceStore(startup_slug=getattr(company, "name", "") or "")
    new_store.chunks = list(new_chunks)

    node_by_question = _collect_nodes_by_question(question_trees)

    total = len(previous_qa_pairs)

    async def _process_one(idx: int, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        question = qa_pair.get("question") or ""
        previous_answer = qa_pair.get("answer") or ""

        relevant, top_chunks = _is_question_relevant_to_new_chunks(question, new_store)
        if not relevant:
            # Preserve verbatim — do NOT touch answer, citations, provenance.
            updated = dict(qa_pair)
        else:
            enriched_answer = await enrich_existing_answer_with_new_chunks(
                question=question,
                previous_answer=previous_answer,
                new_chunks=top_chunks,
                company=company,
                vc_context=vc_context,
            )
            updated = dict(qa_pair)
            updated["answer"] = enriched_answer
            # Union new chunk ids into chunk_ids for provenance rendering.
            prev_ids = list(updated.get("chunk_ids") or [])
            new_ids = [c.chunk_id for c in top_chunks if c.chunk_id]
            seen: set[str] = set()
            merged: list[str] = []
            for cid in prev_ids + new_ids:
                if cid and cid not in seen:
                    seen.add(cid)
                    merged.append(cid)
            updated["chunk_ids"] = merged

            # Extend chunks_preview with the new chunks so the evidence panel
            # can render their text immediately without re-retrieval.
            existing_preview = updated.get("chunks_preview") or ""
            new_preview_parts = [
                f"[{c.chunk_id}]: {c.text[:CHUNK_PREVIEW_CHARS]}"
                f"{'...' if len(c.text) > CHUNK_PREVIEW_CHARS else ''}"
                for c in top_chunks
            ]
            if new_preview_parts:
                joined_new = "\n---\n".join(new_preview_parts)
                updated["chunks_preview"] = (
                    f"{existing_preview}\n---\n{joined_new}"
                    if existing_preview
                    else joined_new
                )

        # Write enriched answer back into the matching tree node.
        node = node_by_question.get(question)
        if node is not None:
            node.answer = updated.get("answer") or previous_answer
            # Keep node.provenance roughly aligned so the snapshot JSON is sane.
            prov = {
                "chunk_ids": updated.get("chunk_ids") or [],
                "chunks_preview": updated.get("chunks_preview") or "",
                "web_search_query": updated.get("web_search_query"),
                "web_search_results": updated.get("web_search_results"),
                "web_search_used": updated.get("web_search_used", False),
                "web_search_decision": updated.get("web_search_decision", "enrich: preserved"),
            }
            try:
                node.provenance = prov
            except Exception:
                pass

        if on_progress:
            try:
                on_progress(idx + 1, total)
            except Exception:
                pass
        return updated

    # Process all pairs concurrently up to MAX_CONCURRENT_LLM_CALLS.
    tasks = [_process_one(i, qa) for i, qa in enumerate(previous_qa_pairs)]
    enriched_pairs = await gather_with_concurrency(tasks, limit=MAX_CONCURRENT_LLM_CALLS)
    return list(enriched_pairs)


def write_qa_pairs_into_trees(
    question_trees: Dict[str, QuestionTree],
    qa_pairs: List[Dict[str, Any]],
) -> None:
    """Write answers from `qa_pairs` back into the matching nodes of `question_trees`.

    Used by the re-evaluation "no new chunks" branch where the previous
    answers are reused verbatim but the tree structure (decomposed separately)
    needs populated `node.answer` fields before downstream stages run.
    """
    if not question_trees or not qa_pairs:
        return
    node_by_question = _collect_nodes_by_question(question_trees)
    for qa in qa_pairs:
        question = qa.get("question") or ""
        node = node_by_question.get(question)
        if node is None:
            continue
        node.answer = qa.get("answer") or ""
        try:
            node.provenance = {
                "chunk_ids": qa.get("chunk_ids") or [],
                "chunks_preview": qa.get("chunks_preview") or "",
                "web_search_query": qa.get("web_search_query"),
                "web_search_results": qa.get("web_search_results"),
                "web_search_used": qa.get("web_search_used", False),
                "web_search_decision": qa.get("web_search_decision", "reused"),
            }
        except Exception:
            pass
