"""Route-aware web-search planning for evidence answering.

The planner replaces the company-anchored query/filter/gate behavior with
question-aware routing: company-specific questions keep the company-first
query and restrictive domain filter, while sector/regulation/competitor/
geography/customer-need/technology questions get company-name-free queries,
broad web access, and a relevance gate that does not require a company
mention. Private-metric and VC-internal-fit questions skip search entirely.

Pure stdlib module: takes scalars, never the Company object, and imports
nothing from the rest of the agent package so it can be unit-tested and
replayed offline in isolation.

Confidentiality (binding): only company name, normalized public domain,
Specter industry taxonomy, Specter HQ geo, and the question text may appear
in queries. ``public_description`` is accepted for forward-compatibility but
deliberately NOT used — ``company.about``/``tagline`` can be derived from
private documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Mirrors _WEB_SEARCH_QUERY_MAX_LEN in agent.evidence_answering.
WEB_SEARCH_QUERY_MAX_LEN = 180

# Appended (in code, not in the editable prompt catalog) to the decomposition
# system prompt so the decomposition LLM tags each node with a route at zero
# extra cost. Also reused by the replay harness to tag persisted benchmark
# questions with the exact same instruction.
ROUTE_TAGGING_INSTRUCTION = """\
Additionally, tag EVERY question node with a "route" field describing where its
answer evidence should come from. Choose exactly one of:
- "company_specific": company-level public facts (funding announcements, founders, \
product, pricing, partnerships, traction announcements, public profiles)
- "sector_market": market size, growth, adoption, demand, TAM/SAM/SOM, sector attractiveness
- "regulation": regulation, compliance, standards, policy, public procurement rules
- "competitors": competitive landscape, alternatives, incumbents, comparable startups
- "geography": regional market readiness, local adoption, local regulation, country attractiveness
- "customer_need": buyer pain, budgets, demand, procurement appetite, willingness to pay
- "technology_validation": technical feasibility, accuracy, benchmarks, scientific \
validation, deployment proof
- "internal_fit": fit with the investing VC's own thesis, stage focus, check size, \
or portfolio strategy (public web adds nothing)
- "skip_public_web": exact private internal metrics (ARR/MRR, burn rate, runway, \
cap table, unit economics) that are unlikely to be publicly available
If unsure, use "company_specific".
"""

MAX_QUERIES_PER_QUESTION = 3


class QuestionRoute(str, Enum):
    SKIP_PUBLIC_WEB = "skip_public_web"
    INTERNAL_FIT = "internal_fit"
    COMPANY_SPECIFIC = "company_specific"
    SECTOR_MARKET = "sector_market"
    REGULATION = "regulation"
    COMPETITORS = "competitors"
    GEOGRAPHY = "geography"
    CUSTOMER_NEED = "customer_need"
    TECHNOLOGY_VALIDATION = "technology_validation"


class RelevancePolicy(str, Enum):
    # Reproduces the legacy _web_results_add_value semantics (company mention
    # required) — used for company_specific.
    COMPANY_MENTION = "company_mention"
    # Route-topic keyword gate; a company mention is NOT required.
    TOPIC_KEYWORDS = "topic_keywords"
    # Skip routes: no search, nothing to gate.
    NONE = "none"


@dataclass(frozen=True)
class SearchQuerySpec:
    query: str
    domain_filter: Optional[tuple[str, ...]]
    purpose: str


@dataclass(frozen=True)
class WebSearchPlan:
    route: QuestionRoute
    queries: tuple[SearchQuerySpec, ...]
    relevance_policy: RelevancePolicy
    skip_reason: Optional[str]
    rationale: str

    @property
    def is_skip(self) -> bool:
        return not self.queries

    def to_telemetry_dict(self) -> dict:
        """JSON-safe representation for provenance/telemetry persistence."""
        return {
            "route": self.route.value,
            "relevance_policy": self.relevance_policy.value,
            "skip_reason": self.skip_reason,
            "rationale": self.rationale,
            "queries": [
                {
                    "query": q.query,
                    "domain_filter": list(q.domain_filter) if q.domain_filter is not None else None,
                    "purpose": q.purpose,
                }
                for q in self.queries
            ],
        }


# Only genuinely-unanswerable private metrics skip. internal_fit was removed as
# a skip route after the f20aa510 Gate-1 replay: those compound "does [company
# fact] fit the VC?" questions are rescued by web search (the company-fact half
# is public), so internal_fit now searches company-anchored instead of skipping.
_SKIP_ROUTE_REASONS = {
    QuestionRoute.SKIP_PUBLIC_WEB: (
        "Exact private company metrics are unlikely to be publicly available."
    ),
}

_TOPIC_ROUTES = {
    QuestionRoute.SECTOR_MARKET,
    QuestionRoute.REGULATION,
    QuestionRoute.COMPETITORS,
    QuestionRoute.GEOGRAPHY,
    QuestionRoute.CUSTOMER_NEED,
    QuestionRoute.TECHNOLOGY_VALIDATION,
}

# Templated root questions (pipeline/stages/constants.py) have known intent.
_STATIC_ROOT_ROUTES = {
    "general_company": QuestionRoute.INTERNAL_FIT,
    "market": QuestionRoute.SECTOR_MARKET,
    "product": QuestionRoute.COMPANY_SPECIFIC,
    "team": QuestionRoute.COMPANY_SPECIFIC,
}

# --- Fallback keyword router -------------------------------------------------
# Deliberately narrow: the dry-run on the f20aa510 benchmark showed broad
# keyword routing misroutes badly ("growth" sends stage questions to
# sector_market, "revenue" sends market questions to skip). The fallback only
# fires on unambiguous signals; everything else stays company_specific so the
# failure mode is the status quo, never a wrong skip.

_FALLBACK_INTERNAL_FIT_RE = re.compile(
    r"(\b(?:vc|investor|fund)'?s?\b[^.?]*\b(?:thesis|strategy|mandate|stage|check\s+size|portfolio)\b)"
    r"|(\binvestment\s+thesis\b)"
    r"|(\bfund'?s?\s+(?:focus|criteria)\b)",
    re.IGNORECASE,
)
# Unambiguous private-metric asks only. Bare "revenue"/"growth" are NOT
# triggers (benchmark collisions: market revenue-growth questions are
# web-answerable; "growth" appears in stage questions).
_FALLBACK_SKIP_RE = re.compile(
    r"\b(?:arr|mrr)\b"
    r"|\bburn\s+rate\b"
    r"|\b(?:cash\s+)?runway\b"
    r"|\bcap\s+table\b"
    r"|\bunit\s+economics\b"
    r"|\bnet\s+revenue\s+retention\b"
    r"|\bchurn\s+rate\b",
    re.IGNORECASE,
)
_FALLBACK_SECTOR_MARKET_RE = re.compile(
    r"\bmarket\s+siz(?:e|ing)\b"
    r"|\btam\b|\bsam\b|\bsom\b|\bcagr\b"
    r"|\btotal\s+addressable\s+market\b"
    r"|\bserviceable\s+(?:available|obtainable)\s+market\b",
    re.IGNORECASE,
)
_FALLBACK_COMPETITORS_RE = re.compile(
    r"\bcompetitors?\b|\bcompetitive\s+landscape\b|\balternatives\b|\bincumbents\b",
    re.IGNORECASE,
)
_FALLBACK_REGULATION_RE = re.compile(r"\bregulations?\b|\bregulatory\b", re.IGNORECASE)


def normalize_route_tag(tag: Optional[str]) -> Optional[QuestionRoute]:
    """Map a (possibly LLM-emitted) route tag string to a QuestionRoute, else None."""
    if not tag or not isinstance(tag, str):
        return None
    cleaned = tag.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return QuestionRoute(cleaned)
    except ValueError:
        return None


def _fallback_keyword_route(question: str) -> tuple[QuestionRoute, str]:
    """Conservative keyword routing; returns (route, rationale-suffix)."""
    text = question or ""
    if _FALLBACK_INTERNAL_FIT_RE.search(text):
        return QuestionRoute.INTERNAL_FIT, "keyword_fallback:internal_fit"
    if _FALLBACK_SKIP_RE.search(text):
        return QuestionRoute.SKIP_PUBLIC_WEB, "keyword_fallback:skip_public_web"
    # sector_market before regulation: "market size, growth and regulation"
    # questions are market questions (handoff §18 case 1).
    if _FALLBACK_SECTOR_MARKET_RE.search(text):
        return QuestionRoute.SECTOR_MARKET, "keyword_fallback:sector_market"
    if _FALLBACK_COMPETITORS_RE.search(text):
        return QuestionRoute.COMPETITORS, "keyword_fallback:competitors"
    if _FALLBACK_REGULATION_RE.search(text):
        return QuestionRoute.REGULATION, "keyword_fallback:regulation"
    return QuestionRoute.COMPANY_SPECIFIC, "keyword_fallback:default"


# --- Query construction -------------------------------------------------------

_QUERY_STOPWORDS = {
    "what", "which", "when", "where", "why", "how", "who", "are", "is", "does", "do",
    "the", "a", "an", "and", "or", "of", "for", "with", "to", "their", "them", "these",
    "those", "main", "direct", "specific", "given", "company", "companys", "its",
    "has", "have", "been", "this", "that", "from", "about",
}

_CURRENT_YEAR_HINT_RE = re.compile(
    r"\bcurrent\b|\brecent\b|\blatest\b|\btoday\b|\bnow\b|\btrends?\b|\bforecast\b",
    re.IGNORECASE,
)


def _tokens(text: Optional[str]) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _company_tokens(company_name: str) -> set[str]:
    return set(_tokens(company_name))


def _question_keywords(question: str, company_name: str) -> list[str]:
    """Order-preserving question keywords with company and boilerplate stripped."""
    company = _company_tokens(company_name)
    out: list[str] = []
    for token in _tokens(question):
        if token in company or token in _QUERY_STOPWORDS:
            continue
        if out and out[-1] == token:
            continue
        out.append(token)
    return out


def _compose_query(parts: list[str], company_name: str, *, strip_company: bool) -> str:
    """Join parts into a deduped, length-capped query string."""
    company = _company_tokens(company_name) if strip_company else set()
    seen: set[str] = set()
    terms: list[str] = []
    for part in parts:
        for token in _tokens(part):
            if token in company or token in seen:
                continue
            seen.add(token)
            terms.append(token)
    query = " ".join(terms)
    if len(query) > WEB_SEARCH_QUERY_MAX_LEN:
        query = query[:WEB_SEARCH_QUERY_MAX_LEN].rsplit(" ", 1)[0]
    return query


def _normalize_company_domain(company_domain: Optional[str]) -> Optional[str]:
    """Mirror the legacy _web_search_domain_filter domain normalization."""
    if not company_domain:
        return None
    raw = company_domain.strip().lower()
    raw = re.sub(r"^https?://(www\.)?", "", raw).split("/")[0]
    return raw or None


def _company_specific_domain_filter(company_domain: Optional[str]) -> tuple[str, ...]:
    domains = ["crunchbase.com", "linkedin.com"]
    normalized = _normalize_company_domain(company_domain)
    if normalized:
        domains.insert(0, normalized)
    return tuple(domains)


def _company_specific_queries(
    question: str, company_name: str, company_domain: Optional[str]
) -> tuple[SearchQuerySpec, ...]:
    keywords = _question_keywords(question, company_name) or ["overview"]
    query = _compose_query([company_name, *keywords], company_name, strip_company=False)
    return (
        SearchQuerySpec(
            query=query,
            domain_filter=_company_specific_domain_filter(company_domain),
            purpose="company-level public facts",
        ),
    )


# (route, geo-aware) -> list of (lead terms, trailing terms, purpose, wants_year_ok)
_TOPIC_QUERY_TEMPLATES: dict[QuestionRoute, list[tuple[str, str, bool]]] = {
    QuestionRoute.SECTOR_MARKET: [
        ("market size growth", "market size and growth", True),
        ("adoption demand market opportunity", "adoption and demand", False),
    ],
    QuestionRoute.REGULATION: [
        ("regulation compliance", "regulation and compliance", True),
        ("policy standards public procurement", "policy and procurement", False),
    ],
    QuestionRoute.COMPETITORS: [
        ("startups competitors", "competitor landscape", False),
        ("companies alternatives comparison", "alternatives and incumbents", False),
        ("venture backed startups funding", "venture-backed comparables", False),
    ],
    QuestionRoute.GEOGRAPHY: [
        ("market adoption", "regional market readiness", False),
        ("regulation public sector procurement", "regional regulation and procurement", False),
    ],
    QuestionRoute.CUSTOMER_NEED: [
        ("customer demand buyers", "buyer demand", False),
        ("procurement budget willingness to pay", "budget and procurement appetite", False),
    ],
    QuestionRoute.TECHNOLOGY_VALIDATION: [
        ("accuracy performance study", "technical validation studies", False),
        ("deployment validation benchmark", "deployment evidence", False),
    ],
}


def _topic_queries(
    route: QuestionRoute,
    question: str,
    company_name: str,
    industry_hint: Optional[str],
    geo_hint: Optional[str],
    current_year: int,
) -> tuple[SearchQuerySpec, ...]:
    """Company-name-free queries for sector-anchored routes; broad web (no filter)."""
    keywords = _question_keywords(question, company_name)
    base = industry_hint.strip() if industry_hint else " ".join(keywords[:6])
    wants_year = bool(_CURRENT_YEAR_HINT_RE.search(question or ""))
    specs: list[SearchQuerySpec] = []
    for route_terms, purpose, year_ok in _TOPIC_QUERY_TEMPLATES[route]:
        parts = [base, " ".join(keywords[:8]), route_terms]
        if geo_hint:
            parts.append(geo_hint)
        if wants_year and year_ok:
            parts.append(str(current_year))
        query = _compose_query(parts, company_name, strip_company=True)
        if not query:
            continue
        specs.append(SearchQuerySpec(query=query, domain_filter=None, purpose=purpose))
    return tuple(specs[:MAX_QUERIES_PER_QUESTION])


def build_web_search_plan(
    *,
    question: str,
    company_name: str,
    company_domain: Optional[str] = None,
    industry_hint: Optional[str] = None,
    geo_hint: Optional[str] = None,
    public_description: Optional[str] = None,
    current_year: int,
    route_tag: Optional[str] = None,
    aspect: Optional[str] = None,
    is_root: bool = False,
) -> WebSearchPlan:
    """Build a route-aware search plan for one question.

    Route priority: decomposition LLM tag > static templated-root map >
    conservative keyword fallback (biased to company_specific).

    ``public_description`` is intentionally unused: about/tagline fields may be
    derived from confidential documents and must never reach search providers.
    """
    del public_description  # confidentiality: never feeds queries

    route = normalize_route_tag(route_tag)
    if route is not None:
        rationale = f"tag:{route.value}"
    elif is_root and aspect in _STATIC_ROOT_ROUTES:
        route = _STATIC_ROOT_ROUTES[aspect]
        rationale = f"static_root:{aspect}"
    else:
        route, rationale = _fallback_keyword_route(question)

    # Gate-1 evidence (f20aa510): internal_fit questions are compound
    # "does [company fact] fit the VC?" and the web rescues the company-fact
    # half, so never skip them — search company-anchored instead. And only honor
    # skip_public_web for genuinely-private metrics (funding stage etc. are
    # public); a skip_public_web tag on anything else searches instead.
    if route == QuestionRoute.INTERNAL_FIT:
        return WebSearchPlan(
            route=route,
            queries=_company_specific_queries(question, company_name, company_domain),
            relevance_policy=RelevancePolicy.COMPANY_MENTION,
            skip_reason=None,
            rationale=rationale,
        )
    if route == QuestionRoute.SKIP_PUBLIC_WEB and not _FALLBACK_SKIP_RE.search(question):
        route = QuestionRoute.COMPANY_SPECIFIC
        rationale = f"{rationale}->company_specific:not_private_metric"

    if route in _SKIP_ROUTE_REASONS:
        return WebSearchPlan(
            route=route,
            queries=(),
            relevance_policy=RelevancePolicy.NONE,
            skip_reason=_SKIP_ROUTE_REASONS[route],
            rationale=rationale,
        )

    if route == QuestionRoute.COMPANY_SPECIFIC:
        return WebSearchPlan(
            route=route,
            queries=_company_specific_queries(question, company_name, company_domain),
            relevance_policy=RelevancePolicy.COMPANY_MENTION,
            skip_reason=None,
            rationale=rationale,
        )

    return WebSearchPlan(
        route=route,
        queries=_topic_queries(route, question, company_name, industry_hint, geo_hint, current_year),
        relevance_policy=RelevancePolicy.TOPIC_KEYWORDS,
        skip_reason=None,
        rationale=rationale,
    )


# --- Relevance gates -----------------------------------------------------------
# The COMPANY_MENTION branch reproduces agent.evidence_answering's
# _web_results_add_value semantics (thresholds and reason strings) so flag=on
# behavior for company_specific questions matches the legacy gate exactly.

_SEARCH_BAD_RESULT_PATTERNS = [
    r"no web search api key configured",
    r"web search failed",
    r"\b429\b",
    r"rate limit",
    r"timed out",
    r"timeout",
    r"no relevant (web )?results",
    r"no search results found",
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

_MIN_RESULT_LEN = 80

# Route-specific evidence terms (handoff §8 table): results must show the
# route's kind of evidence, not merely echo the question.
_ROUTE_EVIDENCE_TERMS: dict[QuestionRoute, set[str]] = {
    QuestionRoute.SECTOR_MARKET: {
        "market", "tam", "sam", "som", "growth", "adoption", "demand", "size",
        "billion", "million", "cagr", "forecast", "tailwinds", "opportunity", "funding",
    },
    QuestionRoute.REGULATION: {
        "regulation", "regulatory", "compliance", "law", "policy", "standards",
        "procurement", "directive", "certification", "licensing",
    },
    QuestionRoute.COMPETITORS: {
        "competitor", "competitors", "competitive", "alternatives", "incumbents",
        "startups", "companies", "landscape", "players", "rivals", "comparable",
    },
    QuestionRoute.GEOGRAPHY: {
        "market", "region", "regional", "country", "adoption", "regulation",
        "procurement", "local", "government",
    },
    QuestionRoute.CUSTOMER_NEED: {
        "customer", "customers", "buyer", "buyers", "demand", "pain", "budget",
        "procurement", "need", "needs", "willingness", "adoption",
    },
    QuestionRoute.TECHNOLOGY_VALIDATION: {
        "accuracy", "performance", "benchmark", "study", "studies", "deployment",
        "validation", "research", "feasibility", "pilot", "trial",
    },
}


def _gate_tokens(text: str) -> set[str]:
    tokens = set(_tokens(text))
    return {t for t in tokens if len(t) >= 3 and t not in _TOKEN_STOPWORDS}


def _company_mention_gate(question: str, company_name: str, web_results: str) -> tuple[bool, str]:
    """Exact reproduction of the legacy _web_results_add_value gate."""
    raw = (web_results or "").strip()
    if _SEARCH_BAD_RESULT_RE.search(raw):
        return (False, "web results indicate failure/noise")
    if len(raw) < _MIN_RESULT_LEN:
        return (False, "web results too short")

    result_tokens = _gate_tokens(raw)
    if not result_tokens:
        return (False, "web results have no usable tokens")

    company_tokens = _gate_tokens(company_name)
    if company_tokens and not (company_tokens & result_tokens):
        return (False, "web results do not mention the company")

    q_tokens = _gate_tokens(question)
    if not q_tokens:
        return (True, "question has no strong tokens; accepting web results")

    overlap = len(q_tokens & result_tokens)
    overlap_ratio = overlap / max(1, len(q_tokens))
    if overlap < 2 and overlap_ratio < 0.20:
        return (False, "web results weakly related to question")

    return (True, "web results relevant to company/question")


def _topic_keywords_gate(
    route: QuestionRoute,
    question: str,
    web_results: str,
    industry_hint: Optional[str],
) -> tuple[bool, str]:
    """Accept route-topic evidence; a company mention is NOT required."""
    raw = (web_results or "").strip()
    if _SEARCH_BAD_RESULT_RE.search(raw):
        return (False, "web results indicate failure/noise")
    if len(raw) < _MIN_RESULT_LEN:
        return (False, "web results too short")

    result_tokens = _gate_tokens(raw)
    if not result_tokens:
        return (False, "web results have no usable tokens")

    topic_tokens = _gate_tokens(question) | _gate_tokens(industry_hint or "")
    topic_overlap = len(topic_tokens & result_tokens)
    route_terms = _ROUTE_EVIDENCE_TERMS.get(route, set())
    route_overlap = len(route_terms & result_tokens)

    if topic_overlap >= 2 and route_overlap >= 1:
        return (True, f"web results provide {route.value} evidence (company mention not required)")
    if topic_overlap >= 4:
        return (True, f"web results strongly match the {route.value} question topic")
    return (False, f"web results do not match the {route.value} question topic")


def evaluate_web_result_relevance(
    *,
    policy: RelevancePolicy,
    route: QuestionRoute,
    question: str,
    company_name: str,
    web_results: str,
    industry_hint: Optional[str] = None,
) -> tuple[bool, str]:
    """Route-aware result gate. Returns (is_useful, reason)."""
    if policy == RelevancePolicy.NONE:
        return (False, f"no search planned for route {route.value}")
    if policy == RelevancePolicy.COMPANY_MENTION:
        return _company_mention_gate(question, company_name, web_results)
    return _topic_keywords_gate(route, question, web_results, industry_hint)
