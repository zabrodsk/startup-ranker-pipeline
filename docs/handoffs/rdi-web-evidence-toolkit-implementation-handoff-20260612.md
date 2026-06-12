# RDI Web Evidence Toolkit Implementation Handoff

Prepared: 2026-06-12  
Owner: Rockaway Deal Intelligence  
Purpose: Give the implementation agent clear instructions to upgrade RDI web-search fallback from company-first search to question-aware web evidence planning.

---

## 1. Executive decision

Do **not** adopt a full deep-research framework as the production runtime for this change.

Build a small in-house **RDI WebEvidencePlanner** that sits in front of the existing web-search provider interface:

```python
search(query: str, *, domain_filter: list[str] | None) -> str
```

Keep the existing Perplexity Sonar provider as default and Brave Search as fallback. Use external frameworks only as reference patterns or optional components after the baseline implementation works.

Recommended stack:

```text
Production path:
  Custom RDI WebEvidencePlanner
  + existing Perplexity Sonar provider
  + existing Brave provider fallback
  + route-specific relevance gates
  + simple cited evidence synthesis block

Optional later:
  Haystack QueryExpander or Haystack search components
  only if the custom baseline proves too limited

Pattern references only:
  GPT-Researcher
  Stanford STORM
  LangChain Local Deep Researcher
  Open Deep Research

Do not prioritize:
  Tavily or Exa as primary architecture
  because they are SaaS APIs and create extra data-retention review work
```

The main problem is not the search provider. The measured failure is strategy and gating:

1. Query construction always anchors on company name.
2. Domain filtering restricts most searches to company domain, Crunchbase, and LinkedIn.
3. Relevance gating rejects useful sector evidence when the startup name is absent.
4. The hybrid prompt further discounts web evidence.

The fix is question-aware routing, sector-aware query generation, and route-aware relevance gates.

---

## 2. Current measured problem

RDI analyzes early-stage startups, often stealth or 2024 to 2026 founded, with very little public footprint.

Measured on job `f20aa510`:

```text
Companies: 10
Questions: 759
Web fallback fired: 471 times, 62 percent of questions
Substantive web-enriched answers: 7.4 percent
Market and sector rescue rate: 8.5 percent
Company financial rescue rate: 5.6 percent
```

Known code locations:

```text
src/agent/evidence_answering.py
  _build_web_search_query           around L338
  _web_search_domain_filter         around L259
  _web_results_add_value            around L278

src/agent/web_search/providers.py
  Existing provider seam

src/agent/prompt_library/defaults.py
  EVIDENCE_HYBRID_SYSTEM_PROMPT

Benchmark data:
  job f20aa510

Related finding:
  docs/handoffs/web-search-gating-ab-finding-20260612.md
```

---

## 3. Non-negotiable constraints

### Confidentiality

Do **not** send document chunks, Specter data, private memo text, founder-uploaded content, CRM notes, or internal RDI evidence snippets to any third-party search or LLM service.

Only these fields may leave the process for query planning or search:

```text
company_name
company_domain
public_description
industry_hint
geo_hint
question_text
known_public_competitor_names, if already public
```

If any of these fields are derived from private source material, sanitize them into generic public labels before sending them outside the process.

Examples:

```text
Allowed:
  "wildfire detection sensors Europe market size 2026"

Not allowed:
  "Company X claims in its confidential deck that utilities in Bavaria have a budget gap for AI wildfire pilots"
```

### Cost discipline

Keep searches bounded:

```text
Max searches per question: 3
Default searches per question: 0 to 2
Existing throttle: 1 concurrent, around 3.5s interval
Per-company search cap: preserve existing cap, currently 100 searches
```

### Integration surface

The output must reduce to:

```python
@dataclass
class WebEvidenceBlock:
    status: Literal["skipped", "no_useful_results", "useful"]
    strategy: str
    queries_tried: list[str]
    evidence_text_with_citations: str
    confidence: Literal["low", "medium", "high"]
```

This block must slot into the existing answer prompt placeholder without requiring a full rewrite of the evidence-answering worker.

---

## 4. Required architecture

Create a small planner module. Suggested file:

```text
src/agent/web_search/planner.py
```

Suggested types:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class QuestionRoute(str, Enum):
    SKIP_PUBLIC_WEB = "skip_public_web"
    COMPANY_SPECIFIC = "company_specific"
    SECTOR_MARKET = "sector_market"
    REGULATION = "regulation"
    COMPETITORS = "competitors"
    GEOGRAPHY = "geography"
    CUSTOMER_NEED = "customer_need"
    TECHNOLOGY_VALIDATION = "technology_validation"


class RelevancePolicy(str, Enum):
    COMPANY_NAME_REQUIRED = "company_name_required"
    COMPANY_DOMAIN_OR_NAME_REQUIRED = "company_domain_or_name_required"
    SECTOR_GEO_EVIDENCE_ALLOWED = "sector_geo_evidence_allowed"
    REGULATORY_EVIDENCE_ALLOWED = "regulatory_evidence_allowed"
    COMPETITOR_EVIDENCE_ALLOWED = "competitor_evidence_allowed"
    TECH_EVIDENCE_ALLOWED = "tech_evidence_allowed"


@dataclass(frozen=True)
class SearchQuerySpec:
    query: str
    domain_filter: list[str] | None
    purpose: str


@dataclass(frozen=True)
class WebSearchPlan:
    route: QuestionRoute
    queries: list[SearchQuerySpec]
    relevance_policy: RelevancePolicy
    skip_reason: str | None
    rationale: str
```

Suggested public function:

```python
def build_web_search_plan(
    *,
    question: str,
    company_name: str,
    company_domain: str | None,
    industry_hint: str | None,
    geo_hint: str | None,
    public_description: str | None,
    current_year: int = 2026,
) -> WebSearchPlan:
    ...
```

Implementation should start deterministic and rule-based. Add LLM-based expansion later only if the rule-based version is insufficient.

---

## 5. Routing policy

### Route: `skip_public_web`

Use when the question is unlikely to be answerable from public web for an early-stage or stealth company.

Typical triggers:

```text
exact ARR
exact revenue
burn rate
cash runway
cap table
round size if unannounced
valuation
customer pipeline details
founder references from private docs
internal KPIs
unit economics
private contract details
```

Behavior:

```text
Run 0 searches.
Return skipped with clear skip_reason.
Do not waste search budget.
```

Example:

```text
Question:
  "What is the company's current ARR?"

Plan:
  route = skip_public_web
  queries = []
  skip_reason = "Exact private financial metrics are unlikely to be publicly available."
```

### Route: `company_specific`

Use when public web could plausibly contain company-level facts.

Typical triggers:

```text
founders
public funding announcements
website claims
pricing page
product page
job listings
LinkedIn footprint
Crunchbase profile
press mentions
public partnerships
```

Behavior:

```text
Use company name and/or domain.
Use restrictive domain filters.
Require company name, company domain, founder name, or exact product match in relevance gate.
```

Example queries:

```text
"<CompanyName>" funding founders
"<CompanyName>" product pricing
site:<company_domain> product customers pricing
```

Domain filters:

```python
[company_domain, "crunchbase.com", "linkedin.com"]
```

### Route: `sector_market`

Use for market size, growth, adoption, tailwinds, sector attractiveness, TAM, and macro demand.

Behavior:

```text
Strip company name by default.
Search sector + geography + market terms.
Allow broad web.
Do not require company name in relevance gate.
```

Example queries:

```text
"wildfire detection sensors market size Europe 2026"
"wildfire detection IoT adoption utilities Europe growth regulation"
"AI wildfire detection startups Europe market demand"
```

Domain filter:

```python
None
```

### Route: `regulation`

Use for regulation, compliance, procurement rules, public-sector constraints, law, standards, and policy risk.

Behavior:

```text
Strip company name by default.
Search sector + geography + regulation terms.
Allow broad web initially.
Optionally add official domains later, but do not overfit in v1.
```

Example queries:

```text
"wildfire detection EU regulation forest monitoring 2026"
"AI wildfire detection compliance Europe public sector procurement"
"remote sensing wildfire monitoring regulation Europe"
```

### Route: `competitors`

Use for competitive landscape, alternatives, category saturation, incumbents, and comparable startups.

Behavior:

```text
Strip company name by default.
Search category + geography + competitors/startups/alternatives.
Allow broad web.
Relevance gate should accept competitor/category evidence even if company name is absent.
```

Example queries:

```text
"wildfire detection AI startups Europe competitors"
"wildfire monitoring sensors companies alternatives"
"venture backed wildfire detection startups Europe"
```

### Route: `geography`

Use for country or regional attractiveness, local market readiness, procurement dynamics, customer base, and local regulatory environment.

Behavior:

```text
Search sector + geo + adoption or regulation terms.
Company name usually not needed.
```

Example queries:

```text
"wildfire detection market Italy Spain Greece public sector"
"forest fire prevention technology Europe adoption regional funding"
```

### Route: `customer_need`

Use for buyer pain, budget pressure, demand, procurement appetite, and customer willingness to pay.

Behavior:

```text
Search customer segment + problem + geography.
Company name usually not needed.
```

Example queries:

```text
"utilities wildfire detection sensors demand Europe"
"forest agencies wildfire monitoring technology procurement Europe"
```

### Route: `technology_validation`

Use for technical feasibility, accuracy, scientific validation, deployment proof, benchmarking, and adoption of the underlying technology.

Behavior:

```text
Search technology + use case + performance/adoption terms.
Company name usually not needed.
```

Example queries:

```text
"AI wildfire detection satellite sensor accuracy study"
"IoT wildfire detection sensors field deployment accuracy"
```

---

## 6. Query generation rules

### Global rules

1. Generate at most 3 queries.
2. Prefer 1 query for narrow company-specific questions.
3. Prefer 2 queries for sector, regulation, customer need, and technology validation.
4. Use 3 queries only for competitor or broader market landscape questions.
5. Do not include company name unless the route requires company-level evidence.
6. Add geography when `geo_hint` is available.
7. Add current year only when the question asks for current market, recent regulation, recent funding, or active trends.
8. Avoid sending private phrases from documents to search providers.

### Query normalization

Implement a helper that removes weak boilerplate from questions:

```text
"Does the company..."
"Is the company..."
"How attractive is..."
"Given the company..."
```

Then compose searches from:

```text
industry_hint
geo_hint
question intent keywords
public description category terms
```

### Example: bad versus good

Bad current query:

```text
Mantic market size growth regulation
```

Good sector query:

```text
wildfire detection sensors market size Europe 2026
```

Bad current relevance rule:

```text
Reject because result does not mention Mantic.
```

Good relevance rule:

```text
Accept if result discusses wildfire detection sensors, Europe, market size, growth, adoption, regulation, or buyer demand.
```

---

## 7. Domain-filter policy

Replace the current one-size-restrictive behavior with route-specific filters.

```python
def domain_filter_for_route(
    route: QuestionRoute,
    company_domain: str | None,
) -> list[str] | None:
    if route == QuestionRoute.COMPANY_SPECIFIC:
        domains = ["crunchbase.com", "linkedin.com"]
        if company_domain:
            domains.insert(0, company_domain)
        return domains

    if route == QuestionRoute.SKIP_PUBLIC_WEB:
        return []

    return None
```

Important behavior:

```text
None means broad web allowed.
[] should not be passed to provider because no search should run.
```

Do not domain-filter sector, regulation, competitor, customer-need, geography, or technology-validation searches in v1.

---

## 8. Relevance gate policy

Replace the current company-name gate with route-specific checks.

Current broken behavior:

```text
Useful sector evidence is rejected because it does not mention the startup.
```

Required behavior:

```python
def web_results_add_value(
    *,
    route: QuestionRoute,
    relevance_policy: RelevancePolicy,
    results_text: str,
    company_name: str,
    company_domain: str | None,
    industry_hint: str | None,
    geo_hint: str | None,
) -> bool:
    ...
```

Rules:

| Route | Accept evidence when |
|---|---|
| `company_specific` | Results mention company name, company domain, founder, exact product, or public profile |
| `sector_market` | Results mention sector/category plus market size, growth, adoption, TAM, demand, funding, or tailwinds |
| `regulation` | Results mention sector/category plus regulation, compliance, standards, law, policy, public procurement, or geography |
| `competitors` | Results mention category plus competitors, alternatives, incumbents, startups, market map, or comparable companies |
| `geography` | Results mention sector/category plus target geography, local adoption, local regulation, or buyer base |
| `customer_need` | Results mention customer segment plus pain, budget, demand, procurement, operational need, or willingness to pay |
| `technology_validation` | Results mention technology/use case plus performance, accuracy, deployment, validation, adoption, studies, or benchmarks |

For v1, implement this with keyword and phrase matching. Keep the logic easy to inspect.

Add telemetry for rejected results:

```text
route
query
relevance_policy
rejection_reason
company_name_present: bool
industry_hint_present: bool
geo_hint_present: bool
```

---

## 9. Prompt update

Update `EVIDENCE_HYBRID_SYSTEM_PROMPT` so the answering model does not automatically distrust web evidence.

Current issue:

```text
The prompt frames web results as "ONLY a fallback".
```

New instruction should be closer to:

```text
Use document evidence as primary evidence for company-specific internal facts.
Use web evidence as valid external evidence for market, sector, regulation, competitor, customer-demand, geography, and technology-validation questions.
Do not infer company-specific private facts from sector evidence.
When web evidence is sector-level, clearly label it as sector-level evidence.
When company-specific evidence is missing, say so directly.
```

Do not allow the model to convert sector evidence into fake company facts.

Good answer pattern:

```text
Company-specific evidence is limited. Sector-level evidence suggests the market is growing because [...]. This supports the attractiveness of the category, but it does not prove the company's traction.
```

Bad answer pattern:

```text
The company is growing quickly because the sector is growing quickly.
```

---

## 10. Candidate tooling decision log

### In-house RDI WebEvidencePlanner

Verdict: **build now**.

Why:

```text
Smallest blast radius.
Best confidentiality posture.
Fits current provider seam.
Directly fixes query planning and relevance gating.
Can be benchmarked against existing job f20aa510.
```

Integration effort: Small to medium.

### Haystack

Verdict: **optional later**.

Useful pieces:

```text
QueryExpander
Brave search component
Perplexity-style web search component if useful
Pipeline/router concepts
```

Do not adopt as the main architecture in phase 1. The RDI problem needs due-diligence-specific routing and gating, which Haystack will not provide out of the box.

Use only if:

```text
The custom query generator is too weak.
The team wants maintained components for multi-query generation.
The integration does not increase confidential data exposure.
```

### GPT-Researcher

Verdict: **extract pattern only**.

Useful ideas:

```text
Planner, multi-search, synthesis, citations, research report structure.
```

Do not use as runtime because:

```text
Too heavy for per-question fallback.
Optimized for full research reports.
Likely too costly and slow for hundreds of fallback calls.
Would require careful confidentiality review.
```

### Stanford STORM

Verdict: **extract pattern only**.

Useful ideas:

```text
Perspective-guided question expansion.
Category-level research before writing.
Useful mental model for sector and competitor questions.
```

Do not use as runtime because:

```text
Designed for long-form article generation.
Not shaped for a small async worker enrichment seam.
Too much complexity for this specific fix.
```

### Tavily and Exa

Verdict: **not primary**.

Why:

```text
They are SaaS APIs.
They may be useful search providers later.
They do not remove the need for RDI-specific routing and gating.
They create data-retention and confidentiality review work.
```

Allowed future usage:

```text
Only send sanitized public facts.
Never send private chunks or Specter data.
Obtain data-retention and contractual approval before production use.
Use behind the same provider interface if added.
```

### LangChain Local Deep Researcher and Open Deep Research

Verdict: **extract bounded-loop pattern only**.

Useful idea:

```text
Search, summarize, reflect on thin results, then issue one follow-up query.
```

RDI adaptation:

```text
Use 0 to 3 searches total.
Do not create open-ended agent loops.
Do not use broad research reports as fallback output.
```

### RAGAS and TruLens

Verdict: **evaluation only**.

Use later for:

```text
faithfulness scoring
citation quality
answer consistency
regression testing
```

Start with a custom replay harness first because job `f20aa510` already gives a real benchmark.

---

## 11. Implementation plan

### Phase 0: Add feature flag

Add a flag so the new planner can run in shadow mode first.

Suggested setting:

```text
RDI_WEB_EVIDENCE_PLANNER=off | shadow | on
```

Behavior:

```text
off:
  Current behavior only.

shadow:
  Current behavior remains production output.
  New planner runs and logs plan, queries, domain filters, and gate decisions.

on:
  New planner controls web search behavior.
```

### Phase 1: Add planner types and deterministic router

Create:

```text
src/agent/web_search/planner.py
tests/unit/test_web_evidence_planner.py
```

Unit tests must cover at least:

```text
market size question -> sector_market, no company name in query
regulation question -> regulation, broad web allowed
competitor question -> competitors, broad web allowed
ARR question -> skip_public_web
funding announcement question -> company_specific, restrictive domains
website/product question -> company_specific, restrictive domains
customer demand question -> customer_need, broad web allowed
technology feasibility question -> technology_validation, broad web allowed
```

### Phase 2: Integrate planner into evidence answering

Modify:

```text
src/agent/evidence_answering.py
```

Replace or wrap:

```text
_build_web_search_query
_web_search_domain_filter
_web_results_add_value
```

Target behavior:

```text
Old methods can remain as fallback for feature flag off.
New planner should be used for feature flag shadow/on.
```

### Phase 3: Add route-specific relevance gate

Add tests for evidence acceptance without company mention.

Required test case:

```text
Question:
  "Is the sector attractive given market size, growth and regulation?"

Company:
  "Mantic"

Industry hint:
  "wildfire detection sensors"

Search result:
  A cited paragraph about wildfire detection sensor market growth in Europe.
  It does not mention Mantic.

Expected:
  Accepted under sector_market.
```

Required negative case:

```text
Question:
  "Has Mantic announced funding?"

Search result:
  A generic paragraph about wildfire startups raising VC funding.
  It does not mention Mantic.

Expected:
  Rejected under company_specific.
```

### Phase 4: Update hybrid prompt

Modify:

```text
src/agent/prompt_library/defaults.py
```

Add instructions that distinguish:

```text
company-specific evidence
sector-level evidence
regulatory evidence
competitor evidence
technology evidence
```

The model must not infer private company traction from sector evidence.

### Phase 5: Add replay harness

Create a replay script:

```text
tools/replay_web_evidence_planner.py
```

Suggested CLI:

```bash
uv run python tools/replay_web_evidence_planner.py \
  --job-id f20aa510 \
  --mode shadow \
  --limit 759 \
  --out artifacts/web-evidence-planner-replay-f20aa510.json
```

The script should produce:

```text
route distribution
old query vs new query examples
searches skipped by route
searches per question
accepted vs rejected web results by route
substantive answer rate by route, if answer regeneration is enabled
sample successes
sample failures
```

### Phase 6: Canary rollout

Suggested rollout:

```text
1. Run unit tests.
2. Run replay on job f20aa510 with no external search calls if existing web results are enough for gate testing.
3. Run shadow mode on one new 10-company batch.
4. Inspect logs manually.
5. Enable planner for one batch only.
6. Compare rescue rate and false positives.
```

Do not enable globally until the replay report is reviewed.

---

## 12. Acceptance criteria

The implementation is acceptable only if all conditions below are met.

### Functional criteria

```text
Sector, market, regulation, competitor, customer-need, geography, and technology-validation questions can generate company-name-free queries.
Company-specific questions still use company-first queries and restrictive domains.
Private financial or internal metric questions can skip web search entirely.
The relevance gate accepts sector evidence without company name when route allows it.
The relevance gate still rejects generic sector evidence for company-specific facts.
The hybrid prompt treats web evidence as valid for external market questions.
```

### Safety criteria

```text
No document chunks are sent to search providers.
No Specter data is sent to external providers unless already public and sanitized.
No private memo text is used in generated search queries.
All outbound query strings are logged for audit.
Feature flag can disable the planner immediately.
```

### Cost criteria

```text
No more than 3 searches per question.
Average searches per fallback question should not materially exceed current behavior without a documented reason.
Skip route should reduce wasted searches on private-company facts.
Existing throttling and per-company caps remain active.
```

### Evaluation criteria

Replay report must include:

```text
baseline web rescue rate
new web rescue rate, if answer regeneration is run
route-level rescue rate
false-positive review samples
company-name-free query examples
company-name-free accepted sector evidence examples
skipped-search examples
cost impact
```

Target outcome:

```text
At least clear directional improvement in sector, market, regulation, competitor, and customer-need questions.
No increase in false positives for company-specific facts.
```

If improvement is weak, keep the feature flag off and preserve the replay report for follow-up tuning.

---

## 13. Telemetry requirements

Add structured telemetry per question:

```json
{
  "company_id": "...",
  "question_id": "...",
  "question_text": "...",
  "planner_enabled": true,
  "planner_mode": "shadow",
  "route": "sector_market",
  "relevance_policy": "sector_geo_evidence_allowed",
  "skip_reason": null,
  "queries_planned": [
    {
      "query": "wildfire detection sensors market size Europe 2026",
      "domain_filter": null,
      "purpose": "market size and growth"
    }
  ],
  "queries_executed": 1,
  "results_add_value": true,
  "rejection_reason": null,
  "company_name_present_in_results": false,
  "industry_hint_present_in_results": true,
  "geo_hint_present_in_results": true,
  "answer_became_substantive": true
}
```

Make sure telemetry is available for offline review.

---

## 14. Suggested deterministic classifier rules

Start with simple keyword rules. Keep them transparent.

```python
PRIVATE_FACT_TERMS = {
    "arr", "revenue", "burn", "runway", "cash", "valuation", "cap table",
    "ownership", "unit economics", "gross margin", "net revenue retention",
    "customer pipeline", "signed contracts", "mrr", "churn"
}

MARKET_TERMS = {
    "market size", "tam", "sam", "som", "growth", "sector attractive",
    "tailwinds", "market opportunity", "demand", "adoption", "market dynamics"
}

REGULATION_TERMS = {
    "regulation", "regulatory", "compliance", "law", "policy", "standards",
    "procurement", "licensing", "certification"
}

COMPETITOR_TERMS = {
    "competitor", "competitive", "alternatives", "incumbents", "landscape",
    "market map", "similar companies", "substitutes"
}

CUSTOMER_NEED_TERMS = {
    "customer need", "pain", "willingness to pay", "budget", "buyer",
    "procurement", "demand", "use case", "workflow"
}

TECH_TERMS = {
    "technology", "technical", "accuracy", "benchmark", "feasibility",
    "performance", "deployment", "validation", "study", "research"
}

COMPANY_PUBLIC_TERMS = {
    "founder", "team", "website", "pricing", "funding", "raised",
    "press", "partnership", "customers announced", "jobs", "linkedin",
    "crunchbase", "product"
}
```

Priority order:

```text
1. regulation
2. competitors
3. sector_market
4. customer_need
5. technology_validation
6. company_specific
7. skip_public_web
```

Exception:

```text
If a question asks for exact private numbers, route to skip_public_web unless the wording explicitly asks for public announcements.
```

---

## 15. Optional LLM query planner prompt for later

Do not start here unless deterministic rules are insufficient. If added, the LLM call must receive only sanitized public fields.

Prompt sketch:

```text
You are planning public web searches for early-stage startup due diligence.

You may use only these public fields:
- company_name
- company_domain
- public_description
- industry_hint
- geo_hint
- question

Do not infer or include private internal facts.
Choose whether the question needs company-specific evidence, sector evidence, regulatory evidence, competitor evidence, customer-demand evidence, technology evidence, geography evidence, or should skip public web.

Return JSON only:
{
  "route": "...",
  "queries": [
    {"query": "...", "domain_filter": null, "purpose": "..."}
  ],
  "relevance_policy": "...",
  "skip_reason": "...",
  "rationale": "..."
}

Rules:
- Max 3 queries.
- Strip company name for sector, regulation, competitor, customer-need, geography, and technology-validation queries.
- Use company name only for company-specific public facts.
- Use no queries when public web is unlikely to answer the question.
```

---

## 16. What not to do

Do not:

```text
Adopt GPT-Researcher as a runtime dependency for this seam.
Adopt STORM as a runtime dependency.
Switch primary search provider to Tavily or Exa in this task.
Send private evidence chunks to any query planner.
Run unbounded research loops.
Use broad web for exact private company metrics.
Require company name for sector-market evidence.
Rewrite the whole evidence-answering pipeline.
Remove existing provider abstraction.
Disable throttling or per-company caps.
```

---

## 17. Deliverables

Required PR deliverables:

```text
1. New planner module.
2. Feature flag: off, shadow, on.
3. Route-specific query generation.
4. Route-specific domain-filter logic.
5. Route-specific relevance gate.
6. Hybrid prompt update.
7. Unit tests for planner and relevance gate.
8. Replay script for job f20aa510.
9. Replay report artifact.
10. Short implementation note in docs/handoffs/.
```

Suggested docs file:

```text
docs/handoffs/web-evidence-planner-implementation-report-20260612.md
```

Implementation note should include:

```text
what changed
why it changed
before and after examples
replay metrics
known limitations
rollout recommendation
```

---

## 18. Minimal example cases for tests

### Case 1: sector market

Input:

```json
{
  "company_name": "Mantic",
  "company_domain": null,
  "industry_hint": "wildfire detection sensors",
  "geo_hint": "Europe",
  "question": "Is the sector attractive given market size, growth and regulation?"
}
```

Expected:

```text
route = sector_market
query does not include "Mantic"
domain_filter = None
relevance_policy = sector_geo_evidence_allowed
```

### Case 2: funding announcement

Input:

```json
{
  "company_name": "Mantic",
  "company_domain": "mantic.ai",
  "industry_hint": "wildfire detection sensors",
  "geo_hint": "Europe",
  "question": "Has the company announced funding?"
}
```

Expected:

```text
route = company_specific
query includes "Mantic"
domain_filter includes "mantic.ai", "crunchbase.com", "linkedin.com"
relevance_policy = company_domain_or_name_required
```

### Case 3: exact ARR

Input:

```json
{
  "company_name": "Mantic",
  "company_domain": "mantic.ai",
  "industry_hint": "wildfire detection sensors",
  "geo_hint": "Europe",
  "question": "What is the company's current ARR?"
}
```

Expected:

```text
route = skip_public_web
queries = []
skip_reason is not null
```

### Case 4: competitor landscape

Input:

```json
{
  "company_name": "Mantic",
  "company_domain": "mantic.ai",
  "industry_hint": "wildfire detection sensors",
  "geo_hint": "Europe",
  "question": "Who are the main competitors and alternatives?"
}
```

Expected:

```text
route = competitors
queries do not include "Mantic" by default
domain_filter = None
relevance_policy = competitor_evidence_allowed
```

---

## 19. Final instruction to implementation agent

Implement the smallest robust version that proves or disproves the thesis.

The thesis:

```text
RDI's web fallback performs poorly because it searches and gates as if every question requires company-specific public evidence. For stealth companies, many due-diligence questions need sector, regulation, customer, technology, or competitor evidence instead. A route-aware planner should materially improve useful web evidence without increasing confidentiality risk.
```

Optimize for:

```text
clear behavior
small code surface
strong logs
safe rollout
replayability
low dependency risk
```

Do not optimize for:

```text
agentic complexity
open-ended research
full market reports
new SaaS search providers
large framework adoption
```
