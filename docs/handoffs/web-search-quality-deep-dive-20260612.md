# Deep Dive: Web-Search Quality Levers (Exploration, no implementation)

Follow-up to `web-search-gating-ab-finding-20260612.md`. Three threads from
Dusan's quality review of job `f20aa510`. Thread 1 (query tooling) has its own
research spec: `web-search-query-tooling-research-spec-20260612.md`. This doc
covers the root-cause analysis plus threads 2 and 3. **Exploration only — no
code changes.**

## Root cause of the 7.4% rescue rate (code-confirmed)

Four compounding company-anchors in `src/agent/evidence_answering.py`:

1. **Query is company-first by design** — `_build_web_search_query` (L338) emits
   `"<CompanyName> <keywords>"` for every search. For stealth companies, the
   anchor term itself guarantees empty/generic results.
2. **Fallback searches are domain-restricted to the company** —
   `_web_search_domain_filter` (L259) returns
   `[company_domain, crunchbase.com, linkedin.com]` for all NON-"heavy"
   questions. The 470 "documents incomplete" searches in arm A ran with this
   filter; only the 6 heavy (market/competitor regex) searches got broad web.
3. **Relevance gate requires a company mention** — `_web_results_add_value`
   (L293-295) rejects results that don't contain the company name. A perfect
   sector-level answer that doesn't name the startup is discarded.
4. **Hybrid prompt demotes web evidence** — `EVIDENCE_HYBRID_SYSTEM_PROMPT`
   says web results are "ONLY a fallback"; the model is licensed to answer
   "Insufficient information available" even with usable web context present.

Implication: prompt tweaks alone won't fix this. Anchors 1–3 prevent useful
context from ever reaching the prompt. Any fix must make query strategy,
domain filter, and the relevance gate **question-aware** (company-anchored vs
sector-anchored), which is exactly what the thread-1 tooling research targets.

## Thread 2 — Skip predictably-unanswerable searches

**Idea:** don't pay for a web search when no public source can answer the
question (e.g., round size / revenue / capital requirement of a `no_funding`
company founded 2025).

### Where it would hook

The search decision is a single choke point: `_answer_node_from_evidence`,
`do_search` block (evidence_answering.py L556-566). Two options:

- **(a) Search-time classifier** (recommended first step): before firing the
  search, classify `(question, company facts)` → `web_answerable: bool`.
  Company facts needed (funding status, founded year, team size) are already on
  the `Company` object from Specter. Start with a curated regex/keyword list +
  company-stage conditions; an LLM micro-classifier is possible but adds a call
  per question — likely unnecessary.
- **(b) Decomposition-time tagging**: have the decomposition LLM tag each
  generated sub-question `answer_source: docs|web|insider_only` (zero extra
  calls — same generation). The gate then reads the tag. More elegant,
  question-intent-aware, but touches the decomposition prompt + tree schema.
  Candidate second iteration.

### The free offline benchmark (key insight)

Job `f20aa510` persisted all 759 QA rows including question text, the search
query used, the raw web results, the decision, and the final answer. So any
classifier can be **dry-run offline with zero new analyses**:

- For each of the 471 searched questions, run the classifier.
- Cost saving = searches it would skip.
- Quality cost = how many of the **35 actually-rescued answers** it would have
  skipped (false positives). Target: 0 of 35.

This gives a measured accept/reject decision for the classifier before any
deployment. Same harness re-usable for thread-1 query-strategy candidates
(replay new queries for a sample, compare answer substance).

### Sizing (from arm A)

Explicit private-fact regex alone matches 36/471 searches (5.6% rescue) — small
but nearly risk-free savings. The bigger pool: many of the 364 "other" searches
are insider-only questions (stage expectations, traction details). Plausible
skip range with stage-aware rules: 30–50% of searches; must be validated against
the 35-rescued set on the benchmark first.

## Thread 3 — Surface evidence coverage alongside scores

**Constraint from Dusan (binding): the scoring logic is finely tuned — do NOT
change it. Exploration only; any implementation needs his explicit sign-off.**

### How scores flow today (traced)

1. `answering` produces tree answers (incl. thin "Insufficient…" ones).
2. `generation_pro`/`generation_contra` build arguments **from those answers** —
   thin evidence already weakens arguments implicitly here.
3. `evaluation` scores each argument on 14 criteria; critique/refinement loops.
4. `total_score = avg_pro - avg_contra` (batch.py:1097);
   `final_decision`; composite/percentile from the ranking stages
   (`compute_composite_rank`); rows assembled in `build_summary_rows`
   (batch.py:1077) → persisted → UI.

Nothing in this chain receives or displays an explicit evidence-coverage
number. A score built on 80% thin answers (orchestrai) and one built on 45%
thin (virtuelni-asistent) look equally confident in the report.

### Options, ordered by invasiveness

| Option | What changes | Scoring impact | Risk |
|---|---|---|---|
| **A. Display-only coverage metric** | Compute `evidence_coverage = 1 - thin_rate` per company from `qa_provenance_rows` at results-build time; add a field to summary rows + a UI column/badge ("Evidence coverage: 47%"). | **None** — pure metadata | Minimal; older runs simply lack the field |
| B. Exec-summary mention | Prompt-only change so the executive summary states coverage and names the unanswered areas | None on numbers | Low; prompt change reviewable in isolation |
| C. Confidence-weighted scores | Feed coverage into composite ranking as a multiplier/interval | **Changes the tuned scoring** | High — explicitly deferred per Dusan |

Recommendation: A (small, reversible, immediately useful for calibrating trust
per company), optionally B later. C is off the table until Dusan wants it.

The thin-answer detector for A already exists (`_answer_indicates_no_evidence`)
and is the same heuristic the pipeline itself uses to trigger searches — so the
displayed metric is consistent with the pipeline's own notion of "unanswered".

## Suggested sequencing

1. **Now (Dusan):** run the thread-1 tooling research with the spec doc.
2. **Cheap next (any session):** build the offline benchmark harness on
   `f20aa510` rows (thread 2a dry-run + thread-1 candidate replay). No staging
   runs needed.
3. **Then:** small PR for thread-3 option A (display-only), if approved.
4. **After research lands:** implement chosen query strategy + answerability
   gate behind env flags, validate on the benchmark, then a fresh 10-company
   staging batch to confirm rescue-rate improvement end-to-end.

## Addendum 2026-06-12 (later): Review of the external research handoff + benchmark dry-run

Dusan's Claude-chat research returned
`~/Downloads/rdi-web-evidence-toolkit-implementation-handoff-20260612.md`:
build an in-house route-aware **WebEvidencePlanner** (8 question routes,
route-specific queries/domain-filters/relevance-gates, off/shadow/on flag,
replay harness on f20aa510, no framework adoption). Architecture verdict:
**endorsed** — it independently matches our root-cause analysis.

**But the proposed Sec.14 deterministic keyword router was dry-run against all
759 real benchmark questions and is NOT production-ready:**

- Route distribution: 60% lands in company_specific (37% in the bare default
  bucket) — the catch-all does most of the work.
- skip_public_web catches only 37/471 searches (7.9%), with 1 false positive
  among the 35 rescued answers ("fundraising, user growth, revenue growth..."
  contains "revenue").
- Keyword collisions misroute badly: "Is the company pre-seed, seed, Series A,
  growth..." → sector_market (hits "growth"); "What has been the market's
  revenue growth..." → skip_public_web (hits "revenue" — kills a perfectly
  web-answerable market question); "security, compliance, governance
  capabilities" → regulation; product-moat questions → competitors.
- VC-fit questions (~105/759, phrasing like "the VC's thesis/stage/check size")
  have no route; 84 fall into the default bucket and get searched. These need
  an internal_fit/skip route — web adds nothing to thesis-match questions.

**Conclusion: keep the handoff's architecture (routes, gates, flag, shadow,
replay, acceptance criteria) but replace the v1 router.** Questions are
LLM-generated at decomposition (610 unique of 759), so subject-blind keyword
matching can't carry routing. Preferred: tag routes at decomposition time
(zero extra LLM calls — extend the decomposition output schema), static tags
for templated root questions, deterministic keyword rules only as fallback.
Confidentiality unchanged: tagging happens inside our LLM calls that already
see the questions.

**Integration gaps found vs the handoff spec:**
1. `Company` model has no geo field → `geo_hint` must be added from Specter HQ
   data at ingest, or omitted in v1. `industry_hint` = `company.industry`
   (Specter taxonomy, public — safe).
2. `company.about`/`tagline` can be document-derived → must NOT feed planner
   queries in v1 (confidentiality). Planner inputs: name, domain, industry only.
3. Shadow mode must be plan-only (log plan, zero extra searches) or it doubles
   Perplexity spend.
4. Multi-query (≤3/question) × 3.5s throttle adds wall-clock; cap accounting
   must count provider calls. Expected net Perplexity cost ≈ flat (skips offset
   multi-query); the win is rescue-rate, not direct search spend.
5. Replay harness is two-stage: stage 1 free (routing + gate decisions vs
   persisted results), stage 2 sampled live (~80-100 sector/regulation/
   competitor questions re-searched + re-answered) to measure rescue uplift.
6. Hybrid-prompt change rides the same flag (select prompt variant at call
   site), so off = byte-identical current behavior.
