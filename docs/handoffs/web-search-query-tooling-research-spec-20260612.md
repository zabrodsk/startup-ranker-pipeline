# Research Spec: Web-Search Query/Enrichment Tooling for RDI Evidence Answering

**Purpose of this document:** hand to Claude (chat) to research existing open-source
tools/libraries/frameworks that could replace or upgrade our hand-rolled web-search
query construction — before we build anything ourselves.

---

## 1. The problem (measured, not hypothetical)

Rockaway Deal Intelligence (RDI) analyzes early-stage startups (often stealth,
2024–2026 founded, <10 employees, no funding announced). The pipeline answers
~75 due-diligence questions per company from (a) the startup's own documents +
Specter data, and (b) a Perplexity Sonar web-search fallback when documents are
insufficient.

Measured on a 10-company batch (759 questions, June 2026):

- 62% of questions exhausted document evidence → web-search fallback fired (471×).
- **Only 7.4% of those web-enriched answers became substantive.** 92.6% still ended
  "Insufficient information available."
- Rescue rate is uniformly poor across question types (market/sector questions
  8.5%, company-financial questions 5.6%).

Root cause — the current implementation anchors everything to the company, which
is self-defeating for stealth companies that have **no web footprint**:

1. **Query construction** (`_build_web_search_query`): always emits
   `"<CompanyName> <keywords>"` — company-first by design.
2. **Domain filter** (`_web_search_domain_filter`): fallback searches are
   restricted to `[company_domain, crunchbase.com, linkedin.com]`. Broad web is
   allowed only for a narrow regex-matched "market/competitor" question class.
3. **Relevance gate** (`_web_results_add_value`): **rejects results that don't
   mention the company name** — so even a perfect sector-level answer (market
   size, regulatory context) is discarded as irrelevant.
4. **Hybrid answer prompt**: instructs the model that web results are "ONLY a
   fallback", further discounting whatever survives the gate.

Net effect: for questions like "Is the sector attractive given market size,
growth and regulation?", we search ~only the company's own site for a company
that has no site presence, reject anything that doesn't name the company, and
then tell the model to distrust it. The fix needs question-aware query
strategies (e.g., sector-anchored queries with the company name *stripped*),
not just prompt tweaks.

## 2. What we are looking for

An open-source tool, library, or framework (or a composable subset of one) that
does **question-aware web research for entity-sparse subjects**. Specifically,
some or all of:

- **Query reformulation / multi-query generation**: turn a due-diligence question
  about a company into 1–3 effective search queries, choosing per question
  whether to anchor on the company, the sector, the geography, or the competitor
  set (e.g., "wildfire detection sensors market size Europe 2026" instead of
  "Mantic market size").
- **Answerability / query routing**: classify whether a question is answerable
  from the public web at all (a stealth company's round size is not) so we can
  skip the search entirely.
- **Result synthesis with citations**: merge multi-query results into a cited
  text block we can inject into our existing answer prompt.
- (Nice-to-have) **Iterative research loops**: follow-up queries when first
  results are thin — bounded to ~2–3 calls.

## 3. Hard requirements

| Requirement | Detail |
|---|---|
| Form factor | Python library importable into an async worker (Python ≥3.11, uv-managed). A self-hostable lightweight service is acceptable; a SaaS-only API is not the goal (we already have Perplexity). |
| License | Permissive preferred (MIT/Apache-2.0/BSD). Flag anything GPL/AGPL. |
| Search backend | Must work with **Perplexity Sonar** (our default; returns LLM-synthesized cited answers, supports `search_domain_filter`) and/or **Brave Search API** (our fallback). Pluggable backends are a big plus. Tools hard-wired to Google scraping are out. |
| LLM calls | Fine to use an LLM for query generation/synthesis — we route through our own LangChain-based clients (OpenAI/Anthropic) with per-stage model policies. Tool must allow bring-your-own-LLM or expose clean seams. |
| Confidentiality | **Must not ship our document chunks or Specter data to third-party services.** Only the question text + public company facts (name, domain, industry) may leave. This is a hard constraint — deal flow is confidential. |
| Cost discipline | Bounded searches per question (≤3), respects our throttling (1 concurrent, ~3.5s interval), per-company cap (currently 100 searches). |
| Integration surface | Output must reduce to a string-with-citations to slot into our existing prompt placeholder, plus optionally a structured "queries tried" list for telemetry. Our provider interface is `search(query: str, *, domain_filter: list[str] | None) -> str`. |
| Maintenance health | Active repo (commits in last 6 months), >500 stars preferred, real issue triage. We don't want to adopt an abandoned grad-school project. |

## 4. Candidate categories + seeds to evaluate (non-exhaustive)

1. **Agentic research frameworks** — can we extract their query-planning module?
   - GPT-Researcher (assafelovic) — query planner + multi-search + synthesis
   - Stanford STORM — perspective-driven question→query expansion
   - LangChain `WebResearchRetriever` / LlamaIndex web tools
   - Open Deep Research re-implementations (HuggingFace smolagents, Firecrawl's, LangGraph examples)
2. **Search-API layers with research-grade answering** (would replace Sonar
   rather than the query builder — evaluate as alternative architecture):
   - Tavily (has OSS SDK; the API itself is SaaS — check data-retention terms vs confidentiality constraint)
   - Exa (neural/semantic search; SaaS — same caveat)
   - Perplexity Sonar deep-research tier (we already pay Perplexity — is a model-tier upgrade cheaper than tooling?)
3. **Query-expansion / reformulation libraries** (lighter weight):
   - HyDE-style hypothetical-document expansion implementations
   - LangChain MultiQueryRetriever pattern (we'd port the prompt, not the dep)
4. **Eval harnesses** for search/RAG quality (to measure rescue-rate offline):
   - RAGAS, TruLens, or simple custom replay — we already have a free benchmark:
     759 persisted Q/A rows with provenance from job `f20aa510`.

## 5. Evaluation criteria for the research

For each candidate, report:

1. What part of our gap it covers (query generation / answerability routing / synthesis / all).
2. License + maintenance health (stars, last release, bus factor).
3. Integration sketch: how it would sit behind our `search(query, domain_filter)`
   provider seam and our LLM clients; estimated integration effort (S/M/L).
4. Confidentiality posture: what data leaves our process, to whom.
5. Cost model: extra LLM calls per question, extra search calls per question.
6. Verdict: adopt / extract-pattern-only / reject, with one-line reason.

**Desired output:** a shortlist of 2–3 options with the above, plus a
recommendation whether to (a) adopt a tool, (b) port a pattern (e.g., copy the
query-planner prompt approach without the dependency), or (c) build the ~200-line
in-house version. Note: our current implementation is ~120 lines; a heavy
framework has to beat "port the pattern" on maintenance, not just capability.

## 6. Context for the researcher

- Code: `src/agent/evidence_answering.py` (`_build_web_search_query` L338,
  `_web_search_domain_filter` L259, `_web_results_add_value` L278);
  providers in `src/agent/web_search/providers.py`; hybrid prompt in
  `src/agent/prompt_library/defaults.py` (`EVIDENCE_HYBRID_SYSTEM_PROMPT`).
- Benchmark data: job `f20aa510` (10 companies, 759 QA rows with
  `web_search_query`, `web_search_results`, `web_search_decision`, answers) —
  lets us replay any new query strategy offline against real questions without
  re-running analyses.
- Related findings: `docs/handoffs/web-search-gating-ab-finding-20260612.md`.
