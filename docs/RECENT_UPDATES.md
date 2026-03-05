# Recent Updates (Since Last Push)

*Last push: v0.0.4 — Person intelligence, deploy improvements*

---

## Summary

This document summarizes the main improvements and fixes since the last push to GitHub. **v0.0.4** adds **Person Intelligence** — on-demand team-member profile enrichment from LinkedIn and public web sources — plus deploy script improvements and pipeline schema updates.

---

## Person Intelligence (New)

On-demand team-member profile enrichment from LinkedIn and public web sources:

- **`PersonIntelService`** — Orchestrates evidence collection, fact extraction, deduplication, and markdown synthesis
- **Providers** — Apify MCP (LinkedIn), user-supplied inputs, web fallback for public profiles
- **Output** — Structured `PersonProfileOutput` with claims, evidence, synthesized sections, and markdown
- **Web API** — `/person-profile/job` and `/person-profile/job/{id}` for async profile jobs
- **Integration** — Team-member metadata enrichment in the web app; person dataclass aligned with Brightdata/LinkedIn schema

### Project layout

```
src/agent/person_intel/
├── service.py      # Orchestrator
├── extract.py      # Fact extraction and normalization
├── dedup.py        # Provenance indexing and deduplication
├── synthesize.py  # Section synthesis (LLM)
├── render_markdown.py
├── models.py       # PersonProfileJobRequest, EvidenceRecord, etc.
└── providers/      # Apify MCP, user inputs, web fallback
```

Tests: `tests/person_intel/` — full coverage for extract, dedup, synthesize, providers, service, and profile schema.

---

## Perplexity / Web Search Improvements

### 1. Web Search Quality Gate

A heuristic filter now decides whether Perplexity/Brave results are worth using before running the hybrid LLM call:

- **Bad-result detection** — Skips results matching patterns like `"no web search api key configured"`, `"web search failed"`, `429`, `rate limit`, `timed out`, `no relevant results`
- **Length check** — Rejects results shorter than 80 characters
- **Company relevance** — Ensures the company name appears in the results (token overlap)
- **Question relevance** — Uses token overlap between question and results; rejects if weakly related (e.g. &lt; 2 overlapping tokens and &lt; 20% overlap ratio)

If the gate fails, the pipeline **falls back to the grounded answer** (documents-only) instead of running the hybrid prompt, saving cost and avoiding low-quality web data.

### 2. Better Search Query Formatting

- Search query changed from `{company_name} {question}` to `"{company_name}" {question}` (quoted company name) for more precise Perplexity results
- Query is now built once and passed through consistently instead of reconstructing in `_run_web_search`

### 3. Higher Per-Company Search Cap

- Default `MAX_PPLX_CALLS_PER_COMPANY` raised from **2 to 100** to favor coverage over cost while development continues
- Can be overridden via `.env` for cost control in production

### 4. Provenance and Observability

- `web_search_used` and `web_search_decision` added to Q&A provenance
- Excel QA Provenance sheet now includes `web_search_used` and `web_search_decision` so you can see when search ran, when it was skipped (cap, bad results, etc.), and why

### 5. Robust Output Handling

- `_coerce_text()` added to normalize model/tool output to plain text (handles list/dict content blocks from some providers)
- Used for grounded and hybrid answers to avoid type errors on non-string responses

---

## Other Fixes and Improvements

### VC Investment Strategy

- `vc_investment_strategy` can arrive as a list (e.g. from form data); added `_ensure_str()` in batch and web app to join lists and avoid `'list' has no attribute 'strip'` errors
- Pydantic field validator in `AnalyzeRequest` to coerce `instructions` and `vc_investment_strategy` to strings

### README Overhaul

- Full rewrite with clearer structure, badges, and tables
- Documented **Web App** (previously undocumented): run, deploy, input modes
- Documented **Specter** CLI and CSV mode
- Added configuration table and project structure reflecting current layout

### Specter Ingest

- Minor refinements to company + people CSV parsing

### Dependencies

- `uv.lock` and `.env.example` updated for current environment

---

## Deploy Script

- `deploy.sh` — Clearer error messages for missing API keys, improved provider validation

---

## Files Changed (v0.0.4)

| File | Summary |
|------|---------|
| `src/agent/person_intel/` | **New** — Person intelligence pipeline (service, extract, dedup, synthesize, providers) |
| `tests/person_intel/` | **New** — Tests for person intel |
| `src/agent/dataclasses/person.py` | Person dataclass aligned with Brightdata/LinkedIn schema |
| `src/agent/dataclasses/company.py` | Company schema updates |
| `src/agent/pipeline/state/schemas.py` | PersonProfileOutput, PersonSubject, PersonClaim, etc. |
| `src/agent/pipeline/state/__init__.py` | State exports |
| `web/app.py` | Person profile job API, team-member enrichment |
| `web/static/index.html` | Person intel UI, cache key |
| `src/agent/batch.py` | Batch updates |
| `src/agent/evidence_answering.py` | Quality gate, provenance |
| `src/agent/ingest/specter_ingest.py` | Parsing refinements |
| `src/agent/web_search/providers.py` | Provider updates |
| `.env.example` | Config updates |
| `deploy.sh` | Deploy script improvements |
| `README.md` | Project structure update |
