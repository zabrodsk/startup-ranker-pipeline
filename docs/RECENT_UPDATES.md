# Recent Updates (Since Last Push)

*Last push: v0.0.3 — Executive summary with dimension summaries, Key Points, Red Flags*

---

## Summary

This document summarizes the main improvements and fixes since the last push to GitHub. **Work is ongoing**, with current focus on improving Perplexity web search quality and robustness.

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

## Current Work in Progress

- **Improving Perplexity searches** — tuning quality gates, query construction, and fallback behavior
- Further refinements to executive summaries and ranking dimensions

---

## Files Changed

| File | Summary |
|------|---------|
| `src/agent/evidence_answering.py` | Quality gate, coerce_text, provenance, query formatting |
| `src/agent/batch.py` | _ensure_str for VC context, QA provenance columns |
| `web/app.py` | VC strategy coercion, Pydantic validators |
| `web/static/index.html` | Minor UI tweaks |
| `src/agent/ingest/specter_ingest.py` | Parsing refinements |
| `src/agent/pipeline/stages/ranking.py` | Ranking adjustments |
| `README.md` | Major rewrite |
| `.env.example` | Config updates |
