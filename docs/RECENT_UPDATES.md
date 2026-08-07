# Recent Updates (Since Last Push)

*Last push: v0.0.6 — Company-centric history, stop finalization, Rockaway branding*

---

## Summary

This document summarizes recent work since v0.0.6. Two headline changes:

1. **Specter MCP integration** — two new intake paths (URL-list and
   pitch-deck auto-augmentation) that pull structured Specter intelligence
   (funding, team, growth signals, investor highlights) into the analysis
   pipeline without requiring CSV uploads.
2. **Production persistence fix** (2026-04-30) — production Supabase had a
   missing non-partial unique constraint on `companies.company_key` that
   silently broke every per-company persist call. Fixed via SQL editor +
   new migration + per-step observability so future silent failures land
   in the `analysis_errors` table.

Earlier v0.0.6 additions (company-centric history, stop finalization,
Specter header sniffing, Rockaway branding) are preserved below.

---

## GPT-5.6 Luna Pipeline Defaults (2026-08-03)

The New Analysis pipeline now defaults every user-facing phase to
`gpt-5.6-luna`. Phase-aware sampling keeps creative generation at temperature
`0.7`, while evidence and decision stages use explicit reasoning effort from
`low` through `high` and omit temperature.

## Flexible Question Allowance (2026-08-07)

Question decomposition receives its allowance before generation: up to 16
strategy-fit, 28 market, 24 product, and 20 team nodes, for a maximum of 88.
There is no minimum or required total. Luna builds a private candidate pool,
ranks candidates by decision value, removes overlap, and stops when additional
questions would not materially improve the assessment. The generated structure
contains only the question tree; coverage tags, priority, rationale, and
evidence-route metadata are no longer mixed into question selection. Evidence
routing happens separately in the downstream web planner. Validation checks the
allowance and essential tree structure but accepts any smaller high-quality
portfolio; generated trees are never truncated. Luna decomposition uses medium
reasoning with temperature omitted.

## Meta Muse Spark Contributor Staging Arm (2026-08-06)

Staging can now route every pipeline phase through
`muse-spark-1.2-contributor` using Meta's OpenAI-compatible Model API. The
phase policy combines temperature and reasoning effort: low reasoning for
decomposition and Q&A, minimal reasoning for creative generation, medium for
evaluation, and high for decision ranking. The model becomes the phase default
when `MODEL_API_KEY` is configured. Contributor data-use terms apply.

---

## Production Persistence Fix (2026-04-30)

A pre-existing production bug surfaced during the Specter MCP cutover and
got fixed in the same week.

### Symptom

Production had never persisted any `companies`, `pitch_decks`, `chunks`, or
per-company `analyses` rows across all time. Only the rollup `analyses` row
(with NULL FKs) ever landed — written by a separate `persist_analysis`
function. User-facing analyses completed and rendered correctly, but
downstream surfaces that traverse the FK graph (company-detail,
chat-with-company, re-evaluation) had nothing to show.

### Root cause

Migration `20260306010000_company_runs.sql` creates a *partial* unique
index `idx_companies_company_key ... WHERE company_key IS NOT NULL`. Most
PostgreSQL versions don't match partial indexes for `ON CONFLICT (column)`
inference unless the predicate is supplied as part of the conflict target.
The application uses the simple form via supabase-py:

```python
client.table("companies").upsert(payload, on_conflict="company_key").execute()
```

→ produces `ON CONFLICT (company_key)` with no predicate → fails on prod
with `42P10 "there is no unique or exclusion constraint matching the
ON CONFLICT specification"`.

The exception was swallowed by `_upsert_company`'s catch which logged to
Python only and returned `None`. With `company_id = None`, the entire
downstream chain (`pitch_decks`, `chunks`, per-company `analyses`,
`company_runs`) was skipped. Testing was unaffected because its table
received a non-partial unique constraint via some earlier setup path.

### Fix

- **SQL editor on prod**: `ALTER TABLE companies ADD CONSTRAINT
  companies_company_key_key UNIQUE (company_key);` — instant fix.
- **New migration** `supabase/migrations/20260430000000_companies_unique_company_key.sql`
  for fresh installs. Idempotent — checks for any existing non-partial
  unique constraint/index on `company_key` first, only adds if missing.
- **Per-step observability in `web/db.py`** — new helper
  `_record_persist_step_failure` writes to both Python logs and the
  `analysis_errors` table. `_upsert_company` and every step in
  `_persist_company_analysis_row` (companies / pitch_decks / chunks /
  analyses / company_runs) now capture exceptions individually with
  operation name + table + exception type, scoped to the offending job.
  Per-chunk failures only log the first occurrence to avoid flooding.
- **Verified end-to-end** on prod 2026-04-30 (job `fae2b416`, Zaitra
  deck): per-company `analyses` row with FKs populated, 1 company, 1
  pitch_deck, 30 chunks, zero `analysis_errors`.

### Critical files

- `supabase/migrations/20260430000000_companies_unique_company_key.sql`
  — the constraint migration
- `web/db.py:46-94` — new `_record_persist_step_failure` helper
- `web/db.py:411` — `_upsert_company` (now uses the new helper)
- `web/db.py:482-590` — `_persist_company_analysis_row` (per-step error
  capture)

---

## Specter MCP Integration (New)

Replaces / complements the CSV-only Specter intake with an OAuth-authenticated
MCP client. Three user-facing surfaces:

### URL-list intake (Specter mode)

Specter mode now accepts a list of URLs in addition to (or instead of) the
company/people CSVs. The worker pipeline calls Specter MCP per URL to produce
a `Company` + `EvidenceStore` mirroring the CSV-derived shape.

### Pitch-deck augmentation (Pitch-deck mode)

A toggle "Augment with Specter intelligence" (default ON, visible in
pitch-deck mode) auto-extracts the company URL from the deck text and merges
Specter MCP chunks into the deck-derived `EvidenceStore`. Extraction logic:

- **Three regex patterns**: `https?://...`, bare domains
  (`acme.com`, `app.acme.io`, `acme.co.uk`), labeled prefixes
  (`Website: foo.com` / `Visit foo.io`).
- **Email-domain detection**: extracts domain from any
  `founder@company.com`; emails get a 3× score multiplier (strong
  "this is OUR domain" signal — pitch decks almost always include
  the founder/contact email of the company being pitched).
- **Position weighting**: 2× for the first 3 chunks (cover and intro
  slides usually carry the URL); full deck scanned by default
  (`max_chunks=50`).
- **Blocklist**: linkedin/twitter/facebook/youtube, gmail/outlook/yahoo,
  vercel/netlify/herokuapp, slack/notion/zoom, crunchbase/wikipedia,
  file-extension TLDs (png/pdf/etc.), and money/metric notation
  (`$9.9M` → `9.9m`, `$2.2Bn` → `2.2bn`, `1.2k`, `10.5x`) — rejected
  because the first label is all-digits or the TLD isn't alphabetic.
- **Brand-stem fallback**: when `find_company(domain)` returns "No
  company found", retry with the brand stem (e.g. `adspawn.com` →
  `adspawn`), then cross-check the returned domain shares the same
  stem. Real-world case: AdSpawn deck references `adspawn.com` but
  Specter indexes the company under `adspawn.io`.

On any failure (no URL, MCP error, disambiguation rejection,
"no record"), the helper falls back to deck-only analysis — never raises.

### Deep-team profiles toggle (both modes)

A second toggle "Fetch deep team profiles (Specter)" (default OFF) flips
`fetch_full_team`:

- **OFF** — 4 MCP calls per company (`find_company`, `get_company_profile`,
  `get_company_intelligence`, and `get_company_funding_rounds`). Founders come
  from the `intelligence.founders` summary list. Recommended for pitch-deck
  mode (deck usually carries founder bios).
- **ON** — Same 4 calls + one `get_person_profile` per founder/key
  person. Adds ~60% more MCP calls but yields full LinkedIn-grade
  career history, education, and seniority per person. Recommended for
  URL-list mode where there is no deck context.

### OAuth + persistent refresh tokens

`scripts/specter_oauth_login.py` is a one-shot CLI that runs the
Authorization Code + PKCE flow, opening a browser, and prints the resulting
`SPECTER_MCP_CLIENT_ID` and `SPECTER_MCP_REFRESH_TOKEN` for the operator to
paste into env vars.

Specter rotates the refresh token on every refresh. To survive Railway
redeploys, the rotated token persists to a new RLS-locked
**`mcp_secrets`** Supabase table (single row keyed by
`secret_key='specter_mcp_refresh_token'`). Migration:
`supabase/migrations/20260428000000_mcp_secrets.sql`.

### Disambiguation safeguards

- **Domain root check** — `find_company('scribe.com')` returning
  "Shopscribe" is rejected as `SpecterDisambiguationError`.
- **Cross-company brand-stem check** — When the brand-stem fallback
  returns a company whose domain doesn't share the same stem, that's a
  cross-company collision (e.g. brand-stem search for `acme` returning
  domain `bigtech.com`); raised as `SpecterDisambiguationError`.
- **Fast-fail on "No company found"** — Specter explicitly saying "no
  match" raises `SpecterCompanyNotFoundError` (subclass of
  `SpecterMCPError`), which the retry loop in `_call_tool` does NOT
  retry. Saves ~6s of wasted backoff per definitively-missing company.
- The augmentation helper logs `Specter has no record of <url> —
  proceeding with deck only` on this error (informational, not an
  outage).

### New env vars

- `SPECTER_MCP_URL` — defaults to `https://mcp.tryspecter.com/mcp`
- `SPECTER_MCP_CLIENT_ID`
- `SPECTER_MCP_REFRESH_TOKEN`

See `.env.example` and the README's environment table.

### Critical files

- **NEW** `src/agent/ingest/specter_mcp_client.py` — OAuth token
  manager, MCP tool wrappers (`find_company`, `get_company_profile`,
  `get_company_intelligence`, `get_company_funding_rounds` (normalized through
  the compatibility `get_company_financials` method),
  `get_person_profile`), chunk builders, `fetch_specter_company()`
  with brand-stem fallback.
- **NEW** `src/agent/ingest/specter_augmentation.py` —
  `extract_company_url()` and `augment_with_specter()`. Self-contained
  helper; never raises.
- **NEW** `scripts/specter_oauth_login.py` — One-shot OAuth helper.
- **NEW** `supabase/migrations/20260428000000_mcp_secrets.sql` —
  RLS-locked secrets table.
- **NEW** `tests/test_specter_mcp_client.py`,
  `tests/test_specter_augmentation.py` — 49 unit tests; network-free.
- `web/app.py` — `AnalyzeRequest.use_specter_mcp` (default True),
  `AnalyzeRequest.fetch_full_team` (default False), threaded through
  `_run_analysis` → `_run_document_analysis`; augmentation runs in
  both single-file and multi-file branches of pitch-deck flow.
- `web/static/index.html` — Two new toggles (parent: "Augment with
  Specter intelligence"; child: "Fetch deep team profiles") visible in
  pitch-deck and URL-list modes.
- `src/agent/specter_company_worker.py` — Accepts new
  `--fetch-full-team` CLI flag for URL-mode tasks.
- `src/agent/specter_batch_worker.py` — Reads `fetch_full_team` from
  `run_config` and forwards it.
- `web/db.py` — `get_mcp_secret()` / `set_mcp_secret()` for the
  rotated-token persistence layer.

---

## Company-Centric History (New)

Per-company run history for grouped UI and history views across jobs.

- **`company_key`** — Normalized key on `companies` from domain or name for deduplication
- **`company_runs`** — New table storing per-company run records (job_id, decision, scores, result payload)
- **`list_company_histories()`** — API endpoint returning saved runs grouped by company; backfills from analyses when needed
- **Migration** — `supabase/migrations/20260306010000_company_runs.sql`

### Setup

Apply migrations in order; see [`supabase/README.md`](../supabase/README.md).

---

## Stop Finalization (New)

When a job is stopped mid-run, partial results are now finalized instead of discarded:

- Ranking and Excel export run for completed companies
- User sees "Partial results ready — N/M companies ranked"
- `_finalize_stopped_results()` builds and persists partial results; `allow_stopped` flag on progress helpers allows messages during finalization

---

## Specter Detection

Improved detection of Specter company + people CSV/Excel pairs:

- **Header sniffing** — Checks tabular headers for company markers (`company name`, `founders`, `industry`, `domain`) vs people markers
- Works when filenames alone do not indicate Specter format

---

## Rockaway Deal Intelligence Branding

- App title and branding updated from "Startup Ranker"
- FastAPI app title: "Rockaway Deal Intelligence"

---

## Other Changes

- **Job list** — `_list_jobs_for_ui()` merges in-memory jobs with Supabase saved jobs for unified history
- **Pause/resume** — Only transitions to `running` when status is actually `paused`
- **Sample data** — `deals/sample_startup/` replaced with `deals/sample_company/`
- **Railway** — `.railwayignore` added for deployment exclusions

---

## Worker-Backed Specter Hardening (Production)

Recent production work focused on making the dedicated Specter worker path
behave consistently across the worker, the saved-job overview, and the results
screen.

- **Newest queued jobs first** — The worker now prefers the newest queued
  Specter jobs when claiming work, avoiding starvation behind old rows.
- **Batch snapshots only** — Saved-run loading now treats only `analyses` rows
  with `company_id IS NULL` as terminal batch snapshots. Per-company rows no
  longer make a batch look complete early.
- **No early result opening** — Worker-backed runs return `409 Conflict` from
  `/api/analyses/<job_id>` while still active, instead of serving partial
  persisted company results.
- **Overview consistency** — The Analysis overview no longer promotes
  `has_results` to `DONE` on the client side for active worker-backed runs.
- **Stale worker detection** — Saved runs with stale worker heartbeats are
  marked interrupted rather than remaining indefinitely queued/running.
- **Idle memory reclaim** — Railway web production now uses
  `RESTART_ON_IDLE_AFTER_ANALYSIS=true` so the web process can recycle after
  terminal analyses and reclaim idle RSS.
- **Lower idle polling cost** — Railway worker production now uses
  `SPECTER_WORKER_POLL_SECONDS=10` to reduce idle polling overhead.
- **Analysis overview as source of truth** — New analyses now navigate
  straight to the Analysis overview. That page is the primary live monitoring
  surface for active and completed runs instead of the dedicated progress page.
- **Server-authoritative saved report opening** — Finished run cards now always
  offer **Open results**, and the browser asks the server directly when the
  user clicks. The client no longer decides report availability from stale
  local browser state.
- **Run naming and navigation polish** — Optional run names are supported for
  all analysis types, and the main header now highlights the active section
  consistently across New Analysis, Analysis, Companies, and results flows.
  The Analysis overview header also exposes the same manual **Refresh** action
  as the Companies page for an immediate server sync.

## Model Catalog

- Added OpenRouter model selection for `openrouter/hunter-alpha`
- OpenRouter now uses dedicated `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL`
  configuration instead of sharing the OpenAI key path

## Model Catalog Expansion (2026-03-15)

The production model catalog now exposes additional Gemini and OpenAI options
without changing the existing default routing behavior.

- Added Gemini model: `gemini-2.5-flash-lite`
- Added Gemini models: `gemini-2.5-flash`, `gemini-3.1-pro-preview`
- Added OpenAI models: `o4-mini`, `gpt-5.2`, `gpt-5.4`
- Preserved existing defaults for:
  - budget tier -> `gemini-3.1-flash-lite-preview`
  - balanced tier -> `claude-haiku-4-5-20251001`
  - premium tier -> `gpt-5`
- Tightened premium-family routing so "Claude" and "GPT-5" phase options still
  resolve to the intended provider families even after adding new balanced and
  premium entries
- Updated README model lists and catalog validation tests to match the new
  production options

## GPT-5.4 Mini/Nano Pipeline Defaults (2026-03-17)

The New Analysis pipeline defaults now prefer GPT-5.4 mini/nano models with
phase-aware OpenAI sampling.

- Added selectable OpenAI models: `gpt-5.4-mini`, `gpt-5.4-nano`
- Replaced selectable `gpt-5-mini` / `gpt-5-nano` entries for new runs while
  keeping legacy label/cost compatibility for historical rows
- New per-phase defaults:
  - decomposition -> `gpt-5.4-mini`
  - answering -> `gpt-5.4-nano`
  - generation -> `gpt-5.4-mini`
  - evaluation -> `gpt-5.4-mini`
  - ranking -> `gpt-5.4-mini`
- OpenAI requests now support mixed `temperature` + `reasoning.effort`
  depending on phase
- Added a narrow `gpt-5.4-nano` fallback that retries in temperature-only mode
  if the API rejects the reasoning parameter

---

## Files Changed (v0.0.6)

| File | Summary |
|------|---------|
| `web/db.py` | Company runs, company_key, list_company_histories, backfill logic |
| `web/app.py` | Stop finalization, job list merge, Specter sniffing |
| `web/static/index.html` | Company-centric history UI, grouped runs |
| `supabase/migrations/20260306010000_company_runs.sql` | **New** — company_runs table, company_key |
| `CHANGELOG.md` | v0.0.6 entry |
