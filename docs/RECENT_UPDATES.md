# Recent Updates (Since Last Push)

*Last push: v0.0.6 — Company-centric history, stop finalization, Rockaway branding*

---

## Summary

This document summarizes the main improvements in **v0.0.6**. Key additions: **company-centric history** with `company_runs` table and grouped UI, **stop finalization** (partial results when stopping mid-run), improved **Specter detection** via header sniffing, and **Rockaway Deal Intelligence** branding.

The current branch also adds per-phase model routing creativity controls in New Analysis. Supported models show a `Creativity` slider, while models with fixed sampling keep the selector but hide the slider.

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

## Per-Phase Creativity Controls

The analysis setup screen now lets you tune sampling per stage.

- **Per-stage routing** — decomposition, answering, generation, evaluation, and ranking each keep their own model selection
- **Creativity slider** — shown only for models that support temperature-style control
- **Auto behavior** — slider defaults preserve the existing stage sampling until the user changes it
- **Unsupported models** — the slider stays hidden and the UI explains that the model uses fixed sampling for that stage

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

The clean `main` line now carries the new GPT-5.4 mini/nano pipeline defaults
for New Analysis without removing legacy label/cost compatibility for old saved
`gpt-5-mini` / `gpt-5-nano` runs.

- Replaced new selectable OpenAI mini/nano entries with:
  - `gpt-5.4-mini`
  - `gpt-5.4-nano`
- New default phase mapping:
  - decomposition -> `gpt-5.4-mini`
  - answering -> `gpt-5.4-nano`
  - generation -> `gpt-5.4-mini`
  - evaluation -> `gpt-5.4-mini`
  - ranking -> `gpt-5.4-mini`
- Added phase-aware OpenAI sampling:
  - base `gpt-5` keeps the existing temperature-mode behavior
  - `gpt-5.4-mini` / `gpt-5.4-nano` can send `reasoning.effort`, temperature, or both depending on the stage
  - non-`gpt-5` non-reasoning OpenAI models keep the stage-requested temperature unchanged
- Added a narrow fallback for `gpt-5.4-nano`: if OpenAI rejects the
  `reasoning` parameter for that model, the app retries once in
  temperature-only mode and records the fallback in telemetry

## Railway Environment Mapping And Live Verification (2026-03-18)

The current Railway naming is easy to misread, so the canonical environment
mapping is now:

- Staging:
  - project: `startup-ranker-mvp`
  - URL: `https://rockaway-deal-intelligence.up.railway.app`
- Production:
  - project: `rockaway-deal-intelligence-prod`
  - URL: `https://startup-ranker-web-production.up.railway.app`

Verified live `/api/config` on March 18, 2026:

- Staging is serving the GPT-5.4 mini/nano rollout:
  - available OpenAI models include `gpt-5.4-mini` and `gpt-5.4-nano`
  - phase defaults are:
    - decomposition -> `gpt-5.4-mini`
    - answering -> `gpt-5.4-nano`
    - generation -> `gpt-5.4-mini`
    - evaluation -> `gpt-5.4-mini`
    - ranking -> `gpt-5.4-mini`
- Production is still serving the prior defaults:
  - available OpenAI mini models are still `gpt-5-mini` and `gpt-5-nano`
  - phase defaults are:
    - decomposition -> `gemini-3.1-pro-preview`
    - answering -> `gemini-2.5-flash`
    - generation -> `gpt-5.2`
    - evaluation -> `o4-mini`
    - ranking -> `gpt-5.2`

---

## Files Changed (v0.0.6)

| File | Summary |
|------|---------|
| `web/db.py` | Company runs, company_key, list_company_histories, backfill logic |
| `web/app.py` | Stop finalization, job list merge, Specter sniffing |
| `web/static/index.html` | Company-centric history UI, grouped runs |
| `supabase/migrations/20260306010000_company_runs.sql` | **New** — company_runs table, company_key |
| `CHANGELOG.md` | v0.0.6 entry |
