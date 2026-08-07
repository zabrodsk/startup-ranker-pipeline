# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed — flexible investment-question allowance (2026-08-07)

- Replaced the exact 76-question requirement with an upfront maximum allowance
  of 88 nodes: up to 16 strategy-fit, 28 market, 24 product, and 20 team nodes.
  Luna may stop below any allowance when additional questions add little
  decision value; generated trees are never truncated after the fact.
- Simplified generation to high-quality, independently answerable,
  non-duplicative questions that collectively cover each category. Removed
  coverage tags, decision rationale, priority, and evidence-route metadata from
  decomposition. Evidence routing now remains a separate downstream concern.
- Deterministic validation now checks only the maximum allowance, root identity,
  uniqueness, connectivity, and parentage. Invalid structures regenerate in
  full; returning fewer than the allowance is valid.
- Raised GPT-5.6 Luna decomposition from low to medium reasoning while keeping
  temperature omitted; all other Luna phase defaults are unchanged.

### Added — Meta Muse Spark Contributor staging model (2026-08-06)

- Added `muse-spark-1.2-contributor` through Meta's OpenAI-compatible Model API,
  including structured-output selection, cost telemetry, phase-specific
  temperature/reasoning defaults, and the New Analysis model selector.
- The Meta model becomes the per-phase default only when `MODEL_API_KEY` is
  configured; existing Luna defaults remain the credential-aware fallback.
- The Contributor catalog entry and environment docs warn that its prompts and
  outputs may be used to improve future Meta models.

### Fixed — Specter MCP financial tool compatibility (2026-08-03)

- Specter removed the legacy `get_company_financials` MCP tool and replaced it
  with split finance tools. The client now calls `get_company_funding_rounds`
  and normalizes the response into the existing evidence schema, restoring URL
  quality preflight and worker enrichment without increasing per-company MCP
  call count.

### Fixed — production persistence (2026-04-30)

- **Missing non-partial UNIQUE constraint on `companies.company_key`** —
  Production Supabase had only the partial unique index from
  migration `20260306010000_company_runs.sql`
  (`WHERE company_key IS NOT NULL`). PostgreSQL's
  `ON CONFLICT (company_key)` inference doesn't match partial indexes
  without an explicit predicate, so every prod company upsert raised
  `42P10 "no unique or exclusion constraint matching"`, which
  `_upsert_company`'s catch silently swallowed. Net result on prod:
  zero `companies` / `pitch_decks` / `chunks` / per-company `analyses`
  rows across all time — only the rollup `analyses` row (with NULL
  FKs, written by the separate `persist_analysis` rollup function)
  ever landed.
- New migration `supabase/migrations/20260430000000_companies_unique_company_key.sql`
  adds the constraint idempotently. Applied on prod via SQL editor;
  testing was unaffected (already had a working constraint).
- **Per-step observability in `web/db.py` persistence path** — a new
  helper `_record_persist_step_failure` writes failures to BOTH Python
  logs (Railway) AND the `analysis_errors` SQL-queryable table.
  `_upsert_company` and every step in `_persist_company_analysis_row`
  (companies / pitch_decks / chunks / analyses / company_runs) now
  capture exceptions individually with operation name + table + error
  type. Per-chunk failures only log the first occurrence to avoid
  flooding. Future silent persistence bugs are queryable post-hoc via
  the SQL editor — no need to tail Railway logs in real time.
- Verified end-to-end on prod 2026-04-30 (job `fae2b416`): Zaitra deck
  produced 1 company + 1 pitch_deck + 30 chunks + per-company analyses
  row with populated FKs. Zero `analysis_errors` rows.

### Added — Specter MCP integration

A major intake-and-enrichment feature: any pitch deck or URL list can now be
augmented with structured Specter intelligence (funding, team, growth signals,
investor highlights) without uploading CSVs.

- **URL-list intake mode** — Specter mode now accepts a list of URLs in
  addition to (or instead of) the company/people CSVs. Worker pipeline
  fetches each company by URL via the Specter MCP tools `find_company`,
  `get_company_profile`, `get_company_intelligence`, `get_company_financials`
  (plus optional `get_person_profile` per founder).
- **Pitch-deck augmentation** — In pitch-deck mode, a new toggle "Augment
  with Specter intelligence" (default ON) extracts the company URL from the
  deck text (regex over scheme URLs, bare domains, labeled patterns
  `Website:` / `Visit:`, and email addresses), calls the Specter MCP, and
  merges returned chunks into the deck-derived `EvidenceStore`. Failures
  fall back gracefully to deck-only analysis.
- **Deep-team toggle** — New "Fetch deep team profiles (Specter)" toggle
  (default OFF) flips `fetch_full_team` between the lightweight founders
  summary list (3 MCP calls per company) and the full
  `get_person_profile` fan-out (~60% more MCP calls; full LinkedIn-grade
  career history, education, seniority per person). Visible in both
  pitch-deck and URL-list modes.
- **OAuth + persistent refresh tokens** — `scripts/specter_oauth_login.py`
  one-shot CLI mints refresh tokens via Authorization Code + PKCE.
  Rotated refresh tokens are persisted in a new RLS-locked
  `mcp_secrets` Supabase table so they survive Railway redeploys.
- **Brand-stem fallback** — When `find_company(domain)` returns "No
  company found" (e.g. deck has `adspawn.com` but Specter indexes
  `adspawn.io`), automatically retry with the brand stem (`adspawn`).
  Cross-checks the returned domain shares the same stem before accepting;
  raises `SpecterDisambiguationError` on cross-company collisions.
- **Disambiguation safeguards** — `find_company('scribe.com')` returning
  "Shopscribe" is caught by the domain-root mismatch check;
  `SpecterCompanyNotFoundError` is a distinct subclass so the retry loop
  in `_call_tool` fast-fails on definitive "no match" answers (saves
  ~6 seconds of wasted backoff per missing company).
- **Email-domain detection in deck text** — `_EMAIL_DOMAIN_RE` extracts
  the domain part of any `founder@company.com` address; emails get the
  same 3× score multiplier as labeled patterns since pitch decks almost
  always include the company's own contact email.
- **Money/metric notation hardening** — Bare-domain regex requires the
  TLD to be alphabetic-only and ≥2 chars, and `_is_blocked()` rejects
  domains whose first label is all-digits — kills false matches on
  `$9.9M`, `$4.8MM`, `$2.2Bn`, `1.2k`, `10.5x` etc.
- **Full-deck coverage** — Default `max_chunks` raised from 10 to 50 so
  contact-slide-only URLs (e.g. on slide 17) are found.
- **Tests** — 49 new unit tests across
  `tests/test_specter_mcp_client.py` and
  `tests/test_specter_augmentation.py`. All network-free; the real MCP
  server is gated on `SPECTER_MCP_REFRESH_TOKEN` being set.

### Added — schema

- **`mcp_secrets`** — RLS-locked single-row-per-key table (key,
  value_text, created_at, updated_at). Stores the rotated Specter refresh
  token. Migration: `supabase/migrations/20260428000000_mcp_secrets.sql`.

### Changed

- `AnalyzeRequest` gains `use_specter_mcp: bool = True` and
  `fetch_full_team: bool = False` fields (web/app.py).
- `_run_analysis` and `_run_document_analysis` thread both flags through;
  augmentation hooks run in **both** the single-file and multi-file
  branches of the pitch-deck pipeline.
- `specter_company_worker` accepts new `--fetch-full-team` CLI flag for
  URL-mode tasks; `specter_batch_worker` reads `fetch_full_team` from
  `run_config` and forwards it.
- `_unwrap_tool_result` inspects MCP tool error text and raises
  `SpecterCompanyNotFoundError` (not generic `SpecterMCPError`) when
  Specter explicitly returns "No company found".

### Env vars

- New: `SPECTER_MCP_URL`, `SPECTER_MCP_CLIENT_ID`,
  `SPECTER_MCP_REFRESH_TOKEN`. Required when MCP augmentation is in
  use; optional otherwise (the augmentation toggle silently no-ops).
  See `.env.example`.

## [0.0.6] - 2026-03-07

### Added

- **Company-centric history** — New `company_runs` table and `company_key` for deduplication; grouped UI/history views by company
  - `companies.company_key` — Normalized key from domain or name for cross-job deduplication
  - `company_runs` — Per-company run records (job, decision, scores, result payload) for history
  - `list_company_histories()` — API to list saved runs grouped by company with backfill from analyses
  - New migration `20260306010000_company_runs.sql`
- **Stop finalization** — When a job is stopped mid-run, partial results are now finalized (ranking + Excel export) instead of discarded
  - `_finalize_stopped_results()` builds and persists partial results; user sees "Partial results ready — N/M companies ranked"
- **Specter detection** — Improved detection via tabular header sniffing (company vs people markers) in addition to filename patterns
- **Job list from Supabase** — `_list_jobs_for_ui()` merges in-memory jobs with saved Supabase jobs for unified history
- **Railway ignore** — `.railwayignore` for deployment exclusions

### Changed

- **Rockaway Deal Intelligence** — App title and branding updated from "Startup Ranker"
- Pause/resume flow — Only transition to `running` when status is actually `paused` (avoids redundant updates)
- `_append_progress()` — New `allow_stopped` flag to append progress when job is stopped (for finalization messages)
- Sample deal data — `deals/sample_startup/` replaced with `deals/sample_company/`

## [0.0.5] - 2026-03-06

### Added

- **Supabase persistence** — Optional persistent storage for analyses, Excel exports, job telemetry, and person profile jobs via Supabase (Postgres + Storage)
  - New `web/db.py` module with full CRUD for jobs, analyses, companies, chunks, events, errors, model executions, and person profile jobs
  - Excel files uploaded to Supabase Storage bucket `analysis-exports`; served from Storage when local file is missing (e.g. after restart)
  - Completed jobs loaded from Supabase on startup in addition to local JSON
  - New API endpoints: `GET /api/analyses/{job_id}`, `GET /api/companies/{company_name}/analyses`
- **Job control (pause / resume / stop)** — New `POST /api/jobs/{job_id}/control` endpoint with cooperative checkpoints throughout the analysis pipeline
  - Jobs can be paused, resumed, or stopped mid-run from the UI or API
  - Status model extended: `paused`, `stopped` states alongside `pending`, `running`, `done`, `error`
- **Scoring heartbeat & timeout** — `_await_with_heartbeat()` wrapper in `batch.py` gives periodic progress updates and a hard wall-clock timeout (default 420s) for the LangGraph scoring step
- **LLM timeout & retry config** — All LLM providers now accept `LLM_REQUEST_TIMEOUT_SECONDS` (default 90s) and `LLM_MAX_RETRIES` (default 2) env vars, passed to Gemini, OpenAI, OpenRouter, and Anthropic clients
- **Runtime version tags** — `APP_VERSION`, `PROMPT_VERSION`, `PIPELINE_VERSION`, `SCHEMA_VERSION` env vars persisted with each job for traceability
- **Model execution telemetry** — Per-company LLM call metadata (provider, model, latency, status, errors) tracked in `model_executions` table
- **Source file metadata** — Uploaded files now include `mime_type`, `sha256`, and `local_path`; persisted to `source_files` table
- **Supabase migrations** — `supabase/migrations/20260306000000_extended_persistence.sql` adds analysis_events, job_controls, job_status_history, analysis_errors, source_files, model_executions, and person_profile_jobs tables

### Changed

- Analysis jobs now run in a dedicated thread to keep the FastAPI event loop responsive for control actions
- Progress reporting refactored through `_append_progress()` and `_set_job_status()` helpers with optional DB persistence
- `pyproject.toml` adds `supabase>=2.0.0` dependency

## [0.0.4] - 2026-03-05

### Added

- **Person Intelligence** — On-demand team-member profile enrichment (LinkedIn, web)
- Deploy script improvements, provider validation

## [0.0.3] - 2026-03-04

### Added

- **Executive Summary**: Human-readable summaries for the three scoring dimensions (Strategy Fit, Team, Potential), plus Key Points and Red Flags sections
  - New pipeline stage `generate_executive_summary` runs after `compute_composite_rank`
  - `CompanyRankingResult` now includes: `strategy_fit_summary`, `team_summary`, `potential_summary`, `key_points`, `red_flags`
  - UI renders Summary, KEY POINTS, and RED FLAGS (N) in both single-company and batch result cards
  - Excel export includes key_points and red_flags columns

### Changed

- Extended ranking prompts with `EXECUTIVE_SUMMARY_SYSTEM` and `EXECUTIVE_SUMMARY_USER`
- Added `ExecutiveSummaryOutput` Pydantic schema for LLM structured output

## [0.0.2] - 2026-03-04

### Added

- Answer-triggered Perplexity web search: only queries the API when the LLM answer indicates no evidence (e.g. "Unknown from provided documents")
- `WEB_SEARCH_TRIGGER` env var: `answer` (default) or `no_chunks` for trigger mode
- `_answer_indicates_no_evidence()` pattern matching for lack-of-evidence detection

### Changed

- Default LLM: Gemini 3.1 Flash-Lite (`gemini-3-flash-lite-preview`)
- Evidence answering flow: run grounded LLM call first, then conditionally call Perplexity when answer indicates no evidence
- Web app, Specter ingest, ranking stage, prompt library integrations
