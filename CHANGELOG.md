# Changelog

All notable changes to this project will be documented in this file.

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
