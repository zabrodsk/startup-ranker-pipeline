# Recent Updates (Since Last Push)

*Last push: v0.0.5 — Supabase persistence, job control, LLM reliability*

---

## Summary

This document summarizes the main improvements since the last push to GitHub. **v0.0.5** adds **Supabase persistence** for analyses and telemetry, **job control** (pause/resume/stop), **scoring heartbeat & timeout**, and **LLM timeout/retry configuration** across all providers.

---

## Supabase Persistence (New)

Optional persistent storage that survives restarts and enables cross-session access to completed analyses.

- **`web/db.py`** — Full Supabase client module with CRUD for jobs, analyses, companies, chunks, events, errors, model executions, source files, and person profile jobs
- **Storage** — Excel exports uploaded to `analysis-exports` bucket; served from Storage when local file is missing
- **Startup** — Completed jobs loaded from Supabase on app startup (in addition to local JSON)
- **Migrations** — `supabase/migrations/20260306000000_extended_persistence.sql` adds 7 new tables

### New API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyses/{job_id}` | GET | Return analysis results for a completed job |
| `/api/companies/{company_name}/analyses` | GET | Return analyses for a company by name |

### Setup

See [`supabase/README.md`](../supabase/README.md) for project details and migration instructions.

---

## Job Control (New)

Running analyses can now be paused, resumed, or stopped mid-run.

- **`POST /api/jobs/{job_id}/control`** — Accepts `{"action": "pause" | "resume" | "stop"}`
- Cooperative checkpoints (`_cooperate_with_job_control()`) throughout the analysis pipeline check for pause/stop requests
- Status model extended with `paused` and `stopped` states
- Control state persisted to Supabase `job_controls` table when configured

---

## Scoring Heartbeat & Timeout

The LangGraph scoring step (argument generation + ranking) now runs inside `_await_with_heartbeat()`:

- **Heartbeat** — Periodic progress updates every 20s (configurable via `SCORING_HEARTBEAT_SECONDS`)
- **Timeout** — Hard wall-clock timeout of 420s (configurable via `SCORING_TIMEOUT_SECONDS`)
- Prevents indefinite hangs from slow LLM responses or network issues

---

## LLM Timeout & Retry Configuration

All four LLM providers (Gemini, OpenAI, OpenRouter, Anthropic) now accept:

| Env Var | Default | Description |
|---------|---------|-------------|
| `LLM_REQUEST_TIMEOUT_SECONDS` | `90` | Per-request timeout passed to the provider client |
| `LLM_MAX_RETRIES` | `2` | Max retries on transient failures |

---

## Runtime Version Tags

New env vars persisted with each job for traceability:

- `APP_VERSION` (default: `dev`)
- `PROMPT_VERSION` (default: `v1`)
- `PIPELINE_VERSION` (default: `v1`)
- `SCHEMA_VERSION` (default: `20260306000000`)

---

## Model Execution Telemetry

Per-company LLM call metadata tracked in the `model_executions` table:

- Provider, model, request timeout, max retries
- Latency (ms), token counts (prompt/completion/total)
- Status (`done` / `error`), error messages

---

## Other Changes

### Threading Model

Analysis jobs now run in a dedicated thread (`threading.Thread`) to keep the FastAPI event loop responsive for pause/resume/stop controls.

### Progress Reporting

Refactored through `_append_progress()` and `_set_job_status()` helpers with optional DB persistence and stop-guard logic.

### Source File Metadata

Uploaded files now include `mime_type`, `sha256`, and `local_path` in the upload response and `source_files` table.

### Dependencies

- `supabase>=2.0.0` added to `pyproject.toml`

---

## Files Changed (v0.0.5)

| File | Summary |
|------|---------|
| `web/db.py` | **New** — Supabase persistence module |
| `supabase/` | **New** — README + migrations for extended schema |
| `web/app.py` | Job control, Supabase integration, threading, telemetry |
| `web/static/index.html` | UI updates for job control and new features |
| `src/agent/batch.py` | Heartbeat/timeout wrapper, cooperative job control |
| `src/agent/llm.py` | Timeout and retry config for all providers |
| `.env.example` | Supabase, version tags, session/job store paths |
| `pyproject.toml` | `supabase>=2.0.0` dependency |
| `CHANGELOG.md` | v0.0.5 entry |
