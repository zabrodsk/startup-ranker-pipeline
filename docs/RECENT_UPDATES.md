# Recent Updates (Since Last Push)

*Last push: v0.0.6 — Company-centric history, stop finalization, Rockaway branding*

---

## Summary

This document summarizes the main improvements in **v0.0.6**. Key additions: **company-centric history** with `company_runs` table and grouped UI, **stop finalization** (partial results when stopping mid-run), improved **Specter detection** via header sniffing, and **Rockaway Deal Intelligence** branding.

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

## Files Changed (v0.0.6)

| File | Summary |
|------|---------|
| `web/db.py` | Company runs, company_key, list_company_histories, backfill logic |
| `web/app.py` | Stop finalization, job list merge, Specter sniffing |
| `web/static/index.html` | Company-centric history UI, grouped runs |
| `supabase/migrations/20260306010000_company_runs.sql` | **New** — company_runs table, company_key |
| `CHANGELOG.md` | v0.0.6 entry |
