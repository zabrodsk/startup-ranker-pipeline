# Supabase Schema

Persistent storage for Rockaway Deal Intelligence analyses.

## Project (created via MCP)

- **Project:** Rockaway Deal Intelligence  
- **ID:** ykxtuqcfhpauddnbxqyq  
- **URL:** https://ykxtuqcfhpauddnbxqyq.supabase.co  
- **Region:** eu-central-1  

## Setup

1. Get your **service_role** key from [Project API Settings](https://supabase.com/dashboard/project/ykxtuqcfhpauddnbxqyq/settings/api).
2. Add to `.env`:
   ```
   SUPABASE_URL=https://ykxtuqcfhpauddnbxqyq.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=<your_service_role_key>
   ```
3. Migrations to apply:
   - `20250304000000_init_schema.sql`
   - `20260306000000_extended_persistence.sql`
   - `20260306010000_company_runs.sql`
   - `20260313000000_enable_rls_phase1_internal_tables.sql`
   - `20260313000001_enable_rls_phase2_app_tables.sql`
   Tables include companies, jobs, pitch_decks, chunks, analyses, analysis_events, job_controls, job_status_history, analysis_errors, source_files, model_executions, person_profile_jobs, company_runs.
4. The app auto-creates the `analysis-exports` bucket on first persist. Or create it manually in [Storage](https://supabase.com/dashboard/project/ykxtuqcfhpauddnbxqyq/storage/buckets).

## Behavior

When configured, the app will:

- Persist completed analyses to Supabase (companies, jobs, pitch_decks, chunks, analyses)
- Upload Excel exports to Storage bucket `analysis-exports`
- Load completed jobs from Supabase on startup (in addition to JSON file)
- Serve Excel downloads from Storage when local file is missing (e.g. after restart)
- Fall back to Supabase for `/api/status/{job_id}` when job not in memory

## Security model

- RLS is enabled on all application tables in `public`.
- No `anon` or `authenticated` policies are created by default.
- The app is expected to access Supabase through the backend using `SUPABASE_SERVICE_ROLE_KEY`.
- Browser clients should keep using the password-protected API instead of querying Supabase tables directly.

## Preflight and rollback

- Preflight before production rollout:
  ```bash
  python scripts/supabase_rls_preflight.py
  ```
- The preflight uses the same `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` env vars as the app backend.
- Manual rollback SQL lives in `supabase/migrations/rollbacks/20260313_disable_rls_on_app_tables.sql`.
- The rollback file is intentionally not part of the normal migration sequence; do not run it with `supabase db push`.

## API Endpoints

- `GET /api/analyses/{job_id}` – Return analysis results for a completed job
- `GET /api/companies/{company_name}/analyses` – Return analyses for a company by name (requires Supabase)
