-- Targeted query-performance indexes for the hottest analyses/jobs reads.
-- Keep these transaction-friendly for normal `supabase db push` usage.

CREATE INDEX IF NOT EXISTS idx_analyses_job_legacy_created_at
    ON analyses(job_id_legacy, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analyses_status_created_at
    ON analyses(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at
    ON jobs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_jobs_input_mode_created_at
    ON jobs(input_mode, created_at DESC);
