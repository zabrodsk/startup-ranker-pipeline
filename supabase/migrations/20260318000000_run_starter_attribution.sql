-- Persist immutable starter attribution for analysis jobs and company run history.

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS started_by_user_id TEXT,
    ADD COLUMN IF NOT EXISTS started_by_email TEXT,
    ADD COLUMN IF NOT EXISTS started_by_display_name TEXT,
    ADD COLUMN IF NOT EXISTS started_by_label TEXT;

ALTER TABLE company_runs
    ADD COLUMN IF NOT EXISTS started_by_user_id TEXT,
    ADD COLUMN IF NOT EXISTS started_by_email TEXT,
    ADD COLUMN IF NOT EXISTS started_by_display_name TEXT,
    ADD COLUMN IF NOT EXISTS started_by_label TEXT;

CREATE INDEX IF NOT EXISTS idx_jobs_started_by_user_id
    ON jobs(started_by_user_id)
    WHERE started_by_user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_company_runs_started_by_user_id
    ON company_runs(started_by_user_id)
    WHERE started_by_user_id IS NOT NULL;
