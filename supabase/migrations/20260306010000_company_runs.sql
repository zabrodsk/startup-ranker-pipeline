-- Company-centric history for Specter and pitch deck runs
-- Deduplicates companies and stores per-company run records for grouped UI/history views.

ALTER TABLE companies
    ADD COLUMN IF NOT EXISTS company_key TEXT;

UPDATE companies
SET company_key = COALESCE(
    NULLIF('domain:' || lower(regexp_replace(regexp_replace(regexp_replace(COALESCE(domain, ''), '^https?://', ''), '^www\\.', ''), '/+$', '')), 'domain:'),
    'name:' || lower(regexp_replace(COALESCE(name, ''), '[^a-zA-Z0-9]+', '-', 'g'))
)
WHERE company_key IS NULL;

WITH ranked AS (
    SELECT id, company_key,
           row_number() OVER (PARTITION BY company_key ORDER BY created_at, id) AS rn
    FROM companies
    WHERE company_key IS NOT NULL
)
UPDATE companies c
SET company_key = ranked.company_key || '--legacy-' || ranked.rn
FROM ranked
WHERE c.id = ranked.id
  AND ranked.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_companies_company_key
    ON companies(company_key)
    WHERE company_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS company_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    company_key TEXT NOT NULL,
    company_name TEXT NOT NULL,
    startup_slug TEXT,
    input_order INT,
    decision TEXT,
    total_score DOUBLE PRECISION,
    composite_score DOUBLE PRECISION,
    bucket TEXT,
    mode TEXT,
    run_created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    result_payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_company_runs_job_company
    ON company_runs(job_id_legacy, company_key);

CREATE INDEX IF NOT EXISTS idx_company_runs_company_key
    ON company_runs(company_key);

CREATE INDEX IF NOT EXISTS idx_company_runs_run_created_at
    ON company_runs(run_created_at DESC);
