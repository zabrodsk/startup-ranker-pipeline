-- Repair a production schema drift that made result reconstruction fail when
-- application code selected the LeadGen source identity column before the
-- broader machine-lifecycle migration had been applied.

ALTER TABLE public.company_runs
    ADD COLUMN IF NOT EXISTS source_company_key TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.company_runs'::regclass
          AND conname = 'company_runs_source_company_key_check'
    ) THEN
        ALTER TABLE public.company_runs
            ADD CONSTRAINT company_runs_source_company_key_check
            CHECK (
                source_company_key IS NULL
                OR (
                    length(source_company_key) BETWEEN 10 AND 260
                    AND source_company_key = lower(source_company_key)
                    AND source_company_key ~ '^domain:[a-z0-9][a-z0-9.-]{1,251}[a-z0-9]$'
                )
            ) NOT VALID;
    END IF;
END $$;

ALTER TABLE public.company_runs
    VALIDATE CONSTRAINT company_runs_source_company_key_check;

CREATE INDEX IF NOT EXISTS idx_company_runs_job_source_company
    ON public.company_runs(job_id_legacy, source_company_key)
    WHERE source_company_key IS NOT NULL;
