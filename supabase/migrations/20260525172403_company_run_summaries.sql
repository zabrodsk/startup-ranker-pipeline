-- Lightweight Companies page summaries and lazy per-company details.

ALTER TABLE company_runs
    ADD COLUMN IF NOT EXISTS company_lookup_key TEXT,
    ADD COLUMN IF NOT EXISTS strategy_fit_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS team_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS upside_score DOUBLE PRECISION;

WITH extracted AS (
    SELECT
        id,
        COALESCE(
            NULLIF(company_lookup_key, ''),
            CASE
                WHEN NULLIF(trim(company_name), '') IS NOT NULL THEN
                    'name:' || COALESCE(NULLIF(trim(both '-' from regexp_replace(lower(company_name), '[^a-z0-9]+', '-', 'g')), ''), 'unknown')
                WHEN NULLIF(trim(startup_slug), '') IS NOT NULL THEN
                    'slug:' || COALESCE(NULLIF(trim(both '-' from regexp_replace(lower(startup_slug), '[^a-z0-9]+', '-', 'g')), ''), 'unknown')
                ELSE
                    COALESCE(NULLIF(regexp_replace(lower(COALESCE(company_key, '')), '--legacy-[0-9]+$', ''), ''), 'name:unknown')
            END
        ) AS normalized_lookup_key,
        COALESCE(
            NULLIF(result_payload #>> '{ranking_result,strategy_fit_score}', ''),
            NULLIF(result_payload #>> '{summary_rows,0,strategy_fit_score}', '')
        ) AS strategy_fit_text,
        COALESCE(
            NULLIF(result_payload #>> '{ranking_result,team_score}', ''),
            NULLIF(result_payload #>> '{summary_rows,0,team_score}', '')
        ) AS team_text,
        COALESCE(
            NULLIF(result_payload #>> '{ranking_result,risk_adjusted_potential_score}', ''),
            NULLIF(result_payload #>> '{ranking_result,upside_score}', ''),
            NULLIF(result_payload #>> '{summary_rows,0,risk_adjusted_potential_score}', ''),
            NULLIF(result_payload #>> '{summary_rows,0,upside_score}', '')
        ) AS upside_text
    FROM company_runs
)
UPDATE company_runs cr
SET
    company_lookup_key = extracted.normalized_lookup_key,
    strategy_fit_score = COALESCE(
        cr.strategy_fit_score,
        CASE WHEN extracted.strategy_fit_text ~ '^[[:space:]]*-?[0-9]+(\.[0-9]+)?[[:space:]]*$' THEN extracted.strategy_fit_text::double precision END
    ),
    team_score = COALESCE(
        cr.team_score,
        CASE WHEN extracted.team_text ~ '^[[:space:]]*-?[0-9]+(\.[0-9]+)?[[:space:]]*$' THEN extracted.team_text::double precision END
    ),
    upside_score = COALESCE(
        cr.upside_score,
        CASE WHEN extracted.upside_text ~ '^[[:space:]]*-?[0-9]+(\.[0-9]+)?[[:space:]]*$' THEN extracted.upside_text::double precision END
    )
FROM extracted
WHERE cr.id = extracted.id;

CREATE INDEX IF NOT EXISTS idx_company_runs_lookup_created_at
    ON company_runs(company_lookup_key, run_created_at DESC)
    WHERE company_lookup_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_company_runs_composite_summary
    ON company_runs(composite_score DESC NULLS LAST, total_score DESC NULLS LAST, run_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_company_runs_strategy_fit_summary
    ON company_runs(strategy_fit_score DESC NULLS LAST, run_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_company_runs_team_summary
    ON company_runs(team_score DESC NULLS LAST, run_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_company_runs_upside_summary
    ON company_runs(upside_score DESC NULLS LAST, run_created_at DESC);

CREATE OR REPLACE FUNCTION public.company_run_summaries(
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0,
    p_sort TEXT DEFAULT 'latest'
)
RETURNS TABLE (
    company_lookup_key TEXT,
    company_name TEXT,
    run_count BIGINT,
    latest_job_id TEXT,
    latest_startup_slug TEXT,
    latest_decision TEXT,
    latest_total_score DOUBLE PRECISION,
    latest_composite_score DOUBLE PRECISION,
    latest_bucket TEXT,
    latest_mode TEXT,
    latest_input_order INTEGER,
    latest_run_at TIMESTAMPTZ,
    latest_strategy_fit_score DOUBLE PRECISION,
    latest_team_score DOUBLE PRECISION,
    latest_upside_score DOUBLE PRECISION,
    latest_started_by_user_id TEXT,
    latest_started_by_email TEXT,
    latest_started_by_display_name TEXT,
    latest_started_by_label TEXT,
    total_count BIGINT
)
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $$
    WITH normalized AS (
        SELECT
            COALESCE(
                NULLIF(company_lookup_key, ''),
                CASE
                    WHEN NULLIF(trim(company_name), '') IS NOT NULL THEN
                        'name:' || COALESCE(NULLIF(trim(both '-' from regexp_replace(lower(company_name), '[^a-z0-9]+', '-', 'g')), ''), 'unknown')
                    WHEN NULLIF(trim(startup_slug), '') IS NOT NULL THEN
                        'slug:' || COALESCE(NULLIF(trim(both '-' from regexp_replace(lower(startup_slug), '[^a-z0-9]+', '-', 'g')), ''), 'unknown')
                    ELSE
                        COALESCE(NULLIF(regexp_replace(lower(COALESCE(company_key, '')), '--legacy-[0-9]+$', ''), ''), 'name:unknown')
                END
            ) AS lookup_key,
            company_runs.*
        FROM company_runs
    ),
    ranked AS (
        SELECT
            normalized.*,
            row_number() OVER (
                PARTITION BY lookup_key
                ORDER BY run_created_at DESC NULLS LAST, created_at DESC NULLS LAST, job_id_legacy DESC
            ) AS row_num,
            count(*) OVER (PARTITION BY lookup_key) AS grouped_run_count
        FROM normalized
        WHERE lookup_key IS NOT NULL AND lookup_key <> ''
    ),
    latest AS (
        SELECT
            lookup_key AS company_lookup_key,
            company_name,
            grouped_run_count AS run_count,
            job_id_legacy AS latest_job_id,
            startup_slug AS latest_startup_slug,
            decision AS latest_decision,
            total_score AS latest_total_score,
            composite_score AS latest_composite_score,
            bucket AS latest_bucket,
            mode AS latest_mode,
            input_order AS latest_input_order,
            COALESCE(run_created_at, created_at) AS latest_run_at,
            strategy_fit_score AS latest_strategy_fit_score,
            team_score AS latest_team_score,
            upside_score AS latest_upside_score,
            started_by_user_id AS latest_started_by_user_id,
            started_by_email AS latest_started_by_email,
            started_by_display_name AS latest_started_by_display_name,
            started_by_label AS latest_started_by_label
        FROM ranked
        WHERE row_num = 1
    ),
    counted AS (
        SELECT latest.*, count(*) OVER () AS total_count
        FROM latest
    )
    SELECT *
    FROM counted
    ORDER BY
        CASE WHEN lower(COALESCE(p_sort, 'latest')) = 'top_scored' THEN latest_composite_score END DESC NULLS LAST,
        CASE WHEN lower(COALESCE(p_sort, 'latest')) = 'top_scored' THEN latest_total_score END DESC NULLS LAST,
        CASE WHEN lower(COALESCE(p_sort, 'latest')) = 'strategy_fit' THEN latest_strategy_fit_score END DESC NULLS LAST,
        CASE WHEN lower(COALESCE(p_sort, 'latest')) = 'team' THEN latest_team_score END DESC NULLS LAST,
        CASE WHEN lower(COALESCE(p_sort, 'latest')) = 'upside' THEN latest_upside_score END DESC NULLS LAST,
        CASE
            WHEN lower(COALESCE(p_sort, 'latest')) NOT IN ('top_scored', 'strategy_fit', 'team', 'upside')
            THEN latest_run_at
        END DESC NULLS LAST,
        latest_run_at DESC NULLS LAST,
        lower(company_name) ASC
    LIMIT LEAST(GREATEST(COALESCE(p_limit, 200), 1), 500)
    OFFSET GREATEST(COALESCE(p_offset, 0), 0)
$$;

CREATE OR REPLACE FUNCTION public.company_run_detail(
    p_company_lookup_key TEXT
)
RETURNS TABLE (
    company_id UUID,
    company_key TEXT,
    company_lookup_key TEXT,
    company_name TEXT,
    startup_slug TEXT,
    job_id_legacy TEXT,
    decision TEXT,
    total_score DOUBLE PRECISION,
    composite_score DOUBLE PRECISION,
    strategy_fit_score DOUBLE PRECISION,
    team_score DOUBLE PRECISION,
    upside_score DOUBLE PRECISION,
    bucket TEXT,
    mode TEXT,
    input_order INTEGER,
    run_created_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    started_by_user_id TEXT,
    started_by_email TEXT,
    started_by_display_name TEXT,
    started_by_label TEXT,
    result_payload JSONB
)
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $$
    WITH normalized AS (
        SELECT
            COALESCE(
                NULLIF(company_lookup_key, ''),
                CASE
                    WHEN NULLIF(trim(company_name), '') IS NOT NULL THEN
                        'name:' || COALESCE(NULLIF(trim(both '-' from regexp_replace(lower(company_name), '[^a-z0-9]+', '-', 'g')), ''), 'unknown')
                    WHEN NULLIF(trim(startup_slug), '') IS NOT NULL THEN
                        'slug:' || COALESCE(NULLIF(trim(both '-' from regexp_replace(lower(startup_slug), '[^a-z0-9]+', '-', 'g')), ''), 'unknown')
                    ELSE
                        COALESCE(NULLIF(regexp_replace(lower(COALESCE(company_key, '')), '--legacy-[0-9]+$', ''), ''), 'name:unknown')
                END
            ) AS lookup_key,
            company_runs.*
        FROM company_runs
    )
    SELECT
        company_id,
        company_key,
        lookup_key AS company_lookup_key,
        company_name,
        startup_slug,
        job_id_legacy,
        decision,
        total_score,
        composite_score,
        strategy_fit_score,
        team_score,
        upside_score,
        bucket,
        mode,
        input_order,
        run_created_at,
        created_at,
        started_by_user_id,
        started_by_email,
        started_by_display_name,
        started_by_label,
        result_payload
    FROM normalized
    WHERE lookup_key = lower(trim(COALESCE(p_company_lookup_key, '')))
    ORDER BY run_created_at DESC NULLS LAST, created_at DESC NULLS LAST, job_id_legacy DESC
    LIMIT 200
$$;

REVOKE EXECUTE ON FUNCTION public.company_run_summaries(INTEGER, INTEGER, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.company_run_detail(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.company_run_summaries(INTEGER, INTEGER, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.company_run_detail(TEXT) TO service_role;

NOTIFY pgrst, 'reload schema';
