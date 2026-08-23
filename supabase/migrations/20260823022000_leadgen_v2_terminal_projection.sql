-- Recover exact-domain v2 results produced before the worker persisted the
-- immutable source_company_key, while keeping cross-domain results closed.
CREATE OR REPLACE FUNCTION public.get_leadgen_machine_v2_lifecycle(p_intake_id TEXT)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_row public.leadgen_machine_v2_intakes%ROWTYPE;
    v_bundle public.leadgen_evidence_bundles%ROWTYPE;
    v_job_status TEXT;
    v_job_status_at TIMESTAMPTZ;
    v_worker_status TEXT;
    v_analysis_status TEXT;
    v_analysis_status_at TIMESTAMPTZ;
    v_pipeline_version TEXT;
    v_scoring_version TEXT;
    v_company_run public.company_runs%ROWTYPE;
    v_company_run_count INTEGER := 0;
    v_requested_company_key TEXT;
    v_state TEXT;
    v_completed_at TIMESTAMPTZ;
    v_terminal_result JSONB;
BEGIN
    SELECT * INTO v_row
      FROM public.leadgen_machine_v2_intakes
      WHERE intake_id = p_intake_id;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    SELECT * INTO v_bundle
      FROM public.leadgen_evidence_bundles
      WHERE bundle_sha256 = v_row.evidence_bundle_sha256;
    v_state := v_row.lifecycle_state;
    v_requested_company_key := 'domain:' || v_row.canonical_domain;

    IF v_row.job_id IS NOT NULL THEN
        SELECT j.run_config #>> '{worker_state,status}',
               j.pipeline_version,
               j.run_config ->> 'rdi_scoring_version'
          INTO v_worker_status, v_pipeline_version, v_scoring_version
          FROM public.jobs AS j
          WHERE j.job_id_legacy = v_row.job_id
          LIMIT 1;

        SELECT h.status, h.created_at
          INTO v_job_status, v_job_status_at
          FROM public.job_status_history AS h
          WHERE h.job_id_legacy = v_row.job_id
          ORDER BY h.created_at DESC
          LIMIT 1;

        SELECT a.status, a.created_at
          INTO v_analysis_status, v_analysis_status_at
          FROM public.analyses AS a
          WHERE a.job_id_legacy = v_row.job_id
            AND a.company_id IS NULL
          ORDER BY a.created_at DESC
          LIMIT 1;

        SELECT count(*)::INTEGER
          INTO v_company_run_count
          FROM public.company_runs AS cr
          WHERE cr.job_id_legacy = v_row.job_id
            AND (
                cr.source_company_key = v_requested_company_key
                OR (
                    cr.source_company_key IS NULL
                    AND cr.company_key = v_requested_company_key
                )
            );

        IF v_company_run_count = 1 THEN
            SELECT cr.* INTO v_company_run
              FROM public.company_runs AS cr
              WHERE cr.job_id_legacy = v_row.job_id
                AND (
                    cr.source_company_key = v_requested_company_key
                    OR (
                        cr.source_company_key IS NULL
                        AND cr.company_key = v_requested_company_key
                    )
                )
              LIMIT 1;
        END IF;

        IF COALESCE(v_analysis_status, v_job_status, v_worker_status) = 'done'
           AND v_company_run_count = 1
           AND v_company_run.company_id IS NOT NULL
           AND COALESCE(v_company_run.decision, '') NOT IN ('error', 'timeout') THEN
            v_state := 'succeeded';
            v_completed_at := COALESCE(
                v_company_run.created_at,
                v_company_run.run_created_at
            );
            v_terminal_result := jsonb_build_object(
              'external_company_id', v_row.external_company_id::TEXT,
              'rdi_company_id', v_company_run.company_id::TEXT,
              'composite_score', COALESCE(
                  v_company_run.result_payload #>> '{ranking_result,composite_score}',
                  v_company_run.result_payload #>> '{summary_rows,0,composite_score}',
                  v_company_run.composite_score::TEXT
              ),
              'strategy_fit_score', COALESCE(
                  v_company_run.result_payload #>> '{ranking_result,strategy_fit_score}',
                  v_company_run.result_payload #>> '{summary_rows,0,strategy_fit_score}'
              ),
              'team_score', COALESCE(
                  v_company_run.result_payload #>> '{ranking_result,team_score}',
                  v_company_run.result_payload #>> '{summary_rows,0,team_score}'
              ),
              'upside_score', COALESCE(
                  v_company_run.result_payload #>> '{ranking_result,upside_score}',
                  v_company_run.result_payload #>> '{summary_rows,0,upside_score}'
              ),
              'rdi_bucket', COALESCE(
                  v_company_run.result_payload #>> '{ranking_result,bucket}',
                  v_company_run.result_payload #>> '{summary_rows,0,bucket}'
              ),
              'completed_at', v_completed_at,
              'pipeline_version', v_pipeline_version,
              'scoring_version', v_scoring_version
            );
        ELSIF COALESCE(v_analysis_status, v_job_status, v_worker_status)
              IN ('done', 'error', 'interrupted') THEN
            v_state := 'failed';
            v_completed_at := COALESCE(
                v_row.completed_at,
                v_analysis_status_at,
                v_job_status_at
            );
        ELSIF COALESCE(v_analysis_status, v_job_status, v_worker_status) = 'stopped' THEN
            v_state := 'cancelled';
            v_completed_at := COALESCE(
                v_row.completed_at,
                v_analysis_status_at,
                v_job_status_at
            );
        ELSIF COALESCE(v_job_status, v_worker_status)
              IN ('claimed', 'running', 'finalizing') THEN
            v_state := 'running';
        ELSIF COALESCE(v_job_status, v_worker_status) = 'queued' THEN
            v_state := 'queued';
        END IF;
    END IF;

    RETURN to_jsonb(v_row) || jsonb_build_object(
      'external_company_id', v_row.external_company_id::TEXT,
      'bundle_schema_version', v_bundle.schema_version,
      'parent_bundle_sha256', v_bundle.parent_bundle_sha256,
      'rdi_company_id', CASE
          WHEN v_state = 'succeeded' THEN v_company_run.company_id::TEXT
          ELSE NULL
      END,
      'lifecycle_state', v_state,
      'completed_at', COALESCE(v_completed_at, v_row.completed_at),
      'terminal_result', v_terminal_result,
      'safe_error_code', CASE
          WHEN v_state IN ('failed', 'cancelled')
              THEN COALESCE(v_row.safe_error_code, 'analysis_failed')
          ELSE v_row.safe_error_code
      END,
      'safe_error_class', CASE
          WHEN v_state IN ('failed', 'cancelled')
              THEN 'terminal_analysis_failure'
          ELSE NULL
      END,
      'safe_error_message', CASE
          WHEN v_state IN ('failed', 'cancelled')
              THEN 'The RDI analysis ended without a publishable result.'
          ELSE NULL
      END
    );
END;
$$;

REVOKE ALL ON FUNCTION public.get_leadgen_machine_v2_lifecycle(TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_leadgen_machine_v2_lifecycle(TEXT)
    TO service_role;
