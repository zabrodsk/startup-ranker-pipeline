-- Stable per-company LeadGen machine lifecycle.
-- Apply to staging before deploying the matching machine-route code.

ALTER TABLE public.company_runs
    ADD COLUMN IF NOT EXISTS source_company_key TEXT
        CHECK (
            source_company_key IS NULL
            OR (
                length(source_company_key) BETWEEN 10 AND 260
                AND source_company_key = lower(source_company_key)
                AND source_company_key ~ '^domain:[a-z0-9][a-z0-9.-]{1,251}[a-z0-9]$'
            )
        );

CREATE INDEX IF NOT EXISTS idx_company_runs_job_source_company
    ON public.company_runs(job_id_legacy, source_company_key)
    WHERE source_company_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.leadgen_machine_intakes (
    intake_id TEXT PRIMARY KEY
        CHECK (intake_id ~ '^rdi-intake-[0-9a-f]{32}$'),
    contract_version TEXT NOT NULL
        CHECK (contract_version = 'rdi.leadgen-machine.v1'),
    idempotency_identity TEXT NOT NULL UNIQUE
        CHECK (idempotency_identity ~ '^[0-9a-f]{64}$'),
    payload_hash TEXT NOT NULL
        CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    external_company_id TEXT NOT NULL
        CHECK (external_company_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    canonical_domain TEXT NOT NULL
        CHECK (
            canonical_domain = lower(canonical_domain)
            AND length(canonical_domain) BETWEEN 3 AND 253
            AND canonical_domain !~ '^www\.'
            AND canonical_domain !~ '[/:[:space:][:cntrl:]]'
        ),
    campaign_id TEXT NOT NULL
        CHECK (campaign_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    iteration_id TEXT NOT NULL
        CHECK (iteration_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    source_run_id TEXT NOT NULL
        CHECK (source_run_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    batch_id TEXT NOT NULL
        CHECK (batch_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    idempotency_key TEXT NOT NULL
        CHECK (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    target_environment TEXT NOT NULL
        CHECK (target_environment IN ('staging', 'production')),
    provenance_reference TEXT NOT NULL
        CHECK (
            length(provenance_reference) BETWEEN 1 AND 512
            AND provenance_reference !~ '[[:cntrl:]]'
        ),
    rdi_company_id UUID,
    rdi_correlation_id TEXT NOT NULL UNIQUE
        CHECK (rdi_correlation_id ~ '^rdi-correlation-[0-9a-f]{32}$'),
    intake_status TEXT NOT NULL DEFAULT 'accepted'
        CHECK (intake_status IN ('accepted', 'rejected')),
    lifecycle_state TEXT NOT NULL DEFAULT 'accepted'
        CHECK (
            lifecycle_state IN (
                'accepted', 'rejected', 'start_fenced', 'uncertain',
                'queued', 'running', 'succeeded', 'failed', 'cancelled'
            )
        ),
    approval_required BOOLEAN NOT NULL DEFAULT false
        CHECK (approval_required = false),
    rejection_code TEXT
        CHECK (rejection_code IS NULL OR rejection_code ~ '^[a-z][a-z0-9_]{0,63}$'),
    job_id TEXT UNIQUE
        CHECK (job_id IS NULL OR job_id ~ '^rdi-job-[0-9a-f]{32}$'),
    start_actor TEXT
        CHECK (start_actor IS NULL OR start_actor = 'service:rockaway-leadgen'),
    safe_error_code TEXT
        CHECK (safe_error_code IS NULL OR safe_error_code ~ '^[a-z][a-z0-9_]{0,63}$'),
    safe_error_class TEXT
        CHECK (safe_error_class IS NULL OR safe_error_class ~ '^[a-z][a-z0-9_]{0,63}$'),
    safe_error_message TEXT
        CHECK (
            safe_error_message IS NULL
            OR (
                length(safe_error_message) BETWEEN 1 AND 240
                AND safe_error_message !~ '[[:cntrl:]]'
            )
        ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    CHECK (
        lifecycle_state IN ('accepted', 'rejected')
        OR (job_id IS NOT NULL AND start_actor = 'service:rockaway-leadgen')
    ),
    CHECK (
        lifecycle_state NOT IN ('succeeded', 'failed', 'cancelled', 'rejected')
        OR completed_at IS NOT NULL
        OR lifecycle_state = 'rejected'
    )
);

CREATE TABLE IF NOT EXISTS public.leadgen_machine_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intake_id TEXT NOT NULL,
    event_type TEXT NOT NULL
        CHECK (event_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    lifecycle_state TEXT NOT NULL
        CHECK (
            lifecycle_state IN (
                'accepted', 'rejected', 'start_fenced', 'uncertain',
                'queued', 'running', 'succeeded', 'failed', 'cancelled'
            )
        ),
    actor TEXT,
    job_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT leadgen_machine_events_intake_fk
        FOREIGN KEY (intake_id)
        REFERENCES public.leadgen_machine_intakes(intake_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_leadgen_machine_intakes_environment_state
    ON public.leadgen_machine_intakes(
        target_environment,
        lifecycle_state
    );

CREATE INDEX IF NOT EXISTS idx_leadgen_machine_intakes_external_company
    ON public.leadgen_machine_intakes(external_company_id);

CREATE INDEX IF NOT EXISTS idx_leadgen_machine_intakes_domain
    ON public.leadgen_machine_intakes(canonical_domain);

CREATE INDEX IF NOT EXISTS idx_leadgen_machine_events_intake_created
    ON public.leadgen_machine_events(intake_id, created_at DESC);

ALTER TABLE public.leadgen_machine_intakes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leadgen_machine_events ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.create_leadgen_machine_intake(
    p_record JSONB
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
BEGIN
    INSERT INTO public.leadgen_machine_intakes (
        intake_id,
        contract_version,
        idempotency_identity,
        payload_hash,
        external_company_id,
        canonical_domain,
        campaign_id,
        iteration_id,
        source_run_id,
        batch_id,
        idempotency_key,
        target_environment,
        provenance_reference,
        rdi_company_id,
        rdi_correlation_id,
        intake_status,
        lifecycle_state,
        approval_required,
        rejection_code,
        created_at,
        updated_at
    ) VALUES (
        p_record ->> 'intake_id',
        p_record ->> 'contract_version',
        p_record ->> 'idempotency_identity',
        p_record ->> 'payload_hash',
        p_record ->> 'external_company_id',
        p_record ->> 'canonical_domain',
        p_record ->> 'campaign_id',
        p_record ->> 'iteration_id',
        p_record ->> 'source_run_id',
        p_record ->> 'batch_id',
        p_record ->> 'idempotency_key',
        p_record ->> 'target_environment',
        p_record ->> 'provenance_reference',
        NULLIF(p_record ->> 'rdi_company_id', '')::UUID,
        p_record ->> 'rdi_correlation_id',
        COALESCE(p_record ->> 'intake_status', 'accepted'),
        COALESCE(p_record ->> 'lifecycle_state', 'accepted'),
        COALESCE((p_record ->> 'approval_required')::BOOLEAN, false),
        p_record ->> 'rejection_code',
        COALESCE((p_record ->> 'created_at')::TIMESTAMPTZ, now()),
        COALESCE((p_record ->> 'updated_at')::TIMESTAMPTZ, now())
    )
    ON CONFLICT (idempotency_identity) DO NOTHING
    RETURNING * INTO v_row;

    IF FOUND THEN
        INSERT INTO public.leadgen_machine_events (
            intake_id,
            event_type,
            lifecycle_state,
            actor,
            payload
        ) VALUES (
            v_row.intake_id,
            'intake_created',
            v_row.lifecycle_state,
            'service:rockaway-leadgen',
            jsonb_build_object(
                'contract_version', v_row.contract_version,
                'payload_hash', v_row.payload_hash
            )
        );
        RETURN jsonb_build_object('action', 'created') || to_jsonb(v_row);
    END IF;

    SELECT *
    INTO v_row
    FROM public.leadgen_machine_intakes
    WHERE idempotency_identity = p_record ->> 'idempotency_identity';

    IF v_row.payload_hash <> p_record ->> 'payload_hash' THEN
        RETURN jsonb_build_object('action', 'conflict') || to_jsonb(v_row);
    END IF;
    RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
END;
$$;

CREATE OR REPLACE FUNCTION public.reserve_leadgen_machine_start(
    p_intake_id TEXT,
    p_target_environment TEXT,
    p_job_id TEXT,
    p_actor TEXT,
    p_global_limit INTEGER
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
    v_started_count INTEGER;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen' THEN
        RAISE EXCEPTION 'invalid machine actor' USING ERRCODE = '42501';
    END IF;
    IF p_target_environment NOT IN ('staging', 'production') THEN
        RAISE EXCEPTION 'invalid target environment' USING ERRCODE = '22023';
    END IF;
    IF p_global_limit < 1 OR p_global_limit > 100 THEN
        RAISE EXCEPTION 'invalid global start limit' USING ERRCODE = '22023';
    END IF;

    SELECT *
    INTO v_row
    FROM public.leadgen_machine_intakes
    WHERE intake_id = p_intake_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('action', 'unknown');
    END IF;
    IF v_row.target_environment <> p_target_environment THEN
        RETURN jsonb_build_object('action', 'environment_mismatch') || to_jsonb(v_row);
    END IF;
    IF v_row.lifecycle_state IN ('start_fenced', 'uncertain', 'queued', 'running') THEN
        RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
    END IF;
    IF v_row.lifecycle_state IN ('rejected', 'failed', 'cancelled', 'succeeded') THEN
        RETURN jsonb_build_object('action', 'terminal_invalid') || to_jsonb(v_row);
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended(v_row.target_environment, 0)
    );

    SELECT count(*)
    INTO v_started_count
    FROM public.leadgen_machine_intakes
    WHERE target_environment = v_row.target_environment
      AND lifecycle_state IN (
          'start_fenced', 'uncertain', 'queued', 'running',
          'succeeded', 'failed', 'cancelled'
      );

    IF v_started_count >= p_global_limit THEN
        RETURN jsonb_build_object('action', 'rate_limited') || to_jsonb(v_row);
    END IF;

    UPDATE public.leadgen_machine_intakes
    SET lifecycle_state = 'start_fenced',
        job_id = p_job_id,
        start_actor = p_actor,
        started_at = COALESCE(started_at, now()),
        updated_at = now()
    WHERE intake_id = p_intake_id
    RETURNING * INTO v_row;

    INSERT INTO public.leadgen_machine_events (
        intake_id,
        event_type,
        lifecycle_state,
        actor,
        job_id,
        payload
    ) VALUES (
        v_row.intake_id,
        'start_fenced',
        v_row.lifecycle_state,
        p_actor,
        p_job_id,
        jsonb_build_object(
            'target_environment', v_row.target_environment,
            'campaign_id', v_row.campaign_id,
            'global_limit', p_global_limit,
            'started_count_before', v_started_count
        )
    );

    RETURN jsonb_build_object('action', 'reserved') || to_jsonb(v_row);
END;
$$;

CREATE OR REPLACE FUNCTION public.release_leadgen_machine_start(
    p_intake_id TEXT,
    p_job_id TEXT,
    p_actor TEXT
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen' THEN
        RAISE EXCEPTION 'invalid machine actor' USING ERRCODE = '42501';
    END IF;

    SELECT *
    INTO v_row
    FROM public.leadgen_machine_intakes
    WHERE intake_id = p_intake_id
    FOR UPDATE;

    IF NOT FOUND
       OR v_row.job_id IS DISTINCT FROM p_job_id
       OR v_row.start_actor IS DISTINCT FROM p_actor
       OR v_row.lifecycle_state <> 'start_fenced' THEN
        RETURN NULL;
    END IF;

    UPDATE public.leadgen_machine_intakes
    SET lifecycle_state = 'accepted',
        job_id = NULL,
        start_actor = NULL,
        started_at = NULL,
        safe_error_code = NULL,
        safe_error_class = NULL,
        safe_error_message = NULL,
        updated_at = now()
    WHERE intake_id = p_intake_id
    RETURNING * INTO v_row;

    INSERT INTO public.leadgen_machine_events (
        intake_id,
        event_type,
        lifecycle_state,
        actor,
        job_id,
        payload
    ) VALUES (
        v_row.intake_id,
        'start_released',
        v_row.lifecycle_state,
        p_actor,
        p_job_id,
        jsonb_build_object('reason', 'definite_no_start')
    );

    RETURN to_jsonb(v_row);
END;
$$;

CREATE OR REPLACE FUNCTION public.finalize_leadgen_machine_start(
    p_intake_id TEXT,
    p_job_id TEXT,
    p_lifecycle_state TEXT,
    p_safe_error_code TEXT,
    p_safe_error_class TEXT,
    p_safe_error_message TEXT,
    p_actor TEXT
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen' THEN
        RAISE EXCEPTION 'invalid machine actor' USING ERRCODE = '42501';
    END IF;
    IF p_lifecycle_state NOT IN ('queued', 'uncertain', 'failed') THEN
        RAISE EXCEPTION 'invalid start outcome' USING ERRCODE = '22023';
    END IF;

    SELECT *
    INTO v_row
    FROM public.leadgen_machine_intakes
    WHERE intake_id = p_intake_id
    FOR UPDATE;

    IF NOT FOUND OR v_row.job_id IS DISTINCT FROM p_job_id THEN
        RETURN NULL;
    END IF;
    IF v_row.lifecycle_state = p_lifecycle_state THEN
        RETURN to_jsonb(v_row);
    END IF;
    IF v_row.lifecycle_state NOT IN ('start_fenced', 'uncertain') THEN
        RETURN NULL;
    END IF;

    UPDATE public.leadgen_machine_intakes
    SET lifecycle_state = p_lifecycle_state,
        safe_error_code = p_safe_error_code,
        safe_error_class = p_safe_error_class,
        safe_error_message = p_safe_error_message,
        completed_at = CASE
            WHEN p_lifecycle_state = 'failed' THEN COALESCE(completed_at, now())
            ELSE completed_at
        END,
        updated_at = now()
    WHERE intake_id = p_intake_id
    RETURNING * INTO v_row;

    INSERT INTO public.leadgen_machine_events (
        intake_id,
        event_type,
        lifecycle_state,
        actor,
        job_id,
        payload
    ) VALUES (
        v_row.intake_id,
        'start_' || p_lifecycle_state,
        v_row.lifecycle_state,
        p_actor,
        p_job_id,
        jsonb_build_object(
            'safe_error_code', p_safe_error_code,
            'safe_error_class', p_safe_error_class
        )
    );

    RETURN to_jsonb(v_row);
END;
$$;

CREATE OR REPLACE FUNCTION public.get_leadgen_machine_lifecycle(
    p_intake_id TEXT
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
    v_job_status TEXT;
    v_job_status_at TIMESTAMPTZ;
    v_worker_status TEXT;
    v_analysis_status TEXT;
    v_analysis_status_at TIMESTAMPTZ;
    v_pipeline_version TEXT;
    v_scoring_version TEXT;
    v_company_run public.company_runs%ROWTYPE;
    v_company_run_count INTEGER := 0;
    v_state TEXT;
    v_completed_at TIMESTAMPTZ;
    v_terminal_result JSONB;
BEGIN
    SELECT *
    INTO v_row
    FROM public.leadgen_machine_intakes
    WHERE intake_id = p_intake_id;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    v_state := v_row.lifecycle_state;
    IF v_row.job_id IS NOT NULL THEN
        SELECT
            j.run_config #>> '{worker_state,status}',
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
          AND cr.source_company_key = 'domain:' || v_row.canonical_domain;

        IF v_company_run_count = 1 THEN
            SELECT cr.*
            INTO v_company_run
            FROM public.company_runs AS cr
            WHERE cr.job_id_legacy = v_row.job_id
              AND cr.source_company_key = 'domain:' || v_row.canonical_domain
            LIMIT 1;
        END IF;

        IF COALESCE(v_analysis_status, v_job_status, v_worker_status) = 'done'
           AND v_company_run_count = 1
           AND v_company_run.company_key IS NOT NULL
           AND v_company_run.company_id IS NOT NULL
           AND COALESCE(v_company_run.decision, '') NOT IN ('error', 'timeout') THEN
            v_state := 'succeeded';
            v_completed_at := COALESCE(v_company_run.created_at, v_company_run.run_created_at);
            v_terminal_result := jsonb_build_object(
                'external_company_id', v_row.external_company_id,
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
        ELSIF COALESCE(v_analysis_status, v_job_status, v_worker_status) = 'done' THEN
            v_state := 'failed';
            v_completed_at := COALESCE(
                v_row.completed_at,
                v_analysis_status_at,
                v_job_status_at
            );
        ELSIF COALESCE(v_analysis_status, v_job_status, v_worker_status) IN ('error', 'interrupted') THEN
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
        ELSIF COALESCE(v_job_status, v_worker_status) IN ('claimed', 'running', 'finalizing') THEN
            v_state := 'running';
        ELSIF COALESCE(v_job_status, v_worker_status) = 'queued' THEN
            v_state := 'queued';
        END IF;
    END IF;

    RETURN to_jsonb(v_row) || jsonb_build_object(
        'rdi_company_id', CASE
            WHEN v_state = 'succeeded' AND v_company_run.company_id IS NOT NULL
                THEN v_company_run.company_id::TEXT
            ELSE v_row.rdi_company_id::TEXT
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
                THEN COALESCE(v_row.safe_error_class, 'terminal_analysis_failure')
            ELSE v_row.safe_error_class
        END,
        'safe_error_message', CASE
            WHEN v_state IN ('failed', 'cancelled')
                THEN 'The RDI analysis ended without a publishable result.'
            ELSE v_row.safe_error_message
        END
    );
END;
$$;

REVOKE ALL ON TABLE public.leadgen_machine_intakes
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON TABLE public.leadgen_machine_events
    FROM PUBLIC, anon, authenticated, service_role;

REVOKE ALL ON FUNCTION public.create_leadgen_machine_intake(JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.reserve_leadgen_machine_start(TEXT, TEXT, TEXT, TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.release_leadgen_machine_start(TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.finalize_leadgen_machine_start(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.get_leadgen_machine_lifecycle(TEXT)
    FROM PUBLIC, anon, authenticated, service_role;

GRANT EXECUTE ON FUNCTION public.create_leadgen_machine_intake(JSONB)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.reserve_leadgen_machine_start(TEXT, TEXT, TEXT, TEXT, INTEGER)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.release_leadgen_machine_start(TEXT, TEXT, TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.finalize_leadgen_machine_start(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.get_leadgen_machine_lifecycle(TEXT)
    TO service_role;
