-- Additive LeadGen machine v2 contract: immutable evidence bundles and durable
-- quota-pending intake. Apply to testing before enabling
-- RDI_LEADGEN_MACHINE_V2_ENABLED.

BEGIN;

CREATE TABLE public.leadgen_evidence_bundles (
    bundle_sha256 TEXT PRIMARY KEY
        CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
    schema_version TEXT NOT NULL
        CHECK (schema_version = 'frozen-leadgen-evidence-bundle-v1'),
    external_company_id UUID NOT NULL,
    canonical_domain TEXT NOT NULL CHECK (
        canonical_domain = lower(canonical_domain)
        AND length(canonical_domain) BETWEEN 3 AND 253
        AND canonical_domain !~ '^www\.'
        AND canonical_domain !~ '[/:[:space:][:cntrl:]]'
    ),
    requires_specter_mcp BOOLEAN NOT NULL,
    parent_bundle_sha256 TEXT REFERENCES public.leadgen_evidence_bundles(bundle_sha256),
    authorization_sha256 TEXT NOT NULL CHECK (authorization_sha256 ~ '^[0-9a-f]{64}$'),
    byte_size INTEGER NOT NULL CHECK (byte_size BETWEEN 2 AND 25000000),
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CHECK (payload ->> 'schema_version' = schema_version),
    CHECK (payload ->> 'external_company_id' = external_company_id::TEXT),
    CHECK (payload ->> 'canonical_domain' = canonical_domain),
    CHECK ((payload ->> 'requires_specter_mcp')::BOOLEAN = requires_specter_mcp)
);

CREATE TABLE public.leadgen_machine_v2_intakes (
    intake_id TEXT PRIMARY KEY CHECK (intake_id ~ '^rdi-v2-intake-[0-9a-f]{32}$'),
    contract_version TEXT NOT NULL CHECK (contract_version = 'rdi.leadgen-machine.v2'),
    idempotency_identity TEXT NOT NULL UNIQUE CHECK (idempotency_identity ~ '^[0-9a-f]{64}$'),
    payload_hash TEXT NOT NULL CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    external_company_id UUID NOT NULL,
    canonical_domain TEXT NOT NULL CHECK (
        canonical_domain = lower(canonical_domain)
        AND length(canonical_domain) BETWEEN 3 AND 253
        AND canonical_domain !~ '^www\.'
        AND canonical_domain !~ '[/:[:space:][:cntrl:]]'
    ),
    campaign_id TEXT NOT NULL CHECK (campaign_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    iteration_id TEXT NOT NULL CHECK (iteration_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    source_run_id TEXT NOT NULL CHECK (source_run_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    batch_id TEXT NOT NULL CHECK (batch_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    idempotency_key TEXT NOT NULL CHECK (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    target_environment TEXT NOT NULL CHECK (target_environment IN ('staging', 'production')),
    leadgen_business_date DATE NOT NULL,
    business_timezone TEXT NOT NULL CHECK (business_timezone = 'Europe/Prague'),
    evidence_bundle_sha256 TEXT NOT NULL
        REFERENCES public.leadgen_evidence_bundles(bundle_sha256) ON DELETE RESTRICT,
    requires_specter_mcp BOOLEAN NOT NULL,
    lifecycle_state TEXT NOT NULL DEFAULT 'intake_pending' CHECK (
        lifecycle_state IN (
            'intake_pending', 'pending_provider_quota', 'ready_after_provider_recovery',
            'start_reserved', 'uncertain', 'queued', 'running',
            'succeeded', 'failed', 'cancelled'
        )
    ),
    wait_reason TEXT CHECK (wait_reason IS NULL OR wait_reason ~ '^[a-z][a-z0-9_]{0,63}$'),
    blocked_until TIMESTAMPTZ,
    actual_start_business_date DATE,
    job_id TEXT UNIQUE CHECK (job_id IS NULL OR job_id ~ '^rdi-v2-job-[0-9a-f]{32}$'),
    start_actor TEXT CHECK (start_actor IS NULL OR start_actor = 'service:rockaway-leadgen'),
    safe_error_code TEXT CHECK (safe_error_code IS NULL OR safe_error_code ~ '^[a-z][a-z0-9_]{0,63}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    CHECK (
        lifecycle_state IN ('intake_pending', 'pending_provider_quota', 'ready_after_provider_recovery')
        OR (actual_start_business_date IS NOT NULL AND job_id IS NOT NULL AND start_actor IS NOT NULL)
    ),
    CHECK (
        lifecycle_state <> 'pending_provider_quota'
        OR (requires_specter_mcp AND wait_reason IS NOT NULL)
    )
);

CREATE TABLE public.leadgen_machine_v2_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intake_id TEXT NOT NULL REFERENCES public.leadgen_machine_v2_intakes(intake_id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL CHECK (event_type ~ '^[a-z][a-z0-9_]{0,63}$'),
    lifecycle_state TEXT NOT NULL,
    actor TEXT,
    job_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX idx_leadgen_evidence_bundles_company_domain
    ON public.leadgen_evidence_bundles(external_company_id, canonical_domain, stored_at DESC);
CREATE INDEX idx_leadgen_evidence_bundles_parent
    ON public.leadgen_evidence_bundles(parent_bundle_sha256)
    WHERE parent_bundle_sha256 IS NOT NULL;
CREATE INDEX idx_leadgen_machine_v2_environment_state
    ON public.leadgen_machine_v2_intakes(target_environment, lifecycle_state, updated_at);
CREATE INDEX idx_leadgen_machine_v2_bundle
    ON public.leadgen_machine_v2_intakes(evidence_bundle_sha256);
CREATE INDEX idx_leadgen_machine_v2_actual_start
    ON public.leadgen_machine_v2_intakes(target_environment, actual_start_business_date, lifecycle_state)
    WHERE actual_start_business_date IS NOT NULL;
CREATE INDEX idx_leadgen_machine_v2_events_intake_created
    ON public.leadgen_machine_v2_events(intake_id, created_at DESC);

ALTER TABLE public.leadgen_evidence_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leadgen_machine_v2_intakes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leadgen_machine_v2_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY leadgen_evidence_bundles_deny_clients
    ON public.leadgen_evidence_bundles
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY leadgen_machine_v2_intakes_deny_clients
    ON public.leadgen_machine_v2_intakes
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY leadgen_machine_v2_events_deny_clients
    ON public.leadgen_machine_v2_events
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

CREATE FUNCTION public.reject_leadgen_bundle_mutation()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY INVOKER SET search_path = public, pg_temp AS $$
BEGIN
    RAISE EXCEPTION 'LeadGen evidence bundles are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER leadgen_evidence_bundles_immutable
    BEFORE UPDATE OR DELETE ON public.leadgen_evidence_bundles
    FOR EACH ROW EXECUTE FUNCTION public.reject_leadgen_bundle_mutation();

CREATE FUNCTION public.put_leadgen_machine_v2_evidence_bundle(p_record JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY INVOKER SET search_path = public, pg_temp AS $$
DECLARE v_row public.leadgen_evidence_bundles%ROWTYPE;
BEGIN
    INSERT INTO public.leadgen_evidence_bundles (
        bundle_sha256, schema_version, external_company_id, canonical_domain,
        requires_specter_mcp, parent_bundle_sha256, authorization_sha256,
        byte_size, payload, created_at
    ) VALUES (
        p_record ->> 'bundle_sha256', p_record ->> 'schema_version',
        (p_record ->> 'external_company_id')::UUID, p_record ->> 'canonical_domain',
        (p_record ->> 'requires_specter_mcp')::BOOLEAN,
        NULLIF(p_record ->> 'parent_bundle_sha256', ''),
        p_record ->> 'authorization_sha256', (p_record ->> 'byte_size')::INTEGER,
        p_record -> 'payload', (p_record ->> 'created_at')::TIMESTAMPTZ
    ) ON CONFLICT (bundle_sha256) DO NOTHING RETURNING * INTO v_row;
    IF FOUND THEN
        RETURN jsonb_build_object('action', 'created') || to_jsonb(v_row);
    END IF;
    SELECT * INTO v_row FROM public.leadgen_evidence_bundles
      WHERE bundle_sha256 = p_record ->> 'bundle_sha256';
    IF v_row.payload <> p_record -> 'payload'
       OR v_row.authorization_sha256 <> p_record ->> 'authorization_sha256' THEN
        RETURN jsonb_build_object('action', 'conflict') || to_jsonb(v_row);
    END IF;
    RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
END;
$$;

CREATE FUNCTION public.create_leadgen_machine_v2_intake(p_record JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY INVOKER SET search_path = public, pg_temp AS $$
DECLARE
    v_row public.leadgen_machine_v2_intakes%ROWTYPE;
    v_bundle public.leadgen_evidence_bundles%ROWTYPE;
BEGIN
    SELECT * INTO v_row FROM public.leadgen_machine_v2_intakes
      WHERE idempotency_identity = p_record ->> 'idempotency_identity';
    IF FOUND THEN
        IF v_row.payload_hash <> p_record ->> 'payload_hash' THEN
            RETURN jsonb_build_object('action', 'conflict') || to_jsonb(v_row);
        END IF;
        RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
    END IF;

    SELECT * INTO v_bundle FROM public.leadgen_evidence_bundles
      WHERE bundle_sha256 = p_record ->> 'evidence_bundle_sha256';
    IF NOT FOUND THEN RETURN jsonb_build_object('action', 'bundle_missing'); END IF;
    IF v_bundle.external_company_id::TEXT <> p_record ->> 'external_company_id'
       OR v_bundle.canonical_domain <> p_record ->> 'canonical_domain'
       OR v_bundle.payload #>> '{authorization,source_run_id}' <> p_record ->> 'source_run_id'
       OR v_bundle.payload #>> '{authorization,company_id}' <> p_record ->> 'external_company_id'
       OR v_bundle.payload #>> '{authorization,canonical_domain}' <> p_record ->> 'canonical_domain' THEN
        RETURN jsonb_build_object('action', 'bundle_mismatch');
    END IF;

    INSERT INTO public.leadgen_machine_v2_intakes (
        intake_id, contract_version, idempotency_identity, payload_hash,
        external_company_id, canonical_domain, campaign_id, iteration_id,
        source_run_id, batch_id, idempotency_key, target_environment,
        leadgen_business_date, business_timezone, evidence_bundle_sha256,
        requires_specter_mcp, lifecycle_state, created_at, updated_at
    ) VALUES (
        p_record ->> 'intake_id', p_record ->> 'contract_version',
        p_record ->> 'idempotency_identity', p_record ->> 'payload_hash',
        (p_record ->> 'external_company_id')::UUID, p_record ->> 'canonical_domain',
        p_record ->> 'campaign_id', p_record ->> 'iteration_id',
        p_record ->> 'source_run_id', p_record ->> 'batch_id',
        p_record ->> 'idempotency_key', p_record ->> 'target_environment',
        (p_record ->> 'leadgen_business_date')::DATE, p_record ->> 'business_timezone',
        p_record ->> 'evidence_bundle_sha256', v_bundle.requires_specter_mcp,
        'intake_pending', (p_record ->> 'created_at')::TIMESTAMPTZ,
        (p_record ->> 'updated_at')::TIMESTAMPTZ
    ) ON CONFLICT (idempotency_identity) DO NOTHING RETURNING * INTO v_row;

    IF NOT FOUND THEN
        SELECT * INTO v_row FROM public.leadgen_machine_v2_intakes
          WHERE idempotency_identity = p_record ->> 'idempotency_identity';
        IF v_row.payload_hash <> p_record ->> 'payload_hash' THEN
            RETURN jsonb_build_object('action', 'conflict') || to_jsonb(v_row);
        END IF;
        RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
    END IF;
    INSERT INTO public.leadgen_machine_v2_events(intake_id, event_type, lifecycle_state, actor, payload)
      VALUES (v_row.intake_id, 'intake_created', v_row.lifecycle_state,
              'service:rockaway-leadgen', jsonb_build_object('bundle_sha256', v_row.evidence_bundle_sha256));
    RETURN jsonb_build_object('action', 'created') || to_jsonb(v_row);
END;
$$;

CREATE FUNCTION public.reserve_leadgen_machine_v2_start(
    p_intake_id TEXT,
    p_target_environment TEXT,
    p_actual_start_business_date DATE,
    p_business_timezone TEXT,
    p_job_id TEXT,
    p_actor TEXT,
    p_daily_start_limit INTEGER
) RETURNS JSONB LANGUAGE plpgsql SECURITY INVOKER SET search_path = public, pg_temp AS $$
DECLARE
    v_row public.leadgen_machine_v2_intakes%ROWTYPE;
    v_bundle public.leadgen_evidence_bundles%ROWTYPE;
    v_gate public.specter_mcp_quota_gate%ROWTYPE;
    v_started INTEGER;
    v_capacity JSONB;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen' OR p_target_environment NOT IN ('staging', 'production')
       OR p_business_timezone <> 'Europe/Prague' OR p_daily_start_limit NOT BETWEEN 1 AND 20
       OR p_actual_start_business_date <> (clock_timestamp() AT TIME ZONE 'Europe/Prague')::DATE THEN
        RAISE EXCEPTION 'invalid machine v2 start scope' USING ERRCODE = '22023';
    END IF;

    -- Gate lock is always acquired before the daily-cap lock.
    PERFORM pg_advisory_xact_lock(hashtextextended('specter_mcp_gate:' || p_target_environment, 0));
    SELECT * INTO v_row FROM public.leadgen_machine_v2_intakes
      WHERE intake_id = p_intake_id FOR UPDATE;
    IF NOT FOUND THEN RETURN jsonb_build_object('action', 'unknown'); END IF;
    IF v_row.target_environment <> p_target_environment THEN
        RETURN jsonb_build_object('action', 'environment_mismatch') || to_jsonb(v_row);
    END IF;
    SELECT * INTO v_bundle FROM public.leadgen_evidence_bundles
      WHERE bundle_sha256 = v_row.evidence_bundle_sha256;
    IF NOT FOUND OR v_bundle.external_company_id <> v_row.external_company_id
       OR v_bundle.canonical_domain <> v_row.canonical_domain THEN
        RETURN jsonb_build_object('action', 'bundle_mismatch') || to_jsonb(v_row);
    END IF;

    SELECT * INTO v_gate FROM public.specter_mcp_quota_gate
      WHERE target_environment = p_target_environment;
    IF v_row.requires_specter_mcp AND FOUND AND v_gate.enforcement_enabled
       AND v_gate.state IN ('blocked', 'probing') THEN
        IF v_row.lifecycle_state IN ('intake_pending', 'ready_after_provider_recovery', 'pending_provider_quota') THEN
            UPDATE public.leadgen_machine_v2_intakes SET
                lifecycle_state = 'pending_provider_quota',
                wait_reason = COALESCE(v_gate.reason_code, 'specter_mcp_quota_exhausted'),
                blocked_until = v_gate.blocked_until, updated_at = clock_timestamp()
              WHERE intake_id = p_intake_id RETURNING * INTO v_row;
        END IF;
        RETURN jsonb_build_object('action', 'pending_provider_quota', 'daily_started_count', 0)
               || to_jsonb(v_row);
    END IF;

    IF v_row.lifecycle_state IN ('start_reserved', 'uncertain', 'queued', 'running', 'succeeded', 'failed', 'cancelled') THEN
        RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
    END IF;
    IF v_row.lifecycle_state = 'pending_provider_quota' THEN
        UPDATE public.leadgen_machine_v2_intakes SET lifecycle_state = 'ready_after_provider_recovery',
          wait_reason = NULL, blocked_until = NULL, updated_at = clock_timestamp()
          WHERE intake_id = p_intake_id RETURNING * INTO v_row;
        INSERT INTO public.leadgen_machine_v2_events(intake_id,event_type,lifecycle_state,actor,payload)
          VALUES (p_intake_id,'provider_recovered',v_row.lifecycle_state,p_actor,'{}');
    END IF;

    PERFORM pg_advisory_xact_lock(hashtextextended(p_target_environment || ':' || p_actual_start_business_date::TEXT, 0));
    SELECT (
        (SELECT count(*) FROM public.leadgen_machine_intakes
          WHERE target_environment = p_target_environment
            AND business_date = p_actual_start_business_date
            AND lifecycle_state IN ('start_fenced','uncertain','queued','running','succeeded','failed','cancelled'))
        +
        (SELECT count(*) FROM public.leadgen_machine_v2_intakes
          WHERE target_environment = p_target_environment
            AND actual_start_business_date = p_actual_start_business_date
            AND lifecycle_state IN ('start_reserved','uncertain','queued','running','succeeded','failed','cancelled'))
    )::INTEGER INTO v_started;
    v_capacity := jsonb_build_object('daily_start_limit',p_daily_start_limit,
      'daily_started_count',v_started,'daily_remaining_capacity',greatest(p_daily_start_limit-v_started,0));
    IF v_started >= p_daily_start_limit THEN
        RETURN jsonb_build_object('action','rate_limited') || to_jsonb(v_row) || v_capacity;
    END IF;

    UPDATE public.leadgen_machine_v2_intakes SET lifecycle_state='start_reserved',
      actual_start_business_date=p_actual_start_business_date, job_id=p_job_id,
      start_actor=p_actor, started_at=COALESCE(started_at,clock_timestamp()),
      wait_reason=NULL, blocked_until=NULL, updated_at=clock_timestamp()
      WHERE intake_id=p_intake_id RETURNING * INTO v_row;
    v_started := v_started + 1;
    INSERT INTO public.leadgen_machine_v2_events(intake_id,event_type,lifecycle_state,actor,job_id,payload)
      VALUES (p_intake_id,'start_reserved',v_row.lifecycle_state,p_actor,p_job_id,
        jsonb_build_object('actual_start_business_date',p_actual_start_business_date,'daily_started_count',v_started));
    RETURN jsonb_build_object('action','reserved','daily_start_limit',p_daily_start_limit,
      'daily_started_count',v_started,'daily_remaining_capacity',greatest(p_daily_start_limit-v_started,0),
      'bundle_payload',v_bundle.payload) || to_jsonb(v_row);
END;
$$;

CREATE FUNCTION public.finalize_leadgen_machine_v2_start(
    p_intake_id TEXT, p_job_id TEXT, p_lifecycle_state TEXT,
    p_actor TEXT, p_safe_error_code TEXT DEFAULT NULL
) RETURNS JSONB LANGUAGE plpgsql SECURITY INVOKER SET search_path = public, pg_temp AS $$
DECLARE v_row public.leadgen_machine_v2_intakes%ROWTYPE;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen'
       OR p_lifecycle_state NOT IN ('start_reserved','uncertain','queued','running','succeeded','failed','cancelled') THEN
        RAISE EXCEPTION 'invalid machine v2 finalize request' USING ERRCODE='22023';
    END IF;
    UPDATE public.leadgen_machine_v2_intakes SET lifecycle_state=p_lifecycle_state,
      safe_error_code=p_safe_error_code, updated_at=clock_timestamp(),
      completed_at=CASE WHEN p_lifecycle_state IN ('succeeded','failed','cancelled')
                        THEN clock_timestamp() ELSE completed_at END
      WHERE intake_id=p_intake_id AND job_id=p_job_id RETURNING * INTO v_row;
    IF NOT FOUND THEN RETURN jsonb_build_object('action','unknown'); END IF;
    INSERT INTO public.leadgen_machine_v2_events(intake_id,event_type,lifecycle_state,actor,job_id,payload)
      VALUES (p_intake_id,'start_finalized',v_row.lifecycle_state,p_actor,p_job_id,
              jsonb_build_object('safe_error_code',p_safe_error_code));
    RETURN jsonb_build_object('action','updated') || to_jsonb(v_row);
END;
$$;

CREATE FUNCTION public.release_leadgen_machine_v2_start(
    p_intake_id TEXT, p_job_id TEXT, p_actor TEXT, p_lifecycle_state TEXT,
    p_wait_reason TEXT DEFAULT NULL, p_blocked_until TIMESTAMPTZ DEFAULT NULL
) RETURNS JSONB LANGUAGE plpgsql SECURITY INVOKER SET search_path = public, pg_temp AS $$
DECLARE v_row public.leadgen_machine_v2_intakes%ROWTYPE;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen'
       OR p_lifecycle_state NOT IN ('intake_pending','pending_provider_quota')
       OR (p_lifecycle_state='pending_provider_quota' AND (p_wait_reason IS NULL OR p_blocked_until IS NULL))
       OR (p_lifecycle_state='intake_pending' AND (p_wait_reason IS NOT NULL OR p_blocked_until IS NOT NULL)) THEN
        RAISE EXCEPTION 'invalid machine v2 release request' USING ERRCODE='22023';
    END IF;
    UPDATE public.leadgen_machine_v2_intakes SET
      lifecycle_state=p_lifecycle_state, actual_start_business_date=NULL,
      job_id=NULL, start_actor=NULL, started_at=NULL,
      wait_reason=p_wait_reason, blocked_until=p_blocked_until,
      safe_error_code=NULL, updated_at=clock_timestamp()
      WHERE intake_id=p_intake_id AND job_id=p_job_id AND lifecycle_state='start_reserved'
      RETURNING * INTO v_row;
    IF NOT FOUND THEN RETURN jsonb_build_object('action','unknown'); END IF;
    INSERT INTO public.leadgen_machine_v2_events(intake_id,event_type,lifecycle_state,actor,payload)
      VALUES (p_intake_id,'start_released',v_row.lifecycle_state,p_actor,
              jsonb_build_object('wait_reason',p_wait_reason,'blocked_until',p_blocked_until));
    RETURN jsonb_build_object('action','released') || to_jsonb(v_row);
END;
$$;

CREATE FUNCTION public.get_leadgen_machine_v2_evidence_bundle(p_bundle_sha256 TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY INVOKER SET search_path = public, pg_temp AS $$
    SELECT to_jsonb(b)
    FROM public.leadgen_evidence_bundles b
    WHERE b.bundle_sha256=p_bundle_sha256;
$$;

CREATE FUNCTION public.get_leadgen_machine_v2_lifecycle(p_intake_id TEXT)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY INVOKER SET search_path = public, pg_temp AS $$
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
    v_state TEXT;
    v_completed_at TIMESTAMPTZ;
    v_terminal_result JSONB;
BEGIN
    SELECT * INTO v_row FROM public.leadgen_machine_v2_intakes WHERE intake_id=p_intake_id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    SELECT * INTO v_bundle FROM public.leadgen_evidence_bundles
      WHERE bundle_sha256=v_row.evidence_bundle_sha256;
    v_state := v_row.lifecycle_state;

    IF v_row.job_id IS NOT NULL THEN
        SELECT j.run_config #>> '{worker_state,status}', j.pipeline_version,
               j.run_config ->> 'rdi_scoring_version'
          INTO v_worker_status, v_pipeline_version, v_scoring_version
          FROM public.jobs j WHERE j.job_id_legacy=v_row.job_id LIMIT 1;
        SELECT h.status,h.created_at INTO v_job_status,v_job_status_at
          FROM public.job_status_history h WHERE h.job_id_legacy=v_row.job_id
          ORDER BY h.created_at DESC LIMIT 1;
        SELECT a.status,a.created_at INTO v_analysis_status,v_analysis_status_at
          FROM public.analyses a WHERE a.job_id_legacy=v_row.job_id AND a.company_id IS NULL
          ORDER BY a.created_at DESC LIMIT 1;
        SELECT count(*)::INTEGER INTO v_company_run_count
          FROM public.company_runs cr WHERE cr.job_id_legacy=v_row.job_id
            AND cr.source_company_key='domain:' || v_row.canonical_domain;
        IF v_company_run_count=1 THEN
            SELECT cr.* INTO v_company_run FROM public.company_runs cr
              WHERE cr.job_id_legacy=v_row.job_id
                AND cr.source_company_key='domain:' || v_row.canonical_domain LIMIT 1;
        END IF;

        IF COALESCE(v_analysis_status,v_job_status,v_worker_status)='done'
           AND v_company_run_count=1 AND v_company_run.company_id IS NOT NULL
           AND COALESCE(v_company_run.decision,'') NOT IN ('error','timeout') THEN
            v_state := 'succeeded';
            v_completed_at := COALESCE(v_company_run.created_at,v_company_run.run_created_at);
            v_terminal_result := jsonb_build_object(
              'external_company_id',v_row.external_company_id::TEXT,
              'rdi_company_id',v_company_run.company_id::TEXT,
              'composite_score',COALESCE(v_company_run.result_payload #>> '{ranking_result,composite_score}',
                 v_company_run.result_payload #>> '{summary_rows,0,composite_score}',v_company_run.composite_score::TEXT),
              'strategy_fit_score',COALESCE(v_company_run.result_payload #>> '{ranking_result,strategy_fit_score}',
                 v_company_run.result_payload #>> '{summary_rows,0,strategy_fit_score}'),
              'team_score',COALESCE(v_company_run.result_payload #>> '{ranking_result,team_score}',
                 v_company_run.result_payload #>> '{summary_rows,0,team_score}'),
              'upside_score',COALESCE(v_company_run.result_payload #>> '{ranking_result,upside_score}',
                 v_company_run.result_payload #>> '{summary_rows,0,upside_score}'),
              'rdi_bucket',COALESCE(v_company_run.result_payload #>> '{ranking_result,bucket}',
                 v_company_run.result_payload #>> '{summary_rows,0,bucket}'),
              'completed_at',v_completed_at,'pipeline_version',v_pipeline_version,
              'scoring_version',v_scoring_version
            );
        ELSIF COALESCE(v_analysis_status,v_job_status,v_worker_status) IN ('done','error','interrupted') THEN
            v_state := 'failed';
            v_completed_at := COALESCE(v_row.completed_at,v_analysis_status_at,v_job_status_at);
        ELSIF COALESCE(v_analysis_status,v_job_status,v_worker_status)='stopped' THEN
            v_state := 'cancelled';
            v_completed_at := COALESCE(v_row.completed_at,v_analysis_status_at,v_job_status_at);
        ELSIF COALESCE(v_job_status,v_worker_status) IN ('claimed','running','finalizing') THEN
            v_state := 'running';
        ELSIF COALESCE(v_job_status,v_worker_status)='queued' THEN
            v_state := 'queued';
        END IF;
    END IF;

    RETURN to_jsonb(v_row) || jsonb_build_object(
      'external_company_id',v_row.external_company_id::TEXT,
      'bundle_schema_version',v_bundle.schema_version,
      'parent_bundle_sha256',v_bundle.parent_bundle_sha256,
      'rdi_company_id',CASE WHEN v_state='succeeded' THEN v_company_run.company_id::TEXT ELSE NULL END,
      'lifecycle_state',v_state,
      'completed_at',COALESCE(v_completed_at,v_row.completed_at),
      'terminal_result',v_terminal_result,
      'safe_error_code',CASE WHEN v_state IN ('failed','cancelled')
          THEN COALESCE(v_row.safe_error_code,'analysis_failed') ELSE v_row.safe_error_code END,
      'safe_error_class',CASE WHEN v_state IN ('failed','cancelled')
          THEN 'terminal_analysis_failure' ELSE NULL END,
      'safe_error_message',CASE WHEN v_state IN ('failed','cancelled')
          THEN 'The RDI analysis ended without a publishable result.' ELSE NULL END
    );
END;
$$;

REVOKE ALL ON TABLE public.leadgen_evidence_bundles, public.leadgen_machine_v2_intakes,
    public.leadgen_machine_v2_events FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON public.leadgen_evidence_bundles TO service_role;
GRANT SELECT, INSERT, UPDATE ON public.leadgen_machine_v2_intakes TO service_role;
GRANT SELECT, INSERT ON public.leadgen_machine_v2_events TO service_role;

REVOKE ALL ON FUNCTION public.reject_leadgen_bundle_mutation() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.put_leadgen_machine_v2_evidence_bundle(JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.create_leadgen_machine_v2_intake(JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.reserve_leadgen_machine_v2_start(TEXT,TEXT,DATE,TEXT,TEXT,TEXT,INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.finalize_leadgen_machine_v2_start(TEXT,TEXT,TEXT,TEXT,TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.release_leadgen_machine_v2_start(TEXT,TEXT,TEXT,TEXT,TEXT,TIMESTAMPTZ) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_leadgen_machine_v2_evidence_bundle(TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_leadgen_machine_v2_lifecycle(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.put_leadgen_machine_v2_evidence_bundle(JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.create_leadgen_machine_v2_intake(JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.reserve_leadgen_machine_v2_start(TEXT,TEXT,DATE,TEXT,TEXT,TEXT,INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.finalize_leadgen_machine_v2_start(TEXT,TEXT,TEXT,TEXT,TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.release_leadgen_machine_v2_start(TEXT,TEXT,TEXT,TEXT,TEXT,TIMESTAMPTZ) TO service_role;
GRANT EXECUTE ON FUNCTION public.get_leadgen_machine_v2_evidence_bundle(TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.get_leadgen_machine_v2_lifecycle(TEXT) TO service_role;

COMMIT;
