BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;

CREATE TABLE public.specter_quota_broker_daily (
    target_environment TEXT NOT NULL CHECK (target_environment IN ('staging', 'production')),
    business_date DATE NOT NULL,
    business_timezone TEXT NOT NULL DEFAULT 'Europe/Prague'
        CHECK (business_timezone = 'Europe/Prague'),
    circuit_state TEXT NOT NULL DEFAULT 'closed'
        CHECK (circuit_state IN ('closed', 'open', 'probing')),
    reason_code TEXT CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$'),
    observed_limit INTEGER NOT NULL DEFAULT 250
        CHECK (observed_limit BETWEEN 50 AND 1000),
    safety_reserve INTEGER NOT NULL DEFAULT 25
        CHECK (safety_reserve BETWEEN 0 AND 250),
    company_cap INTEGER NOT NULL DEFAULT 8
        CHECK (company_cap BETWEEN 1 AND 32),
    founder_profile_cap INTEGER NOT NULL DEFAULT 3
        CHECK (founder_profile_cap BETWEEN 0 AND 16),
    scheduled_import_allowance INTEGER NOT NULL DEFAULT 40
        CHECK (scheduled_import_allowance BETWEEN 0 AND 250),
    recovery_allowance INTEGER NOT NULL DEFAULT 5
        CHECK (recovery_allowance BETWEEN 0 AND 50),
    retry_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (target_environment, business_date)
);

CREATE TABLE public.specter_quota_authorizations (
    authorization_id TEXT PRIMARY KEY
        CHECK (authorization_id ~ '^specter-auth-[0-9a-f]{64}$'),
    target_environment TEXT NOT NULL CHECK (target_environment IN ('staging', 'production')),
    business_date DATE NOT NULL,
    business_timezone TEXT NOT NULL CHECK (business_timezone = 'Europe/Prague'),
    consumer TEXT NOT NULL CHECK (consumer ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    company_ref TEXT CHECK (company_ref IS NULL OR length(company_ref) BETWEEN 1 AND 255),
    operation TEXT NOT NULL CHECK (operation ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    quota_class TEXT NOT NULL CHECK (
        quota_class IN (
            'flex', 'flexible_pool', 'manual_batch', 'promoted_candidate_refresh',
            'scheduled_import', 'recovery_probe', 'autonomous_campaign'
        )
    ),
    idempotency_key TEXT NOT NULL CHECK (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$'),
    intake_id TEXT,
    status TEXT NOT NULL CHECK (
        status IN (
            'reserved', 'denied', 'committed', 'released',
            'provider_quota_exhausted', 'provider_unavailable'
        )
    ),
    reason_code TEXT CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$'),
    estimated_remaining INTEGER NOT NULL CHECK (estimated_remaining >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    reserved_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    finalized_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (target_environment, business_date, consumer, operation, idempotency_key)
);

CREATE TABLE public.specter_quota_authorization_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    authorization_id TEXT NOT NULL
        REFERENCES public.specter_quota_authorizations(authorization_id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL CHECK (event_type IN ('reserve', 'deny', 'commit', 'release')),
    status TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

ALTER TABLE public.specter_quota_broker_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.specter_quota_authorizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.specter_quota_authorization_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY specter_quota_broker_daily_deny_clients
    ON public.specter_quota_broker_daily
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY specter_quota_authorizations_deny_clients
    ON public.specter_quota_authorizations
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY specter_quota_authorization_events_deny_clients
    ON public.specter_quota_authorization_events
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

CREATE OR REPLACE FUNCTION public._specter_quota_current_prague_date()
RETURNS DATE
LANGUAGE sql
STABLE
SET search_path TO public, pg_temp
AS $$
    SELECT (clock_timestamp() AT TIME ZONE 'Europe/Prague')::DATE;
$$;

CREATE OR REPLACE FUNCTION public._specter_quota_retry_at_for_next_prague_day()
RETURNS TIMESTAMPTZ
LANGUAGE sql
STABLE
SET search_path TO public, pg_temp
AS $$
    SELECT ((date_trunc('day', clock_timestamp() AT TIME ZONE 'Europe/Prague') + interval '1 day')
        AT TIME ZONE 'Europe/Prague');
$$;

CREATE OR REPLACE FUNCTION public._specter_quota_external_status(p_status TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
SET search_path TO public, pg_temp
AS $$
    SELECT CASE p_status
        WHEN 'reserved' THEN 'authorized'
        WHEN 'denied' THEN 'deferred'
        ELSE p_status
    END;
$$;

CREATE OR REPLACE FUNCTION public._specter_quota_payload(
    p_auth public.specter_quota_authorizations,
    p_daily public.specter_quota_broker_daily,
    p_enforcement_enabled BOOLEAN
) RETURNS JSONB
LANGUAGE sql
STABLE
SET search_path TO public, pg_temp
AS $$
    SELECT jsonb_build_object(
        'provider', 'specter_mcp',
        'authorization_id', p_auth.authorization_id,
        'status', public._specter_quota_external_status(p_auth.status),
        'status_internal', p_auth.status,
        'target_environment', p_auth.target_environment,
        'business_date', p_auth.business_date::TEXT,
        'circuit_state', p_daily.circuit_state,
        'reason', COALESCE(p_auth.reason_code, p_daily.reason_code),
        'reason_code', COALESCE(p_auth.reason_code, p_daily.reason_code),
        'retry_at', p_daily.retry_at,
        'estimated_remaining', p_auth.estimated_remaining,
        'quota_remaining', p_auth.estimated_remaining,
        'accepting_new_analyses',
            (NOT p_enforcement_enabled) OR p_daily.circuit_state = 'closed',
        'state', CASE p_daily.circuit_state
            WHEN 'closed' THEN 'open'
            WHEN 'open' THEN 'blocked'
            ELSE 'probing'
        END
    );
$$;

CREATE OR REPLACE FUNCTION public._specter_quota_circuit_payload(
    p_daily public.specter_quota_broker_daily,
    p_estimated_remaining INTEGER,
    p_enforcement_enabled BOOLEAN
) RETURNS JSONB
LANGUAGE sql
STABLE
SET search_path TO public, pg_temp
AS $$
    SELECT jsonb_build_object(
        'provider', 'specter_mcp',
        'target_environment', p_daily.target_environment,
        'business_date', p_daily.business_date::TEXT,
        'circuit_state', p_daily.circuit_state,
        'reason', p_daily.reason_code,
        'reason_code', p_daily.reason_code,
        'retry_at', p_daily.retry_at,
        'estimated_remaining', p_estimated_remaining,
        'quota_remaining', p_estimated_remaining,
        'accepting_new_analyses',
            (NOT p_enforcement_enabled) OR p_daily.circuit_state = 'closed',
        'state', CASE p_daily.circuit_state
            WHEN 'closed' THEN 'open'
            WHEN 'open' THEN 'blocked'
            ELSE 'probing'
        END
    );
$$;

CREATE OR REPLACE FUNCTION public._specter_quota_write_event(
    p_authorization_id TEXT,
    p_event_type TEXT,
    p_status TEXT,
    p_payload JSONB
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO public, pg_temp
AS $$
BEGIN
    INSERT INTO public.specter_quota_authorization_events(
        authorization_id, event_type, status, payload
    ) VALUES (
        p_authorization_id, p_event_type, p_status, COALESCE(p_payload, '{}'::JSONB)
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.get_specter_quota_broker_circuit(
    p_target_environment TEXT,
    p_business_date DATE,
    p_enforcement_enabled BOOLEAN
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO public, pg_temp
AS $$
DECLARE
    v_daily public.specter_quota_broker_daily%ROWTYPE;
    v_active_count INTEGER := 0;
    v_remaining INTEGER := 250;
BEGIN
    SELECT * INTO v_daily
      FROM public.specter_quota_broker_daily
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date;
    IF NOT FOUND THEN
        INSERT INTO public.specter_quota_broker_daily(target_environment, business_date)
        VALUES (p_target_environment, p_business_date)
        ON CONFLICT (target_environment, business_date) DO NOTHING;
        SELECT * INTO v_daily
          FROM public.specter_quota_broker_daily
          WHERE target_environment = p_target_environment
            AND business_date = p_business_date;
    END IF;

    SELECT count(*)::INTEGER INTO v_active_count
      FROM public.specter_quota_authorizations
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
        AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');
    v_remaining := greatest(v_daily.observed_limit - v_active_count, 0);

    RETURN public._specter_quota_circuit_payload(v_daily, v_remaining, p_enforcement_enabled);
END;
$$;

CREATE OR REPLACE FUNCTION public.reserve_specter_quota_authorization(
    p_target_environment TEXT,
    p_business_date DATE,
    p_business_timezone TEXT,
    p_consumer TEXT,
    p_company_ref TEXT,
    p_operation TEXT,
    p_quota_class TEXT,
    p_idempotency_key TEXT,
    p_enforcement_enabled BOOLEAN,
    p_remaining_rdi_slots INTEGER,
    p_actor TEXT,
    p_metadata JSONB DEFAULT '{}'::JSONB
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO public, pg_temp
AS $$
DECLARE
    v_daily public.specter_quota_broker_daily%ROWTYPE;
    v_auth public.specter_quota_authorizations%ROWTYPE;
    v_active_count INTEGER := 0;
    v_company_count INTEGER := 0;
    v_founder_count INTEGER := 0;
    v_import_count INTEGER := 0;
    v_recovery_count INTEGER := 0;
    v_campaign_count INTEGER := 0;
    v_campaign_reserve_target INTEGER := 0;
    v_floor INTEGER := 0;
    v_reason TEXT := NULL;
    v_effective_company_ref TEXT := NULLIF(trim(COALESCE(p_company_ref, '')), '');
    v_estimated_remaining INTEGER := 0;
    v_authorization_material TEXT;
BEGIN
    IF p_target_environment NOT IN ('staging', 'production')
       OR p_business_timezone <> 'Europe/Prague'
       OR p_business_date <> public._specter_quota_current_prague_date() THEN
        RAISE EXCEPTION 'invalid specter quota scope' USING ERRCODE = '22023';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('specter-quota:' || p_target_environment || ':' || p_business_date::TEXT, 0)
    );

    INSERT INTO public.specter_quota_broker_daily(target_environment, business_date)
    VALUES (p_target_environment, p_business_date)
    ON CONFLICT (target_environment, business_date) DO NOTHING;

    SELECT * INTO v_daily
      FROM public.specter_quota_broker_daily
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
      FOR UPDATE;

    SELECT * INTO v_auth
      FROM public.specter_quota_authorizations
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
        AND consumer = p_consumer
        AND operation = p_operation
        AND idempotency_key = p_idempotency_key;
    IF FOUND THEN
        RETURN public._specter_quota_payload(v_auth, v_daily, p_enforcement_enabled);
    END IF;

    SELECT count(*)::INTEGER INTO v_active_count
      FROM public.specter_quota_authorizations
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
        AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');
    IF v_effective_company_ref IS NOT NULL THEN
        SELECT count(*)::INTEGER INTO v_company_count
          FROM public.specter_quota_authorizations
          WHERE target_environment = p_target_environment
            AND business_date = p_business_date
            AND company_ref = v_effective_company_ref
            AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');
        SELECT count(*)::INTEGER INTO v_founder_count
          FROM public.specter_quota_authorizations
          WHERE target_environment = p_target_environment
            AND business_date = p_business_date
            AND company_ref = v_effective_company_ref
            AND operation = 'get_person_profile'
            AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');
    END IF;
    SELECT count(*)::INTEGER INTO v_import_count
      FROM public.specter_quota_authorizations
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
        AND quota_class = 'scheduled_import'
        AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');
    SELECT count(*)::INTEGER INTO v_recovery_count
      FROM public.specter_quota_authorizations
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
        AND quota_class = 'recovery_probe'
        AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');
    SELECT count(*)::INTEGER INTO v_campaign_count
      FROM public.specter_quota_authorizations
      WHERE target_environment = p_target_environment
        AND business_date = p_business_date
        AND quota_class = 'autonomous_campaign'
        AND status IN ('reserved', 'committed', 'provider_quota_exhausted', 'provider_unavailable');

    v_campaign_reserve_target := least(160, greatest(COALESCE(p_remaining_rdi_slots, 20), 0) * 8);

    IF p_enforcement_enabled AND v_daily.circuit_state <> 'closed' THEN
        v_reason := COALESCE(v_daily.reason_code, 'specter_mcp_quota_exhausted');
    ELSIF v_company_count >= v_daily.company_cap THEN
        v_reason := 'company_cap_exhausted';
    ELSIF p_operation = 'get_person_profile' AND v_founder_count >= v_daily.founder_profile_cap THEN
        v_reason := 'founder_profile_cap_exhausted';
    ELSIF p_quota_class = 'scheduled_import' AND v_import_count >= v_daily.scheduled_import_allowance THEN
        v_reason := 'scheduled_import_allowance_exhausted';
    ELSIF p_quota_class = 'recovery_probe' AND v_recovery_count >= v_daily.recovery_allowance THEN
        v_reason := 'recovery_allowance_exhausted';
    ELSE
        IF p_quota_class = 'autonomous_campaign' THEN
            v_floor := v_daily.safety_reserve
                + greatest(v_daily.scheduled_import_allowance - v_import_count, 0)
                + greatest(v_daily.recovery_allowance - v_recovery_count, 0);
        ELSIF p_quota_class = 'scheduled_import' THEN
            v_floor := v_daily.safety_reserve
                + greatest(v_daily.recovery_allowance - v_recovery_count, 0)
                + greatest(v_campaign_reserve_target - v_campaign_count, 0);
        ELSIF p_quota_class = 'recovery_probe' THEN
            v_floor := v_daily.safety_reserve
                + greatest(v_daily.scheduled_import_allowance - v_import_count, 0)
                + greatest(v_campaign_reserve_target - v_campaign_count, 0);
        ELSE
            v_floor := v_daily.safety_reserve
                + greatest(v_daily.scheduled_import_allowance - v_import_count, 0)
                + greatest(v_daily.recovery_allowance - v_recovery_count, 0)
                + greatest(v_campaign_reserve_target - v_campaign_count, 0);
        END IF;
        IF v_active_count >= (v_daily.observed_limit - v_floor) THEN
            v_reason := 'quota_estimate_exhausted';
        END IF;
    END IF;

    v_estimated_remaining := greatest(
        v_daily.observed_limit - v_active_count - CASE WHEN v_reason IS NULL THEN 1 ELSE 0 END,
        0
    );
    v_authorization_material := concat_ws(
        E'\x1f',
        p_target_environment,
        p_business_date::TEXT,
        p_consumer,
        COALESCE(v_effective_company_ref, ''),
        p_operation,
        p_quota_class,
        p_idempotency_key
    );

    INSERT INTO public.specter_quota_authorizations(
        authorization_id, target_environment, business_date, business_timezone,
        consumer, company_ref, operation, quota_class, idempotency_key, intake_id,
        status, reason_code, estimated_remaining, metadata
    ) VALUES (
        'specter-auth-' || encode(digest(v_authorization_material, 'sha256'), 'hex'),
        p_target_environment, p_business_date, p_business_timezone,
        p_consumer, v_effective_company_ref, p_operation, p_quota_class, p_idempotency_key,
        NULLIF(COALESCE(p_metadata ->> 'intake_id', ''), ''),
        CASE WHEN v_reason IS NULL THEN 'reserved' ELSE 'denied' END,
        v_reason,
        v_estimated_remaining,
        COALESCE(p_metadata, '{}'::JSONB)
    )
    RETURNING * INTO v_auth;

    PERFORM public._specter_quota_write_event(
        v_auth.authorization_id,
        CASE WHEN v_reason IS NULL THEN 'reserve' ELSE 'deny' END,
        v_auth.status,
        jsonb_build_object(
            'consumer', v_auth.consumer,
            'operation', v_auth.operation,
            'quota_class', v_auth.quota_class,
            'company_ref', v_auth.company_ref,
            'reason_code', v_auth.reason_code,
            'estimated_remaining', v_auth.estimated_remaining
        )
    );

    RETURN public._specter_quota_payload(v_auth, v_daily, p_enforcement_enabled);
END;
$$;

CREATE OR REPLACE FUNCTION public.commit_specter_quota_authorization(
    p_authorization_id TEXT,
    p_target_environment TEXT,
    p_operation TEXT,
    p_outcome TEXT,
    p_provider_quota_error BOOLEAN DEFAULT FALSE,
    p_reason_code TEXT DEFAULT NULL,
    p_intake_id TEXT DEFAULT NULL,
    p_actor TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO public, pg_temp
AS $$
DECLARE
    v_auth public.specter_quota_authorizations%ROWTYPE;
    v_daily public.specter_quota_broker_daily%ROWTYPE;
    v_final_status TEXT;
BEGIN
    IF p_outcome NOT IN ('succeeded', 'failed') THEN
        RAISE EXCEPTION 'invalid specter quota commit outcome' USING ERRCODE = '22023';
    END IF;

    SELECT * INTO v_auth
      FROM public.specter_quota_authorizations
      WHERE authorization_id = p_authorization_id
        AND target_environment = p_target_environment
        AND operation = p_operation
      FOR UPDATE;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    IF (v_auth.intake_id IS NOT NULL AND p_intake_id IS NULL)
       OR (p_intake_id IS NOT NULL AND COALESCE(v_auth.intake_id, p_intake_id) <> p_intake_id) THEN
        RAISE EXCEPTION 'specter quota authorization intake mismatch' USING ERRCODE = '22023';
    END IF;

    SELECT * INTO v_daily
      FROM public.specter_quota_broker_daily
      WHERE target_environment = v_auth.target_environment
        AND business_date = v_auth.business_date
      FOR UPDATE;

    IF v_auth.status <> 'reserved' THEN
        RETURN public._specter_quota_payload(v_auth, v_daily, true);
    END IF;

    v_final_status := CASE
        WHEN p_provider_quota_error THEN 'provider_quota_exhausted'
        WHEN p_outcome = 'succeeded' THEN 'committed'
        WHEN p_provider_quota_error THEN 'provider_quota_exhausted'
        ELSE 'provider_unavailable'
    END;

    UPDATE public.specter_quota_authorizations
      SET status = v_final_status,
          reason_code = CASE
              WHEN p_provider_quota_error THEN COALESCE(NULLIF(p_reason_code, ''), 'specter_mcp_quota_exhausted')
              WHEN v_final_status = 'provider_unavailable' THEN COALESCE(NULLIF(p_reason_code, ''), 'specter_mcp_unavailable')
              ELSE reason_code
          END,
          finalized_at = clock_timestamp(),
          updated_at = clock_timestamp()
      WHERE authorization_id = p_authorization_id
      RETURNING * INTO v_auth;

    IF v_final_status = 'provider_quota_exhausted' THEN
        UPDATE public.specter_quota_broker_daily
          SET circuit_state = 'open',
              reason_code = COALESCE(v_auth.reason_code, 'specter_mcp_quota_exhausted'),
              retry_at = public._specter_quota_retry_at_for_next_prague_day(),
              updated_at = clock_timestamp()
          WHERE target_environment = v_auth.target_environment
            AND business_date = v_auth.business_date
          RETURNING * INTO v_daily;
    ELSIF v_final_status = 'provider_unavailable' THEN
        UPDATE public.specter_quota_broker_daily
          SET circuit_state = CASE WHEN circuit_state = 'closed' THEN 'probing' ELSE circuit_state END,
              reason_code = COALESCE(v_auth.reason_code, 'specter_mcp_unavailable'),
              retry_at = CASE WHEN circuit_state = 'closed' THEN clock_timestamp() + interval '5 minutes' ELSE retry_at END,
              updated_at = clock_timestamp()
          WHERE target_environment = v_auth.target_environment
            AND business_date = v_auth.business_date
          RETURNING * INTO v_daily;
    END IF;

    PERFORM public._specter_quota_write_event(
        v_auth.authorization_id,
        'commit',
        v_auth.status,
        jsonb_build_object(
            'operation', v_auth.operation,
            'reason_code', v_auth.reason_code
        )
    );

    RETURN public._specter_quota_payload(v_auth, v_daily, true);
END;
$$;

CREATE OR REPLACE FUNCTION public.release_specter_quota_authorization(
    p_authorization_id TEXT,
    p_target_environment TEXT,
    p_operation TEXT,
    p_reason_code TEXT DEFAULT NULL,
    p_intake_id TEXT DEFAULT NULL,
    p_actor TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO public, pg_temp
AS $$
DECLARE
    v_auth public.specter_quota_authorizations%ROWTYPE;
    v_daily public.specter_quota_broker_daily%ROWTYPE;
BEGIN
    SELECT * INTO v_auth
      FROM public.specter_quota_authorizations
      WHERE authorization_id = p_authorization_id
        AND target_environment = p_target_environment
        AND operation = p_operation
      FOR UPDATE;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    IF (v_auth.intake_id IS NOT NULL AND p_intake_id IS NULL)
       OR (p_intake_id IS NOT NULL AND COALESCE(v_auth.intake_id, p_intake_id) <> p_intake_id) THEN
        RAISE EXCEPTION 'specter quota authorization intake mismatch' USING ERRCODE = '22023';
    END IF;

    SELECT * INTO v_daily
      FROM public.specter_quota_broker_daily
      WHERE target_environment = v_auth.target_environment
        AND business_date = v_auth.business_date;

    IF v_auth.status <> 'reserved' THEN
        RETURN public._specter_quota_payload(v_auth, v_daily, true);
    END IF;

    UPDATE public.specter_quota_authorizations
      SET status = 'released',
          reason_code = COALESCE(NULLIF(p_reason_code, ''), reason_code),
          finalized_at = clock_timestamp(),
          updated_at = clock_timestamp()
      WHERE authorization_id = p_authorization_id
      RETURNING * INTO v_auth;

    PERFORM public._specter_quota_write_event(
        v_auth.authorization_id,
        'release',
        v_auth.status,
        jsonb_build_object(
            'operation', v_auth.operation,
            'reason_code', v_auth.reason_code
        )
    );

    RETURN public._specter_quota_payload(v_auth, v_daily, true);
END;
$$;

REVOKE ALL ON TABLE public.specter_quota_broker_daily,
    public.specter_quota_authorizations,
    public.specter_quota_authorization_events
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON public.specter_quota_broker_daily TO service_role;
GRANT SELECT, INSERT, UPDATE ON public.specter_quota_authorizations TO service_role;
GRANT SELECT, INSERT ON public.specter_quota_authorization_events TO service_role;

REVOKE ALL ON FUNCTION public._specter_quota_write_event(TEXT, TEXT, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_specter_quota_broker_circuit(TEXT, DATE, BOOLEAN)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.reserve_specter_quota_authorization(TEXT, DATE, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BOOLEAN, INTEGER, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.commit_specter_quota_authorization(TEXT, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.release_specter_quota_authorization(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.get_specter_quota_broker_circuit(TEXT, DATE, BOOLEAN) TO service_role;
GRANT EXECUTE ON FUNCTION public.reserve_specter_quota_authorization(TEXT, DATE, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BOOLEAN, INTEGER, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.commit_specter_quota_authorization(TEXT, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.release_specter_quota_authorization(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT) TO service_role;

COMMIT;
