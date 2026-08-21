-- Durable, service-only circuit breaker for the shared Specter MCP quota.
-- Browser and machine clients read this state only through the RDI backend.

CREATE TABLE IF NOT EXISTS public.specter_mcp_quota_gate (
    target_environment TEXT PRIMARY KEY
        CHECK (target_environment IN ('staging', 'production')),
    state TEXT NOT NULL DEFAULT 'open'
        CHECK (state IN ('open', 'blocked', 'probing')),
    enforcement_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    blocked_at TIMESTAMPTZ,
    blocked_until TIMESTAMPTZ,
    next_probe_at TIMESTAMPTZ,
    reason_code TEXT,
    reset_hint TEXT,
    source_component TEXT,
    source_job_id TEXT,
    probe_lease_token TEXT,
    probe_lease_until TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT specter_mcp_quota_gate_block_fields CHECK (
        state = 'open'
        OR (
            blocked_at IS NOT NULL
            AND blocked_until IS NOT NULL
            AND next_probe_at IS NOT NULL
        )
    ),
    CONSTRAINT specter_mcp_quota_gate_probe_fields CHECK (
        state <> 'probing'
        OR (
            probe_lease_token IS NOT NULL
            AND probe_lease_until IS NOT NULL
        )
    )
);

ALTER TABLE public.specter_mcp_quota_gate ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.specter_mcp_quota_gate
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.specter_mcp_quota_gate
    TO service_role;

CREATE OR REPLACE FUNCTION public.get_specter_mcp_quota_gate(
    p_target_environment TEXT,
    p_enforcement_enabled BOOLEAN
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    v_row public.specter_mcp_quota_gate%ROWTYPE;
    v_retry_after BIGINT := 0;
BEGIN
    IF p_target_environment NOT IN ('staging', 'production') THEN
        RAISE EXCEPTION 'invalid Specter MCP gate environment';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('specter_mcp_gate:' || p_target_environment, 0)
    );

    UPDATE public.specter_mcp_quota_gate
    SET enforcement_enabled = p_enforcement_enabled,
        updated_at = v_now
    WHERE target_environment = p_target_environment
      AND enforcement_enabled IS DISTINCT FROM p_enforcement_enabled;

    SELECT * INTO v_row
    FROM public.specter_mcp_quota_gate
    WHERE target_environment = p_target_environment;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'provider', 'specter_mcp',
            'target_environment', p_target_environment,
            'state', 'open',
            'enforcement_enabled', p_enforcement_enabled,
            'accepting_new_analyses', TRUE,
            'quota_remaining', 'unknown',
            'blocked_until', NULL,
            'next_probe_at', NULL,
            'retry_after_seconds', 0,
            'reason_code', NULL,
            'observed_at', v_now
        );
    END IF;

    IF v_row.state IN ('blocked', 'probing') THEN
        v_retry_after := GREATEST(
            0,
            CEIL(EXTRACT(EPOCH FROM (COALESCE(v_row.next_probe_at, v_now) - v_now)))::BIGINT
        );
    END IF;

    RETURN jsonb_build_object(
        'provider', 'specter_mcp',
        'target_environment', v_row.target_environment,
        'state', v_row.state,
        'enforcement_enabled', v_row.enforcement_enabled,
        'accepting_new_analyses', (
            NOT v_row.enforcement_enabled OR v_row.state = 'open'
        ),
        'quota_remaining', 'unknown',
        'blocked_until', v_row.blocked_until,
        'next_probe_at', v_row.next_probe_at,
        'retry_after_seconds', v_retry_after,
        'reason_code', v_row.reason_code,
        'reset_hint', v_row.reset_hint,
        'source_component', v_row.source_component,
        'source_job_id', v_row.source_job_id,
        'probe_lease_until', v_row.probe_lease_until,
        'observed_at', v_now,
        'updated_at', v_row.updated_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.trip_specter_mcp_quota_gate(
    p_target_environment TEXT,
    p_enforcement_enabled BOOLEAN,
    p_reason_code TEXT,
    p_reset_hint TEXT,
    p_source_component TEXT,
    p_source_job_id TEXT DEFAULT NULL,
    p_retry_after_seconds INTEGER DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    v_blocked_until TIMESTAMPTZ;
BEGIN
    IF p_target_environment NOT IN ('staging', 'production') THEN
        RAISE EXCEPTION 'invalid Specter MCP gate environment';
    END IF;
    IF p_reason_code IS NULL OR length(p_reason_code) NOT BETWEEN 1 AND 64 THEN
        RAISE EXCEPTION 'invalid Specter MCP gate reason';
    END IF;
    IF p_source_component IS NULL OR length(p_source_component) NOT BETWEEN 1 AND 64 THEN
        RAISE EXCEPTION 'invalid Specter MCP gate source';
    END IF;

    -- Serialize gate trips against machine intake/start checks for this stack.
    PERFORM pg_advisory_xact_lock(
        hashtextextended('specter_mcp_gate:' || p_target_environment, 0)
    );
    IF p_retry_after_seconds IS NOT NULL
       AND p_retry_after_seconds NOT BETWEEN 1 AND 3600 THEN
        RAISE EXCEPTION 'invalid Specter MCP retry interval';
    END IF;

    v_blocked_until := CASE
        WHEN p_retry_after_seconds IS NOT NULL
            THEN v_now + make_interval(secs => p_retry_after_seconds)
        ELSE (
            date_trunc('day', v_now AT TIME ZONE 'UTC')
            + INTERVAL '1 day 5 minutes'
        ) AT TIME ZONE 'UTC'
    END;

    INSERT INTO public.specter_mcp_quota_gate (
        target_environment,
        state,
        enforcement_enabled,
        blocked_at,
        blocked_until,
        next_probe_at,
        reason_code,
        reset_hint,
        source_component,
        source_job_id,
        probe_lease_token,
        probe_lease_until,
        updated_at
    ) VALUES (
        p_target_environment,
        'blocked',
        p_enforcement_enabled,
        v_now,
        v_blocked_until,
        v_blocked_until,
        left(p_reason_code, 64),
        NULLIF(left(COALESCE(p_reset_hint, ''), 64), ''),
        left(p_source_component, 64),
        NULLIF(left(COALESCE(p_source_job_id, ''), 128), ''),
        NULL,
        NULL,
        v_now
    )
    ON CONFLICT (target_environment) DO UPDATE
    SET state = 'blocked',
        enforcement_enabled = EXCLUDED.enforcement_enabled,
        blocked_at = CASE
            WHEN public.specter_mcp_quota_gate.state IN ('blocked', 'probing')
                 AND public.specter_mcp_quota_gate.blocked_until > v_now
                THEN public.specter_mcp_quota_gate.blocked_at
            ELSE v_now
        END,
        blocked_until = GREATEST(
            COALESCE(public.specter_mcp_quota_gate.blocked_until, EXCLUDED.blocked_until),
            EXCLUDED.blocked_until
        ),
        next_probe_at = GREATEST(
            COALESCE(public.specter_mcp_quota_gate.next_probe_at, EXCLUDED.next_probe_at),
            EXCLUDED.next_probe_at
        ),
        reason_code = EXCLUDED.reason_code,
        reset_hint = EXCLUDED.reset_hint,
        source_component = EXCLUDED.source_component,
        source_job_id = EXCLUDED.source_job_id,
        probe_lease_token = NULL,
        probe_lease_until = NULL,
        updated_at = v_now;

    RETURN public.get_specter_mcp_quota_gate(
        p_target_environment,
        p_enforcement_enabled
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.acquire_specter_mcp_quota_probe(
    p_target_environment TEXT,
    p_enforcement_enabled BOOLEAN,
    p_probe_lease_token TEXT,
    p_lease_seconds INTEGER DEFAULT 60
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    v_row public.specter_mcp_quota_gate%ROWTYPE;
BEGIN
    IF p_target_environment NOT IN ('staging', 'production')
       OR p_probe_lease_token IS NULL
       OR length(p_probe_lease_token) NOT BETWEEN 16 AND 64
       OR p_lease_seconds NOT BETWEEN 10 AND 300 THEN
        RAISE EXCEPTION 'invalid Specter MCP probe lease request';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('specter_mcp_gate:' || p_target_environment, 0)
    );

    INSERT INTO public.specter_mcp_quota_gate (
        target_environment, state, enforcement_enabled, updated_at
    ) VALUES (
        p_target_environment, 'open', p_enforcement_enabled, v_now
    )
    ON CONFLICT (target_environment) DO NOTHING;

    SELECT * INTO v_row
    FROM public.specter_mcp_quota_gate
    WHERE target_environment = p_target_environment
    FOR UPDATE;

    UPDATE public.specter_mcp_quota_gate
    SET enforcement_enabled = p_enforcement_enabled,
        updated_at = v_now
    WHERE target_environment = p_target_environment;

    IF v_row.state = 'open' OR NOT p_enforcement_enabled THEN
        RETURN public.get_specter_mcp_quota_gate(
            p_target_environment,
            p_enforcement_enabled
        ) || jsonb_build_object('action', 'open');
    END IF;

    IF v_row.state = 'blocked' AND v_row.next_probe_at > v_now THEN
        RETURN public.get_specter_mcp_quota_gate(
            p_target_environment,
            p_enforcement_enabled
        ) || jsonb_build_object('action', 'blocked');
    END IF;

    IF v_row.state = 'probing' AND v_row.probe_lease_until > v_now THEN
        RETURN public.get_specter_mcp_quota_gate(
            p_target_environment,
            p_enforcement_enabled
        ) || jsonb_build_object('action', 'leased');
    END IF;

    UPDATE public.specter_mcp_quota_gate
    SET state = 'probing',
        probe_lease_token = p_probe_lease_token,
        probe_lease_until = v_now + make_interval(secs => p_lease_seconds),
        updated_at = v_now
    WHERE target_environment = p_target_environment;

    RETURN public.get_specter_mcp_quota_gate(
        p_target_environment,
        p_enforcement_enabled
    ) || jsonb_build_object(
        'action', 'acquired',
        'probe_lease_token', p_probe_lease_token
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.finish_specter_mcp_quota_probe(
    p_target_environment TEXT,
    p_enforcement_enabled BOOLEAN,
    p_probe_lease_token TEXT,
    p_succeeded BOOLEAN,
    p_reason_code TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_now TIMESTAMPTZ := clock_timestamp();
    v_row public.specter_mcp_quota_gate%ROWTYPE;
BEGIN
    IF p_target_environment NOT IN ('staging', 'production') THEN
        RAISE EXCEPTION 'invalid Specter MCP gate environment';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended('specter_mcp_gate:' || p_target_environment, 0)
    );

    SELECT * INTO v_row
    FROM public.specter_mcp_quota_gate
    WHERE target_environment = p_target_environment
    FOR UPDATE;

    IF NOT FOUND
       OR v_row.state <> 'probing'
       OR v_row.probe_lease_token IS DISTINCT FROM p_probe_lease_token THEN
        RETURN public.get_specter_mcp_quota_gate(
            p_target_environment,
            p_enforcement_enabled
        ) || jsonb_build_object('action', 'stale');
    END IF;

    IF p_succeeded THEN
        UPDATE public.specter_mcp_quota_gate
        SET state = 'open',
            enforcement_enabled = p_enforcement_enabled,
            blocked_at = NULL,
            blocked_until = NULL,
            next_probe_at = NULL,
            reason_code = NULL,
            reset_hint = NULL,
            source_component = 'recovery_probe',
            source_job_id = NULL,
            probe_lease_token = NULL,
            probe_lease_until = NULL,
            updated_at = v_now
        WHERE target_environment = p_target_environment;
    ELSE
        UPDATE public.specter_mcp_quota_gate
        SET state = 'blocked',
            enforcement_enabled = p_enforcement_enabled,
            blocked_until = v_now + INTERVAL '5 minutes',
            next_probe_at = v_now + INTERVAL '5 minutes',
            reason_code = left(
                COALESCE(NULLIF(p_reason_code, ''), 'specter_mcp_recovery_probe_failed'),
                64
            ),
            source_component = 'recovery_probe',
            source_job_id = NULL,
            probe_lease_token = NULL,
            probe_lease_until = NULL,
            updated_at = v_now
        WHERE target_environment = p_target_environment;
    END IF;

    RETURN public.get_specter_mcp_quota_gate(
        p_target_environment,
        p_enforcement_enabled
    ) || jsonb_build_object(
        'action', CASE WHEN p_succeeded THEN 'opened' ELSE 'reblocked' END
    );
END;
$$;

-- The trigger is the race-safe backstop for machine intake creation and start
-- reservation. Exact idempotent replays do not INSERT/UPDATE and remain valid.
CREATE OR REPLACE FUNCTION public.enforce_specter_mcp_gate_on_machine_intake()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_gate public.specter_mcp_quota_gate%ROWTYPE;
    v_existing public.leadgen_machine_intakes%ROWTYPE;
BEGIN
    IF TG_OP = 'UPDATE'
       AND NOT (
           OLD.lifecycle_state = 'accepted'
           AND NEW.lifecycle_state = 'start_fenced'
       ) THEN
        RETURN NEW;
    END IF;


    PERFORM pg_advisory_xact_lock(
        hashtextextended('specter_mcp_gate:' || NEW.target_environment, 0)
    );

    -- BEFORE INSERT fires ahead of ON CONFLICT. Permit a pre-existing
    -- idempotency identity so the RPC can return its established lifecycle
    -- (or its normal payload-conflict response) while blocking only new rows.
    IF TG_OP = 'INSERT' THEN
        SELECT * INTO v_existing
        FROM public.leadgen_machine_intakes
        WHERE idempotency_identity = NEW.idempotency_identity;
        IF FOUND THEN
            RETURN NEW;
        END IF;
    END IF;

    SELECT * INTO v_gate
    FROM public.specter_mcp_quota_gate
    WHERE target_environment = NEW.target_environment;

    IF FOUND
       AND v_gate.enforcement_enabled
       AND v_gate.state IN ('blocked', 'probing') THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'specter_mcp_quota_gate_blocked',
            DETAIL = jsonb_build_object(
                'blocked_until', v_gate.blocked_until,
                'next_probe_at', v_gate.next_probe_at,
                'reason_code', v_gate.reason_code
            )::TEXT;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS leadgen_machine_intake_specter_gate_insert
    ON public.leadgen_machine_intakes;
CREATE TRIGGER leadgen_machine_intake_specter_gate_insert
    BEFORE INSERT ON public.leadgen_machine_intakes
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_specter_mcp_gate_on_machine_intake();

DROP TRIGGER IF EXISTS leadgen_machine_intake_specter_gate_start
    ON public.leadgen_machine_intakes;
CREATE TRIGGER leadgen_machine_intake_specter_gate_start
    BEFORE UPDATE OF lifecycle_state ON public.leadgen_machine_intakes
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_specter_mcp_gate_on_machine_intake();

REVOKE ALL ON FUNCTION public.get_specter_mcp_quota_gate(TEXT, BOOLEAN)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.trip_specter_mcp_quota_gate(
    TEXT, BOOLEAN, TEXT, TEXT, TEXT, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.acquire_specter_mcp_quota_probe(
    TEXT, BOOLEAN, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.finish_specter_mcp_quota_probe(
    TEXT, BOOLEAN, TEXT, BOOLEAN, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.enforce_specter_mcp_gate_on_machine_intake()
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.get_specter_mcp_quota_gate(TEXT, BOOLEAN)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.trip_specter_mcp_quota_gate(
    TEXT, BOOLEAN, TEXT, TEXT, TEXT, TEXT, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.acquire_specter_mcp_quota_probe(
    TEXT, BOOLEAN, TEXT, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.finish_specter_mcp_quota_probe(
    TEXT, BOOLEAN, TEXT, BOOLEAN, TEXT
) TO service_role;
