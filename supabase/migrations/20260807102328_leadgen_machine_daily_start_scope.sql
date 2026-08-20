-- Correct the LeadGen machine safety ceiling from a lifetime environment cap
-- to an immutable Europe/Prague daily scope. This migration is forward-only
-- because the prior lifecycle migration is already applied in staging.
--
-- Emergency rollback: set RDI_LEADGEN_AUTOSTART_ENABLED=false. Do not remove
-- scope columns or delete lifecycle rows; status/result reconciliation remains
-- safe while new starts are disabled.

BEGIN;

CREATE TABLE IF NOT EXISTS public.leadgen_machine_daily_scopes (
    target_environment TEXT NOT NULL
        CHECK (target_environment IN ('staging', 'production')),
    business_date DATE NOT NULL,
    business_timezone TEXT NOT NULL DEFAULT 'Europe/Prague'
        CHECK (business_timezone = 'Europe/Prague'),
    canonical_campaign_id TEXT NOT NULL
        CHECK (canonical_campaign_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target_environment, business_date)
);

ALTER TABLE public.leadgen_machine_intakes
    ADD COLUMN IF NOT EXISTS business_date DATE,
    ADD COLUMN IF NOT EXISTS business_timezone TEXT;

INSERT INTO public.leadgen_machine_daily_scopes (
    target_environment,
    business_date,
    business_timezone,
    canonical_campaign_id,
    created_at
)
SELECT
    target_environment,
    (created_at AT TIME ZONE 'Europe/Prague')::DATE,
    'Europe/Prague',
    min(campaign_id),
    min(created_at)
FROM public.leadgen_machine_intakes
GROUP BY
    target_environment,
    (created_at AT TIME ZONE 'Europe/Prague')::DATE
ON CONFLICT (target_environment, business_date) DO NOTHING;

UPDATE public.leadgen_machine_intakes
SET business_date = (created_at AT TIME ZONE 'Europe/Prague')::DATE,
    business_timezone = 'Europe/Prague'
WHERE business_date IS NULL
   OR business_timezone IS NULL;

ALTER TABLE public.leadgen_machine_intakes
    ALTER COLUMN business_date SET NOT NULL,
    ALTER COLUMN business_timezone SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'leadgen_machine_intakes_business_timezone_check'
          AND conrelid = 'public.leadgen_machine_intakes'::regclass
    ) THEN
        ALTER TABLE public.leadgen_machine_intakes
            ADD CONSTRAINT leadgen_machine_intakes_business_timezone_check
            CHECK (business_timezone = 'Europe/Prague');
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'leadgen_machine_intakes_daily_scope_fk'
          AND conrelid = 'public.leadgen_machine_intakes'::regclass
    ) THEN
        ALTER TABLE public.leadgen_machine_intakes
            ADD CONSTRAINT leadgen_machine_intakes_daily_scope_fk
            FOREIGN KEY (target_environment, business_date)
            REFERENCES public.leadgen_machine_daily_scopes (
                target_environment,
                business_date
            )
            ON DELETE RESTRICT;
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_leadgen_machine_intakes_daily_scope_state
    ON public.leadgen_machine_intakes (
        target_environment,
        business_date,
        lifecycle_state
    );

ALTER TABLE public.leadgen_machine_daily_scopes ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.enforce_leadgen_machine_scope_immutability()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NEW.target_environment IS DISTINCT FROM OLD.target_environment
       OR NEW.business_date IS DISTINCT FROM OLD.business_date
       OR NEW.business_timezone IS DISTINCT FROM OLD.business_timezone
       OR NEW.campaign_id IS DISTINCT FROM OLD.campaign_id THEN
        RAISE EXCEPTION 'LeadGen machine daily scope is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS leadgen_machine_intake_scope_immutable
    ON public.leadgen_machine_intakes;
CREATE TRIGGER leadgen_machine_intake_scope_immutable
    BEFORE UPDATE ON public.leadgen_machine_intakes
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_leadgen_machine_scope_immutability();

CREATE OR REPLACE FUNCTION public.create_leadgen_machine_intake(
    p_record JSONB
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
    v_business_date DATE;
    v_target_environment TEXT;
    v_campaign_id TEXT;
BEGIN
    SELECT *
    INTO v_row
    FROM public.leadgen_machine_intakes
    WHERE idempotency_identity = p_record ->> 'idempotency_identity';

    IF FOUND THEN
        IF v_row.payload_hash <> p_record ->> 'payload_hash' THEN
            RETURN jsonb_build_object('action', 'conflict') || to_jsonb(v_row);
        END IF;
        RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row);
    END IF;

    v_target_environment := p_record ->> 'target_environment';
    v_campaign_id := p_record ->> 'campaign_id';
    BEGIN
        v_business_date := (p_record ->> 'business_date')::DATE;
    EXCEPTION WHEN invalid_text_representation OR datetime_field_overflow THEN
        RAISE EXCEPTION 'invalid LeadGen machine business date'
            USING ERRCODE = '22023';
    END;

    IF v_target_environment NOT IN ('staging', 'production')
       OR p_record ->> 'business_timezone' <> 'Europe/Prague'
       OR v_business_date IS NULL
       OR v_business_date <> (clock_timestamp() AT TIME ZONE 'Europe/Prague')::DATE THEN
        RAISE EXCEPTION 'LeadGen machine intake is outside the current Prague business date'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            v_target_environment || ':' || v_business_date::TEXT,
            0
        )
    );

    INSERT INTO public.leadgen_machine_daily_scopes (
        target_environment,
        business_date,
        business_timezone,
        canonical_campaign_id
    ) VALUES (
        v_target_environment,
        v_business_date,
        'Europe/Prague',
        v_campaign_id
    )
    ON CONFLICT (target_environment, business_date) DO NOTHING;

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
        business_date,
        business_timezone,
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
        v_campaign_id,
        p_record ->> 'iteration_id',
        p_record ->> 'source_run_id',
        p_record ->> 'batch_id',
        p_record ->> 'idempotency_key',
        v_target_environment,
        v_business_date,
        'Europe/Prague',
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
                'payload_hash', v_row.payload_hash,
                'business_date', v_row.business_date,
                'business_timezone', v_row.business_timezone
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

REVOKE ALL ON FUNCTION public.reserve_leadgen_machine_start(
    TEXT, TEXT, TEXT, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated, service_role;
DROP FUNCTION IF EXISTS public.reserve_leadgen_machine_start(
    TEXT, TEXT, TEXT, TEXT, INTEGER
);

CREATE FUNCTION public.reserve_leadgen_machine_start(
    p_intake_id TEXT,
    p_target_environment TEXT,
    p_business_date DATE,
    p_business_timezone TEXT,
    p_job_id TEXT,
    p_actor TEXT,
    p_daily_start_limit INTEGER
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.leadgen_machine_intakes%ROWTYPE;
    v_started_count INTEGER;
    v_capacity JSONB;
BEGIN
    IF p_actor <> 'service:rockaway-leadgen' THEN
        RAISE EXCEPTION 'invalid machine actor' USING ERRCODE = '42501';
    END IF;
    IF p_target_environment NOT IN ('staging', 'production') THEN
        RAISE EXCEPTION 'invalid target environment' USING ERRCODE = '22023';
    END IF;
    IF p_business_date IS NULL OR p_business_timezone <> 'Europe/Prague' THEN
        RAISE EXCEPTION 'invalid daily scope' USING ERRCODE = '22023';
    END IF;
    IF p_daily_start_limit < 1 OR p_daily_start_limit > 20 THEN
        RAISE EXCEPTION 'invalid daily start limit' USING ERRCODE = '22023';
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
    IF v_row.business_date <> p_business_date
       OR v_row.business_timezone <> p_business_timezone THEN
        RETURN jsonb_build_object('action', 'scope_mismatch') || to_jsonb(v_row);
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            v_row.target_environment || ':' || v_row.business_date::TEXT,
            0
        )
    );

    SELECT count(*)::INTEGER
    INTO v_started_count
    FROM public.leadgen_machine_intakes
    WHERE target_environment = v_row.target_environment
      AND business_date = v_row.business_date
      AND lifecycle_state IN (
          'start_fenced', 'uncertain', 'queued', 'running',
          'succeeded', 'failed', 'cancelled'
      );

    v_capacity := jsonb_build_object(
        'daily_start_limit', p_daily_start_limit,
        'daily_started_count', v_started_count,
        'daily_remaining_capacity', greatest(
            p_daily_start_limit - v_started_count,
            0
        )
    );

    IF v_row.lifecycle_state IN ('start_fenced', 'uncertain', 'queued', 'running') THEN
        RETURN jsonb_build_object('action', 'existing') || to_jsonb(v_row) || v_capacity;
    END IF;
    IF v_row.lifecycle_state IN ('rejected', 'failed', 'cancelled', 'succeeded') THEN
        RETURN jsonb_build_object('action', 'terminal_invalid') || to_jsonb(v_row) || v_capacity;
    END IF;
    IF v_row.business_date <>
       (clock_timestamp() AT TIME ZONE 'Europe/Prague')::DATE THEN
        RETURN jsonb_build_object('action', 'business_date_closed') || to_jsonb(v_row) || v_capacity;
    END IF;
    IF v_started_count >= p_daily_start_limit THEN
        RETURN jsonb_build_object('action', 'rate_limited') || to_jsonb(v_row) || v_capacity;
    END IF;

    UPDATE public.leadgen_machine_intakes
    SET lifecycle_state = 'start_fenced',
        job_id = p_job_id,
        start_actor = p_actor,
        started_at = COALESCE(started_at, now()),
        updated_at = now()
    WHERE intake_id = p_intake_id
    RETURNING * INTO v_row;

    v_started_count := v_started_count + 1;
    v_capacity := jsonb_build_object(
        'daily_start_limit', p_daily_start_limit,
        'daily_started_count', v_started_count,
        'daily_remaining_capacity', greatest(
            p_daily_start_limit - v_started_count,
            0
        )
    );

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
            'business_date', v_row.business_date,
            'business_timezone', v_row.business_timezone,
            'campaign_id', v_row.campaign_id,
            'daily_start_limit', p_daily_start_limit,
            'daily_started_count', v_started_count,
            'daily_remaining_capacity', greatest(
                p_daily_start_limit - v_started_count,
                0
            )
        )
    );

    RETURN jsonb_build_object('action', 'reserved') || to_jsonb(v_row) || v_capacity;
END;
$$;

CREATE OR REPLACE FUNCTION public.get_leadgen_machine_daily_capacity(
    p_target_environment TEXT,
    p_business_date DATE,
    p_daily_start_limit INTEGER
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_started_count INTEGER;
BEGIN
    IF p_target_environment NOT IN ('staging', 'production')
       OR p_business_date IS NULL
       OR p_daily_start_limit < 1
       OR p_daily_start_limit > 20 THEN
        RAISE EXCEPTION 'invalid daily capacity scope' USING ERRCODE = '22023';
    END IF;

    SELECT count(*)::INTEGER
    INTO v_started_count
    FROM public.leadgen_machine_intakes
    WHERE target_environment = p_target_environment
      AND business_date = p_business_date
      AND lifecycle_state IN (
          'start_fenced', 'uncertain', 'queued', 'running',
          'succeeded', 'failed', 'cancelled'
      );

    RETURN jsonb_build_object(
        'target_environment', p_target_environment,
        'business_date', p_business_date,
        'business_timezone', 'Europe/Prague',
        'daily_start_limit', p_daily_start_limit,
        'daily_started_count', v_started_count,
        'daily_remaining_capacity', greatest(
            p_daily_start_limit - v_started_count,
            0
        )
    );
END;
$$;

REVOKE ALL ON TABLE public.leadgen_machine_daily_scopes
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON TABLE public.leadgen_machine_intakes
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON TABLE public.leadgen_machine_events
    FROM PUBLIC, anon, authenticated, service_role;

REVOKE ALL ON FUNCTION public.enforce_leadgen_machine_scope_immutability()
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.create_leadgen_machine_intake(JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.reserve_leadgen_machine_start(
    TEXT, TEXT, DATE, TEXT, TEXT, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.get_leadgen_machine_daily_capacity(
    TEXT, DATE, INTEGER
) FROM PUBLIC, anon, authenticated, service_role;

GRANT EXECUTE ON FUNCTION public.create_leadgen_machine_intake(JSONB)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.reserve_leadgen_machine_start(
    TEXT, TEXT, DATE, TEXT, TEXT, TEXT, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.get_leadgen_machine_daily_capacity(
    TEXT, DATE, INTEGER
) TO service_role;

COMMIT;
