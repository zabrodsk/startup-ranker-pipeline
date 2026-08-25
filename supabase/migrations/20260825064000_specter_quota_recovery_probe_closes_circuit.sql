BEGIN;

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
    ELSIF v_final_status = 'committed'
          AND v_auth.quota_class = 'recovery_probe'
          AND v_daily.circuit_state = 'probing' THEN
        UPDATE public.specter_quota_broker_daily
          SET circuit_state = 'closed',
              reason_code = NULL,
              retry_at = NULL,
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

COMMIT;
