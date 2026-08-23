-- The v2 reservation function must count legacy v1 starts to enforce one
-- combined daily cap. The legacy table intentionally has no direct
-- service_role grant, so keep that cross-generation read behind this narrowly
-- exposed, input-validating RPC instead of broadening table access.

ALTER FUNCTION public.reserve_leadgen_machine_v2_start(
    TEXT, TEXT, DATE, TEXT, TEXT, TEXT, INTEGER
) SECURITY DEFINER;

ALTER FUNCTION public.reserve_leadgen_machine_v2_start(
    TEXT, TEXT, DATE, TEXT, TEXT, TEXT, INTEGER
) SET search_path TO public, pg_temp;

REVOKE ALL ON FUNCTION public.reserve_leadgen_machine_v2_start(
    TEXT, TEXT, DATE, TEXT, TEXT, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.reserve_leadgen_machine_v2_start(
    TEXT, TEXT, DATE, TEXT, TEXT, TEXT, INTEGER
) TO service_role;
