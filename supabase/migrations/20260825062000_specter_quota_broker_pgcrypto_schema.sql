BEGIN;

ALTER FUNCTION public.reserve_specter_quota_authorization(
    TEXT, DATE, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BOOLEAN, INTEGER, TEXT, JSONB
) SET search_path TO public, extensions, pg_temp;

COMMIT;
