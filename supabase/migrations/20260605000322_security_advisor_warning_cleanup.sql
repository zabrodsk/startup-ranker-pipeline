-- Resolve Supabase Security Advisor warnings that are safe to fix in schema.
--
-- 1. Function Search Path Mutable:
--    The feedback trigger function only writes NEW.updated_at and calls now(),
--    so an empty search_path is safe and avoids role-mutable lookup behavior.
--
-- 2. RLS Policy Always True:
--    data_room_files is accessed by the backend with the service-role key. The
--    service_role already bypasses RLS, so the prior unrestricted policy was
--    unnecessary and made the table look publicly permissive if grants changed.

ALTER FUNCTION public.set_feedback_items_updated_at()
  SET search_path = '';

DROP POLICY IF EXISTS "Service role full access on data_room_files"
  ON public.data_room_files;

REVOKE ALL ON TABLE public.data_room_files FROM anon, authenticated;
