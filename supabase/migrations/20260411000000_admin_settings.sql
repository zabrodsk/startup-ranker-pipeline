-- Admin settings: generic JSONB key/value store for admin-managed configuration.
-- Primary use case: persisted per-stage pipeline model defaults for portal re-evaluation.
--
-- Access model: service role (backend) only. RLS denies all public access; the backend
-- connects with the Supabase service role key which bypasses RLS by design.

CREATE TABLE IF NOT EXISTS admin_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by TEXT
);

ALTER TABLE admin_settings ENABLE ROW LEVEL SECURITY;

-- Deny-all to public; backend uses service role key, which bypasses RLS.
DROP POLICY IF EXISTS admin_settings_service_only ON admin_settings;
CREATE POLICY admin_settings_service_only ON admin_settings
    FOR ALL TO public USING (false) WITH CHECK (false);
