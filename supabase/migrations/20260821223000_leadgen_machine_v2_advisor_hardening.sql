-- Follow-up for testing environments that received the initial v2 migration
-- before the advisor-driven indexes and explicit client-deny policies landed.
BEGIN;

CREATE INDEX IF NOT EXISTS idx_leadgen_evidence_bundles_parent
    ON public.leadgen_evidence_bundles(parent_bundle_sha256)
    WHERE parent_bundle_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_leadgen_machine_v2_bundle
    ON public.leadgen_machine_v2_intakes(evidence_bundle_sha256);

DROP POLICY IF EXISTS leadgen_evidence_bundles_deny_clients
    ON public.leadgen_evidence_bundles;
CREATE POLICY leadgen_evidence_bundles_deny_clients
    ON public.leadgen_evidence_bundles
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

DROP POLICY IF EXISTS leadgen_machine_v2_intakes_deny_clients
    ON public.leadgen_machine_v2_intakes;
CREATE POLICY leadgen_machine_v2_intakes_deny_clients
    ON public.leadgen_machine_v2_intakes
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

DROP POLICY IF EXISTS leadgen_machine_v2_events_deny_clients
    ON public.leadgen_machine_v2_events;
CREATE POLICY leadgen_machine_v2_events_deny_clients
    ON public.leadgen_machine_v2_events
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

COMMIT;
