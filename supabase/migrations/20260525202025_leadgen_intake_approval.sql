-- Human approval gate for LeadGen batches before Deal Intelligence execution.
-- These tables are written/read only by the backend service-role client.

CREATE TABLE IF NOT EXISTS public.leadgen_intake_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id TEXT NOT NULL UNIQUE,
    generated_at TIMESTAMPTZ,
    scoring_version TEXT,
    source TEXT NOT NULL DEFAULT 'leadgen',
    thesis_key TEXT NOT NULL DEFAULT 'rockaway',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'queued', 'approved', 'partially_approved', 'rejected')),
    lead_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    payload_hash TEXT NOT NULL,
    summary JSONB NOT NULL DEFAULT '{}',
    job_id_legacy TEXT,
    approved_at TIMESTAMPTZ,
    approved_by_user_id TEXT,
    approved_by_email TEXT,
    approved_by_display_name TEXT,
    approved_by_label TEXT,
    rejected_at TIMESTAMPTZ,
    rejected_by_user_id TEXT,
    rejected_by_email TEXT,
    rejected_by_display_name TEXT,
    rejected_by_label TEXT,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.leadgen_intake_leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intake_id UUID NOT NULL REFERENCES public.leadgen_intake_batches(id) ON DELETE CASCADE,
    input_order INTEGER NOT NULL,
    company_name TEXT,
    domain TEXT,
    url TEXT,
    leadgen_score DOUBLE PRECISION,
    leadgen_bucket TEXT,
    thesis_key TEXT,
    thesis_status TEXT,
    eligible BOOLEAN NOT NULL DEFAULT false,
    approval_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (approval_status IN ('pending', 'approved', 'rejected', 'duplicate', 'invalid', 'ineligible')),
    rejection_reason TEXT,
    duplicate_of_url TEXT,
    raw_lead JSONB NOT NULL DEFAULT '{}',
    raw_score JSONB NOT NULL DEFAULT '{}',
    context_payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.leadgen_intake_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intake_id UUID NOT NULL REFERENCES public.leadgen_intake_batches(id) ON DELETE CASCADE,
    lead_id UUID REFERENCES public.leadgen_intake_leads(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    actor_user_id TEXT,
    actor_email TEXT,
    actor_display_name TEXT,
    actor_label TEXT,
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.leadgen_intake_batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leadgen_intake_leads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leadgen_intake_events ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_batches_status_created
    ON public.leadgen_intake_batches(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_batches_job_id
    ON public.leadgen_intake_batches(job_id_legacy)
    WHERE job_id_legacy IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_leads_intake_order
    ON public.leadgen_intake_leads(intake_id, input_order);

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_leads_status
    ON public.leadgen_intake_leads(intake_id, approval_status);

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_leads_domain
    ON public.leadgen_intake_leads(domain)
    WHERE domain IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_leads_score
    ON public.leadgen_intake_leads(leadgen_score DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_leadgen_intake_events_intake_created
    ON public.leadgen_intake_events(intake_id, created_at DESC);

REVOKE ALL ON TABLE public.leadgen_intake_batches FROM anon, authenticated;
REVOKE ALL ON TABLE public.leadgen_intake_leads FROM anon, authenticated;
REVOKE ALL ON TABLE public.leadgen_intake_events FROM anon, authenticated;
GRANT ALL ON TABLE public.leadgen_intake_batches TO service_role;
GRANT ALL ON TABLE public.leadgen_intake_leads TO service_role;
GRANT ALL ON TABLE public.leadgen_intake_events TO service_role;

NOTIFY pgrst, 'reload schema';
