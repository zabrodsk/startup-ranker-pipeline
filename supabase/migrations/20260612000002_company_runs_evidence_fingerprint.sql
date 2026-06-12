-- Duplicate-run gate (Sprint 3): evidence fingerprint on company_runs.
--
-- evidence_fingerprint identifies the evidence inputs of one company
-- analysis (see src/agent/dup_fingerprint.py for the versioned schemes).
-- The gate looks up recent non-failed rows by fingerprint (and, for
-- identity-based modes, company_lookup_key) within a configurable window.
--
-- NOTE: the recency index uses run_created_at — the upsert writes
-- run_created_at, NOT created_at.

ALTER TABLE company_runs ADD COLUMN IF NOT EXISTS evidence_fingerprint TEXT;

CREATE INDEX IF NOT EXISTS idx_company_runs_lookup_recent
    ON company_runs(company_lookup_key, run_created_at DESC);

CREATE INDEX IF NOT EXISTS idx_company_runs_evidence_fingerprint
    ON company_runs(evidence_fingerprint)
    WHERE evidence_fingerprint IS NOT NULL;

COMMENT ON COLUMN company_runs.evidence_fingerprint IS
    'Duplicate-run gate: hash of the evidence inputs (doc-v1 | specter-csv-v1 | identity-v1 schemes).';
