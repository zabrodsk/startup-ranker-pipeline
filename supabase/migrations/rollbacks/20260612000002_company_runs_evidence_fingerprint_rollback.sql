-- Rollback for 20260612000002_company_runs_evidence_fingerprint.sql (Sprint 3).

DROP INDEX IF EXISTS idx_company_runs_evidence_fingerprint;
DROP INDEX IF EXISTS idx_company_runs_lookup_recent;
ALTER TABLE company_runs DROP COLUMN IF EXISTS evidence_fingerprint;
