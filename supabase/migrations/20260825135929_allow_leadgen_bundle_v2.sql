BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

ALTER TABLE public.leadgen_evidence_bundles
    DROP CONSTRAINT leadgen_evidence_bundles_schema_version_check;

ALTER TABLE public.leadgen_evidence_bundles
    ADD CONSTRAINT leadgen_evidence_bundles_schema_version_check
    CHECK (
        schema_version IN (
            'frozen-leadgen-evidence-bundle-v1',
            'frozen-leadgen-evidence-bundle-v2'
        )
    ) NOT VALID;

ALTER TABLE public.leadgen_evidence_bundles
    VALIDATE CONSTRAINT leadgen_evidence_bundles_schema_version_check;

COMMIT;
