-- Durable cross-worker decomposition-tree cache (Sprint 3 W11).
--
-- The decomposition prompt sees ONLY the question and the industry
-- (decomposition.py), so the cache key drops the company: same-industry
-- companies share rows. The key embeds a prompt signature covering
-- decomposition.system + decomposition.user + ROUTE_TAGGING_INSTRUCTION so
-- any prompt change invalidates cleanly.
--
-- company_name and hit_count are OBSERVABILITY ONLY: they record which
-- company first stored a row and how often rows are reused, so cross-company
-- reuse can be measured before anyone widens reliance on it.
--
-- Service-role only: RLS enabled with no policies (mcp_secrets pattern).

CREATE TABLE IF NOT EXISTS decomposition_tree_cache (
    query_hash TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    industry TEXT NOT NULL,
    aspect TEXT NOT NULL,
    prompt_signature TEXT NOT NULL,
    tree JSONB NOT NULL,
    company_name TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_decomposition_tree_cache_created_at
    ON decomposition_tree_cache(created_at DESC);

ALTER TABLE decomposition_tree_cache ENABLE ROW LEVEL SECURITY;

-- No policies are defined. With RLS enabled and no policies, only the
-- service-role key (which bypasses RLS) can read/write this table.

COMMENT ON TABLE decomposition_tree_cache IS
    'Service-role-only cache of decomposed question trees keyed by sha256("dtree-v3|question|industry|aspect|prompt_signature").';
COMMENT ON COLUMN decomposition_tree_cache.company_name IS
    'Observability only: company whose run stored this row (cache keys are company-free).';
COMMENT ON COLUMN decomposition_tree_cache.hit_count IS
    'Observability only: best-effort count of cache hits on this row.';
