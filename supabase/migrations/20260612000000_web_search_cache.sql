-- Durable cross-run web-search result cache (Sprint 3 W13).
--
-- Planner topic routes emit company-name-free queries, so identical sector
-- questions across companies hit the same cache row; legacy company-anchored
-- queries hit on same-company re-runs. TTL is enforced at read time by the
-- accessor (RDI_WEB_SEARCH_CACHE_TTL_DAYS, default 14); no background cleanup
-- in v1.
--
-- Service-role only: RLS enabled with no policies (mcp_secrets pattern).

CREATE TABLE IF NOT EXISTS web_search_cache (
    query_hash TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    domain_filter JSONB,
    results TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_web_search_cache_created_at
    ON web_search_cache(created_at DESC);

ALTER TABLE web_search_cache ENABLE ROW LEVEL SECURITY;

-- No policies are defined. With RLS enabled and no policies, only the
-- service-role key (which bypasses RLS) can read/write this table.

COMMENT ON TABLE web_search_cache IS
    'Service-role-only cache of web-search provider results keyed by sha256("wsc-v1|provider|query|domain_filter").';
COMMENT ON COLUMN web_search_cache.query_hash IS
    'sha256 of "wsc-v1|" + provider + "|" + query + "|" + canonical JSON of sorted domain_filter.';
