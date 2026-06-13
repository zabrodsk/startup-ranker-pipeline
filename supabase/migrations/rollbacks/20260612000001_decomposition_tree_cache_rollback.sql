-- Rollback for 20260612000001_decomposition_tree_cache.sql (Sprint 3 W11).
-- The table is a pure cache; dropping it loses no source-of-truth data.

DROP INDEX IF EXISTS idx_decomposition_tree_cache_created_at;
DROP TABLE IF EXISTS decomposition_tree_cache;
