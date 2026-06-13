-- Rollback for 20260612000000_web_search_cache.sql (Sprint 3 W13).
-- The table is a pure cache; dropping it loses no source-of-truth data.

DROP INDEX IF EXISTS idx_web_search_cache_created_at;
DROP TABLE IF EXISTS web_search_cache;
