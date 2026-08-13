"""Durable cross-run web-search result cache (Sprint 3 W13).

Backed by the service-role-only Supabase table web_search_cache. Planner topic
routes emit company-name-free queries, so identical sector questions across
companies share cache rows; legacy company-anchored queries hit on
same-company re-runs.

Fail-open by design: an unconfigured database or any exception is a cache
miss on lookup and a no-op on store. There is NO local-JSON fallback — the
cache is either the shared durable one or nothing.
"""

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# Cache mode (Sprint 3 W13):
#   "off" = no lookups, no stores — byte-identical legacy behavior (default)
#   "on"  = consult/populate the web_search_cache table around provider calls
# Read at call time (not import time) so tests and rollout can flip it via env.
RDI_WEB_SEARCH_CACHE_ENV = "RDI_WEB_SEARCH_CACHE"
RDI_WEB_SEARCH_CACHE_TTL_DAYS_ENV = "RDI_WEB_SEARCH_CACHE_TTL_DAYS"
_CACHE_MODES = ("off", "on")
_DEFAULT_CACHE_MODE = "off"
_DEFAULT_TTL_DAYS = 14
_WARNED_INVALID_CACHE_VALUES: set[str] = set()

_KEY_VERSION_PREFIX = "wsc-v1"


def _warn_once(env_name: str, raw: str, fallback: object) -> None:
    marker = f"{env_name}={raw}"
    if marker not in _WARNED_INVALID_CACHE_VALUES:
        _WARNED_INVALID_CACHE_VALUES.add(marker)
        logger.warning("Invalid %s=%r; falling back to %r", env_name, raw, fallback)


def _cache_mode() -> str:
    """Return the active web-search-cache mode."""
    raw = os.getenv(RDI_WEB_SEARCH_CACHE_ENV, _DEFAULT_CACHE_MODE)
    mode = raw.strip().lower()
    if mode not in _CACHE_MODES:
        _warn_once(RDI_WEB_SEARCH_CACHE_ENV, raw, _DEFAULT_CACHE_MODE)
        return _DEFAULT_CACHE_MODE
    return mode


def is_web_search_cache_enabled() -> bool:
    """Return True when the web-search-cache flag is on."""
    return _cache_mode() == "on"


def cache_ttl_days() -> int:
    """Return the cache TTL in days (default 14; invalid values fall back)."""
    raw = os.getenv(RDI_WEB_SEARCH_CACHE_TTL_DAYS_ENV, str(_DEFAULT_TTL_DAYS))
    try:
        ttl = int(raw.strip())
    except (ValueError, AttributeError):
        _warn_once(RDI_WEB_SEARCH_CACHE_TTL_DAYS_ENV, raw, _DEFAULT_TTL_DAYS)
        return _DEFAULT_TTL_DAYS
    if ttl <= 0:
        _warn_once(RDI_WEB_SEARCH_CACHE_TTL_DAYS_ENV, raw, _DEFAULT_TTL_DAYS)
        return _DEFAULT_TTL_DAYS
    return ttl


def compute_query_hash(
    provider: str, query: str, domain_filter: list[str] | None
) -> str:
    """Versioned cache key over provider, query, and canonical domain filter."""
    canonical_filter = json.dumps(sorted(domain_filter or []), separators=(",", ":"))
    raw = f"{_KEY_VERSION_PREFIX}|{provider}|{query}|{canonical_filter}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def lookup(provider: str, query: str, domain_filter: list[str] | None) -> str | None:
    """Return cached result text, or None on off/miss/expired/error."""
    if not is_web_search_cache_enabled():
        return None
    try:
        from web import db

        return db.get_web_search_cache_entry(
            compute_query_hash(provider, query, domain_filter),
            ttl_days=cache_ttl_days(),
        )
    except Exception:
        logger.warning("web-search cache lookup failed; treating as miss", exc_info=True)
        return None


def lookup_provider(
    provider: str, query: str, domain_filter: list[str] | None
) -> str | None:
    """Return the actual provider stored with a cache entry, when available."""
    if not is_web_search_cache_enabled():
        return None
    try:
        from web import db

        return db.get_web_search_cache_provider(
            compute_query_hash(provider, query, domain_filter),
            ttl_days=cache_ttl_days(),
        )
    except Exception:
        logger.warning(
            "web-search cache provider lookup failed; provider is unknown",
            exc_info=True,
        )
        return None


def store(
    provider: str,
    query: str,
    domain_filter: list[str] | None,
    results: str,
    actual_provider: str | None = None,
) -> None:
    """Best-effort store of a provider result; failures are swallowed."""
    if not is_web_search_cache_enabled():
        return
    try:
        from web import db

        db.upsert_web_search_cache_entry(
            query_hash=compute_query_hash(provider, query, domain_filter),
            provider=actual_provider or provider,
            query=query,
            domain_filter=sorted(domain_filter or []),
            results=results,
        )
    except Exception:
        logger.warning("web-search cache store failed; result not cached", exc_info=True)
