"""Durable cross-worker store for decomposed question trees (Sprint 3 W11).

The decomposition prompt sees ONLY the question and the industry
(decomposition.py builds it from decomposition.user with {question} and
{industry} placeholders, plus ROUTE_TAGGING_INSTRUCTION appended in code), so
the cache key safely drops the company: same-industry companies share rows.
The key embeds a full prompt signature so any prompt change — including the
route-tagging instruction the local signature historically missed —
invalidates cleanly.

RDI_DECOMP_CACHE_STORE selects the backend:
  "local"    = the existing per-container JSON cache, byte-identical (default)
  "supabase" = the shared decomposition_tree_cache table; silently falls back
               to local when the database is not configured (CLI mode keeps
               working offline)

Fail-open: any lookup error is a miss, any store error is a no-op.
"""

import hashlib
import logging
import os

from agent.dataclasses.question_tree import QuestionTree
from agent.prompt_library.manager import get_prompt
from agent.web_search.planner import ROUTE_TAGGING_INSTRUCTION

logger = logging.getLogger(__name__)

RDI_DECOMP_CACHE_STORE_ENV = "RDI_DECOMP_CACHE_STORE"
_STORE_MODES = ("local", "supabase")
_DEFAULT_STORE_MODE = "local"
_WARNED_INVALID_STORE_MODE: set[str] = set()

_KEY_VERSION_PREFIX = "dtree-v3"


def _store_mode() -> str:
    """Return the configured decomposition-cache backend."""
    raw = os.getenv(RDI_DECOMP_CACHE_STORE_ENV, _DEFAULT_STORE_MODE)
    mode = raw.strip().lower()
    if mode not in _STORE_MODES:
        if mode not in _WARNED_INVALID_STORE_MODE:
            _WARNED_INVALID_STORE_MODE.add(mode)
            logger.warning(
                "Invalid %s=%r; falling back to %r",
                RDI_DECOMP_CACHE_STORE_ENV, raw, _DEFAULT_STORE_MODE,
            )
        return _DEFAULT_STORE_MODE
    return mode


def use_supabase_store() -> bool:
    """True when the supabase backend is selected AND the DB is configured."""
    if _store_mode() != "supabase":
        return False
    try:
        from web import db

        return bool(db.is_configured())
    except Exception:
        return False


def compute_prompt_signature(prompt_overrides: dict | None = None) -> str:
    """Signature over everything the decomposition LLM call is prompted with.

    Covers decomposition.system, decomposition.user, AND
    ROUTE_TAGGING_INSTRUCTION (appended at call time in decomposition.py) —
    the local-cache signature historically omitted the instruction.
    """
    raw = (
        str(get_prompt("decomposition.system", prompt_overrides))
        + "\n||\n"
        + str(get_prompt("decomposition.user", prompt_overrides))
        + "\n||\n"
        + ROUTE_TAGGING_INSTRUCTION
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_query_hash(
    question: str, industry: str, aspect: str, prompt_signature: str
) -> str:
    """Company-free versioned cache key (see module docstring for why)."""
    raw = (
        f"{_KEY_VERSION_PREFIX}|{question}|{(industry or '').strip().lower()}"
        f"|{aspect}|{prompt_signature}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def lookup(
    *,
    question: str,
    industry: str,
    aspect: str,
    company_name: str,
    prompt_overrides: dict | None = None,
) -> QuestionTree | None:
    """Return a cached QuestionTree, or None on miss/error (fail-open)."""
    try:
        from web import db

        query_hash = compute_query_hash(
            question, industry, str(aspect), compute_prompt_signature(prompt_overrides)
        )
        entry = db.get_decomposition_tree_cache_entry(query_hash)
        if not entry:
            return None
        tree = QuestionTree(**entry["tree"])
    except Exception:
        logger.warning(
            "decomposition cache lookup failed; treating as miss", exc_info=True
        )
        return None

    # Observability (best-effort, must never undo the hit): log whether the
    # row was stored by a different company and bump its hit counter.
    try:
        stored_company = str(entry.get("company_name") or "").strip()
        cross_company = bool(stored_company) and (
            stored_company.lower() != (company_name or "").strip().lower()
        )
        logger.info(
            "decomposition cache hit (cross_company=%s) aspect=%s", cross_company, aspect
        )
        db.increment_decomposition_tree_cache_hit(query_hash, entry.get("hit_count"))
    except Exception:
        pass
    return tree


def store(
    *,
    question: str,
    industry: str,
    aspect: str,
    company_name: str,
    tree: QuestionTree,
    prompt_overrides: dict | None = None,
) -> None:
    """Best-effort store of a freshly decomposed tree; failures are swallowed."""
    try:
        from web import db

        prompt_signature = compute_prompt_signature(prompt_overrides)
        db.upsert_decomposition_tree_cache_entry(
            query_hash=compute_query_hash(
                question, industry, str(aspect), prompt_signature
            ),
            question=question,
            industry=industry,
            aspect=str(aspect),
            prompt_signature=prompt_signature,
            tree=tree.model_dump(),
            company_name=company_name,
        )
    except Exception:
        logger.warning(
            "decomposition cache store failed; tree not cached", exc_info=True
        )
