"""Skip predictably-unanswerable web searches (Sprint 4 workstream C).

The "documents incomplete" web-search trigger fires whenever a grounded answer
looks thin, but for early-stage/stealth companies most of those searches cannot
be rescued (no public web footprint). This predictor suppresses the unrescuable
ones.

Safety property: it does NOT invent its own keyword logic. It delegates the
skip judgment to ``build_web_search_plan(...).is_skip`` — the same conservative,
benchmark-tuned router the WebEvidencePlanner uses. So this gate and the planner
can never disagree about what is "unanswerable", and this gate inherits the
planner's already-validated 0/35 false-positive guarantee. It skips only the two
planner skip routes (private internal metrics, VC-internal-fit); every other
route and all ambiguity fall through to do-not-skip (the status quo).

Pure/local: no LLM, no network. Inputs are restricted to the same public fields
the planner accepts (name, public domain, industry taxonomy, question, route
tag) — never document chunks, answers, or other private content (confidentiality).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from .planner import build_web_search_plan

logger = logging.getLogger(__name__)

# Mode (Sprint 4 C):
#   "off"    = predictor never consulted — byte-identical legacy behavior (default)
#   "shadow" = prediction recorded in provenance; the search still runs
#   "on"     = a predicted-skip suppresses the search (no cap slot consumed)
# Read at call time (not import time) so tests and rollout can flip it via env.
RDI_SKIP_UNANSWERABLE_SEARCH_ENV = "RDI_SKIP_UNANSWERABLE_SEARCH"
_SKIP_MODES = ("off", "shadow", "on")
_DEFAULT_SKIP_MODE = "off"
_WARNED_INVALID_SKIP_VALUES: set[str] = set()


def skip_unanswerable_mode() -> str:
    """Return the active skip-unanswerable mode (off|shadow|on); default off."""
    raw = os.getenv(RDI_SKIP_UNANSWERABLE_SEARCH_ENV, _DEFAULT_SKIP_MODE)
    mode = raw.strip().lower()
    if mode not in _SKIP_MODES:
        marker = f"{RDI_SKIP_UNANSWERABLE_SEARCH_ENV}={raw}"
        if marker not in _WARNED_INVALID_SKIP_VALUES:
            _WARNED_INVALID_SKIP_VALUES.add(marker)
            logger.warning(
                "Invalid %s=%r; falling back to %r",
                RDI_SKIP_UNANSWERABLE_SEARCH_ENV,
                raw,
                _DEFAULT_SKIP_MODE,
            )
        return _DEFAULT_SKIP_MODE
    return mode


def predict_search_unanswerable(
    *,
    question: str,
    route_tag: str | None = None,
    aspect: str | None = None,
    company_name: str,
    company_domain: str | None = None,
    industry: str | None = None,
    is_root: bool = False,
) -> tuple[bool, str]:
    """Return ``(skip, reason)`` for one "documents incomplete" question.

    Delegates to ``build_web_search_plan`` so the predictor and the planner can
    never disagree on what is unanswerable. ``skip`` is True only when the plan
    routes to a skip route (private internal metrics / VC-internal-fit); every
    other route and all ambiguity return False (search as before).
    """
    plan = build_web_search_plan(
        question=question,
        company_name=company_name,
        company_domain=company_domain,
        industry_hint=industry,
        current_year=datetime.now().year,
        route_tag=route_tag,
        aspect=aspect,
        is_root=is_root,
    )
    if plan.is_skip:
        return (True, f"{plan.route.value}: {plan.skip_reason}")
    return (False, f"answerable route {plan.route.value}")
