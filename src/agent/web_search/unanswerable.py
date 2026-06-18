"""Skip predictably-unanswerable / low-value web searches.

Two pure, local, confidentiality-bounded predictors live here. Both delegate
their routing to ``build_web_search_plan`` (the benchmark-tuned WebEvidencePlanner
router) rather than inventing keyword logic, and both accept only the same public
scalars the planner accepts — company name, normalized public domain, Specter
industry taxonomy, the question text, its route tag, and its aspect band — never
document chunks, answers, memo text, or ``company.about``/``tagline``.

``predict_search_unanswerable`` (Sprint 4 workstream C, env
``RDI_SKIP_UNANSWERABLE_SEARCH``): the conservative gate behind the "documents
incomplete" trigger. It skips ONLY the planner's single skip route (exact private
internal metrics — ARR/MRR/burn/runway/cap table); every other route and all
ambiguity fall through to do-not-skip. So it inherits the planner's validated
0/35 false-positive guarantee and can never skip a question the planner searches.
(``internal_fit`` was removed from the planner's skip set after the f20aa510
replay — those compound "does [public company fact] fit the VC?" questions are
rescued by web — so this conservative gate no longer skips them either.)

``predict_targeted_skip`` (the per-run "Off / Targeted / Full" intensity knob):
the broader, OPT-IN predictor. The user has explicitly chosen less web, so the
safety bar is "the user asked for it", not "never drop a real rescue" — it MAY
suppress searches the conservative gate keeps. It skips the planner's private-
metric route, re-adds ``internal_fit``, and additionally skips the ``team`` aspect
band, where public web is near-total waste. Because the team arm is an exact
``aspect == "team"`` match (mutually exclusive with ``market``/``product``),
Targeted can never drop a market or product question via the aspect arm.

Pure/local: no LLM, no network.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from .planner import QuestionRoute, build_web_search_plan

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
    routes to a skip route (exact private internal metrics); every other route
    and all ambiguity return False (search as before).
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


# --- Targeted ("Off / Targeted / Full") web-search predictor -----------------
#
# Targeted intensity fires web search only on the market/product external-
# landscape band and suppresses it where it is near-total waste. Unlike
# predict_search_unanswerable (which inherits the planner's 0/35 "never drop a
# real rescue" guarantee), Targeted is opt-in, so it MAY drop searches the
# conservative gate keeps:
#   * the planner's private-metric skip route (skip_public_web), and
#   * the internal_fit route (which the planner itself searches company-anchored
#     on the always-on path, but which Targeted re-adds as a suppression), and
#   * any question whose aspect band is exactly "team".
# The team arm is route-independent because route COMPANY_SPECIFIC is overloaded
# (both product=win and team=waste route to it); keying on aspect == "team"
# (mutually exclusive with market/product) makes "never drop a market/product
# question via the aspect arm" a theorem, not an empirical hope.
TARGETED_SKIP_ROUTES = frozenset(
    {QuestionRoute.SKIP_PUBLIC_WEB, QuestionRoute.INTERNAL_FIT}
)
_TARGETED_TEAM_ASPECT = "team"


def predict_targeted_skip(
    *,
    question: str,
    route_tag: str | None = None,
    aspect: str | None = None,
    company_name: str,
    company_domain: str | None = None,
    industry: str | None = None,
    is_root: bool = False,
) -> tuple[bool, str]:
    """Return ``(skip, reason)`` for the Targeted web-search intensity.

    Skip iff the planner routes the question to a waste route (``skip_public_web``
    private metrics or ``internal_fit`` VC-thesis) OR the question's aspect band
    is ``team``. Same signature and confidentiality envelope as
    ``predict_search_unanswerable``: only the public scalars below — never chunks,
    answers, memo text, or ``company.about``/``tagline`` — reach this function.

    The aspect arm is an exact ``== "team"`` match (null-safe: a legacy
    ``aspect=None`` row is not team -> not skipped) and is mutually exclusive
    with ``market``/``product``, so Targeted never drops a market or product
    question via the team arm.
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
    route_skip = plan.route in TARGETED_SKIP_ROUTES
    team_skip = aspect == _TARGETED_TEAM_ASPECT
    if route_skip or team_skip:
        arms: list[str] = []
        if team_skip:
            arms.append("team aspect")
        if route_skip:
            arms.append(f"{plan.route.value} route")
        return (True, "targeted skip — " + " + ".join(arms))
    return (False, f"targeted: answerable route {plan.route.value}")


# Per-run web-search intensity (the "Off / Targeted / Full" knob):
#   "off"      = no web search at all (enforced upstream as use_web_search=False)
#   "targeted" = web only on the market/product band; predict_targeted_skip gates
#                the "documents incomplete" path
#   "full"     = today's behavior, byte-identical (default)
# RDI_WEB_SEARCH_MODE is the FLEET default, consulted only when no explicit
# per-run mode is threaded; an explicit per-run mode always wins. Read at call
# time (not import time), warn-once on an invalid value.
RDI_WEB_SEARCH_MODE_ENV = "RDI_WEB_SEARCH_MODE"
WEB_SEARCH_MODES = ("off", "targeted", "full")
_DEFAULT_WEB_SEARCH_MODE = "full"
_WARNED_INVALID_WEB_SEARCH_VALUES: set[str] = set()


def _normalize_web_search_mode(raw: str | None) -> str | None:
    """Return a valid mode for ``raw``, or None when absent/blank/invalid."""
    if raw is None:
        return None
    mode = raw.strip().lower()
    if mode in WEB_SEARCH_MODES:
        return mode
    return None


def _env_web_search_mode() -> str:
    """Fleet-default web-search mode from RDI_WEB_SEARCH_MODE; default full."""
    raw = os.getenv(RDI_WEB_SEARCH_MODE_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_WEB_SEARCH_MODE
    mode = _normalize_web_search_mode(raw)
    if mode is None:
        marker = f"{RDI_WEB_SEARCH_MODE_ENV}={raw}"
        if marker not in _WARNED_INVALID_WEB_SEARCH_VALUES:
            _WARNED_INVALID_WEB_SEARCH_VALUES.add(marker)
            logger.warning(
                "Invalid %s=%r; falling back to %r",
                RDI_WEB_SEARCH_MODE_ENV,
                raw,
                _DEFAULT_WEB_SEARCH_MODE,
            )
        return _DEFAULT_WEB_SEARCH_MODE
    return mode


def resolve_web_search_mode(explicit: str | None = None) -> str:
    """Resolve the effective web-search mode (off|targeted|full).

    Precedence: a valid explicit per-run ``explicit`` wins; else the
    ``RDI_WEB_SEARCH_MODE`` fleet default; else ``"full"`` (today's behavior).
    An invalid explicit value is ignored (falls through to env/default) and, like
    the env path, never raises — call-time, warn-once.
    """
    chosen = _normalize_web_search_mode(explicit)
    if chosen is not None:
        return chosen
    if explicit is not None and explicit.strip():
        marker = f"explicit={explicit}"
        if marker not in _WARNED_INVALID_WEB_SEARCH_VALUES:
            _WARNED_INVALID_WEB_SEARCH_VALUES.add(marker)
            logger.warning(
                "Invalid explicit web_search_mode=%r; falling back to env/default",
                explicit,
            )
    return _env_web_search_mode()
