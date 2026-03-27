from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from agent.llm_policy import get_alternate_premium_selection
from agent.rate_limit import is_authentication_api_error
from agent.run_context import use_phase_llm

T = TypeVar("T")


def invoke_with_phase_fallback(
    selection: dict[str, Any] | None,
    invoke: Callable[[], T],
) -> T:
    with use_phase_llm(selection):
        try:
            return invoke()
        except Exception as exc:
            fallback = _get_phase_fallback_selection(selection, exc)
            if fallback is None:
                raise

    with use_phase_llm(fallback):
        return invoke()


async def ainvoke_with_phase_fallback(
    selection: dict[str, Any] | None,
    invoke: Callable[[], Awaitable[T]],
) -> T:
    with use_phase_llm(selection):
        try:
            return await invoke()
        except Exception as exc:
            fallback = _get_phase_fallback_selection(selection, exc)
            if fallback is None:
                raise

    with use_phase_llm(fallback):
        return await invoke()


def _get_phase_fallback_selection(
    selection: dict[str, Any] | None,
    exc: Exception,
) -> dict[str, Any] | None:
    if not is_authentication_api_error(exc):
        return None
    fallback = get_alternate_premium_selection(selection)
    if fallback == selection:
        return None
    return fallback
