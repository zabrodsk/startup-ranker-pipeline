from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Semaphore
from typing import Any, Awaitable, Iterable

from agent.run_context import (
    get_current_collector,
    get_current_llm_request_settings,
    get_current_llm_selection,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_ms(name: str, default_ms: int) -> float:
    return max(0.0, _env_int(name, default_ms) / 1000.0)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    base_delay_sec: float
    max_delay_sec: float
    jitter_sec: float


class InvocationThrottle:
    """Shared concurrency and pacing guard for outbound provider calls."""

    def __init__(
        self,
        *,
        max_concurrent: int,
        min_interval_sec: float,
        start_jitter_sec: float,
    ):
        self._state_lock = Lock()
        self._sync_semaphore = Semaphore(max(1, max_concurrent))
        self._min_interval_sec = max(0.0, min_interval_sec)
        self._start_jitter_sec = max(0.0, start_jitter_sec)
        self._next_allowed_at = 0.0

    def _reserve_slot(self) -> float:
        with self._state_lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed_at - now)
            if delay <= 0:
                gap = self._min_interval_sec + random.uniform(0.0, self._start_jitter_sec)
                self._next_allowed_at = now + gap
            return delay

    def _cooldown(self, delay_sec: float) -> None:
        if delay_sec <= 0:
            return
        with self._state_lock:
            self._next_allowed_at = max(
                self._next_allowed_at,
                time.monotonic() + delay_sec,
            )

    async def acquire_async(self) -> None:
        # Use the process-wide threading semaphore for async callers too.
        # This avoids binding a module-global asyncio.Semaphore to whichever
        # event loop imported the module first, which breaks once analysis jobs
        # run on dedicated worker-thread event loops.
        await asyncio.to_thread(self._sync_semaphore.acquire)
        while True:
            delay = self._reserve_slot()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    def release_async(self) -> None:
        self._sync_semaphore.release()

    def acquire_sync(self) -> None:
        self._sync_semaphore.acquire()
        while True:
            delay = self._reserve_slot()
            if delay <= 0:
                return
            time.sleep(delay)

    def release_sync(self) -> None:
        self._sync_semaphore.release()

    async def impose_async_cooldown(self, delay_sec: float) -> None:
        self._cooldown(delay_sec)

    def impose_sync_cooldown(self, delay_sec: float) -> None:
        self._cooldown(delay_sec)


_LLM_THROTTLE = InvocationThrottle(
    max_concurrent=max(1, _env_int("LLM_MAX_CONCURRENT", 2)),
    min_interval_sec=_env_ms("LLM_MIN_INTERVAL_MS", 1500),
    start_jitter_sec=_env_ms("LLM_JITTER_MS", 250),
)

_WEB_SEARCH_THROTTLE = InvocationThrottle(
    max_concurrent=max(1, _env_int("WEB_SEARCH_MAX_CONCURRENT", 1)),
    min_interval_sec=_env_ms("WEB_SEARCH_MIN_INTERVAL_MS", 3000),
    start_jitter_sec=_env_ms("WEB_SEARCH_JITTER_MS", 350),
)

_LLM_RETRY_POLICY = RetryPolicy(
    max_retries=max(0, _env_int("LLM_MAX_RETRIES", 6)),
    base_delay_sec=_env_ms("LLM_BACKOFF_BASE_MS", 2000),
    max_delay_sec=_env_ms("LLM_BACKOFF_MAX_MS", 45000),
    jitter_sec=_env_ms("LLM_BACKOFF_JITTER_MS", 750),
)

_WEB_SEARCH_RETRY_POLICY = RetryPolicy(
    max_retries=max(0, _env_int("WEB_SEARCH_MAX_RETRIES", 5)),
    base_delay_sec=_env_ms("WEB_SEARCH_BACKOFF_BASE_MS", 2500),
    max_delay_sec=_env_ms("WEB_SEARCH_BACKOFF_MAX_MS", 60000),
    jitter_sec=_env_ms("WEB_SEARCH_BACKOFF_JITTER_MS", 1000),
)


def llm_throttle() -> InvocationThrottle:
    return _LLM_THROTTLE


def web_search_throttle() -> InvocationThrottle:
    return _WEB_SEARCH_THROTTLE


def llm_retry_policy() -> RetryPolicy:
    return _LLM_RETRY_POLICY


def web_search_retry_policy() -> RetryPolicy:
    return _WEB_SEARCH_RETRY_POLICY


class ThrottledRunnable:
    """Proxy that adds shared pacing and retry behavior around LangChain runnables."""

    def __init__(
        self,
        runnable: Any,
        *,
        throttle: InvocationThrottle,
        retry_policy: RetryPolicy,
        fallback_builder: Any | None = None,
    ):
        self._runnable = runnable
        self._throttle = throttle
        self._retry_policy = retry_policy
        self._fallback_builder = fallback_builder
        self._fallback_applied = False

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "ThrottledRunnable":
        fallback_builder = None
        if self._fallback_builder is not None:
            def fallback_builder():
                base = self._fallback_builder()
                if base is None:
                    return None
                return base.with_structured_output(*args, **kwargs)
        return ThrottledRunnable(
            self._runnable.with_structured_output(*args, **kwargs),
            throttle=self._throttle,
            retry_policy=self._retry_policy,
            fallback_builder=fallback_builder,
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> "ThrottledRunnable":
        fallback_builder = None
        if self._fallback_builder is not None:
            def fallback_builder():
                base = self._fallback_builder()
                if base is None:
                    return None
                return base.bind_tools(*args, **kwargs)
        return ThrottledRunnable(
            self._runnable.bind_tools(*args, **kwargs),
            throttle=self._throttle,
            retry_policy=self._retry_policy,
            fallback_builder=fallback_builder,
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            await self._throttle.acquire_async()
            started = time.monotonic()
            try:
                return await self._runnable.ainvoke(*args, **kwargs)
            except Exception as exc:
                selection = get_current_llm_selection() or {}
                collector = get_current_collector()
                latency_ms = int((time.monotonic() - started) * 1000)
                if self._try_reasoning_fallback(exc, selection, collector, latency_ms):
                    continue
                retryable = is_retryable_api_error(exc)
                if attempt >= self._retry_policy.max_retries or not retryable:
                    if collector:
                        collector.record_execution_event(
                            service="llm",
                            status="error",
                            provider=selection.get("provider"),
                            model=selection.get("model"),
                            latency_ms=latency_ms,
                            max_retries=self._retry_policy.max_retries,
                            error_message=str(exc)[:500],
                            metadata={
                                "attempt": attempt + 1,
                                "retryable": retryable,
                                "status_code": _extract_status_code(exc),
                                "error_type": exc.__class__.__name__,
                            },
                        )
                    raise
                delay = compute_retry_delay(exc, attempt, self._retry_policy)
                if collector:
                    collector.record_execution_event(
                        service="llm",
                        status="retrying",
                        provider=selection.get("provider"),
                        model=selection.get("model"),
                        latency_ms=latency_ms,
                        max_retries=self._retry_policy.max_retries,
                        error_message=str(exc)[:500],
                        metadata={
                            "attempt": attempt + 1,
                            "retry_delay_seconds": delay,
                            "status_code": _extract_status_code(exc),
                            "error_type": exc.__class__.__name__,
                            "rate_limited": is_rate_limit_error(exc),
                        },
                    )
                if is_rate_limit_error(exc):
                    await self._throttle.impose_async_cooldown(delay)
            finally:
                self._throttle.release_async()

            attempt += 1
            await asyncio.sleep(delay)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            self._throttle.acquire_sync()
            started = time.monotonic()
            try:
                return self._runnable.invoke(*args, **kwargs)
            except Exception as exc:
                selection = get_current_llm_selection() or {}
                collector = get_current_collector()
                latency_ms = int((time.monotonic() - started) * 1000)
                if self._try_reasoning_fallback(exc, selection, collector, latency_ms):
                    continue
                retryable = is_retryable_api_error(exc)
                if attempt >= self._retry_policy.max_retries or not retryable:
                    if collector:
                        collector.record_execution_event(
                            service="llm",
                            status="error",
                            provider=selection.get("provider"),
                            model=selection.get("model"),
                            latency_ms=latency_ms,
                            max_retries=self._retry_policy.max_retries,
                            error_message=str(exc)[:500],
                            metadata={
                                "attempt": attempt + 1,
                                "retryable": retryable,
                                "status_code": _extract_status_code(exc),
                                "error_type": exc.__class__.__name__,
                            },
                        )
                    raise
                delay = compute_retry_delay(exc, attempt, self._retry_policy)
                if collector:
                    collector.record_execution_event(
                        service="llm",
                        status="retrying",
                        provider=selection.get("provider"),
                        model=selection.get("model"),
                        latency_ms=latency_ms,
                        max_retries=self._retry_policy.max_retries,
                        error_message=str(exc)[:500],
                        metadata={
                            "attempt": attempt + 1,
                            "retry_delay_seconds": delay,
                            "status_code": _extract_status_code(exc),
                            "error_type": exc.__class__.__name__,
                            "rate_limited": is_rate_limit_error(exc),
                        },
                    )
                if is_rate_limit_error(exc):
                    self._throttle.impose_sync_cooldown(delay)
            finally:
                self._throttle.release_sync()

            attempt += 1
            time.sleep(delay)

    def _try_reasoning_fallback(
        self,
        exc: Exception,
        selection: dict[str, Any],
        collector: Any,
        latency_ms: int,
    ) -> bool:
        if self._fallback_applied or self._fallback_builder is None:
            return False
        if not _supports_reasoning_fallback(exc, selection):
            return False

        fallback_runnable = self._fallback_builder()
        if fallback_runnable is None:
            return False

        self._runnable = fallback_runnable
        self._fallback_applied = True
        if collector:
            collector.record_execution_event(
                service="llm",
                status="retrying",
                provider=selection.get("provider"),
                model=selection.get("model"),
                latency_ms=latency_ms,
                max_retries=self._retry_policy.max_retries,
                error_message=str(exc)[:500],
                metadata={
                    "attempt": 1,
                    "retry_delay_seconds": 0.0,
                    "status_code": _extract_status_code(exc),
                    "error_type": exc.__class__.__name__,
                    "reasoning_fallback_applied": True,
                },
            )
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


def wrap_llm(
    runnable: Any,
    *,
    fallback_builder: Any | None = None,
) -> ThrottledRunnable:
    return ThrottledRunnable(
        runnable,
        throttle=llm_throttle(),
        retry_policy=llm_retry_policy(),
        fallback_builder=fallback_builder,
    )


async def gather_with_concurrency(
    awaitables: Iterable[Awaitable[Any]],
    *,
    limit: int | None = None,
) -> list[Any]:
    items = list(awaitables)
    if not items:
        return []

    cap = limit if limit is not None else max(1, _env_int("PIPELINE_STAGE_MAX_CONCURRENCY", 2))
    cap = max(1, cap)
    semaphore = asyncio.Semaphore(cap)

    async def _run(awaitable: Awaitable[Any]) -> Any:
        async with semaphore:
            return await awaitable

    return await asyncio.gather(*(_run(item) for item in items))


def run_with_sync_retries(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    throttle = web_search_throttle()
    retry_policy = web_search_retry_policy()
    attempt = 0

    while True:
        throttle.acquire_sync()
        try:
            return callable_(*args, **kwargs)
        except Exception as exc:
            if attempt >= retry_policy.max_retries or not is_retryable_api_error(exc):
                raise
            delay = compute_retry_delay(exc, attempt, retry_policy)
            if is_rate_limit_error(exc):
                throttle.impose_sync_cooldown(delay)
        finally:
            throttle.release_sync()

        attempt += 1
        time.sleep(delay)


_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "acceleration",
)

_RETRYABLE_MARKERS = _RATE_LIMIT_MARKERS + (
    "timed out",
    "timeout",
    "connection reset",
    "connection error",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)

_AUTHENTICATION_MARKERS = (
    "authentication_error",
    "authentication error",
    "invalid x-api-key",
    "invalid api key",
    "unauthorized",
    "forbidden",
    "permission denied",
    "invalid credentials",
)


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in {429, 529}:
        return True

    error_text = _exception_text(exc)
    return any(marker in error_text for marker in _RATE_LIMIT_MARKERS)


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
        return True

    name = exc.__class__.__name__.lower()
    if any(token in name for token in ("ratelimit", "timeout", "connection", "overloaded")):
        return True

    error_text = _exception_text(exc)
    return any(marker in error_text for marker in _RETRYABLE_MARKERS)


def is_authentication_api_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in {401, 403}:
        return True

    name = exc.__class__.__name__.lower()
    if any(token in name for token in ("authentication", "unauthorized", "forbidden", "permission")):
        return True

    error_text = _exception_text(exc)
    return any(marker in error_text for marker in _AUTHENTICATION_MARKERS)


def _supports_reasoning_fallback(
    exc: Exception,
    selection: dict[str, Any],
) -> bool:
    if selection.get("provider") != "openai" or selection.get("model") != "gpt-5.4-nano":
        return False

    request_settings = get_current_llm_request_settings() or {}
    if not request_settings.get("effective_reasoning_effort"):
        return False

    status_code = _extract_status_code(exc)
    if status_code != 400:
        return False

    error_text = _exception_text(exc)
    if "reasoning" not in error_text:
        return False
    return any(
        marker in error_text
        for marker in (
            "unsupported_parameter",
            "unsupported parameter",
            "unsupported_value",
            "unsupported value",
            "does not support",
            "not supported",
            "unknown parameter",
        )
    )


def compute_retry_delay(exc: Exception, attempt: int, retry_policy: RetryPolicy) -> float:
    header_delay = _extract_retry_after_seconds(exc)
    if header_delay is not None:
        delay = header_delay
    else:
        delay = min(
            retry_policy.max_delay_sec,
            retry_policy.base_delay_sec * (2 ** attempt),
        )

    if retry_policy.jitter_sec > 0:
        delay += random.uniform(0.0, retry_policy.jitter_sec)
    return max(retry_policy.base_delay_sec, delay)


def _extract_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code

    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    for key in (
        "retry-after",
        "retry-after-ms",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-input-tokens-reset",
        "anthropic-ratelimit-output-tokens-reset",
        "anthropic-ratelimit-tokens-reset",
    ):
        value = headers.get(key)
        parsed = _parse_retry_hint(value, key)
        if parsed is not None:
            return parsed
    return None


def _parse_retry_hint(value: str | None, key: str) -> float | None:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    if key == "retry-after-ms":
        try:
            return max(0.0, float(text) / 1000.0)
        except ValueError:
            return None

    if key == "retry-after":
        try:
            return max(0.0, float(text))
        except ValueError:
            return None

    try:
        reset_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)

    return max(0.0, (reset_at - datetime.now(timezone.utc)).total_seconds())


def _exception_text(exc: Exception) -> str:
    parts = [str(exc), exc.__class__.__name__]
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return " ".join(part.lower() for part in parts if part)
