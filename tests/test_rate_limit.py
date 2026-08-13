import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

import agent.rate_limit as rate_limit
from agent.rate_limit import InvocationThrottle, RetryPolicy


async def _acquire_and_release(throttle: InvocationThrottle) -> None:
    await throttle.acquire_async()
    throttle.release_async()


def test_invocation_throttle_async_acquire_works_across_event_loops():
    throttle = InvocationThrottle(
        max_concurrent=1,
        min_interval_sec=0.0,
        start_jitter_sec=0.0,
    )

    asyncio.run(_acquire_and_release(throttle))
    asyncio.run(_acquire_and_release(throttle))


def test_sync_retry_deadline_stops_before_an_unbounded_retry(monkeypatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr(
        rate_limit,
        "web_search_throttle",
        lambda: InvocationThrottle(
            max_concurrent=1,
            min_interval_sec=0.0,
            start_jitter_sec=0.0,
        ),
    )
    monkeypatch.setattr(
        rate_limit,
        "web_search_retry_policy",
        lambda: RetryPolicy(
            max_retries=5,
            base_delay_sec=10.0,
            max_delay_sec=10.0,
            jitter_sec=0.0,
        ),
    )

    def timeout_call(*_args, timeout: float, **_kwargs):
        calls.append(timeout)
        raise TimeoutError("provider timed out")

    with pytest.raises(TimeoutError, match="deadline exceeded"):
        rate_limit.run_with_sync_retries(
            timeout_call,
            timeout=30,
            max_elapsed_seconds=0.1,
        )

    assert len(calls) == 1
    assert 0 < calls[0] <= 0.1


def test_sync_retry_deadline_includes_throttle_wait(monkeypatch) -> None:
    throttle = InvocationThrottle(
        max_concurrent=1,
        min_interval_sec=0.0,
        start_jitter_sec=0.0,
    )
    assert throttle.acquire_sync()
    monkeypatch.setattr(rate_limit, "web_search_throttle", lambda: throttle)
    calls: list[str] = []
    attempts: list[str] = []

    try:
        with pytest.raises(TimeoutError, match="waiting for capacity"):
            rate_limit.run_with_sync_retries(
                lambda: calls.append("called"),
                max_elapsed_seconds=0.01,
                on_attempt_started=lambda: attempts.append("started"),
            )
    finally:
        throttle.release_sync()

    assert calls == []
    assert attempts == []
