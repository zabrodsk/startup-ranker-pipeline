import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.rate_limit import InvocationThrottle


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
