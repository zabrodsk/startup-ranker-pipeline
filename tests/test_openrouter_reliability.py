from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import agent.llm as llm_module  # noqa: E402
import agent.rate_limit as rate_limit_module  # noqa: E402
from agent.batch import _await_with_heartbeat  # noqa: E402
from agent.model_benchmark import (  # noqa: E402
    benchmark_completeness_issues,
    run_openrouter_route_qualification,
)
from agent.model_profiles import resolve_model_profile  # noqa: E402
from agent.rate_limit import (  # noqa: E402
    RetryPolicy,
    compute_retry_delay,
    gather_with_concurrency,
)


def test_concurrent_llm_failure_cancels_and_drains_sibling_requests() -> None:
    async def run() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def slow_sibling() -> None:
            sibling_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        async def fail_after_sibling_starts() -> None:
            await sibling_started.wait()
            raise RuntimeError("provider overloaded")

        with pytest.raises(RuntimeError, match="provider overloaded"):
            await gather_with_concurrency(
                [slow_sibling(), fail_after_sibling_starts()],
                limit=2,
            )

        assert sibling_cancelled.is_set()

    asyncio.run(run())


def test_openrouter_retry_after_header_controls_backoff() -> None:
    class Response:
        headers = {"retry-after-ms": "1750"}

    class RateLimitError(Exception):
        status_code = 429
        response = Response()

    delay = compute_retry_delay(
        RateLimitError("engine_overloaded"),
        attempt=0,
        retry_policy=RetryPolicy(
            max_retries=1,
            base_delay_sec=0.25,
            max_delay_sec=10.0,
            jitter_sec=0.0,
        ),
    )

    assert delay == 1.75


def test_provider_retry_hint_cannot_exceed_configured_backoff_ceiling() -> None:
    class Response:
        headers = {"retry-after": "240"}

    class RateLimitError(Exception):
        status_code = 429
        response = Response()

    delay = compute_retry_delay(
        RateLimitError("engine_overloaded"),
        attempt=0,
        retry_policy=RetryPolicy(
            max_retries=1,
            base_delay_sec=2.0,
            max_delay_sec=45.0,
            jitter_sec=0.0,
        ),
    )

    assert delay == 45.0


def test_openrouter_uses_a_longer_configurable_request_timeout(monkeypatch) -> None:
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OPENROUTER_REQUEST_TIMEOUT_SECONDS", raising=False)

    assert llm_module.get_llm_runtime_settings(provider="openai")[
        "request_timeout_seconds"
    ] == 90.0
    assert llm_module.get_llm_runtime_settings(provider="openrouter")[
        "request_timeout_seconds"
    ] == 300.0

    monkeypatch.setenv("OPENROUTER_REQUEST_TIMEOUT_SECONDS", "240")
    assert llm_module.get_llm_runtime_settings(provider="openrouter")[
        "request_timeout_seconds"
    ] == 240.0


def test_benchmark_rejects_terminal_run_with_failed_ranking_dimensions() -> None:
    issues = benchmark_completeness_issues(
        skipped=False,
        final_state={
            "final_decision": "invest",
            "final_arguments": [{"argument_type": "pro", "content": "Evidence"}],
            "ranking_result": {
                "composite_score": 0.0,
                "dimension_scores": [
                    {
                        "dimension": "strategy_fit",
                        "raw_score": 0.0,
                        "critical_gaps": ["Scoring failed due to LLM error"],
                    },
                    {
                        "dimension": "team",
                        "raw_score": 0.0,
                        "critical_gaps": ["Scoring failed due to LLM error"],
                    },
                    {
                        "dimension": "upside",
                        "raw_score": 0.0,
                        "critical_gaps": ["Scoring failed due to LLM error"],
                    },
                ],
            },
        },
        model_executions=[
            {
                "service": "llm",
                "stage": "ranking_dimension_score",
                "status": "error",
            }
        ],
    )

    assert "required_llm_stage_error:ranking_dimension_score" in issues
    assert "ranking_dimension_failed:strategy_fit" in issues
    assert "ranking_dimension_failed:team" in issues
    assert "ranking_dimension_failed:upside" in issues


def test_openrouter_adapter_can_relax_collection_filter_while_retaining_zdr(
    monkeypatch,
) -> None:
    import langchain_openai

    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    llm_module._create_openrouter(
        "moonshotai/kimi-k2.6",
        None,
        90.0,
        0,
        reasoning_enabled=True,
        routing={
            "require_parameters": True,
            "data_collection": "allow",
            "zdr": True,
            "only": ["digitalocean", "together"],
            "allow_fallbacks": True,
        },
    )

    assert captured["extra_body"]["provider"] == {
        "require_parameters": True,
        "data_collection": "allow",
        "zdr": True,
        "only": ["digitalocean", "together"],
        "allow_fallbacks": True,
    }


def test_staging_profiles_use_qualified_same_model_fallback_routes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    kimi = resolve_model_profile("kimi_k26").policy
    hybrid = resolve_model_profile("glm_deepseek_flash").policy

    assert kimi.answering["openrouter_routing"] == {
        "require_parameters": True,
        "data_collection": "allow",
        "zdr": True,
        "order": ["wandb", "modelrun", "novita"],
        "only": ["wandb", "modelrun", "novita"],
        "allow_fallbacks": True,
    }
    assert hybrid.decomposition["openrouter_routing"] == {
        "require_parameters": True,
        "data_collection": "allow",
        "zdr": True,
        "order": ["together", "fireworks"],
        "only": ["together", "fireworks"],
        "allow_fallbacks": True,
    }
    assert hybrid.answering["openrouter_routing"] == {
        "require_parameters": True,
        "data_collection": "allow",
        "zdr": True,
        "order": ["morph", "atlas-cloud", "parasail", "digitalocean"],
        "only": ["morph", "atlas-cloud", "parasail", "digitalocean"],
        "allow_fallbacks": True,
    }


def test_route_qualification_ranks_only_successful_pinned_zdr_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    observed_routing: list[dict[str, object]] = []

    async def fake_invoke(case, routing):
        assert rate_limit_module.llm_retry_policy().max_retries == 1
        observed_routing.append(dict(routing))
        provider = str(routing["only"][0])
        if provider == "together":
            raise RuntimeError("provider overloaded")
        return {
            "structured_ok": True,
            "selected_provider": "DigitalOcean",
            "generation_id": f"{case['id']}-generation",
            "actual_cost_usd": 0.0001,
            "latency_ms": 125,
            "retry_count": 0,
        }

    report = asyncio.run(
        run_openrouter_route_qualification(
            tmp_path,
            invoke_case=fake_invoke,
            calls_per_case=1,
            concurrency=2,
            route_candidates={
                "moonshotai/kimi-k2.6": ("digitalocean", "together"),
            },
        )
    )

    assert report["eligible"] is True
    assert report["recommendations"] == {
        "moonshotai/kimi-k2.6": "digitalocean",
    }
    assert (tmp_path / "route-qualification.json").is_file()
    assert observed_routing
    assert all(routing["data_collection"] == "allow" for routing in observed_routing)
    assert all(routing["zdr"] is True for routing in observed_routing)
    assert all(routing["require_parameters"] is True for routing in observed_routing)
    assert all(routing["allow_fallbacks"] is False for routing in observed_routing)
    assert all(len(routing["only"]) == 1 for routing in observed_routing)


def test_wall_timeout_cancels_and_drains_provider_coroutine() -> None:
    async def run() -> None:
        cancelled = asyncio.Event()

        async def slow_provider_call() -> None:
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(TimeoutError, match="provider work cancelled"):
            await _await_with_heartbeat(
                slow_provider_call(),
                timeout_seconds=1,
                heartbeat_seconds=1,
            )

        assert cancelled.is_set()

    asyncio.run(run())
