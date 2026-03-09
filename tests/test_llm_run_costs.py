import asyncio
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.llm_catalog import available_models_payload, validate_requested_selection
from agent.run_context import (
    RunTelemetryCollector,
    use_company_context,
    use_run_context,
    use_stage_context,
)
from agent import llm as llm_module
from agent import rate_limit as rate_limit_module
from web.app import AnalysisStatus, app


def _login(client: TestClient) -> None:
    response = client.post("/api/login", json={"password": "9876"})
    assert response.status_code == 200
    client.cookies.set("session_id", response.json()["session_id"])


def test_model_catalog_validation_accepts_only_available_entries(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    entry = validate_requested_selection("anthropic", "claude-haiku-4-5-20251001")
    assert entry is not None
    assert entry.provider == "anthropic"
    assert entry.model == "claude-haiku-4-5-20251001"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        validate_requested_selection("anthropic", "claude-haiku-4-5-20251001")
    except ValueError as exc:
        assert "not available" in str(exc).lower()
    else:
        raise AssertionError("Expected unavailable model selection to raise ValueError")


def test_available_models_payload_marks_availability(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    models = available_models_payload()
    gemini = next(item for item in models if item["model"] == "gemini-3.1-flash-lite-preview")
    assert {item["model"] for item in models} == {
        "gemini-3.1-flash-lite-preview",
        "claude-haiku-4-5-20251001",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-4.1-mini",
    }

    assert gemini["available"] is True
    assert gemini["pricing_available"] is True
    assert gemini["summary"] == "Budget speed"
    assert all(
        item["available"] is False
        for item in models
        if item["provider"] == "openai"
    )


def test_model_catalog_validation_accepts_openai_entries_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    entry = validate_requested_selection("openai", "gpt-5-mini")

    assert entry is not None
    assert entry.provider == "openai"
    assert entry.model == "gpt-5-mini"


def test_create_llm_prefers_run_context_selection_over_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")

    called = {}

    def fake_gemini(model, temperature, timeout_s, max_retries):
        called["provider"] = "gemini"
        called["model"] = model
        return object()

    def fake_anthropic(model, temperature, timeout_s, max_retries):
        called["provider"] = "anthropic"
        called["model"] = model
        return object()

    monkeypatch.setattr(llm_module, "_create_gemini", fake_gemini)
    monkeypatch.setattr(llm_module, "_create_anthropic", fake_anthropic)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable: runnable)

    with use_run_context(llm_selection={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}):
        llm_module.create_llm()

    assert called == {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    }


def test_run_telemetry_collector_builds_costs_for_llm_and_perplexity() -> None:
    collector = RunTelemetryCollector()
    collector.record_llm_usage(
        provider="gemini",
        model="gemini-3.1-flash-lite-preview",
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
    )
    collector.record_perplexity_search(metadata={"query": "market size"})

    costs = collector.build_run_costs()

    assert costs["status"] == "complete"
    assert costs["llm_tokens"] == {"prompt": 1_000, "completion": 500, "total": 1_500}
    assert costs["perplexity_search"]["requests"] == 1
    assert costs["llm_usd"] == 0.001
    assert costs["perplexity_usd"] == 0.005
    assert costs["total_usd"] == 0.006
    assert costs["by_model"][0]["label"] == "Gemini 3.1 Flash Lite"


def test_run_telemetry_collector_marks_partial_when_usage_missing() -> None:
    collector = RunTelemetryCollector()
    collector.record_llm_usage(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    costs = collector.build_run_costs()

    assert costs["status"] == "partial"
    assert costs["llm_usd"] is None
    assert costs["total_usd"] is None


def test_throttled_runnable_records_retry_events_with_stage_context() -> None:
    class RetryableError(Exception):
        status_code = 429

    class FakeRunnable:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RetryableError("rate limit")
            return {"ok": True}

    collector = RunTelemetryCollector()
    runnable = rate_limit_module.ThrottledRunnable(
        FakeRunnable(),
        throttle=rate_limit_module.InvocationThrottle(
            max_concurrent=1,
            min_interval_sec=0.0,
            start_jitter_sec=0.0,
        ),
        retry_policy=rate_limit_module.RetryPolicy(
            max_retries=1,
            base_delay_sec=0.0,
            max_delay_sec=0.0,
            jitter_sec=0.0,
        ),
    )

    async def _run():
        with use_run_context(
            llm_selection={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            telemetry_collector=collector,
        ):
            with use_company_context("alpha"):
                with use_stage_context("batch_scoring"):
                    return await runnable.ainvoke("payload")

    assert asyncio.run(_run()) == {"ok": True}
    retry_rows = [row for row in collector.snapshot_model_executions() if row.get("status") == "retrying"]
    assert len(retry_rows) == 1
    assert retry_rows[0]["stage"] == "batch_scoring"
    assert retry_rows[0]["company_slug"] == "alpha"
    assert retry_rows[0]["provider"] == "anthropic"
    assert retry_rows[0]["metadata"]["rate_limited"] is True


def test_extract_usage_metadata_reads_usage_from_generation_message_object() -> None:
    class FakeMessage:
        usage_metadata = {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }

    class FakeGeneration:
        message = FakeMessage()

    class FakeResult:
        generations = [[FakeGeneration()]]
        llm_output = {}
        response_metadata = {}

        def dict(self):
            return {
                "generations": [[{"message": {}}]],
                "llm_output": {},
                "response_metadata": {},
            }

    usage = llm_module._extract_usage_metadata(FakeResult())

    assert usage == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }


def test_api_config_exposes_default_and_available_models(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/config")
        assert response.status_code == 200
        payload = response.json()

    assert payload["default_llm"]["provider"] == "gemini"
    assert payload["default_llm"]["model"] == "gemini-3.1-flash-lite-preview"
    assert payload["default_llm"]["label"] == "Gemini 3.1 Flash Lite"
    providers = {item["provider"] for item in payload["available_models"]}
    assert providers == {"gemini", "anthropic", "openai"}
    assert any(item["model"] == "gemini-3.1-flash-lite-preview" and item["available"] for item in payload["available_models"])
    assert any(item["model"] == "gpt-5-mini" and item["available"] for item in payload["available_models"])


def test_start_analysis_persists_requested_llm_selection(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    started = {"called": False}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            started["called"] = True

    monkeypatch.setattr(web_app_module.threading, "Thread", FakeThread)

    with TestClient(app) as client:
        _login(client)
        upload = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )
        assert upload.status_code == 200
        job_id = upload.json()["job_id"]

        analyze = client.post(
            f"/api/analyze/{job_id}",
            json={
                "input_mode": "pitchdeck",
                "llm_provider": "anthropic",
                "llm_model": "claude-haiku-4-5-20251001",
            },
        )
        assert analyze.status_code == 200
        assert started["called"] is True

        cache = web_app_module._results_cache[job_id]
        assert cache["llm_selection"]["provider"] == "anthropic"
        assert cache["llm_selection"]["model"] == "claude-haiku-4-5-20251001"
        assert cache["run_config"]["llm_provider"] == "anthropic"
        assert cache["run_config"]["llm_model"] == "claude-haiku-4-5-20251001"


def test_start_analysis_normalizes_google_provider_alias(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(web_app_module.threading, "Thread", FakeThread)

    with TestClient(app) as client:
        _login(client)
        upload = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )
        job_id = upload.json()["job_id"]

        analyze = client.post(
            f"/api/analyze/{job_id}",
            json={
                "input_mode": "pitchdeck",
                "llm_provider": "google",
                "llm_model": "gemini-3.1-flash-lite-preview",
            },
        )
        assert analyze.status_code == 200

        cache = web_app_module._results_cache[job_id]
        assert cache["llm_selection"]["provider"] == "gemini"
        assert cache["run_config"]["llm_provider"] == "gemini"


def test_status_returns_saved_run_llm_from_results(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")

    job_id = "job-llm"
    web_app_module._jobs[job_id] = AnalysisStatus(
        job_id=job_id,
        status="done",
        progress="Analysis complete",
        results={},
    )
    web_app_module._results_cache[job_id] = {
        "results": {
            "mode": "single",
            "company_name": "Acme",
            "llm": "GPT-5 mini",
            "llm_selection": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "label": "Claude Haiku 4.5",
            },
            "run_costs": {
                "currency": "USD",
                "status": "complete",
                "total_usd": 0.1,
                "llm_usd": 0.1,
                "perplexity_usd": 0.0,
                "llm_tokens": {"prompt": 100, "completion": 100, "total": 200},
                "perplexity_search": {"requests": 0, "total_usd": 0.0},
                "by_model": [],
            },
        }
    }

    with TestClient(app) as client:
        _login(client)
        response = client.get(f"/api/status/{job_id}")
        assert response.status_code == 200
        payload = response.json()

    assert payload["llm"] == "Claude Haiku 4.5"
