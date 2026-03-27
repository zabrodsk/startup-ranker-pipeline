import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.llm_catalog import (
    available_models_payload,
    validate_chat_requested_selection,
    validate_requested_selection,
)
from agent.llm_policy import (
    build_default_phase_model_policy,
    build_phase_model_policy,
    build_pipeline_policy,
    phase_model_defaults_payload,
    premium_phase_options_payload,
    quality_tiers_payload,
)
from agent.run_context import (
    RunTelemetryCollector,
    set_current_llm_request_settings,
    use_company_context,
    use_phase_llm,
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


def test_upload_creates_pending_job_without_model_rebuild_error() -> None:
    from web import app as web_app_module

    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/upload",
            files={"files": ("deck.txt", io.BytesIO(b"sample content"), "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()
    job_id = payload["job_id"]
    assert job_id
    assert AnalysisStatus.model_fields
    assert job_id in web_app_module._jobs
    assert web_app_module._jobs[job_id].status == "pending"


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


def test_chat_model_validation_accepts_legacy_openai_alias(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    entry = validate_chat_requested_selection("openai", "gpt-5-mini")

    assert entry is not None
    assert entry.provider == "openai"
    assert entry.model == "gpt-5-mini"


def test_available_models_payload_marks_availability(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    models = available_models_payload()
    gemini = next(item for item in models if item["model"] == "gemini-3.1-flash-lite-preview")
    assert {item["model"] for item in models} == {
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview",
        "claude-haiku-4-5-20251001",
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5",
        "gpt-4.1-mini",
        "o4-mini",
        "gpt-5.2",
        "gpt-5.4",
        "openai/gpt-5-mini",
        "openai/gpt-5",
        "openai/gpt-4.1-mini",
        "openrouter/hunter-alpha",
    }

    assert gemini["available"] is True
    assert gemini["selectable"] is True
    assert gemini["pricing_available"] is True
    assert gemini["summary"] == "Budget speed"
    assert gemini["supports_creativity_control"] is False
    assert all(
        item["available"] is False
        for item in models
        if item["provider"] in {"openai", "openrouter"}
    )
    assert all(
        item["available"] is False
        for item in models
        if item["provider"] == "openrouter"
    )
    assert gemini["tier"] == "budget"
    assert next(item for item in models if item["model"] == "gpt-5")["supports_creativity_control"] is True


def test_validate_requested_selection_accepts_new_catalog_models(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    assert validate_requested_selection("gemini", "gemini-2.5-flash") is not None
    assert validate_requested_selection("gemini", "gemini-2.5-flash-lite") is not None
    assert validate_requested_selection("gemini", "gemini-3.1-pro-preview") is not None
    assert validate_requested_selection("openai", "o4-mini") is not None
    assert validate_requested_selection("openai", "gpt-5.2") is not None
    assert validate_requested_selection("openai", "gpt-5.4") is not None
    assert validate_requested_selection("openai", "gpt-5.4-mini") is not None
    assert validate_requested_selection("openai", "gpt-5.4-nano") is not None


def test_build_pipeline_policy_resolves_cheap_and_premium(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    cheap = build_pipeline_policy("cheap")
    assert cheap.answering["provider"] == "gemini"
    assert cheap.decomposition["provider"] == "gemini"
    assert cheap.ranking["provider"] == "gemini"

    premium = build_pipeline_policy(
        "premium",
        {
            "decomposition": "claude",
            "generation": "gpt5",
            "evaluation": "claude",
            "ranking": "gpt5",
        },
    )
    assert premium.answering["provider"] == "gemini"
    assert premium.critique["provider"] == "gemini"
    assert premium.refinement["provider"] == "gemini"
    assert premium.decomposition["provider"] == "anthropic"
    assert premium.generation["model"] == "gpt-5"
    assert premium.evaluation["provider"] == "anthropic"
    assert premium.ranking["model"] == "gpt-5"


def test_build_phase_model_policy_preserves_supported_creativity_and_drops_unsupported(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    policy = build_phase_model_policy(
        {
            "decomposition": {"provider": "openai", "model": "gpt-5", "creativity": 0.9},
            "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "creativity": 0.7},
            "generation": {"provider": "openai", "model": "gpt-4.1-mini", "creativity": 0.6},
            "evaluation": {"provider": "openai", "model": "gpt-5.4-mini", "creativity": 0.8},
            "ranking": {"provider": "openai", "model": "o4-mini", "creativity": 0.4},
        }
    )

    assert policy.decomposition["creativity"] == 0.9
    assert "creativity" not in policy.answering
    assert policy.generation["creativity"] == 0.6
    assert "creativity" not in policy.evaluation
    assert policy.ranking["creativity"] == 0.4
    assert "creativity" not in policy.critique
    assert "creativity" not in policy.refinement


def test_build_pipeline_policy_falls_back_between_claude_and_gpt5(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    premium = build_pipeline_policy("premium", {"evaluation": "gpt5"})

    assert premium.evaluation["provider"] == "anthropic"
    assert premium.evaluation["model"] == "claude-haiku-4-5-20251001"


def test_build_phase_model_policy_resolves_five_user_facing_phases(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    policy = build_phase_model_policy(
        {
            "decomposition": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
            "generation": {"provider": "openai", "model": "gpt-5"},
            "evaluation": {"provider": "openai", "model": "gpt-5.4-mini"},
            "ranking": {"provider": "openai", "model": "gpt-4.1-mini"},
        }
    )

    assert policy.decomposition["provider"] == "anthropic"
    assert policy.answering["provider"] == "gemini"
    assert policy.critique["provider"] == "gemini"
    assert policy.refinement["provider"] == "gemini"
    assert policy.generation["model"] == "gpt-5"
    assert policy.evaluation["model"] == "gpt-5.4-mini"
    assert policy.ranking["model"] == "gpt-4.1-mini"


def test_phase_model_defaults_follow_new_analysis_recommendations(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    defaults = phase_model_defaults_payload()

    assert defaults["decomposition"] == {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
    }
    assert defaults["answering"] == {
        "provider": "openai",
        "model": "gpt-5.4-nano",
        "label": "GPT-5.4 nano",
    }
    assert defaults["generation"] == {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
    }
    assert defaults["evaluation"] == {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
    }
    assert defaults["ranking"] == {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
    }


def test_build_phase_model_policy_requires_all_user_facing_phases(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    try:
        build_phase_model_policy(
            {
                "decomposition": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
                "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
            }
        )
    except ValueError as exc:
        assert "missing required phases" in str(exc)
    else:
        raise AssertionError("Expected incomplete phase_models payload to raise ValueError")


def test_is_authentication_api_error_detects_status_and_message() -> None:
    class StatusAuthError(Exception):
        status_code = 401

    class TextAuthError(Exception):
        pass

    assert rate_limit_module.is_authentication_api_error(StatusAuthError("bad key")) is True
    assert (
        rate_limit_module.is_authentication_api_error(
            TextAuthError("authentication_error: invalid x-api-key")
        )
        is True
    )
    assert rate_limit_module.is_authentication_api_error(Exception("timeout")) is False


def test_model_catalog_validation_accepts_openai_entries_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    entry = validate_requested_selection("openai", "gpt-5.4-mini")

    assert entry is not None
    assert entry.provider == "openai"
    assert entry.model == "gpt-5.4-mini"


def test_model_catalog_validation_accepts_openrouter_entries_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    try:
        validate_requested_selection("openrouter", "openrouter/hunter-alpha")
    except ValueError as exc:
        assert "structured-output analysis runs" in str(exc)
    else:
        raise AssertionError("Expected incompatible structured-output model to raise ValueError")


def test_model_catalog_validation_accepts_openrouter_gpt5_mini_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    entry = validate_requested_selection("openrouter", "openai/gpt-5-mini")

    assert entry is not None
    assert entry.provider == "openrouter"
    assert entry.model == "openai/gpt-5-mini"


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
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}):
        llm_module.create_llm()

    assert called == {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    }


def test_create_chat_llm_always_uses_fixed_gemini_model(monkeypatch) -> None:
    called = {}

    def fake_gemini(model, temperature, timeout_s, max_retries):
        called["model"] = model
        called["temperature"] = temperature
        return object()

    monkeypatch.setattr(llm_module, "_create_gemini", fake_gemini)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL_NAME", "claude-haiku-4-5-20251001")

    llm_module.create_chat_llm()

    assert called["model"] == "gemini-3.1-flash-lite-preview"


def test_create_llm_uses_openrouter_selection_and_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    called = {}

    def fake_openrouter(model, temperature, timeout_s, max_retries):
        called["provider"] = "openrouter"
        called["model"] = model

        return object()

    monkeypatch.setattr(llm_module, "_create_openrouter", fake_openrouter)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openrouter", "model": "openrouter/hunter-alpha"}):
        llm_module.create_llm()

    assert called == {
        "provider": "openrouter",
        "model": "openrouter/hunter-alpha",
    }


def test_create_llm_respects_requested_temperature_for_gpt5_by_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_GPT5_TEMPERATURE_MODE", raising=False)

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5"}):
        llm_module.create_llm(temperature=0.0)

    assert called == {"model": "gpt-5", "temperature": 0.0, "reasoning_effort": None}


def test_create_llm_can_force_temperature_one_for_gpt5(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_GPT5_TEMPERATURE_MODE", "force_one")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5"}):
        llm_module.create_llm(temperature=0.0)

    assert called == {"model": "gpt-5", "temperature": 1.0, "reasoning_effort": None}


def test_create_llm_passes_requested_temperature_for_non_gpt5_openai(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_GPT5_TEMPERATURE_MODE", "force_one")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-4.1-mini"}):
        llm_module.create_llm(temperature=0.3)

    assert called == {"model": "gpt-4.1-mini", "temperature": 0.3, "reasoning_effort": None}


def test_create_llm_prefers_selection_creativity_for_supported_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-4.1-mini", "creativity": 0.8}):
        llm_module.create_llm(temperature=0.3)

    assert called == {"model": "gpt-4.1-mini", "temperature": 0.8, "reasoning_effort": None}


def test_create_llm_passes_requested_temperature_for_o4_mini(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "o4-mini"}):
        llm_module.create_llm(temperature=0.0)

    assert called == {"model": "o4-mini", "temperature": 0.0, "reasoning_effort": None}


def test_create_llm_maps_gpt54_mini_decomposition_to_temperature_plus_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5.4-mini"}):
        with use_stage_context("decomposition"):
            llm_module.create_llm(temperature=0.2)

    assert called == {"model": "gpt-5.4-mini", "temperature": 0.5, "reasoning_effort": "none"}


def test_create_llm_ignores_selection_creativity_for_unsupported_reasoning_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5.4-mini", "creativity": 0.9}):
        with use_stage_context("decomposition"):
            llm_module.create_llm(temperature=0.2)

    assert called == {"model": "gpt-5.4-mini", "temperature": 0.5, "reasoning_effort": "none"}


def test_create_llm_maps_gpt54_nano_answering_to_temperature_plus_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5.4-nano"}):
        with use_stage_context("answering"):
            llm_module.create_llm(temperature=0.2)

    assert called == {"model": "gpt-5.4-nano", "temperature": 0.0, "reasoning_effort": "none"}


def test_create_llm_maps_gpt54_mini_ranking_upside_to_temperature_plus_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5.4-mini"}):
        with use_stage_context("ranking_upside_score"):
            llm_module.create_llm(temperature=0.0)

    assert called == {"model": "gpt-5.4-mini", "temperature": 0.7, "reasoning_effort": "none"}


def test_create_llm_maps_gpt52_ranking_upside_to_temperature_plus_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5.2"}):
        with use_stage_context("ranking_upside_score"):
            llm_module.create_llm(temperature=0.0)

    assert called == {"model": "gpt-5.2", "temperature": 0.7, "reasoning_effort": "none"}


def test_create_llm_maps_gpt54_mini_evaluation_to_reasoning_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    called = {}

    def fake_openai(model, temperature, timeout_s, max_retries, reasoning_effort=None):
        called["model"] = model
        called["temperature"] = temperature
        called["reasoning_effort"] = reasoning_effort
        return object()

    monkeypatch.setattr(llm_module, "_create_openai", fake_openai)
    monkeypatch.setattr(llm_module, "wrap_llm", lambda runnable, **kwargs: runnable)

    with use_run_context(llm_selection={"provider": "openai", "model": "gpt-5.4-mini"}):
        with use_stage_context("evaluation"):
            llm_module.create_llm(temperature=0.0)

    assert called == {"model": "gpt-5.4-mini", "temperature": None, "reasoning_effort": "medium"}


def test_use_phase_llm_temporarily_overrides_selection() -> None:
    from agent.run_context import get_current_llm_selection

    with use_run_context(llm_selection={"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"}):
        assert get_current_llm_selection()["provider"] == "gemini"
        with use_phase_llm({"provider": "openai", "model": "gpt-5"}):
            assert get_current_llm_selection()["provider"] == "openai"
            assert get_current_llm_selection()["model"] == "gpt-5"
        assert get_current_llm_selection()["provider"] == "gemini"


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


def test_run_telemetry_collector_keeps_known_spend_when_some_usage_missing() -> None:
    collector = RunTelemetryCollector()
    collector.record_llm_usage(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
    )
    collector.record_llm_usage(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    costs = collector.build_run_costs()

    assert costs["status"] == "partial"
    assert costs["llm_usd"] == 0.0035
    assert costs["total_usd"] == 0.0035
    assert costs["by_model"][0]["label"] == "Claude Haiku 4.5"
    assert costs["by_model"][0]["usd"] == 0.0035
    assert costs["by_model"][0]["pricing_available"] is False
    assert costs["by_model"][0]["partial"] is True


def test_run_costs_ignore_non_done_llm_events_for_partial_status() -> None:
    collector = RunTelemetryCollector()
    collector.record_llm_usage(
        provider="openai",
        model="gpt-5-nano",
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
    )
    collector.record_execution_event(
        service="llm",
        status="retrying",
        provider="openai",
        model="gpt-5-nano",
        metadata={"attempt": 1},
    )
    collector.record_execution_event(
        service="llm",
        status="error",
        provider="openai",
        model="gpt-5-nano",
        error_message="rate limit",
    )

    costs = collector.build_run_costs()

    assert costs["status"] == "complete"
    assert costs["llm_usd"] == 0.00025
    assert costs["total_usd"] == 0.00025
    assert costs["by_model"][0]["label"] == "GPT-5 nano"
    assert costs["by_model"][0]["partial"] is False
    assert costs["by_model"][0]["pricing_available"] is True


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


def test_estimate_usage_metadata_returns_token_counts_when_provider_usage_missing() -> None:
    usage = llm_module._estimate_usage_metadata(
        provider="openai",
        model="gpt-5",
        prompt_text="Summarize the startup traction.",
        completion_text="The company has strong month-over-month growth.",
    )

    assert usage is not None
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_telemetry_callback_estimates_usage_when_response_omits_metadata() -> None:
    collector = RunTelemetryCollector()
    run_id = uuid4()

    class FakeGeneration:
        message = AIMessage(content="Short answer")

    class FakeResult:
        generations = [[FakeGeneration()]]
        llm_output = {}
        response_metadata = {}

        def dict(self):
            return {
                "generations": [[{"message": {"content": "Short answer"}}]],
                "llm_output": {},
                "response_metadata": {},
            }

    with use_run_context(
        llm_selection={"provider": "openai", "model": "gpt-5"},
        telemetry_collector=collector,
    ):
        set_current_llm_request_settings(
            {
                "requested_temperature": 0.0,
                "effective_temperature": 0.0,
                "sampling_mode": "respect_requested",
                "requested_reasoning_effort": None,
                "effective_reasoning_effort": None,
            }
        )
        llm_module._TELEMETRY_CALLBACK.on_chat_model_start(
            {},
            [[HumanMessage(content="What is the moat?")]],
            run_id=run_id,
        )
        llm_module._TELEMETRY_CALLBACK.on_llm_end(FakeResult(), run_id=run_id)

    costs = collector.build_run_costs()
    rows = collector.snapshot_model_executions()

    assert costs["status"] == "complete"
    assert costs["llm_usd"] is not None
    assert rows[0]["prompt_tokens"] > 0
    assert rows[0]["completion_tokens"] > 0
    assert rows[0]["metadata"]["estimated_usage"] is True
    assert rows[0]["metadata"]["requested_temperature"] == 0.0
    assert rows[0]["metadata"]["effective_temperature"] == 0.0
    assert rows[0]["metadata"]["sampling_mode"] == "respect_requested"
    assert rows[0]["metadata"]["requested_reasoning_effort"] is None
    assert rows[0]["metadata"]["effective_reasoning_effort"] is None


def test_api_config_exposes_default_and_available_models(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/config")
        assert response.status_code == 200
        payload = response.json()

    assert payload["default_llm"]["provider"] == "gemini"
    assert payload["default_llm"]["model"] == "gemini-3.1-flash-lite-preview"
    assert payload["default_llm"]["label"] == "Gemini 3.1 Flash Lite"
    providers = {item["provider"] for item in payload["available_models"]}
    assert providers == {"gemini", "anthropic", "openai", "openrouter"}
    assert any(item["model"] == "gemini-3.1-flash-lite-preview" and item["available"] for item in payload["available_models"])
    assert any(item["model"] == "gpt-5.4-mini" and item["available"] for item in payload["available_models"])
    assert any(
        item["model"] == "openrouter/hunter-alpha"
        and item["available"]
        and not item["selectable"]
        and item["supports_structured_output"] is False
        for item in payload["available_models"]
    )
    assert payload["phase_model_defaults"] == phase_model_defaults_payload()
    assert payload["quality_tiers"] == quality_tiers_payload()
    assert payload["premium_phase_options"] == premium_phase_options_payload()


def test_start_analysis_requires_supabase_identity_when_auth_is_configured(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setattr(
        web_app_module,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            get_authenticated_supabase_user=lambda _token: None,
            insert_analysis_event=lambda *args, **kwargs: True,
            insert_job_status_history=lambda *args, **kwargs: True,
        ),
    )

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
            json={"input_mode": "pitchdeck"},
        )

    assert analyze.status_code == 401
    assert "Supabase sign-in required" in analyze.json()["detail"]
    web_app_module._jobs.pop(job_id, None)
    web_app_module._results_cache.pop(job_id, None)


def test_start_analysis_persists_supabase_starter_identity(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    started = {"called": False}
    upserted = {}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            started["called"] = True

    monkeypatch.setattr(web_app_module.threading, "Thread", FakeThread)
    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(web_app_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        web_app_module,
        "db",
        SimpleNamespace(
            is_configured=lambda: True,
            get_authenticated_supabase_user=lambda token: {
                "id": "user-123",
                "email": "jane@example.com",
                "user_metadata": {"full_name": "Jane Doe"},
            } if token == "good-token" else None,
            upsert_job=lambda job_id, **kwargs: upserted.update({"job_id": job_id, **kwargs}) or "job-uuid",
            upsert_job_control=lambda *args, **kwargs: True,
            insert_analysis_event=lambda *args, **kwargs: True,
            insert_job_status_history=lambda *args, **kwargs: True,
        ),
    )

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
            headers={"Authorization": "Bearer good-token"},
            json={
                "input_mode": "pitchdeck",
                "llm_provider": "anthropic",
                "llm_model": "claude-haiku-4-5-20251001",
            },
        )
        assert analyze.status_code == 200
        assert started["called"] is True

        cache = web_app_module._results_cache[job_id]
        assert cache["run_config"]["started_by_user_id"] == "user-123"
        assert cache["run_config"]["started_by_email"] == "jane@example.com"
        assert cache["run_config"]["started_by_display_name"] == "Jane Doe"
        assert cache["run_config"]["started_by_label"] == "Jane Doe"
        assert web_app_module._jobs[job_id].started_by_label == "Jane Doe"
        assert upserted["run_config"]["started_by_label"] == "Jane Doe"
    web_app_module._jobs.pop(job_id, None)
    web_app_module._results_cache.pop(job_id, None)


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


def test_start_analysis_persists_quality_tier_selection(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

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
        job_id = upload.json()["job_id"]

        analyze = client.post(
            f"/api/analyze/{job_id}",
            json={
                "input_mode": "pitchdeck",
                "quality_tier": "premium",
                "premium_phase_models": {
                    "decomposition": "claude",
                    "generation": "gpt5",
                    "evaluation": "claude",
                    "ranking": "gpt5",
                },
            },
        )
        assert analyze.status_code == 200
        assert started["called"] is True

        cache = web_app_module._results_cache[job_id]
        assert cache["quality_tier"] == "premium"
        assert cache["llm_selection"]["provider"] == "gemini"
        assert cache["premium_phase_models"]["decomposition"] == "claude"
        assert cache["effective_phase_models"] == {
            "decomposition": "claude",
            "answering": "gemini",
            "generation": "gpt5",
            "critique": "gemini",
            "evaluation": "claude",
            "refinement": "gemini",
            "ranking": "gpt5",
        }
        assert cache["run_config"]["quality_tier"] == "premium"
        assert cache["run_config"]["llm_provider"] == "gemini"
        assert cache["run_config"]["llm"] == "Premium tier · QA Gemini · Critical phases mixed"


def test_start_analysis_persists_phase_model_selection(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

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
        job_id = upload.json()["job_id"]

        analyze = client.post(
            f"/api/analyze/{job_id}",
            json={
                "input_mode": "pitchdeck",
                "phase_models": {
                    "decomposition": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
                    "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
                    "generation": {"provider": "openai", "model": "gpt-5", "creativity": 0.8},
                    "evaluation": {"provider": "openai", "model": "gpt-5.4-mini", "creativity": 0.9},
                    "ranking": {"provider": "openai", "model": "gpt-4.1-mini", "creativity": 0.4},
                },
            },
        )
        assert analyze.status_code == 200
        assert started["called"] is True

        cache = web_app_module._results_cache[job_id]
        assert cache["phase_models"]["decomposition"]["provider"] == "anthropic"
        assert cache["llm_selection"]["provider"] == "gemini"
        assert cache["quality_tier"] is None
        assert cache["effective_phase_models"]["critique"]["provider"] == "gemini"
        assert cache["effective_phase_models"]["generation"]["creativity"] == 0.8
        assert "creativity" not in cache["effective_phase_models"]["evaluation"]
        assert cache["effective_phase_models"]["ranking"]["creativity"] == 0.4
        assert cache["effective_phase_models"]["ranking"]["model"] == "gpt-4.1-mini"
        assert cache["run_config"]["phase_models"]["generation"]["model"] == "gpt-5"
        assert cache["run_config"]["phase_models"]["generation"]["creativity"] == 0.8
        assert cache["run_config"]["llm"] == (
            "Per-phase · D Claude Haiku 4.5 · A Gemini 3.1 Flash Lite"
            " · G GPT-5 · E GPT-5.4 mini · R GPT-4.1 mini"
        )


def test_start_analysis_defaults_to_phase_model_policy_when_no_selection_is_provided(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

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
        job_id = upload.json()["job_id"]

        analyze = client.post(
            f"/api/analyze/{job_id}",
            json={"input_mode": "pitchdeck"},
        )
        assert analyze.status_code == 200
        assert started["called"] is True

        cache = web_app_module._results_cache[job_id]
        expected_policy = build_default_phase_model_policy()
        expected_effective = expected_policy.as_dict()
        expected_defaults = phase_model_defaults_payload()
        assert cache["llm_selection"] == expected_effective["answering"]
        assert cache["phase_models"] == expected_defaults
        assert cache["effective_phase_models"] == expected_effective
        assert cache["quality_tier"] is None
        assert cache["run_config"]["phase_models"] == expected_defaults
        assert cache["run_config"]["llm"] == (
            "Per-phase · D GPT-5.4 mini · A GPT-5.4 nano"
            " · G GPT-5.4 mini · E GPT-5.4 mini · R GPT-5.4 mini"
        )


def test_start_analysis_rejects_incomplete_phase_model_selection(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            raise AssertionError("analysis thread should not start for invalid phase_models")

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
                "phase_models": {
                    "decomposition": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
                    "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
                },
            },
        )

    assert analyze.status_code == 422
    assert "phase_models is missing required phases" in analyze.text


def test_start_analysis_rejects_invalid_creativity(monkeypatch) -> None:
    from web import app as web_app_module

    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            raise AssertionError("analysis thread should not start for invalid creativity")

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
                "phase_models": {
                    "decomposition": {"provider": "openai", "model": "gpt-5", "creativity": 9},
                    "answering": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
                    "generation": {"provider": "openai", "model": "gpt-5"},
                    "evaluation": {"provider": "openai", "model": "gpt-5.4-mini"},
                    "ranking": {"provider": "openai", "model": "gpt-4.1-mini"},
                },
            },
        )

    assert analyze.status_code == 422
    assert "Creativity must be between 0.0 and 2.0." in analyze.text


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
