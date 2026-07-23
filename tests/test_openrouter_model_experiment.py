from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import agent.llm as llm_module  # noqa: E402
from agent.llm_catalog import available_models_payload  # noqa: E402
from agent.model_profiles import (  # noqa: E402
    model_profiles_payload,
    resolve_model_profile,
)
from agent.run_context import (  # noqa: E402
    RunTelemetryCollector,
    use_phase_llm,
    use_run_context,
    use_stage_context,
)
from web import app as web_app_module  # noqa: E402


@pytest.fixture(autouse=True)
def _staging_environment(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")


def test_openrouter_experiment_models_require_flag_and_dedicated_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", raising=False)

    models = {item["model"]: item for item in available_models_payload()}

    for model in (
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    ):
        assert models[model]["available"] is False
        assert models[model]["selectable"] is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")

    enabled = {item["model"]: item for item in available_models_payload()}
    assert enabled["moonshotai/kimi-k2.6"]["supports_temperature_control"] is False
    assert enabled["moonshotai/kimi-k2.6"]["supports_reasoning_toggle"] is True
    assert enabled["moonshotai/kimi-k2.6"]["reasoning_effort_options"] == []
    assert enabled["z-ai/glm-5.2"]["reasoning_effort_options"] == ["high", "xhigh"]
    assert enabled["deepseek/deepseek-v4-flash"]["selectable"] is True
    assert enabled["deepseek/deepseek-v4-pro"]["selectable"] is True

    monkeypatch.setenv("APP_ENV", "production")
    production = {item["model"]: item for item in available_models_payload()}
    assert production["moonshotai/kimi-k2.6"]["selectable"] is False


def test_openrouter_adapter_enforces_privacy_reasoning_and_strict_schema(monkeypatch) -> None:
    import langchain_openai

    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def with_structured_output(self, schema, **kwargs):
            captured["structured"] = {"schema": schema, **kwargs}
            return "structured"

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")

    model = llm_module._create_openrouter(
        "moonshotai/kimi-k2.6",
        None,
        90.0,
        0,
        reasoning_enabled=True,
        routing={"only": ["deepinfra"], "allow_fallbacks": False},
    )

    init = captured["init"]
    assert init["api_key"] == "openrouter-key"
    assert "temperature" not in init
    assert init["extra_body"]["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "only": ["deepinfra"],
        "allow_fallbacks": False,
    }
    assert init["extra_body"]["reasoning"] == {"enabled": True}
    assert "reasoning" not in init
    assert init["default_headers"]["X-OpenRouter-Metadata"] == "enabled"

    assert model.with_structured_output({"type": "object"}) == "structured"
    assert captured["structured"]["method"] == "json_schema"
    assert captured["structured"]["strict"] is True

    monkeypatch.delenv("OPENROUTER_API_KEY")
    try:
        llm_module._create_openrouter("z-ai/glm-5.2", 0.3, 90.0, 0)
    except ValueError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("OPENAI_API_KEY must not authorize OpenRouter")


def test_openrouter_adapter_preserves_router_metadata(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    model = llm_module._create_openrouter(
        "deepseek/deepseek-v4-flash",
        0.2,
        90.0,
        0,
        reasoning_effort="high",
    )

    result = model._runnable._create_chat_result(
        {
            "id": "gen-openrouter-raw",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek/deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "openrouter_metadata": {
                "endpoints": {
                    "available": [{"provider": "Pinned Provider", "selected": True}]
                }
            },
        }
    )

    assert result.llm_output["openrouter_metadata"]["endpoints"]["available"] == [
        {"provider": "Pinned Provider", "selected": True}
    ]


def test_named_profiles_resolve_immutable_seven_stage_policies(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    profiles = {item["id"]: item for item in model_profiles_payload()}
    assert set(profiles) == {"gpt_current", "kimi_k26", "glm_deepseek_flash"}
    assert all(item["available"] for item in profiles.values())

    kimi = resolve_model_profile("kimi_k26")
    assert set(kimi.phase_models) == {
        "decomposition",
        "answering",
        "generation",
        "critique",
        "evaluation",
        "refinement",
        "ranking",
    }
    assert {selection["model"] for selection in kimi.phase_models.values()} == {
        "moonshotai/kimi-k2.6"
    }
    assert kimi.phase_models["decomposition"]["reasoning_enabled"] is True
    assert kimi.phase_models["answering"]["reasoning_enabled"] is False
    assert kimi.phase_models["generation"]["reasoning_enabled"] is False
    assert kimi.phase_models["critique"]["reasoning_enabled"] is True
    assert kimi.phase_models["evaluation"]["reasoning_enabled"] is True
    assert kimi.phase_models["refinement"]["reasoning_enabled"] is True
    assert kimi.phase_models["ranking"]["stage_settings"] == {
        "ranking_dimension_score": {"reasoning_enabled": True},
        "ranking_upside_score": {"reasoning_enabled": False},
        "ranking_executive_summary": {"reasoning_enabled": True},
    }
    assert all("temperature" not in selection for selection in kimi.phase_models.values())

    hybrid = resolve_model_profile("glm_deepseek_flash")
    assert hybrid.phase_models["critique"]["model"] == "z-ai/glm-5.2"
    assert hybrid.phase_models["refinement"]["model"] == "deepseek/deepseek-v4-flash"
    assert hybrid.phase_models["evaluation"]["temperature"] == 0.1
    assert hybrid.phase_models["refinement"]["temperature"] is None
    assert hybrid.phase_models["ranking"]["stage_settings"] == {
        "ranking_dimension_score": {"temperature": 0.1, "reasoning_effort": "high"},
        "ranking_upside_score": {"temperature": 0.7, "reasoning_enabled": False},
        "ranking_executive_summary": {"temperature": 0.3, "reasoning_effort": "high"},
    }
    assert hybrid.openrouter_routing == {
        "require_parameters": True,
        "data_collection": "allow",
        "zdr": True,
    }


def test_profile_stage_matrix_emits_exact_openrouter_parameters(monkeypatch) -> None:
    import langchain_openai

    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    captured: list[dict[str, object]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)

    cases = (
        # Profile B: Kimi fixed sampling, thinking only for analytical judgment.
        ("kimi_k26", "decomposition", "decomposition", 0.5, None, {"enabled": True}),
        ("kimi_k26", "answering", "answering", 0.2, None, {"enabled": False}),
        ("kimi_k26", "generation", "generation_pro", 0.5, None, {"enabled": False}),
        ("kimi_k26", "generation", "generation_contra", 0.5, None, {"enabled": False}),
        ("kimi_k26", "critique", "critique", 0.5, None, {"enabled": True}),
        ("kimi_k26", "evaluation", "evaluation", 0.0, None, {"enabled": True}),
        ("kimi_k26", "refinement", "refinement", 0.7, None, {"enabled": True}),
        ("kimi_k26", "ranking", "ranking_dimension_score", 0.0, None, {"enabled": True}),
        ("kimi_k26", "ranking", "ranking_upside_score", 0.7, None, {"enabled": False}),
        ("kimi_k26", "ranking", "ranking_executive_summary", 0.3, None, {"enabled": True}),
        # Profile C: inexpensive non-thinking extraction/generation, reasoning for judgment.
        ("glm_deepseek_flash", "decomposition", "decomposition", 0.5, 0.5, {"effort": "high"}),
        ("glm_deepseek_flash", "answering", "answering", 0.2, 0.2, {"enabled": False}),
        ("glm_deepseek_flash", "generation", "generation_pro", 0.5, 0.5, {"enabled": False}),
        ("glm_deepseek_flash", "generation", "generation_contra", 0.5, 0.5, {"enabled": False}),
        ("glm_deepseek_flash", "critique", "critique", 0.5, 0.5, {"effort": "high"}),
        ("glm_deepseek_flash", "evaluation", "evaluation", 0.0, 0.1, {"effort": "high"}),
        ("glm_deepseek_flash", "refinement", "refinement", 0.7, None, {"effort": "high"}),
        ("glm_deepseek_flash", "ranking", "ranking_dimension_score", 0.0, 0.1, {"effort": "high"}),
        ("glm_deepseek_flash", "ranking", "ranking_upside_score", 0.7, 0.7, {"enabled": False}),
        ("glm_deepseek_flash", "ranking", "ranking_executive_summary", 0.3, 0.3, {"effort": "high"}),
    )

    for profile_id, phase, stage, requested_temperature, expected_temperature, expected_reasoning in cases:
        selection = dict(getattr(resolve_model_profile(profile_id).policy, phase))
        selection["openrouter_routing"] = {
            "only": ["deepinfra"],
            "allow_fallbacks": False,
        }
        with use_phase_llm(selection), use_stage_context(stage):
            llm_module.create_llm(temperature=requested_temperature)
        outgoing = captured[-1]
        assert outgoing.get("temperature") == expected_temperature
        if expected_temperature is None:
            assert "temperature" not in outgoing
        assert outgoing["extra_body"]["reasoning"] == expected_reasoning
        assert outgoing["extra_body"]["provider"] == {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "only": ["deepinfra"],
            "allow_fallbacks": False,
        }


def test_analysis_api_exposes_profiles_and_rejects_conflicting_selection(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(web_app_module, "_check_session", lambda _session_id: True)
    monkeypatch.setattr(web_app_module, "db", None)

    with TestClient(web_app_module.app) as client:
        config = client.get("/api/config")

    assert config.status_code == 200
    assert [profile["id"] for profile in config.json()["model_profiles"]] == [
        "gpt_current",
        "kimi_k26",
        "glm_deepseek_flash",
    ]

    with pytest.raises(ValueError, match="model_profile_id cannot be combined"):
        web_app_module.AnalyzeRequest(
            model_profile_id="kimi_k26",
            llm_provider="openrouter",
            llm_model="moonshotai/kimi-k2.6",
        )


def test_openrouter_usage_records_actual_cost_and_reasoning_breakdown(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    collector = RunTelemetryCollector()

    class FakeMessage:
        usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        }
        response_metadata = {
            "id": "gen-openrouter-1",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "cost": 0.001234,
                "prompt_tokens_details": {"cached_tokens": 25},
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        }

    class FakeGeneration:
        message = FakeMessage()

    class FakeResult:
        generations = [[FakeGeneration()]]
        llm_output = {
            "openrouter_metadata": {
                "requested": "deepseek/deepseek-v4-flash",
                "strategy": "direct",
                "attempt": 2,
                "endpoints": {
                    "available": [
                        {"provider": "Ignored Provider", "selected": False},
                        {"provider": "Example Provider", "selected": True},
                    ]
                },
            }
        }
        response_metadata = {}

    with use_run_context(
        llm_selection={"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        telemetry_collector=collector,
    ):
        llm_module._TELEMETRY_CALLBACK.on_llm_end(FakeResult(), run_id="run-1")

    row = collector.snapshot_model_executions()[0]
    assert row["metadata"]["actual_cost_usd"] == 0.001234
    assert row["metadata"]["catalog_estimated_cost_usd"] == row["estimated_cost_usd"]
    assert row["metadata"]["generation_id"] == "gen-openrouter-1"
    assert row["metadata"]["selected_provider"] == "Example Provider"
    assert row["metadata"]["reasoning_tokens"] == 12
    assert row["metadata"]["cached_tokens"] == 25
    assert row["metadata"]["provider_retry_count"] == 1
    assert collector.build_run_costs()["llm_usd"] == 0.001234


def test_llm_telemetry_records_success_latency(monkeypatch) -> None:
    collector = RunTelemetryCollector()
    monotonic_values = iter((10.0, 10.25))
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: next(monotonic_values))

    class FakeResult:
        generations = []
        llm_output = {
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            }
        }
        response_metadata = {}

    with use_run_context(
        llm_selection={"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        telemetry_collector=collector,
    ):
        llm_module._TELEMETRY_CALLBACK.on_llm_start({}, ["prompt"], run_id="latency-run")
        llm_module._TELEMETRY_CALLBACK.on_llm_end(FakeResult(), run_id="latency-run")

    assert collector.snapshot_model_executions()[0]["latency_ms"] == 250


def test_admin_pipeline_editor_accepts_all_seven_stages(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    full_policy = resolve_model_profile("glm_deepseek_flash").phase_models
    captured: dict[str, object] = {}

    class FakeDb:
        @staticmethod
        def admin_set_pipeline_model_defaults(value, *, updated_by):
            captured["value"] = value
            captured["updated_by"] = updated_by
            return value

    monkeypatch.setattr(web_app_module, "db", FakeDb())
    user = web_app_module.CurrentUser(
        id="admin-1",
        email="admin@example.com",
        role="admin",
        approved=True,
        display_name="Admin",
    )

    result = asyncio.run(
        web_app_module.admin_put_pipeline_models(
            {"phase_models": full_policy},
            user,
        )
    )

    assert set(result["phase_models"]) == set(full_policy)
    assert captured["value"]["critique"]["model"] == "z-ai/glm-5.2"
    assert captured["value"]["refinement"]["model"] == "deepseek/deepseek-v4-flash"


def test_start_analysis_persists_profile_and_resolved_policy(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OPENROUTER_MODEL_EXPERIMENT", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(web_app_module, "db", None)

    started: dict[str, object] = {}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            started["target"] = target
            started["daemon"] = daemon

        def start(self):
            started["called"] = True

    monkeypatch.setattr(web_app_module.threading, "Thread", FakeThread)
    job_id = "profile1"
    web_app_module._jobs[job_id] = web_app_module.AnalysisStatus(
        job_id=job_id,
        status="pending",
    )
    web_app_module._results_cache[job_id] = {"files": []}

    asyncio.run(
        web_app_module._start_analysis_job(
            job_id,
            web_app_module.AnalyzeRequest(
                model_profile_id="glm_deepseek_flash",
                input_mode="pitchdeck",
                use_specter_mcp=False,
            ),
            require_identity=False,
            skip_quality_preflight=True,
            skip_specter_mcp_preflight=True,
        )
    )

    run_config = web_app_module._results_cache[job_id]["run_config"]
    assert started["called"] is True
    assert run_config["model_profile_id"] == "glm_deepseek_flash"
    assert run_config["model_profile_arm"] == "C"
    assert run_config["phase_models"] is None
    assert run_config["pipeline_models"]["critique"]["model"] == "z-ai/glm-5.2"
    assert run_config["pipeline_models"]["refinement"]["model"] == "deepseek/deepseek-v4-flash"
    assert run_config["openrouter_routing"]["zdr"] is True


def test_staging_intake_uses_named_profiles_and_admin_keeps_advanced_controls() -> None:
    index_html = (ROOT / "web" / "static" / "index.html").read_text()
    portal_html = (ROOT / "web" / "static" / "portal.html").read_text()

    assert "data-model-profile-id" in index_html
    assert "profile.id === 'gpt_current'" in index_html
    assert "if (advancedSection) advancedSection.hidden = true;" in index_html
    assert "model_profile_id: analysisSelection.model_profile_id" in index_html
    assert "const PHASE_ORDER = [\"decomposition\", \"answering\", \"generation\", \"critique\", \"evaluation\", \"refinement\", \"ranking\"]" in portal_html
    assert 'critique: "C"' in portal_html
    assert 'refinement: "F"' in portal_html
    assert '["none","minimal","low","medium","high","xhigh"]' in portal_html
