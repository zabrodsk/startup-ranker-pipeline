import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent import benchmark
from agent.dataclasses.argument import Argument
from agent.run_context import get_current_collector


def test_benchmark_writes_two_scenarios_and_aggregates_runs(monkeypatch, tmp_path):
    input_dir = tmp_path / "acme"
    input_dir.mkdir()
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        benchmark,
        "current_default_selection",
        lambda: {"provider": "openai", "model": "gpt-5"},
    )

    counters = {"force_one": 0, "respect_requested": 0}

    async def fake_evaluate_startup(folder, k=8, config=None, use_web_search=False):
        sampling_mode = os.getenv("OPENAI_GPT5_TEMPERATURE_MODE", "respect_requested")
        counters[sampling_mode] += 1
        collector = get_current_collector()
        collector.record_llm_usage(
            provider="openai",
            model="gpt-5",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            stage="evaluation",
            metadata={
                "requested_temperature": 0.0,
                "effective_temperature": 1.0 if sampling_mode == "force_one" else 0.0,
                "sampling_mode": sampling_mode,
            },
        )
        collector.record_llm_usage(
            provider="openai",
            model="gpt-5",
            prompt_tokens=60,
            completion_tokens=30,
            total_tokens=90,
            stage="ranking_dimension_score",
            metadata={
                "requested_temperature": 0.0,
                "effective_temperature": 1.0 if sampling_mode == "force_one" else 0.0,
                "sampling_mode": sampling_mode,
            },
        )
        run_index = counters[sampling_mode]
        composite = 70 + run_index if sampling_mode == "force_one" else 80 + run_index
        return {
            "slug": folder.name,
            "final_state": {
                "final_decision": "invest" if composite >= 80 else "watch",
                "ranking_result": {
                    "strategy_fit_score": composite - 2,
                    "team_score": composite - 1,
                    "upside_score": composite,
                    "composite_score": composite,
                },
                "final_arguments": [
                    Argument(
                        content=f"Argument {run_index}",
                        argument_type="pro",
                        qa_indices=[0],
                        score=float(composite),
                    )
                ],
            },
        }

    monkeypatch.setattr(benchmark, "evaluate_startup", fake_evaluate_startup)

    asyncio.run(
        benchmark.async_main(
            [
                "--input",
                str(input_dir),
                "--repeats",
                "2",
                "--output",
                str(output_dir),
            ]
        )
    )

    summary = json.loads((output_dir / "summary.json").read_text())

    assert summary["repeat_count"] == 2
    assert {scenario["scenario_name"] for scenario in summary["scenarios"]} == {
        "current_gpt5_sampling",
        "corrected_gpt5_sampling",
    }

    force_one = next(
        scenario for scenario in summary["scenarios"] if scenario["sampling_mode"] == "force_one"
    )
    corrected = next(
        scenario
        for scenario in summary["scenarios"]
        if scenario["sampling_mode"] == "respect_requested"
    )

    assert force_one["repeat_count"] == 2
    assert corrected["repeat_count"] == 2
    assert force_one["ranking_metrics"]["composite_score"]["mean"] == 71.5
    assert corrected["ranking_metrics"]["composite_score"]["mean"] == 81.5
    assert force_one["stage_metrics"]["evaluation"]["llm_calls_total"] == 2
    assert corrected["stage_metrics"]["ranking_dimension_score"]["total_tokens_total"] == 180

    assert (output_dir / "current_gpt5_sampling" / "run_1.json").exists()
    assert (output_dir / "corrected_gpt5_sampling" / "run_2.json").exists()

    run_report = json.loads((output_dir / "corrected_gpt5_sampling" / "run_1.json").read_text())
    assert run_report["scenario_name"] == "corrected_gpt5_sampling"
    assert run_report["sampling_mode"] == "respect_requested"
    assert run_report["stage_telemetry"]["evaluation"]["sampling_modes"] == {
        "respect_requested": 1
    }
