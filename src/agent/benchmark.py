"""Internal benchmark runner for comparing GPT-5 sampling modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.batch import evaluate_startup
from agent.dataclasses.argument import Argument
from agent.dataclasses.config import Config
from agent.llm import _GPT5_TEMPERATURE_MODE_ENV
from agent.llm_catalog import current_default_selection
from agent.run_context import RunTelemetryCollector, use_run_context

SCENARIOS: tuple[tuple[str, str], ...] = (
    ("current_gpt5_sampling", "force_one"),
    ("corrected_gpt5_sampling", "respect_requested"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark current vs corrected GPT-5 temperature handling.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single startup folder to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per scenario (default: 3).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON file with pipeline config overrides.",
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        default=False,
        help="Enable web search during benchmark runs.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of evidence chunks to retrieve per question (default: 8).",
    )
    return parser.parse_args(argv)


def _load_config(path: str | None) -> Config | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    return Config.model_validate(payload)


@contextmanager
def _temporary_env(name: str, value: str) -> Any:
    original = os.getenv(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _safe_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _safe_min(values: list[float]) -> float | None:
    return round(min(values), 6) if values else None


def _safe_max(values: list[float]) -> float | None:
    return round(max(values), 6) if values else None


def _safe_stddev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return round(statistics.pstdev(values), 6)


def _serialize_argument(argument: Argument) -> dict[str, Any]:
    return {
        "type": argument.argument_type,
        "score": argument.score,
        "tracking_id": argument.tracking_id,
        "content": argument.refined_content or argument.content,
        "critique": argument.critique or "",
        "feedback": argument.argument_feedback or "",
        "qa_indices": list(argument.qa_indices or []),
    }


def _serialize_ranking_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return dict(result)


def _group_stage_telemetry(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = str(row.get("stage") or "unknown")
        entry = grouped.setdefault(
            stage,
            {
                "llm_calls": 0,
                "events": 0,
                "statuses": Counter(),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "models": Counter(),
                "sampling_modes": Counter(),
            },
        )
        entry["events"] += 1
        entry["statuses"][str(row.get("status") or "done")] += 1
        provider = row.get("provider")
        model = row.get("model")
        if provider and model:
            entry["models"][f"{provider}:{model}"] += 1
        metadata = row.get("metadata") or {}
        sampling_mode = metadata.get("sampling_mode")
        if sampling_mode:
            entry["sampling_modes"][str(sampling_mode)] += 1

        if row.get("service") != "llm" or str(row.get("status") or "").lower() != "done":
            continue

        entry["llm_calls"] += int(row.get("request_count") or 1)
        entry["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        entry["completion_tokens"] += int(row.get("completion_tokens") or 0)
        entry["total_tokens"] += int(row.get("total_tokens") or 0)
        entry["cost_usd"] += float(row.get("estimated_cost_usd") or 0.0)

    serialized: dict[str, dict[str, Any]] = {}
    for stage, entry in grouped.items():
        serialized[stage] = {
            "llm_calls": entry["llm_calls"],
            "events": entry["events"],
            "statuses": dict(entry["statuses"]),
            "prompt_tokens": entry["prompt_tokens"],
            "completion_tokens": entry["completion_tokens"],
            "total_tokens": entry["total_tokens"],
            "cost_usd": round(entry["cost_usd"], 8),
            "models": dict(entry["models"]),
            "sampling_modes": dict(entry["sampling_modes"]),
        }
    return serialized


def _aggregate_stage_summaries(
    summaries: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs_observed": 0,
            "llm_calls_total": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "cost_usd_total": 0.0,
        }
    )
    for summary in summaries:
        for stage, entry in summary.items():
            target = aggregated[stage]
            target["runs_observed"] += 1
            target["llm_calls_total"] += int(entry.get("llm_calls") or 0)
            target["prompt_tokens_total"] += int(entry.get("prompt_tokens") or 0)
            target["completion_tokens_total"] += int(entry.get("completion_tokens") or 0)
            target["total_tokens_total"] += int(entry.get("total_tokens") or 0)
            target["cost_usd_total"] += float(entry.get("cost_usd") or 0.0)

    serialized: dict[str, dict[str, Any]] = {}
    for stage, entry in aggregated.items():
        runs = max(1, int(entry["runs_observed"]))
        serialized[stage] = {
            **entry,
            "cost_usd_total": round(entry["cost_usd_total"], 8),
            "llm_calls_mean": round(entry["llm_calls_total"] / runs, 6),
            "total_tokens_mean": round(entry["total_tokens_total"] / runs, 6),
            "cost_usd_mean": round(entry["cost_usd_total"] / runs, 8),
        }
    return serialized


def _extract_top_argument_score(final_arguments: list[Argument]) -> float | None:
    scores = [float(argument.score) for argument in final_arguments if argument.score is not None]
    if not scores:
        return None
    return max(scores)


def _build_run_report(
    *,
    scenario_name: str,
    sampling_mode: str,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    wall_clock_seconds: float,
) -> dict[str, Any]:
    final_state = result.get("final_state") or {}
    final_arguments = list(final_state.get("final_arguments") or [])
    top_final_arguments = sorted(
        final_arguments,
        key=lambda argument: argument.score if argument.score is not None else float("-inf"),
        reverse=True,
    )[:5]

    return {
        "scenario_name": scenario_name,
        "sampling_mode": sampling_mode,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "slug": result.get("slug"),
        "final_decision": final_state.get("final_decision", "unknown"),
        "ranking_result": _serialize_ranking_result(final_state.get("ranking_result")),
        "top_argument_score": _extract_top_argument_score(final_arguments),
        "top_final_arguments": [_serialize_argument(argument) for argument in top_final_arguments],
        "run_costs": result.get("run_costs"),
        "stage_telemetry": _group_stage_telemetry(rows),
        "model_executions": rows,
    }


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _safe_mean(values),
        "min": _safe_min(values),
        "max": _safe_max(values),
        "stddev": _safe_stddev(values),
    }


def _aggregate_scenario_reports(
    scenario_name: str,
    sampling_mode: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_distribution = Counter(run.get("final_decision", "unknown") for run in runs)
    ranking_fields = {
        "strategy_fit_score": [],
        "team_score": [],
        "upside_score": [],
        "composite_score": [],
    }
    top_argument_scores: list[float] = []
    total_costs: list[float] = []
    wall_clock_seconds: list[float] = []
    stage_summaries: list[dict[str, dict[str, Any]]] = []

    for run in runs:
        ranking = run.get("ranking_result") or {}
        for field in ranking_fields:
            value = ranking.get(field)
            if isinstance(value, (int, float)):
                ranking_fields[field].append(float(value))
        if isinstance(run.get("top_argument_score"), (int, float)):
            top_argument_scores.append(float(run["top_argument_score"]))
        run_costs = run.get("run_costs") or {}
        total_usd = run_costs.get("total_usd")
        if isinstance(total_usd, (int, float)):
            total_costs.append(float(total_usd))
        wall_clock_seconds.append(float(run.get("wall_clock_seconds") or 0.0))
        stage_summaries.append(run.get("stage_telemetry") or {})

    return {
        "scenario_name": scenario_name,
        "sampling_mode": sampling_mode,
        "repeat_count": len(runs),
        "final_decision_distribution": dict(decision_distribution),
        "ranking_metrics": {
            field: _metric_summary(values)
            for field, values in ranking_fields.items()
        },
        "top_argument_score": _metric_summary(top_argument_scores),
        "total_cost_usd": _metric_summary(total_costs),
        "wall_clock_seconds": _metric_summary(wall_clock_seconds),
        "stage_metrics": _aggregate_stage_summaries(stage_summaries),
        "runs": [run["report_path"] for run in runs],
    }


async def _run_single_scenario(
    *,
    folder: Path,
    repeats: int,
    output_dir: Path,
    config: Config | None,
    use_web_search: bool,
    k: int,
    scenario_name: str,
    sampling_mode: str,
) -> dict[str, Any]:
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for repeat_index in range(1, repeats + 1):
        collector = RunTelemetryCollector()
        started = time.monotonic()
        with _temporary_env(_GPT5_TEMPERATURE_MODE_ENV, sampling_mode):
            with use_run_context(
                llm_selection=current_default_selection(),
                telemetry_collector=collector,
            ):
                result = await evaluate_startup(
                    folder,
                    k=k,
                    config=config,
                    use_web_search=use_web_search,
                )
        wall_clock_seconds = time.monotonic() - started
        rows = collector.snapshot_model_executions()
        run_costs = collector.build_run_costs()
        result["run_costs"] = run_costs
        run_report = _build_run_report(
            scenario_name=scenario_name,
            sampling_mode=sampling_mode,
            result=result,
            rows=rows,
            wall_clock_seconds=wall_clock_seconds,
        )
        report_path = scenario_dir / f"run_{repeat_index}.json"
        report_path.write_text(json.dumps(run_report, indent=2, ensure_ascii=True))
        run_report["report_path"] = str(report_path)
        runs.append(run_report)

    return _aggregate_scenario_reports(scenario_name, sampling_mode, runs)


async def async_main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    folder = Path(args.input)
    if not folder.is_dir():
        raise SystemExit(f"Input folder '{folder}' does not exist.")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be greater than 0.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config)

    scenario_summaries = []
    for scenario_name, sampling_mode in SCENARIOS:
        summary = await _run_single_scenario(
            folder=folder,
            repeats=args.repeats,
            output_dir=output_dir,
            config=config,
            use_web_search=args.use_web_search,
            k=args.k,
            scenario_name=scenario_name,
            sampling_mode=sampling_mode,
        )
        scenario_summaries.append(summary)

    summary_payload = {
        "input_folder": str(folder),
        "repeat_count": args.repeats,
        "use_web_search": bool(args.use_web_search),
        "k": args.k,
        "scenarios": scenario_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True))
    print(f"Benchmark summary written to {summary_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
