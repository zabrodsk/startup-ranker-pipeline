"""Run-scoped context for model selection and telemetry aggregation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator

from agent.llm_catalog import estimate_llm_cost_usd, model_label, serialize_selection

PERPLEXITY_SEARCH_PRICE_PER_REQUEST_USD = 0.005

_llm_selection_var: ContextVar[dict[str, str] | None] = ContextVar("llm_selection", default=None)
_telemetry_collector_var: ContextVar["RunTelemetryCollector | None"] = ContextVar(
    "telemetry_collector",
    default=None,
)
_company_slug_var: ContextVar[str | None] = ContextVar("company_slug", default=None)
_stage_name_var: ContextVar[str | None] = ContextVar("stage_name", default=None)


def get_current_llm_selection() -> dict[str, str] | None:
    return _llm_selection_var.get()


def get_current_company_slug() -> str | None:
    return _company_slug_var.get()


def get_current_collector() -> "RunTelemetryCollector | None":
    return _telemetry_collector_var.get()


def get_current_stage_name() -> str | None:
    return _stage_name_var.get()


@contextmanager
def use_run_context(
    *,
    llm_selection: dict[str, str] | None = None,
    telemetry_collector: "RunTelemetryCollector | None" = None,
) -> Iterator[None]:
    llm_token: Token[dict[str, str] | None] | None = None
    telemetry_token: Token[RunTelemetryCollector | None] | None = None
    if llm_selection is not None:
        llm_token = _llm_selection_var.set(serialize_selection(llm_selection.get("provider"), llm_selection.get("model")))
    if telemetry_collector is not None:
        telemetry_token = _telemetry_collector_var.set(telemetry_collector)
    try:
        yield
    finally:
        if telemetry_token is not None:
            _telemetry_collector_var.reset(telemetry_token)
        if llm_token is not None:
            _llm_selection_var.reset(llm_token)


@contextmanager
def use_company_context(company_slug: str | None) -> Iterator[None]:
    token = _company_slug_var.set((company_slug or "").strip() or None)
    try:
        yield
    finally:
        _company_slug_var.reset(token)


@contextmanager
def use_stage_context(stage_name: str | None) -> Iterator[None]:
    token = _stage_name_var.set((stage_name or "").strip() or None)
    try:
        yield
    finally:
        _stage_name_var.reset(token)


@dataclass
class RunTelemetryCollector:
    """Collects per-call model execution rows and builds run-level cost summaries."""

    model_executions: list[dict[str, Any]] = field(default_factory=list)
    missing_llm_usage: bool = False
    selected_llm: dict[str, str] | None = None

    def record_llm_usage(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        metadata: dict[str, Any] | None = None,
        stage: str | None = None,
    ) -> None:
        provider_norm = (provider or "").strip().lower()
        model_norm = (model or "").strip()
        stage_name = (stage or get_current_stage_name() or "llm_invoke").strip() or "llm_invoke"
        if prompt_tokens is None or completion_tokens is None or total_tokens is None:
            self.missing_llm_usage = True
            self.model_executions.append(
                {
                    "service": "llm",
                    "provider": provider_norm,
                    "model": model_norm,
                    "company_slug": get_current_company_slug(),
                    "stage": stage_name,
                    "status": "done",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "estimated_cost_usd": None,
                    "request_count": 1,
                    "metadata": metadata or {},
                }
            )
            return

        estimated_cost_usd = estimate_llm_cost_usd(
            provider_norm,
            model_norm,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.model_executions.append(
            {
                "service": "llm",
                    "provider": provider_norm,
                    "model": model_norm,
                    "company_slug": get_current_company_slug(),
                    "stage": stage_name,
                    "status": "done",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "request_count": 1,
                "metadata": metadata or {},
            }
        )

    def record_execution_event(
        self,
        *,
        service: str,
        status: str,
        stage: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        request_timeout_seconds: float | None = None,
        max_retries: int | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_executions.append(
            {
                "service": (service or "").strip() or "pipeline",
                "provider": (provider or "").strip().lower() or None,
                "model": (model or "").strip() or None,
                "company_slug": get_current_company_slug(),
                "stage": (stage or get_current_stage_name() or "pipeline").strip() or "pipeline",
                "status": (status or "").strip() or "done",
                "request_timeout_seconds": request_timeout_seconds,
                "max_retries": max_retries,
                "latency_ms": latency_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "estimated_cost_usd": None,
                "request_count": 1,
                "error_message": error_message,
                "metadata": metadata or {},
            }
        )

    def record_perplexity_search(
        self,
        *,
        provider: str = "perplexity",
        model: str = "search_api",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_executions.append(
            {
                "service": "perplexity_search",
                "provider": provider,
                "model": model,
                "company_slug": get_current_company_slug(),
                "stage": "perplexity_search",
                "status": "done",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "estimated_cost_usd": PERPLEXITY_SEARCH_PRICE_PER_REQUEST_USD,
                "request_count": 1,
                "metadata": metadata or {},
            }
        )

    def build_run_costs(self) -> dict[str, Any]:
        llm_prompt_tokens = 0
        llm_completion_tokens = 0
        llm_total_tokens = 0
        llm_cost = 0.0
        llm_cost_known = True
        llm_events_seen = False

        perplexity_requests = 0
        perplexity_cost = 0.0

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.model_executions:
            service = row.get("service")
            if service == "llm":
                llm_events_seen = True
                provider = row.get("provider") or ""
                model = row.get("model") or ""
                key = (provider, model)
                grouped.setdefault(
                    key,
                    {
                        "provider": provider,
                        "model": model,
                        "label": model_label(provider, model),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "usd": 0.0,
                        "pricing_available": True,
                    },
                )
                prompt_tokens = row.get("prompt_tokens")
                completion_tokens = row.get("completion_tokens")
                total_tokens = row.get("total_tokens")
                if isinstance(prompt_tokens, int):
                    llm_prompt_tokens += prompt_tokens
                    grouped[key]["prompt_tokens"] += prompt_tokens
                if isinstance(completion_tokens, int):
                    llm_completion_tokens += completion_tokens
                    grouped[key]["completion_tokens"] += completion_tokens
                if isinstance(total_tokens, int):
                    llm_total_tokens += total_tokens
                    grouped[key]["total_tokens"] += total_tokens
                estimated_cost_usd = row.get("estimated_cost_usd")
                if estimated_cost_usd is None:
                    llm_cost_known = False
                    grouped[key]["pricing_available"] = False
                    grouped[key]["usd"] = None
                else:
                    llm_cost += float(estimated_cost_usd)
                    if grouped[key]["usd"] is not None:
                        grouped[key]["usd"] += float(estimated_cost_usd)
            elif service == "perplexity_search":
                perplexity_requests += int(row.get("request_count") or 1)
                perplexity_cost += float(row.get("estimated_cost_usd") or 0.0)

        if not llm_events_seen and perplexity_requests == 0:
            status = "unavailable"
        elif self.missing_llm_usage or not llm_cost_known:
            status = "partial"
        else:
            status = "complete"

        by_model: list[dict[str, Any]] = []
        for item in grouped.values():
            if isinstance(item["usd"], float):
                item["usd"] = round(item["usd"], 8)
            by_model.append(item)
        by_model.sort(key=lambda item: (item["provider"], item["model"]))

        llm_usd = round(llm_cost, 8) if llm_cost_known else None
        total_usd = None
        if llm_usd is not None:
            total_usd = round(llm_usd + perplexity_cost, 8)
        elif perplexity_requests > 0 and not llm_events_seen:
            total_usd = round(perplexity_cost, 8)

        return {
            "currency": "USD",
            "status": status,
            "total_usd": total_usd,
            "llm_usd": llm_usd,
            "perplexity_usd": round(perplexity_cost, 8),
            "llm_tokens": {
                "prompt": llm_prompt_tokens,
                "completion": llm_completion_tokens,
                "total": llm_total_tokens,
            },
            "perplexity_search": {
                "requests": perplexity_requests,
                "total_usd": round(perplexity_cost, 8),
            },
            "by_model": by_model,
        }

    def snapshot_model_executions(self) -> list[dict[str, Any]]:
        return deepcopy(self.model_executions)
