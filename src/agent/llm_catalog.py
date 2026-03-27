"""Curated model catalog and pricing metadata for run-level LLM selection."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ModelPricing:
    input_per_million_tokens_usd: float
    output_per_million_tokens_usd: float


@dataclass(frozen=True)
class ModelCatalogEntry:
    provider: str
    model: str
    label: str
    summary: str
    tier: Literal["budget", "balanced", "premium"]
    pricing: ModelPricing | None
    required_env: tuple[str, ...]
    supports_structured_output: bool = True
    supports_creativity_control: bool = False
    default_creativity: float | None = None


MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        provider="gemini",
        model="gemini-3.1-flash-lite-preview",
        label="Gemini 3.1 Flash Lite",
        summary="Budget speed",
        tier="budget",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.25,
            output_per_million_tokens_usd=1.50,
        ),
        required_env=("GOOGLE_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        label="Claude Haiku 4.5",
        summary="Sharp writing",
        tier="balanced",
        pricing=ModelPricing(
            input_per_million_tokens_usd=1.00,
            output_per_million_tokens_usd=5.00,
        ),
        required_env=("ANTHROPIC_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5.4-nano",
        label="GPT-5.4 nano",
        summary="Cheapest GPT-5.4",
        tier="budget",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.20,
            output_per_million_tokens_usd=1.25,
        ),
        required_env=("OPENAI_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5.4-mini",
        label="GPT-5.4 mini",
        summary="Strong mini model",
        tier="balanced",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.75,
            output_per_million_tokens_usd=4.50,
        ),
        required_env=("OPENAI_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5",
        label="GPT-5",
        summary="Deep reasoning",
        tier="premium",
        pricing=ModelPricing(
            input_per_million_tokens_usd=1.25,
            output_per_million_tokens_usd=10.00,
        ),
        required_env=("OPENAI_API_KEY",),
        supports_creativity_control=True,
        default_creativity=0.5,
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-4.1-mini",
        label="GPT-4.1 mini",
        summary="Stable fallback",
        tier="balanced",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.80,
            output_per_million_tokens_usd=3.20,
        ),
        required_env=("OPENAI_API_KEY",),
        supports_creativity_control=True,
        default_creativity=0.5,
    ),
    ModelCatalogEntry(
        provider="gemini",
        model="gemini-2.5-flash",
        label="Gemini 2.5 Flash",
        summary="Scale-ready speed",
        tier="balanced",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.30,
            output_per_million_tokens_usd=2.50,
        ),
        required_env=("GOOGLE_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="gemini",
        model="gemini-2.5-flash-lite",
        label="Gemini 2.5 Flash-Lite",
        summary="Cost-efficient throughput",
        tier="budget",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.10,
            output_per_million_tokens_usd=0.40,
        ),
        required_env=("GOOGLE_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="gemini",
        model="gemini-3.1-pro-preview",
        label="Gemini 3.1 Pro Preview",
        summary="Agentic frontier",
        tier="premium",
        pricing=ModelPricing(
            input_per_million_tokens_usd=2.00,
            output_per_million_tokens_usd=12.00,
        ),
        required_env=("GOOGLE_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openai",
        model="o4-mini",
        label="o4-mini",
        summary="Reasoning on a budget",
        tier="balanced",
        pricing=ModelPricing(
            input_per_million_tokens_usd=1.10,
            output_per_million_tokens_usd=4.40,
        ),
        required_env=("OPENAI_API_KEY",),
        supports_creativity_control=True,
        default_creativity=0.5,
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5.2",
        label="GPT-5.2",
        summary="Flagship coding",
        tier="premium",
        pricing=ModelPricing(
            input_per_million_tokens_usd=1.75,
            output_per_million_tokens_usd=14.00,
        ),
        required_env=("OPENAI_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5.4",
        label="GPT-5.4",
        summary="Latest frontier",
        tier="premium",
        pricing=ModelPricing(
            input_per_million_tokens_usd=2.50,
            output_per_million_tokens_usd=15.00,
        ),
        required_env=("OPENAI_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openrouter",
    model="openrouter/hunter-alpha",
        label="Hunter Alpha",
        summary="Experimental reasoning",
        tier="premium",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.0,
            output_per_million_tokens_usd=0.0,
        ),
        required_env=("OPENROUTER_API_KEY",),
        supports_structured_output=False,
    ),
    ModelCatalogEntry(
        provider="openrouter",
        model="openai/gpt-5-mini",
        label="OpenRouter · GPT-5 mini",
        summary="Balanced via OpenRouter",
        tier="balanced",
        pricing=None,
        required_env=("OPENROUTER_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openrouter",
        model="openai/gpt-5",
        label="OpenRouter · GPT-5",
        summary="Deep reasoning via OpenRouter",
        tier="premium",
        pricing=None,
        required_env=("OPENROUTER_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openrouter",
        model="openai/gpt-4.1-mini",
        label="OpenRouter · GPT-4.1 mini",
        summary="Stable fallback via OpenRouter",
        tier="balanced",
        pricing=None,
        required_env=("OPENROUTER_API_KEY",),
    ),
)

_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

_LEGACY_MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5-nano",
        label="GPT-5 nano",
        summary="Legacy GPT-5 nano",
        tier="budget",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.05,
            output_per_million_tokens_usd=0.40,
        ),
        required_env=("OPENAI_API_KEY",),
    ),
    ModelCatalogEntry(
        provider="openai",
        model="gpt-5-mini",
        label="GPT-5 mini",
        summary="Legacy GPT-5 mini",
        tier="balanced",
        pricing=ModelPricing(
            input_per_million_tokens_usd=0.25,
            output_per_million_tokens_usd=2.00,
        ),
        required_env=("OPENAI_API_KEY",),
    ),
)

_PROVIDER_ALIASES = {
    "google": "gemini",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "openai": "openai",
    "openrouter": "openrouter",
}

_CREATIVITY_MIN = 0.0
_CREATIVITY_MAX = 2.0


def normalize_provider(provider: str | None) -> str:
    raw = (provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw)


def _has_required_env(entry: ModelCatalogEntry) -> bool:
    if entry.provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return all(bool(os.getenv(name)) for name in entry.required_env)


def is_selectable_for_analysis(entry: ModelCatalogEntry) -> bool:
    return _has_required_env(entry) and entry.supports_structured_output


def find_model_entry(provider: str | None, model: str | None) -> ModelCatalogEntry | None:
    provider_norm = normalize_provider(provider)
    model_norm = (model or "").strip()
    for entry in MODEL_CATALOG:
        if entry.provider == provider_norm and entry.model == model_norm:
            return entry
    return None


def find_compatible_model_entry(
    provider: str | None,
    model: str | None,
) -> ModelCatalogEntry | None:
    entry = find_model_entry(provider, model)
    if entry is not None:
        return entry

    provider_norm = normalize_provider(provider)
    model_norm = (model or "").strip()
    for legacy in _LEGACY_MODEL_CATALOG:
        if legacy.provider == provider_norm and legacy.model == model_norm:
            return legacy
    return None


def get_tier_default(
    tier: Literal["budget", "balanced", "premium"],
) -> ModelCatalogEntry | None:
    for entry in MODEL_CATALOG:
        if entry.tier == tier and _has_required_env(entry):
            return entry
    return None


def model_label(provider: str | None, model: str | None) -> str:
    entry = find_compatible_model_entry(provider, model)
    if entry:
        return entry.label
    provider_norm = normalize_provider(provider or _DEFAULT_PROVIDER)
    model_norm = (model or _DEFAULT_MODEL).strip()
    return f"{provider_norm} · {model_norm}"


def normalize_creativity(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Creativity must be a number between 0.0 and 2.0.") from exc
    if not (_CREATIVITY_MIN <= normalized <= _CREATIVITY_MAX):
        raise ValueError("Creativity must be between 0.0 and 2.0.")
    return round(normalized, 2)


def supports_selection_creativity_control(provider: str | None, model: str | None) -> bool:
    entry = find_compatible_model_entry(provider, model)
    return bool(entry and entry.supports_creativity_control)


def default_selection_creativity(provider: str | None, model: str | None) -> float | None:
    entry = find_compatible_model_entry(provider, model)
    return entry.default_creativity if entry else None


def serialize_selection(
    provider: str | None,
    model: str | None,
    creativity: Any = None,
) -> dict[str, Any]:
    provider_norm = normalize_provider(provider or _DEFAULT_PROVIDER)
    model_norm = (model or _DEFAULT_MODEL).strip()
    selection: dict[str, Any] = {
        "provider": provider_norm,
        "model": model_norm,
        "label": model_label(provider_norm, model_norm),
    }
    normalized_creativity = normalize_creativity(creativity)
    if normalized_creativity is not None and supports_selection_creativity_control(provider_norm, model_norm):
        selection["creativity"] = normalized_creativity
    return selection


def current_default_selection() -> dict[str, Any]:
    provider = normalize_provider(os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER))
    model = os.getenv("MODEL_NAME", _DEFAULT_MODEL).strip()
    return serialize_selection(provider, model)


def available_models_payload() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for entry in MODEL_CATALOG:
        env_available = _has_required_env(entry)
        selectable = is_selectable_for_analysis(entry)
        if not env_available:
            unavailable_reason = "Missing provider credentials."
        elif not entry.supports_structured_output:
            unavailable_reason = "Not supported for structured-output analysis runs yet."
        else:
            unavailable_reason = ""
        models.append(
            {
                "provider": entry.provider,
                "model": entry.model,
                "label": entry.label,
                "summary": entry.summary,
                "tier": entry.tier,
                "available": env_available,
                "selectable": selectable,
                "pricing_available": entry.pricing is not None,
                "supports_structured_output": entry.supports_structured_output,
                "supports_creativity_control": entry.supports_creativity_control,
                "default_creativity": entry.default_creativity,
                "unavailable_reason": unavailable_reason,
            }
        )
    return models


def available_chat_models_payload() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for entry in MODEL_CATALOG:
        env_available = _has_required_env(entry)
        models.append(
            {
                "provider": entry.provider,
                "model": entry.model,
                "label": entry.label,
                "summary": entry.summary,
                "tier": entry.tier,
                "available": env_available,
                "selectable": env_available,
                "pricing_available": entry.pricing is not None,
                "supports_structured_output": entry.supports_structured_output,
                "supports_creativity_control": entry.supports_creativity_control,
                "default_creativity": entry.default_creativity,
                "unavailable_reason": "" if env_available else "Missing provider credentials.",
            }
        )
    return models


def validate_requested_selection(
    provider: str | None,
    model: str | None,
) -> ModelCatalogEntry | None:
    if not provider and not model:
        return None
    entry = find_model_entry(provider, model)
    if not entry:
        raise ValueError("Unknown LLM model selection.")
    if not _has_required_env(entry):
        raise ValueError(f"{entry.label} is not available in this environment.")
    if not entry.supports_structured_output:
        raise ValueError(
            f"{entry.label} is not supported for structured-output analysis runs yet."
        )
    return entry


def validate_chat_requested_selection(
    provider: str | None,
    model: str | None,
) -> ModelCatalogEntry | None:
    if not provider and not model:
        return None
    entry = find_compatible_model_entry(provider, model)
    if not entry:
        raise ValueError("Unknown chat LLM model selection.")
    if not _has_required_env(entry):
        raise ValueError(f"{entry.label} is not available in this environment.")
    return entry


def estimate_llm_cost_usd(
    provider: str | None,
    model: str | None,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    entry = find_compatible_model_entry(provider, model)
    if not entry or not entry.pricing:
        return None
    prompt_cost = (prompt_tokens / 1_000_000) * entry.pricing.input_per_million_tokens_usd
    completion_cost = (completion_tokens / 1_000_000) * entry.pricing.output_per_million_tokens_usd
    return round(prompt_cost + completion_cost, 8)


def pricing_catalog_payload() -> dict[str, dict[str, Any]]:
    return {
        f"{entry.provider}:{entry.model}": {
            "provider": entry.provider,
            "model": entry.model,
            "label": entry.label,
            "pricing": asdict(entry.pricing) if entry.pricing else None,
        }
        for entry in MODEL_CATALOG
    }
