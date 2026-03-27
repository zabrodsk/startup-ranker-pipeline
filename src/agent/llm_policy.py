"""Pipeline-specific model selection policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from agent.llm_catalog import (
    ModelCatalogEntry,
    normalize_creativity,
    get_tier_default,
    serialize_selection,
    supports_selection_creativity_control,
    validate_requested_selection,
)

PublicQualityTier = Literal["cheap", "premium"]
PremiumPhaseChoice = Literal["claude", "gpt5"]
PipelinePhase = Literal[
    "decomposition",
    "answering",
    "generation",
    "critique",
    "evaluation",
    "refinement",
    "ranking",
]
UserSelectablePhase = Literal[
    "decomposition",
    "answering",
    "generation",
    "evaluation",
    "ranking",
]
CriticalPipelinePhase = Literal["decomposition", "generation", "evaluation", "ranking"]
OpenAIReasoningPhase = Literal[
    "decomposition",
    "answering",
    "generation_pro",
    "generation_contra",
    "critique",
    "evaluation",
    "refinement",
    "ranking_dimension_score",
    "ranking_upside_score",
    "ranking_executive_summary",
]

_CRITICAL_PHASES: tuple[CriticalPipelinePhase, ...] = (
    "decomposition",
    "generation",
    "evaluation",
    "ranking",
)
_USER_SELECTABLE_PHASES: tuple[UserSelectablePhase, ...] = (
    "decomposition",
    "answering",
    "generation",
    "evaluation",
    "ranking",
)
_PREMIUM_FAMILY_TO_TIER: dict[PremiumPhaseChoice, Literal["balanced", "premium"]] = {
    "claude": "balanced",
    "gpt5": "premium",
}
DEFAULT_PREMIUM_PHASE_MODELS: dict[CriticalPipelinePhase, PremiumPhaseChoice] = {
    phase: "gpt5" for phase in _CRITICAL_PHASES
}
PHASE_LABELS: dict[UserSelectablePhase, str] = {
    "decomposition": "Decomposition",
    "answering": "Q&A",
    "generation": "Generation",
    "evaluation": "Evaluation",
    "ranking": "Ranking",
}
PHASE_SHORT_LABELS: dict[UserSelectablePhase, str] = {
    "decomposition": "D",
    "answering": "A",
    "generation": "G",
    "evaluation": "E",
    "ranking": "R",
}


@dataclass(frozen=True)
class PipelineModelPolicy:
    decomposition: dict[str, Any]
    answering: dict[str, Any]
    generation: dict[str, Any]
    critique: dict[str, Any]
    evaluation: dict[str, Any]
    refinement: dict[str, Any]
    ranking: dict[str, Any]

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return asdict(self)


def _serialize_entry(entry: ModelCatalogEntry) -> dict[str, Any]:
    return serialize_selection(entry.provider, entry.model)


def normalize_quality_tier(value: str | None) -> PublicQualityTier | None:
    tier = (value or "").strip().lower()
    if tier in {"cheap", "premium"}:
        return tier
    return None


def normalize_premium_phase_choice(value: str | None) -> PremiumPhaseChoice | None:
    choice = (value or "").strip().lower()
    if choice == "claude":
        return "claude"
    if choice in {"gpt5", "gpt-5"}:
        return "gpt5"
    return None


def normalize_premium_phase_models(
    payload: dict[str, Any] | None,
) -> dict[CriticalPipelinePhase, PremiumPhaseChoice]:
    normalized = dict(DEFAULT_PREMIUM_PHASE_MODELS)
    if not isinstance(payload, dict):
        return normalized
    for phase in _CRITICAL_PHASES:
        choice = normalize_premium_phase_choice(payload.get(phase))
        if choice:
            normalized[phase] = choice
    return normalized


def normalize_phase_models(
    payload: dict[str, Any] | None,
) -> dict[UserSelectablePhase, dict[str, Any]]:
    if not isinstance(payload, dict):
        return default_phase_model_selections()

    normalized: dict[UserSelectablePhase, dict[str, Any]] = {}

    for phase in _USER_SELECTABLE_PHASES:
        raw = payload.get(phase)
        if not isinstance(raw, dict):
            continue
        provider = str(raw.get("provider") or "").strip()
        model = str(raw.get("model") or "").strip()
        if provider and model:
            selection = serialize_selection(provider, model, raw.get("creativity"))
            normalized[phase] = {
                "provider": selection["provider"],
                "model": selection["model"],
                **({"creativity": selection["creativity"]} if selection.get("creativity") is not None else {}),
            }
    if len(normalized) == len(_USER_SELECTABLE_PHASES):
        return normalized

    defaults = default_phase_model_selections()
    defaults.update(normalized)
    return defaults


def coerce_phase_models_payload(
    payload: dict[str, Any] | None,
    *,
    require_all: bool = False,
) -> dict[UserSelectablePhase, dict[str, Any]]:
    if not isinstance(payload, dict):
        if require_all:
            raise ValueError("phase_models must be an object.")
        return default_phase_model_selections()

    invalid_keys = [key for key in payload.keys() if key not in _USER_SELECTABLE_PHASES]
    if invalid_keys:
        raise ValueError("phase_models contains unsupported phases.")

    normalized: dict[UserSelectablePhase, dict[str, Any]] = {}
    missing: list[str] = []
    for phase in _USER_SELECTABLE_PHASES:
        raw = payload.get(phase)
        if not isinstance(raw, dict):
            if require_all:
                missing.append(phase)
            continue
        provider = str(raw.get("provider") or "").strip()
        model = str(raw.get("model") or "").strip()
        if not provider or not model:
            raise ValueError(f"phase_models.{phase} must include provider and model.")
        creativity = normalize_creativity(raw.get("creativity"))
        normalized[phase] = {
            "provider": provider,
            "model": model,
            **({"creativity": creativity} if creativity is not None else {}),
        }

    if require_all and missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"phase_models is missing required phases: {missing_list}.")

    if require_all:
        return normalized

    defaults = default_phase_model_selections()
    defaults.update(normalized)
    return defaults


def _resolve_premium_choice(choice: PremiumPhaseChoice) -> ModelCatalogEntry | None:
    preferred = _default_claude_entry() if choice == "claude" else _default_gpt5_family_entry()
    if preferred:
        return preferred
    fallback_choice: PremiumPhaseChoice = "claude" if choice == "gpt5" else "gpt5"
    return _default_claude_entry() if fallback_choice == "claude" else _default_gpt5_family_entry()


def _required_model_entry(provider: str, model: str) -> ModelCatalogEntry:
    entry = validate_requested_selection(provider, model)
    if entry is None:
        raise ValueError("A provider and model are required for each phase selection.")
    return entry


def _default_claude_entry() -> ModelCatalogEntry | None:
    return _first_available_entry(
        ("anthropic", "claude-haiku-4-5-20251001"),
        include_tier_fallback=False,
    )


def _default_gpt5_family_entry() -> ModelCatalogEntry | None:
    return _first_available_entry(
        ("openai", "gpt-5"),
        ("openai", "gpt-5.2"),
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-4.1-mini"),
        include_tier_fallback=False,
    )


def _first_available_explicit(
    provider: str,
    model: str,
) -> ModelCatalogEntry | None:
    try:
        return _required_model_entry(provider, model)
    except ValueError:
        return None


def _first_available_entry(
    *candidates: tuple[str, str],
    include_tier_fallback: bool = True,
) -> ModelCatalogEntry | None:
    for provider, model in candidates:
        entry = _first_available_explicit(provider, model)
        if entry:
            return entry
    if not include_tier_fallback:
        return None
    for tier in ("budget", "balanced", "premium"):
        entry = get_tier_default(tier)
        if entry:
            return entry
    return None


def default_phase_model_selections() -> dict[UserSelectablePhase, dict[str, Any]]:
    decomposition = _first_available_entry(
        ("openai", "gpt-5.4-mini"),
        ("gemini", "gemini-3.1-pro-preview"),
        ("gemini", "gemini-2.5-flash"),
        ("openai", "gpt-5.2"),
        ("openai", "gpt-5"),
    )
    answering = _first_available_entry(
        ("openai", "gpt-5.4-nano"),
        ("gemini", "gemini-2.5-flash"),
        ("gemini", "gemini-3.1-flash-lite-preview"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("openai", "gpt-5.4-mini"),
    )
    generation = _first_available_entry(
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-5.2"),
        ("openai", "gpt-5"),
        ("gemini", "gemini-3.1-pro-preview"),
    )
    evaluation = _first_available_entry(
        ("openai", "gpt-5.4-mini"),
        ("openai", "o4-mini"),
        ("openai", "gpt-4.1-mini"),
        ("anthropic", "claude-haiku-4-5-20251001"),
    )
    ranking = _first_available_entry(
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-5.2"),
        ("openai", "gpt-5"),
        ("openai", "o4-mini"),
        ("anthropic", "claude-haiku-4-5-20251001"),
    )

    if decomposition is None:
        raise ValueError("No available models are configured for decomposition.")
    if answering is None:
        raise ValueError("No available models are configured for question answering.")
    if generation is None:
        raise ValueError("No available models are configured for generation.")
    if evaluation is None:
        raise ValueError("No available models are configured for evaluation.")
    if ranking is None:
        raise ValueError("No available models are configured for ranking.")

    return {
        "decomposition": _serialize_entry(decomposition),
        "answering": _serialize_entry(answering),
        "generation": _serialize_entry(generation),
        "evaluation": _serialize_entry(evaluation),
        "ranking": _serialize_entry(ranking),
    }


def build_phase_model_policy(
    phase_models: dict[str, Any] | None = None,
) -> PipelineModelPolicy:
    selections = coerce_phase_models_payload(phase_models, require_all=True)
    resolved: dict[UserSelectablePhase, dict[str, Any]] = {}
    for phase in _USER_SELECTABLE_PHASES:
        selection = selections[phase]
        entry = _required_model_entry(selection["provider"], selection["model"])
        resolved_selection = _serialize_entry(entry)
        if supports_selection_creativity_control(entry.provider, entry.model):
            creativity = normalize_creativity(selection.get("creativity"))
            if creativity is not None:
                resolved_selection["creativity"] = creativity
        resolved[phase] = resolved_selection

    answering_selection = dict(resolved["answering"])
    return PipelineModelPolicy(
        decomposition=dict(resolved["decomposition"]),
        answering=answering_selection,
        generation=dict(resolved["generation"]),
        critique=dict(answering_selection),
        evaluation=dict(resolved["evaluation"]),
        refinement=dict(answering_selection),
        ranking=dict(resolved["ranking"]),
    )


def build_default_phase_model_policy() -> PipelineModelPolicy:
    return build_phase_model_policy(default_phase_model_selections())


def build_pipeline_policy(
    quality_tier: PublicQualityTier,
    premium_phase_models: dict[str, Any] | None = None,
) -> PipelineModelPolicy:
    gemini = get_tier_default("budget")
    if quality_tier == "cheap":
        if gemini is None:
            raise ValueError("Cheap tier is unavailable in this environment.")
        selection = _serialize_entry(gemini)
        return PipelineModelPolicy(
            decomposition=selection,
            answering=selection,
            generation=selection,
            critique=selection,
            evaluation=selection,
            refinement=selection,
            ranking=selection,
        )

    if gemini is None:
        raise ValueError("Premium tier requires a Gemini model for answering.")

    choices = normalize_premium_phase_models(premium_phase_models)
    resolved: dict[str, dict[str, Any]] = {}
    for phase in _CRITICAL_PHASES:
        entry = _resolve_premium_choice(choices[phase])
        if entry is None:
            raise ValueError(f"Premium phase '{phase}' is unavailable in this environment.")
        resolved[phase] = _serialize_entry(entry)

    answering_selection = _serialize_entry(gemini)
    return PipelineModelPolicy(
        decomposition=resolved["decomposition"],
        answering=answering_selection,
        generation=resolved["generation"],
        critique=answering_selection,
        evaluation=resolved["evaluation"],
        refinement=answering_selection,
        ranking=resolved["ranking"],
    )


def resolve_effective_phase_choices(
    policy: PipelineModelPolicy,
) -> dict[PipelinePhase, str]:
    return {
        "decomposition": _label_family(policy.decomposition),
        "answering": _label_family(policy.answering),
        "generation": _label_family(policy.generation),
        "critique": _label_family(policy.critique),
        "evaluation": _label_family(policy.evaluation),
        "refinement": _label_family(policy.refinement),
        "ranking": _label_family(policy.ranking),
    }


def resolve_effective_phase_models(
    policy: PipelineModelPolicy,
) -> dict[PipelinePhase, dict[str, Any]]:
    return policy.as_dict()


def _selection_label(selection: dict[str, Any] | None) -> str:
    if not isinstance(selection, dict):
        return "Unknown"
    label = str(selection.get("label") or "").strip()
    if label:
        return label
    provider = selection.get("provider")
    model = selection.get("model")
    fallback = serialize_selection(provider, model)
    return fallback["label"]


def _label_family(selection: dict[str, Any]) -> str:
    provider = selection.get("provider")
    model = selection.get("model", "")
    if provider == "openai" and model == "gpt-5":
        return "gpt5"
    if provider == "anthropic":
        return "claude"
    if provider == "gemini":
        return "gemini"
    return model or provider or "unknown"


def build_phase_policy_display_label(
    effective_phase_models: dict[str, Any] | None,
) -> str:
    if not isinstance(effective_phase_models, dict):
        return "Per-phase routing"

    parts: list[str] = []
    for phase in _USER_SELECTABLE_PHASES:
        selection = effective_phase_models.get(phase)
        if not isinstance(selection, dict):
            continue
        parts.append(f"{PHASE_SHORT_LABELS[phase]} {_selection_label(selection)}")

    return "Per-phase · " + " · ".join(parts) if parts else "Per-phase routing"


def build_tier_display_label(
    quality_tier: PublicQualityTier,
    *,
    effective_phase_models: dict[str, Any] | None = None,
) -> str:
    if quality_tier == "cheap":
        return "Cheap tier · Gemini"

    effective = effective_phase_models or {}
    critical = [effective.get(phase, "gpt5") for phase in _CRITICAL_PHASES]
    unique_critical = {value for value in critical if value}
    if unique_critical == {"gpt5"}:
        return "Premium tier · Gemini + GPT-5"
    if unique_critical == {"claude"}:
        return "Premium tier · Gemini + Claude"
    return "Premium tier · QA Gemini · Critical phases mixed"


def get_alternate_premium_selection(
    selection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    provider = (selection or {}).get("provider")
    family = _label_family(selection or {})
    if provider == "anthropic" or family == "claude":
        fallback = _first_available_entry(
            ("openai", "gpt-5"),
            ("openai", "gpt-5.4-mini"),
            ("openai", "gpt-4.1-mini"),
            include_tier_fallback=False,
        )
    elif provider == "openai" or family == "gpt5":
        fallback = _first_available_entry(
            ("anthropic", "claude-haiku-4-5-20251001"),
            include_tier_fallback=False,
        )
    else:
        return None
    if fallback is None:
        return None
    fallback_selection = _serialize_entry(fallback)
    if (
        fallback_selection.get("provider") == (selection or {}).get("provider")
        and fallback_selection.get("model") == (selection or {}).get("model")
    ):
        return None
    return fallback_selection


def quality_tiers_payload() -> list[dict[str, Any]]:
    cheap_available = get_tier_default("budget") is not None
    claude_available = _default_claude_entry() is not None
    gpt5_available = _default_gpt5_family_entry() is not None
    return [
        {
            "id": "cheap",
            "label": "Cheap",
            "description": "Gemini for all phases",
            "available": cheap_available,
            "degraded": False,
            "llm_label": build_tier_display_label("cheap"),
        },
        {
            "id": "premium",
            "label": "Premium",
            "description": "Gemini for Q&A, Claude or GPT-5 for critical phases",
            "available": cheap_available and (claude_available or gpt5_available),
            "degraded": not (claude_available and gpt5_available),
            "llm_label": "Premium tier · QA Gemini · Critical phases mixed",
        },
    ]


def premium_phase_options_payload() -> dict[str, dict[str, Any]]:
    claude = _default_claude_entry()
    gpt5 = _default_gpt5_family_entry()
    return {
        "claude": {
            "id": "claude",
            "label": claude.label if claude else "Claude",
            "available": claude is not None,
        },
        "gpt5": {
            "id": "gpt5",
            "label": gpt5.label if gpt5 else "GPT-5",
            "available": gpt5 is not None,
        },
    }


def phase_model_defaults_payload() -> dict[UserSelectablePhase, dict[str, Any]]:
    try:
        return default_phase_model_selections()
    except ValueError:
        return {}


def resolve_openai_phase_sampling(
    model: str | None,
    stage_name: str | None,
    requested_temperature: float | None,
) -> dict[str, float | str | None] | None:
    model_norm = (model or "").strip()
    stage = (stage_name or "").strip().lower()

    if model_norm in {"gpt-5.4-mini", "gpt-5.4-nano"}:
        stage_map: dict[str, dict[str, float | str | None]] = {
            "decomposition": {"temperature": 0.5, "reasoning_effort": "none"},
            "answering": {"temperature": 0.0, "reasoning_effort": "none"},
            "generation_pro": {"temperature": 0.7, "reasoning_effort": "none"},
            "generation_contra": {"temperature": 0.7, "reasoning_effort": "none"},
            "critique": {"temperature": 0.5, "reasoning_effort": "none"},
            "evaluation": {"temperature": None, "reasoning_effort": "medium"},
            "refinement": {"temperature": 0.7, "reasoning_effort": "none"},
            "ranking_dimension_score": {"temperature": None, "reasoning_effort": "high"},
            "ranking_upside_score": {"temperature": 0.7, "reasoning_effort": "none"},
            "ranking_executive_summary": {"temperature": None, "reasoning_effort": "high"},
        }
        return stage_map.get(
            stage,
            {
                "temperature": requested_temperature,
                "reasoning_effort": "none",
            },
        )

    if model_norm in {"gpt-5.2", "gpt-5.4"}:
        ranking_effort = "xhigh" if model_norm == "gpt-5.4" else "high"
        stage_map = {
            "decomposition": {"temperature": None, "reasoning_effort": "medium"},
            "answering": {"temperature": None, "reasoning_effort": "low"},
            "generation_pro": {"temperature": None, "reasoning_effort": "medium"},
            "generation_contra": {"temperature": None, "reasoning_effort": "medium"},
            "critique": {"temperature": None, "reasoning_effort": "none"},
            "evaluation": {"temperature": None, "reasoning_effort": "high"},
            "refinement": {"temperature": None, "reasoning_effort": "medium"},
            "ranking_dimension_score": {"temperature": None, "reasoning_effort": ranking_effort},
            "ranking_upside_score": {"temperature": 0.7, "reasoning_effort": "none"},
            "ranking_executive_summary": {"temperature": None, "reasoning_effort": ranking_effort},
        }
        if stage in stage_map:
            return stage_map[stage]
        return {
            "temperature": None,
            "reasoning_effort": "low" if requested_temperature is not None and requested_temperature <= 0.0 else "medium",
        }

    return None


def resolve_openai_reasoning_fallback_temperature(
    model: str | None,
    stage_name: str | None,
) -> float | None:
    if (model or "").strip() != "gpt-5.4-nano":
        return None

    stage = (stage_name or "").strip().lower()
    fallback_map = {
        "answering": 0.0,
        "decomposition": 0.5,
        "generation_pro": 0.7,
        "generation_contra": 0.7,
        "critique": 0.5,
        "refinement": 0.7,
        "evaluation": 0.0,
        "ranking_dimension_score": 0.0,
        "ranking_upside_score": 0.7,
        "ranking_executive_summary": 0.3,
    }
    return fallback_map.get(stage)
