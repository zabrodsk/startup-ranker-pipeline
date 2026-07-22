"""Immutable model profiles for the staging OpenRouter A/B/C experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.llm_catalog import openrouter_model_experiment_enabled
from agent.llm_policy import (
    PipelineModelPolicy,
    build_explicit_pipeline_model_policy,
    resolve_effective_phase_models,
)

OPENROUTER_ROUTING_POLICY: dict[str, Any] = {
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
}


@dataclass(frozen=True)
class ResolvedModelProfile:
    """Resolved immutable experiment arm and its seven-stage policy."""

    id: str
    arm: str
    label: str
    description: str
    policy: PipelineModelPolicy
    phase_models: dict[str, dict[str, Any]]
    openrouter_routing: dict[str, Any]


@dataclass(frozen=True)
class _ProfileDefinition:
    id: str
    arm: str
    label: str
    description: str


_PROFILE_DEFINITIONS: tuple[_ProfileDefinition, ...] = (
    _ProfileDefinition(
        id="gpt_current",
        arm="A",
        label="A · Current GPT Mini/Nano",
        description="Existing GPT-5.4 Mini/Nano routing control.",
    ),
    _ProfileDefinition(
        id="kimi_k26",
        arm="B",
        label="B · Kimi K2.6",
        description="Kimi K2.6 thinking mode for every pipeline stage.",
    ),
    _ProfileDefinition(
        id="glm_deepseek_flash",
        arm="C",
        label="C · GLM 5.2 + DeepSeek V4 Flash",
        description="GLM for judgment stages and DeepSeek Flash for analysis generation.",
    ),
)


def _selection(model: str, *, reasoning_effort: str | None = None) -> dict[str, Any]:
    provider = "openrouter" if "/" in model else "openai"
    value: dict[str, Any] = {"provider": provider, "model": model}
    if reasoning_effort is not None:
        value["reasoning_effort"] = reasoning_effort
    return value


def _profile_phase_models(profile_id: str) -> dict[str, dict[str, Any]]:
    if profile_id == "gpt_current":
        return {
            "decomposition": _selection("gpt-5.4-mini"),
            "answering": _selection("gpt-5.4-nano"),
            "generation": _selection("gpt-5.4-mini"),
            "critique": _selection("gpt-5.4-nano"),
            "evaluation": _selection("gpt-5.4-mini"),
            "refinement": _selection("gpt-5.4-nano"),
            "ranking": _selection("gpt-5.4-mini"),
        }
    if profile_id == "kimi_k26":
        return {
            phase: _selection("moonshotai/kimi-k2.6", reasoning_effort="high")
            for phase in (
                "decomposition",
                "answering",
                "generation",
                "critique",
                "evaluation",
                "refinement",
                "ranking",
            )
        }
    if profile_id == "glm_deepseek_flash":
        def glm() -> dict[str, Any]:
            return _selection("z-ai/glm-5.2", reasoning_effort="high")

        def deepseek() -> dict[str, Any]:
            return _selection(
                "deepseek/deepseek-v4-flash",
                reasoning_effort="high",
            )

        return {
            "decomposition": glm(),
            "answering": deepseek(),
            "generation": deepseek(),
            "critique": glm(),
            "evaluation": glm(),
            "refinement": deepseek(),
            "ranking": glm(),
        }
    raise ValueError("Unknown model profile.")


def resolve_model_profile(profile_id: str | None) -> ResolvedModelProfile:
    """Resolve a staging profile into an immutable seven-stage policy."""
    normalized = (profile_id or "").strip()
    definition = next((item for item in _PROFILE_DEFINITIONS if item.id == normalized), None)
    if definition is None:
        raise ValueError("Unknown model profile.")
    if not openrouter_model_experiment_enabled():
        raise ValueError("OpenRouter model experiment is disabled.")

    policy = build_explicit_pipeline_model_policy(_profile_phase_models(definition.id))
    phase_models = resolve_effective_phase_models(policy)
    uses_openrouter = any(
        selection.get("provider") == "openrouter"
        for selection in phase_models.values()
    )
    return ResolvedModelProfile(
        id=definition.id,
        arm=definition.arm,
        label=definition.label,
        description=definition.description,
        policy=policy,
        phase_models=phase_models,
        openrouter_routing=dict(OPENROUTER_ROUTING_POLICY) if uses_openrouter else {},
    )


def model_profiles_payload() -> list[dict[str, Any]]:
    """Return staging-visible named profiles and availability details."""
    if not openrouter_model_experiment_enabled():
        return []
    payload: list[dict[str, Any]] = []
    for definition in _PROFILE_DEFINITIONS:
        try:
            resolved = resolve_model_profile(definition.id)
        except ValueError as exc:
            payload.append(
                {
                    "id": definition.id,
                    "arm": definition.arm,
                    "label": definition.label,
                    "description": definition.description,
                    "available": False,
                    "unavailable_reason": str(exc),
                    "phase_models": {},
                }
            )
            continue
        payload.append(
            {
                "id": resolved.id,
                "arm": resolved.arm,
                "label": resolved.label,
                "description": resolved.description,
                "available": True,
                "unavailable_reason": "",
                "phase_models": resolved.phase_models,
            }
        )
    return payload
