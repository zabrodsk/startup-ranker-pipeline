"""Runtime prompt library manager with validation and persistence."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent.prompt_library.defaults import (
    ORDERED_PROMPT_IDS,
    PROMPT_DEFINITIONS,
    SCHEMA_VERSION,
    build_default_catalog,
    get_default_values,
)

LIBRARY_PATH = Path(__file__).with_name("library.json")
MAX_TEXT_LENGTH = 80_000
MAX_LIST_ITEM_LENGTH = 400


class PromptLibraryValidationError(ValueError):
    """Raised when prompt catalog input fails validation."""


class PromptLibraryStorageError(RuntimeError):
    """Raised when prompt catalog persistence fails."""


def _catalog_from_values(values: dict[str, Any]) -> dict[str, Any]:
    catalog = build_default_catalog()
    for item in catalog["items"]:
        item["value"] = deepcopy(values[item["id"]])
    return catalog


def _extract_overrides(run_overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not run_overrides or not isinstance(run_overrides, dict):
        return {}
    candidate = run_overrides.get("values")
    if isinstance(candidate, dict):
        return candidate
    if isinstance(run_overrides.get("catalog"), dict):
        run_overrides = run_overrides["catalog"]
    if isinstance(run_overrides.get("items"), list):
        values: dict[str, Any] = {}
        for item in run_overrides["items"]:
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item.get("id") in PROMPT_DEFINITIONS
                and "value" in item
            ):
                values[item["id"]] = item["value"]
        return values
    return {
        key: value
        for key, value in run_overrides.items()
        if isinstance(key, str) and key in PROMPT_DEFINITIONS
    }


def _validate_value(key: str, value: Any) -> None:
    meta = PROMPT_DEFINITIONS[key]
    value_type = meta["type"]

    if value_type == "text":
        if not isinstance(value, str):
            raise PromptLibraryValidationError(f"{key} must be a string.")
        if not value.strip():
            raise PromptLibraryValidationError(f"{key} cannot be empty.")
        if len(value) > MAX_TEXT_LENGTH:
            raise PromptLibraryValidationError(
                f"{key} exceeds maximum length ({MAX_TEXT_LENGTH})."
            )
        for placeholder in meta["required_placeholders"]:
            if placeholder not in value:
                raise PromptLibraryValidationError(
                    f"{key} is missing required placeholder: {placeholder}"
                )
        return

    if value_type == "list":
        if not isinstance(value, list):
            raise PromptLibraryValidationError(f"{key} must be a list.")
        if key == "evaluation.criteria_mapping":
            if len(value) != 14:
                raise PromptLibraryValidationError(
                    "evaluation.criteria_mapping must contain exactly 14 items."
                )
            for idx, entry in enumerate(value):
                if not isinstance(entry, str) or not entry.strip():
                    raise PromptLibraryValidationError(
                        f"evaluation.criteria_mapping[{idx}] must be a non-empty string."
                    )
                if len(entry) > MAX_LIST_ITEM_LENGTH:
                    raise PromptLibraryValidationError(
                        "evaluation.criteria_mapping entries are too long."
                    )
        return

    raise PromptLibraryValidationError(f"Unsupported value type for {key}: {value_type}")


def _validate_values(values: dict[str, Any]) -> None:
    expected_ids = set(ORDERED_PROMPT_IDS)
    actual_ids = set(values.keys())

    missing = sorted(expected_ids - actual_ids)
    if missing:
        raise PromptLibraryValidationError(f"Missing required prompt IDs: {missing}")

    unknown = sorted(actual_ids - expected_ids)
    if unknown:
        raise PromptLibraryValidationError(f"Unknown prompt IDs: {unknown}")

    for key in ORDERED_PROMPT_IDS:
        _validate_value(key, values[key])


def _values_from_catalog(catalog: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        raise PromptLibraryValidationError("Catalog payload must be an object.")
    if strict and catalog.get("schema_version") != SCHEMA_VERSION:
        raise PromptLibraryValidationError(
            f"Unsupported schema_version: {catalog.get('schema_version')}"
        )

    items = catalog.get("items")
    if not isinstance(items, list):
        raise PromptLibraryValidationError("Catalog must contain an items array.")

    seen: set[str] = set()
    values: dict[str, Any] = {}
    allowed_fields = {
        "id",
        "title",
        "stage",
        "category",
        "source_path",
        "description",
        "type",
        "required_placeholders",
        "default_value",
        "value",
    }
    required_fields = allowed_fields

    for item in items:
        if not isinstance(item, dict):
            raise PromptLibraryValidationError("Each catalog item must be an object.")
        prompt_id = item.get("id")
        if not isinstance(prompt_id, str) or prompt_id not in PROMPT_DEFINITIONS:
            raise PromptLibraryValidationError(f"Invalid prompt ID in catalog item: {prompt_id}")
        if prompt_id in seen:
            raise PromptLibraryValidationError(f"Duplicate prompt ID in catalog: {prompt_id}")
        seen.add(prompt_id)

        if strict:
            item_fields = set(item.keys())
            if item_fields != required_fields:
                raise PromptLibraryValidationError(
                    f"Catalog item fields mismatch for {prompt_id}. "
                    f"Expected {sorted(required_fields)}, got {sorted(item_fields)}."
                )
            meta = PROMPT_DEFINITIONS[prompt_id]
            immutable_checks = {
                "title": meta["title"],
                "stage": meta["stage"],
                "category": meta["category"],
                "source_path": meta["source_path"],
                "description": meta["description"],
                "type": meta["type"],
                "required_placeholders": list(meta["required_placeholders"]),
                "default_value": meta["default_value"],
            }
            for field, expected in immutable_checks.items():
                if item.get(field) != expected:
                    raise PromptLibraryValidationError(
                        f"Immutable field changed for {prompt_id}: {field}"
                    )

        if "value" not in item:
            raise PromptLibraryValidationError(f"Catalog item {prompt_id} is missing value.")
        values[prompt_id] = item["value"]

    if strict and seen != set(ORDERED_PROMPT_IDS):
        missing = sorted(set(ORDERED_PROMPT_IDS) - seen)
        extra = sorted(seen - set(ORDERED_PROMPT_IDS))
        if missing:
            raise PromptLibraryValidationError(f"Missing required prompt IDs: {missing}")
        if extra:
            raise PromptLibraryValidationError(f"Unknown prompt IDs in catalog: {extra}")

    return values


def _is_writable() -> bool:
    try:
        LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        test_path = LIBRARY_PATH.parent / ".prompt_library_write_test"
        test_path.write_text("ok")
        test_path.unlink()
        return True
    except OSError:
        return False


def _load_values_from_disk() -> dict[str, Any]:
    values = get_default_values()
    if not LIBRARY_PATH.exists():
        return values
    try:
        raw = json.loads(LIBRARY_PATH.read_text())
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            loaded = _values_from_catalog(raw, strict=False)
        elif isinstance(raw, dict) and isinstance(raw.get("values"), dict):
            loaded = {
                key: value
                for key, value in raw["values"].items()
                if key in PROMPT_DEFINITIONS
            }
        elif isinstance(raw, dict):
            loaded = {key: value for key, value in raw.items() if key in PROMPT_DEFINITIONS}
        else:
            loaded = {}
        for key, value in loaded.items():
            values[key] = value
        _ensure_scoring_prompt_compat(values)
        _validate_values(values)
        return values
    except Exception:
        return get_default_values()


def _ensure_scoring_prompt_compat(values: dict[str, Any]) -> None:
    """Upgrade pre-redesign ranking prompts in-place.

    Runtime prompt catalogs and run-level prompt overrides can lag behind code
    defaults. The scoring redesign adds structured-signal placeholders and new
    output fields; old ranking prompts would silently bypass the new schema.
    """
    defaults = get_default_values()
    if "{scoring_signals}" not in str(values.get("ranking.team.user", "")):
        values["ranking.team.user"] = defaults["ranking.team.user"]
    if "{scoring_signals}" not in str(values.get("ranking.upside.user", "")):
        values["ranking.upside.user"] = defaults["ranking.upside.user"]
    if "founder_archetype" not in str(values.get("ranking.team.system", "")):
        values["ranking.team.system"] = defaults["ranking.team.system"]
    if "risk_adjusted_potential_score" not in str(values.get("ranking.upside.system", "")):
        values["ranking.upside.system"] = defaults["ranking.upside.system"]


def _write_catalog(catalog: dict[str, Any]) -> None:
    tmp_path = LIBRARY_PATH.with_suffix(".json.tmp")
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    tmp_path.replace(LIBRARY_PATH)


def _resolve_values(run_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    values = _load_values_from_disk()
    override_values = _extract_overrides(run_overrides)
    for key, value in override_values.items():
        values[key] = value
    _ensure_scoring_prompt_compat(values)
    _validate_values(values)
    return values


def get_prompt(key: str, run_overrides: dict[str, Any] | None = None) -> Any:
    """Return a single prompt value by key, with optional run-level overrides."""
    values = _resolve_values(run_overrides=run_overrides)
    if key not in values:
        raise KeyError(f"Unknown prompt key: {key}")
    return values[key]


def get_questions(run_overrides: dict[str, Any] | None = None) -> dict[str, str]:
    """Return the 4 root investment questions with optional overrides."""
    return {
        "general_company": str(get_prompt("questions.general_company", run_overrides)),
        "market": str(get_prompt("questions.market", run_overrides)),
        "product": str(get_prompt("questions.product", run_overrides)),
        "team": str(get_prompt("questions.team", run_overrides)),
    }


def get_criteria_mapping(run_overrides: dict[str, Any] | None = None) -> list[str]:
    """Return the 14 evaluation criterion labels with optional overrides."""
    criteria = get_prompt("evaluation.criteria_mapping", run_overrides)
    if not isinstance(criteria, list):
        raise PromptLibraryValidationError("evaluation.criteria_mapping must be a list.")
    return [str(item) for item in criteria]


def get_catalog(run_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the full catalog for UI rendering/editing."""
    values = _resolve_values(run_overrides=run_overrides)
    return _catalog_from_values(values)


def get_catalog_payload(run_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return catalog plus storage metadata for API responses."""
    writable = _is_writable()
    return {
        "schema_version": SCHEMA_VERSION,
        "writable": writable,
        "storage_mode": "global" if writable else "local_fallback",
        "catalog": get_catalog(run_overrides=run_overrides),
    }


def save_catalog(updated_catalog: dict[str, Any]) -> dict[str, Any]:
    """Validate and save the prompt catalog to disk."""
    if not _is_writable():
        raise PromptLibraryStorageError(
            "Prompt library is not writable in this environment."
        )
    values = _values_from_catalog(updated_catalog, strict=True)
    _validate_values(values)
    _write_catalog(_catalog_from_values(values))
    return get_catalog_payload()


def reset_catalog(ids: list[str] | None = None) -> dict[str, Any]:
    """Reset all prompts or selected IDs to defaults and persist."""
    if not _is_writable():
        raise PromptLibraryStorageError(
            "Prompt library is not writable in this environment."
        )
    defaults = get_default_values()
    if ids is None:
        new_values = defaults
    else:
        invalid = [prompt_id for prompt_id in ids if prompt_id not in PROMPT_DEFINITIONS]
        if invalid:
            raise PromptLibraryValidationError(f"Unknown prompt IDs in reset request: {invalid}")
        new_values = _load_values_from_disk()
        for prompt_id in ids:
            new_values[prompt_id] = deepcopy(defaults[prompt_id])
    _validate_values(new_values)
    _write_catalog(_catalog_from_values(new_values))
    return get_catalog_payload()
