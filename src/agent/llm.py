"""Centralized multi-provider LLM factory.

Supports Gemini, OpenAI, Anthropic, and OpenRouter via environment variables.

Environment variables:
    LLM_PROVIDER: One of "gemini", "openai", "anthropic", "openrouter" (default: "gemini")
    MODEL_NAME: Model identifier (default: "gemini-3.1-flash-lite-preview")
    GOOGLE_API_KEY: Required when LLM_PROVIDER=gemini
    OPENAI_API_KEY: Required when LLM_PROVIDER=openai
    OPENROUTER_API_KEY: Required when LLM_PROVIDER=openrouter (falls back to OPENAI_API_KEY)
    ANTHROPIC_API_KEY: Required when LLM_PROVIDER=anthropic
    OPENROUTER_BASE_URL: Optional when LLM_PROVIDER=openrouter (defaults to https://openrouter.ai/api/v1)
    OPENAI_BASE_URL: Legacy fallback for OpenRouter base URL
"""

import os
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
import tiktoken

from agent.llm_catalog import normalize_provider
from agent.rate_limit import wrap_llm
from agent.run_context import (
    get_current_collector,
    get_current_stage_name,
    get_current_llm_request_settings,
    get_current_llm_selection,
    set_current_llm_request_settings,
)

load_dotenv()

_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
_DEFAULT_TIMEOUT_SECONDS = 90.0
_DEFAULT_MAX_RETRIES = 2
_FALLBACK_ENCODING_NAME = "o200k_base"
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_GPT5_TEMPERATURE_MODE_ENV = "OPENAI_GPT5_TEMPERATURE_MODE"
_DEFAULT_GPT5_TEMPERATURE_MODE = "respect_requested"


def _extract_usage_metadata(payload: Any) -> dict[str, int] | None:
    """Normalize provider-specific usage payloads into prompt/completion/total tokens."""
    if payload is None:
        return None

    sources: list[dict[str, Any]] = []
    generations = None
    if isinstance(payload, dict):
        sources.append(payload)
        generations = payload.get("generations")
        if isinstance(payload.get("llm_output"), dict):
            sources.append(payload["llm_output"])
        if isinstance(payload.get("response_metadata"), dict):
            sources.append(payload["response_metadata"])
    else:
        llm_output = getattr(payload, "llm_output", None)
        response_metadata = getattr(payload, "response_metadata", None)
        generations = getattr(payload, "generations", None)
        if isinstance(llm_output, dict):
            sources.append(llm_output)
        if isinstance(response_metadata, dict):
            sources.append(response_metadata)
        if hasattr(payload, "dict"):
            try:
                payload_dict = payload.dict()
            except Exception:
                payload_dict = None
            if isinstance(payload_dict, dict):
                sources.append(payload_dict)
                if generations is None:
                    generations = payload_dict.get("generations")
                if isinstance(payload_dict.get("llm_output"), dict):
                    sources.append(payload_dict["llm_output"])
                if isinstance(payload_dict.get("response_metadata"), dict):
                    sources.append(payload_dict["response_metadata"])

    candidate = None
    for source in sources:
        candidate = (
            source.get("token_usage")
            or source.get("usage_metadata")
            or source.get("usage")
        )
        if isinstance(candidate, dict):
            break
        candidate = None

    if candidate is None and isinstance(generations, list):
        for generation_list in generations:
            if not generation_list:
                continue
            generation = generation_list[0]
            message = getattr(generation, "message", None)
            if message is None and isinstance(generation, dict):
                message = generation.get("message")
            usage_metadata = getattr(message, "usage_metadata", None)
            if usage_metadata is None and isinstance(message, dict):
                usage_metadata = message.get("usage_metadata")
            if isinstance(usage_metadata, dict):
                candidate = usage_metadata
                break

    if not isinstance(candidate, dict):
        return None

    def _read_int(*keys: str) -> int | None:
        for key in keys:
            value = candidate.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return None

    prompt_tokens = _read_int("prompt_tokens", "input_tokens", "prompt_token_count", "input_token_count")
    completion_tokens = _read_int("completion_tokens", "output_tokens", "candidates_token_count", "output_token_count")
    total_tokens = _read_int("total_tokens", "total_token_count")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _estimate_usage_metadata(
    *,
    provider: str | None,
    model: str | None,
    prompt_text: str,
    completion_text: str,
) -> dict[str, int] | None:
    prompt_tokens = _estimate_text_tokens(prompt_text, provider=provider, model=model)
    completion_tokens = _estimate_text_tokens(completion_text, provider=provider, model=model)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _estimate_text_tokens(
    text: str,
    *,
    provider: str | None,
    model: str | None,
) -> int | None:
    try:
        encoding = _encoding_for_selection(provider, model)
        return len(encoding.encode(text or ""))
    except Exception:
        normalized = (text or "").strip()
        if not normalized:
            return 0
        return max(1, round(len(normalized) / 4))


def _encoding_for_selection(provider: str | None, model: str | None):
    provider_norm = normalize_provider(provider or _DEFAULT_PROVIDER)
    model_norm = (model or _DEFAULT_MODEL).strip()
    if provider_norm == "openai":
        try:
            return tiktoken.encoding_for_model(model_norm)
        except KeyError:
            return tiktoken.get_encoding(_FALLBACK_ENCODING_NAME)
    return tiktoken.get_encoding(_FALLBACK_ENCODING_NAME)


def _render_message_text(messages: list[list[BaseMessage]]) -> str:
    rendered_batches: list[str] = []
    for batch in messages:
        batch_parts: list[str] = []
        for message in batch:
            role = getattr(message, "type", None) or getattr(message, "role", None) or message.__class__.__name__
            content = _coerce_message_content(getattr(message, "content", ""))
            batch_parts.append(f"{role}: {content}")
        rendered_batches.append("\n".join(batch_parts))
    return "\n\n".join(part for part in rendered_batches if part).strip()


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _extract_completion_text(payload: Any) -> str:
    generations = getattr(payload, "generations", None)
    if generations is None and isinstance(payload, dict):
        generations = payload.get("generations")
    if not isinstance(generations, list):
        return ""

    parts: list[str] = []
    for generation_list in generations:
        if not generation_list:
            continue
        generation = generation_list[0]
        message = getattr(generation, "message", None)
        if message is None and isinstance(generation, dict):
            message = generation.get("message")
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            parts.append(_coerce_message_content(content))
            continue
        text = getattr(generation, "text", None)
        if text is None and isinstance(generation, dict):
            text = generation.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(part for part in parts if part).strip()


class _TelemetryCallbackHandler(BaseCallbackHandler):
    """Collect token usage from LangChain callback payloads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._prompt_text_by_run_id: dict[Any, str] = {}

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs: Any) -> Any:
        prompt_text = "\n\n".join(str(prompt or "") for prompt in prompts or []).strip()
        with self._lock:
            self._prompt_text_by_run_id[run_id] = prompt_text
        return None

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs: Any) -> Any:
        with self._lock:
            self._prompt_text_by_run_id[run_id] = _render_message_text(messages or [])
        return None

    def on_llm_end(self, response, *, run_id=None, **kwargs: Any) -> Any:
        collector = get_current_collector()
        selection = get_current_llm_selection()
        request_settings = get_current_llm_request_settings() or {}
        if not collector or not selection:
            return None
        usage = _extract_usage_metadata(response)
        estimated = False
        if usage is None and run_id is not None:
            with self._lock:
                prompt_text = self._prompt_text_by_run_id.pop(run_id, "")
            usage = _estimate_usage_metadata(
                provider=selection.get("provider"),
                model=selection.get("model"),
                prompt_text=prompt_text,
                completion_text=_extract_completion_text(response),
            )
            estimated = usage is not None
        elif run_id is not None:
            with self._lock:
                self._prompt_text_by_run_id.pop(run_id, None)
        metadata = {"estimated_usage": estimated}
        for key in (
            "requested_temperature",
            "effective_temperature",
            "sampling_mode",
            "requested_reasoning_effort",
            "effective_reasoning_effort",
        ):
            if key in request_settings:
                metadata[key] = request_settings[key]

        collector.record_llm_usage(
            provider=selection["provider"],
            model=selection["model"],
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            total_tokens=(usage or {}).get("total_tokens"),
            metadata=metadata,
        )
        return None

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs: Any) -> Any:
        with self._lock:
            self._prompt_text_by_run_id.pop(run_id, None)
        return None


_TELEMETRY_CALLBACK = _TelemetryCallbackHandler()


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _read_nonnegative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= 0 else default
    except Exception:
        return default


def _normalize_gpt5_temperature_mode(value: str | None) -> str:
    mode = (value or "").strip().lower()
    if mode in {"respect_requested", "force_one"}:
        return mode
    return _DEFAULT_GPT5_TEMPERATURE_MODE


def _is_exact_gpt5_model(model: str) -> bool:
    return (model or "").strip() == "gpt-5"


def _is_reasoning_effort_model(model: str) -> bool:
    return (model or "").strip() in {"gpt-5.2", "gpt-5.4"}


def _resolve_reasoning_effort_for_stage(
    model: str,
    stage_name: str | None,
    requested_temperature: float,
) -> str:
    stage = (stage_name or "").strip().lower()
    ranking_effort = "xhigh" if model == "gpt-5.4" else "high"
    stage_map = {
        "decomposition": "medium",
        "answering": "low",
        "generation_pro": "medium",
        "generation_contra": "medium",
        "evaluation": "high",
        "ranking_dimension_score": ranking_effort,
        "ranking_executive_summary": ranking_effort,
    }
    if stage in stage_map:
        return stage_map[stage]
    if requested_temperature <= 0.0:
        return "low"
    return "medium"


def _resolve_openai_temperature(
    model: str,
    requested_temperature: float,
) -> dict[str, Any]:
    if _is_reasoning_effort_model(model):
        effort = _resolve_reasoning_effort_for_stage(
            model,
            get_current_stage_name(),
            requested_temperature,
        )
        return {
            "requested_temperature": requested_temperature,
            "effective_temperature": 1.0,
            "sampling_mode": "reasoning_effort",
            "requested_reasoning_effort": effort,
            "effective_reasoning_effort": effort,
        }

    if not _is_exact_gpt5_model(model):
        return {
            "requested_temperature": requested_temperature,
            "effective_temperature": 1.0,
            "sampling_mode": "default_one",
        }

    sampling_mode = _normalize_gpt5_temperature_mode(os.getenv(_GPT5_TEMPERATURE_MODE_ENV))
    effective_temperature = 1.0 if sampling_mode == "force_one" else requested_temperature
    return {
        "requested_temperature": requested_temperature,
        "effective_temperature": effective_temperature,
        "sampling_mode": sampling_mode,
    }


def create_llm(temperature: float = 0.0) -> BaseChatModel:
    """Create an LLM instance based on environment configuration.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).

    Returns:
        A LangChain chat model instance for the configured provider.

    Raises:
        ValueError: If the provider is unknown or required keys are missing.
    """
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")

    selection = get_current_llm_selection() or {}
    provider = normalize_provider(selection.get("provider") or os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER))
    model = selection.get("model") or os.getenv("MODEL_NAME", _DEFAULT_MODEL)
    runtime = get_llm_runtime_settings()
    timeout_s = runtime["request_timeout_seconds"]
    max_retries = runtime["max_retries"]

    request_settings = {
        "requested_temperature": temperature,
        "effective_temperature": temperature,
        "sampling_mode": "requested",
        "requested_reasoning_effort": None,
        "effective_reasoning_effort": None,
        "provider": provider,
        "model": model,
    }
    if provider == "openai":
        request_settings.update(_resolve_openai_temperature(model, temperature))
    set_current_llm_request_settings(request_settings)
    effective_temperature = float(request_settings["effective_temperature"])

    if provider == "gemini":
        return wrap_llm(_create_gemini(model, temperature, timeout_s, max_retries))
    elif provider == "openai":
        return wrap_llm(
            _create_openai(
                model,
                effective_temperature,
                timeout_s,
                max_retries,
                reasoning_effort=request_settings.get("effective_reasoning_effort"),
            )
        )
    elif provider == "openrouter":
        return wrap_llm(_create_openrouter(model, temperature, timeout_s, max_retries))
    elif provider == "anthropic":
        return wrap_llm(_create_anthropic(model, temperature, timeout_s, max_retries))
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Supported: gemini, openai, anthropic, openrouter"
        )


def get_llm_runtime_settings() -> dict[str, float | int]:
    """Return the active runtime request controls for the current process."""
    return {
        "request_timeout_seconds": _read_positive_float_env(
            "LLM_REQUEST_TIMEOUT_SECONDS",
            _DEFAULT_TIMEOUT_SECONDS,
        ),
        "max_retries": _read_nonnegative_int_env(
            "LLM_MAX_RETRIES",
            _DEFAULT_MAX_RETRIES,
        ),
    }


def _create_gemini(
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        timeout=timeout_s,
        max_retries=max_retries,
        callbacks=[_TELEMETRY_CALLBACK],
    )


def _create_openai(
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        request_timeout=timeout_s,
        max_retries=max_retries,
        callbacks=[_TELEMETRY_CALLBACK],
    )


def _create_openrouter(
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = (
        os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or _DEFAULT_OPENROUTER_BASE_URL
    )
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter"
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        request_timeout=timeout_s,
        max_retries=max_retries,
        callbacks=[_TELEMETRY_CALLBACK],
    )


def _create_anthropic(
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
        default_request_timeout=timeout_s,
        max_retries=max_retries,
        callbacks=[_TELEMETRY_CALLBACK],
    )
