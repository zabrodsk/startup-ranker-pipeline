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

from agent.llm_catalog import normalize_creativity, normalize_provider, supports_selection_creativity_control
from agent.llm_policy import (
    resolve_openai_phase_sampling,
    resolve_openai_reasoning_fallback_temperature,
)
from agent.rate_limit import wrap_llm
from agent.run_context import (
    get_current_collector,
    get_current_llm_request_settings,
    get_current_llm_selection,
    get_current_stage_name,
    set_current_llm_request_settings,
)

load_dotenv()

_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
_DEFAULT_TIMEOUT_SECONDS = 90.0
_DEFAULT_MAX_RETRIES = 2
_CHAT_PROVIDER = "gemini"
_CHAT_MODEL = "gemini-3.1-flash-lite-preview"
_FALLBACK_ENCODING_NAME = "o200k_base"
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_APP_NAME = "Deal Intelligence"
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
    if provider_norm in {"openai", "openrouter"}:
        lookup_model = model_norm.rsplit("/", 1)[-1]
        try:
            return tiktoken.encoding_for_model(lookup_model)
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
                # Skip OpenAI Responses API reasoning blocks — they carry
                # no human-readable text and will otherwise stringify into
                # user-visible output.
                if item.get("type") == "reasoning":
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                # Any other dict shape — drop silently.
            else:
                # Unknown non-dict, non-str block — drop silently instead
                # of serialising the repr.
                continue
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
            "reasoning_fallback_applied",
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
    return (model or "").strip() in {"gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}


def _resolve_openai_request_settings(
    model: str,
    requested_temperature: float | None,
    requested_reasoning_effort: str | None,
    *,
    selected_temperature: float | None = None,
    selected_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if _is_reasoning_effort_model(model):
        resolved = resolve_openai_phase_sampling(
            model,
            get_current_stage_name(),
            requested_temperature,
            requested_reasoning_effort=requested_reasoning_effort,
            selected_temperature=selected_temperature,
            selected_reasoning_effort=selected_reasoning_effort,
        )
        resolved_temperature = requested_temperature
        resolved_reasoning_effort = requested_reasoning_effort
        if resolved:
            resolved_temperature = resolved.get("temperature")
            resolved_reasoning_effort = resolved.get("reasoning_effort")
        sampling_mode = (
            "temperature_plus_reasoning"
            if resolved_temperature is not None and resolved_reasoning_effort is not None
            else "reasoning_effort"
        )
        return {
            "requested_temperature": requested_temperature,
            "effective_temperature": resolved_temperature,
            "sampling_mode": sampling_mode,
            "requested_reasoning_effort": resolved_reasoning_effort,
            "effective_reasoning_effort": resolved_reasoning_effort,
            "reasoning_fallback_applied": False,
        }

    sampling_mode = _normalize_gpt5_temperature_mode(os.getenv(_GPT5_TEMPERATURE_MODE_ENV))
    effective_temperature = requested_temperature
    if _is_exact_gpt5_model(model):
        effective_temperature = 1.0 if sampling_mode == "force_one" else requested_temperature
    else:
        sampling_mode = "requested"
    return {
        "requested_temperature": requested_temperature,
        "effective_temperature": effective_temperature,
        "sampling_mode": sampling_mode,
        "requested_reasoning_effort": requested_reasoning_effort,
        "effective_reasoning_effort": requested_reasoning_effort,
        "reasoning_fallback_applied": False,
    }


def create_llm(
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    """Create an LLM instance based on environment configuration.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).
        reasoning_effort: Optional reasoning effort for OpenAI reasoning-capable models.

    Returns:
        A LangChain chat model instance for the configured provider.

    Raises:
        ValueError: If the provider is unknown or required keys are missing.
    """
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")

    selection = get_current_llm_selection() or {}
    provider = normalize_provider(selection.get("provider") or os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER))
    model = selection.get("model") or os.getenv("MODEL_NAME", _DEFAULT_MODEL)
    selection_creativity = None
    if supports_selection_creativity_control(provider, model):
        selection_creativity = normalize_creativity(selection.get("creativity"))
    selection_temperature = selection.get("temperature")
    selection_reasoning_effort = selection.get("reasoning_effort")
    requested_temperature = selection_creativity if selection_creativity is not None else (
        selection_temperature if selection_temperature is not None else temperature
    )
    requested_reasoning_effort = (
        selection_reasoning_effort if selection_reasoning_effort is not None else reasoning_effort
    )
    runtime = get_llm_runtime_settings()
    timeout_s = runtime["request_timeout_seconds"]
    max_retries = runtime["max_retries"]

    request_settings = {
        "requested_temperature": requested_temperature,
        "effective_temperature": requested_temperature,
        "sampling_mode": "selection_creativity" if selection_creativity is not None else "requested",
        "requested_reasoning_effort": requested_reasoning_effort,
        "effective_reasoning_effort": requested_reasoning_effort,
        "reasoning_fallback_applied": False,
        "provider": provider,
        "model": model,
        "selection_creativity": selection_creativity,
    }
    if provider == "openai":
        request_settings.update(
            _resolve_openai_request_settings(
                model,
                requested_temperature,
                requested_reasoning_effort,
                selected_temperature=selection_temperature,
                selected_reasoning_effort=selection_reasoning_effort,
            )
        )
    set_current_llm_request_settings(request_settings)
    effective_temperature = request_settings["effective_temperature"]

    if provider == "gemini":
        return wrap_llm(_create_gemini(model, requested_temperature or 0.0, timeout_s, max_retries))
    elif provider == "openai":
        fallback_builder = _build_openai_reasoning_fallback_builder(
            model,
            timeout_s,
            max_retries,
        )
        return wrap_llm(
            _create_openai(
                model,
                effective_temperature,
                timeout_s,
                max_retries,
                reasoning_effort=request_settings.get("effective_reasoning_effort"),
            ),
            fallback_builder=fallback_builder,
        )
    elif provider == "openrouter":
        return wrap_llm(_create_openrouter(model, requested_temperature or 0.0, timeout_s, max_retries))
    elif provider == "anthropic":
        return wrap_llm(_create_anthropic(model, requested_temperature or 0.0, timeout_s, max_retries))
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Supported: gemini, openai, anthropic, openrouter"
        )


def chat_llm_selection() -> dict[str, str]:
    """Return the fixed model selection for company chat."""
    return {
        "provider": normalize_provider(_CHAT_PROVIDER),
        "model": _CHAT_MODEL,
    }


def create_chat_llm(temperature: float = 0.2) -> BaseChatModel:
    """Create the fixed Gemini model used by company chat."""
    runtime = get_llm_runtime_settings()
    timeout_s = runtime["request_timeout_seconds"]
    max_retries = runtime["max_retries"]
    return wrap_llm(_create_gemini(_CHAT_MODEL, temperature, timeout_s, max_retries))


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
    temperature: float | None,
    timeout_s: float,
    max_retries: int,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "request_timeout": timeout_s,
        "max_retries": max_retries,
        "callbacks": [_TELEMETRY_CALLBACK],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["model_kwargs"] = {"reasoning": {"effort": reasoning_effort}}

    return ChatOpenAI(**kwargs)


def _build_openai_reasoning_fallback_builder(
    model: str,
    timeout_s: float,
    max_retries: int,
):
    if model != "gpt-5.4-nano":
        return None

    def _builder():
        stage_name = get_current_stage_name()
        fallback_temperature = resolve_openai_reasoning_fallback_temperature(
            model,
            stage_name,
        )
        if fallback_temperature is None:
            return None

        current_settings = dict(get_current_llm_request_settings() or {})
        current_settings.update(
            {
                "requested_temperature": fallback_temperature,
                "effective_temperature": fallback_temperature,
                "sampling_mode": "reasoning_fallback_temperature_only",
                "effective_reasoning_effort": None,
                "reasoning_fallback_applied": True,
            }
        )
        set_current_llm_request_settings(current_settings)
        return _create_openai(
            model,
            fallback_temperature,
            timeout_s,
            max_retries,
            reasoning_effort=None,
        )

    return _builder


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
    referer = os.getenv("OPENROUTER_SITE_URL") or os.getenv("APP_BASE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME") or _OPENROUTER_DEFAULT_APP_NAME
    default_headers = {"X-Title": app_name}
    if referer:
        default_headers["HTTP-Referer"] = referer

    if model.startswith("openai/gpt-5") or model.startswith("gpt-5"):
        temperature = 1

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        request_timeout=timeout_s,
        max_retries=max_retries,
        default_headers=default_headers,
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
