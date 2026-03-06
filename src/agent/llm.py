"""Centralized multi-provider LLM factory.

Supports Gemini, OpenAI, Anthropic, and OpenRouter via environment variables.

Environment variables:
    LLM_PROVIDER: One of "gemini", "openai", "anthropic", "openrouter" (default: "gemini")
    MODEL_NAME: Model identifier (default: "gemini-3.1-flash-lite-preview")
    GOOGLE_API_KEY: Required when LLM_PROVIDER=gemini
    OPENAI_API_KEY: Required when LLM_PROVIDER=openai or openrouter
    ANTHROPIC_API_KEY: Required when LLM_PROVIDER=anthropic
    OPENAI_BASE_URL: Required when LLM_PROVIDER=openrouter
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from agent.rate_limit import wrap_llm

load_dotenv()

_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
_DEFAULT_TIMEOUT_SECONDS = 90.0
_DEFAULT_MAX_RETRIES = 2


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

    provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    model = os.getenv("MODEL_NAME", _DEFAULT_MODEL)
    timeout_s = _read_positive_float_env("LLM_REQUEST_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
    max_retries = _read_nonnegative_int_env("LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES)

    if provider == "gemini":
        return wrap_llm(_create_gemini(model, temperature, timeout_s, max_retries))
    elif provider == "openai":
        return wrap_llm(_create_openai(model, temperature, timeout_s, max_retries))
    elif provider == "openrouter":
        return wrap_llm(_create_openrouter(model, temperature, timeout_s, max_retries))
    elif provider == "anthropic":
        return wrap_llm(_create_anthropic(model, temperature, timeout_s, max_retries))
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Supported: gemini, openai, anthropic, openrouter"
        )


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
    )


def _create_openai(
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if model == "gpt-5-mini":
        temperature = 1

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        request_timeout=timeout_s,
        max_retries=max_retries,
    )


def _create_openrouter(
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError(
            "OPENAI_API_KEY and OPENAI_BASE_URL are required when LLM_PROVIDER=openrouter"
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        request_timeout=timeout_s,
        max_retries=max_retries,
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
    )
