"""Centralized multi-provider LLM factory.

Supports Gemini, OpenAI, Anthropic, and OpenRouter via environment variables.

Environment variables:
    LLM_PROVIDER: One of "gemini", "openai", "anthropic", "openrouter" (default: "gemini")
    MODEL_NAME: Model identifier (default: "gemini-2.0-flash")
    GOOGLE_API_KEY: Required when LLM_PROVIDER=gemini
    OPENAI_API_KEY: Required when LLM_PROVIDER=openai or openrouter
    ANTHROPIC_API_KEY: Required when LLM_PROVIDER=anthropic
    OPENAI_BASE_URL: Required when LLM_PROVIDER=openrouter
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "gemini-2.5-flash"


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

    if provider == "gemini":
        return _create_gemini(model, temperature)
    elif provider == "openai":
        return _create_openai(model, temperature)
    elif provider == "openrouter":
        return _create_openrouter(model, temperature)
    elif provider == "anthropic":
        return _create_anthropic(model, temperature)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Supported: gemini, openai, anthropic, openrouter"
        )


def _create_gemini(model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )


def _create_openai(model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if model == "gpt-5-mini":
        temperature = 1

    return ChatOpenAI(model=model, temperature=temperature)


def _create_openrouter(model: str, temperature: float) -> BaseChatModel:
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
    )


def _create_anthropic(model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
