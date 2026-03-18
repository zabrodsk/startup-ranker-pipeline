"""Centralized LLM configuration for the agent package.

Delegates to agent.llm.create_llm() for multi-provider support.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from agent.llm import create_llm


def get_llm(
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    """Get a configured LLM instance.

    Args:
        temperature: Controls randomness in responses (0.0 = deterministic).
        reasoning_effort: Optional OpenAI reasoning effort override.

    Returns:
        Configured chat model instance for the active provider.
    """
    return create_llm(temperature=temperature, reasoning_effort=reasoning_effort)
