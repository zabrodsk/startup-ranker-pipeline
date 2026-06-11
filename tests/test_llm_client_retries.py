"""Sprint 1 (W10): inner LangChain-client retries default to 0, decoupled from
the outer ThrottledRunnable policy (LLM_MAX_RETRIES)."""

from agent.llm import get_llm_runtime_settings


def test_inner_client_retries_default_to_zero(monkeypatch):
    monkeypatch.delenv("LLM_CLIENT_MAX_RETRIES", raising=False)
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)

    assert get_llm_runtime_settings()["max_retries"] == 0


def test_inner_client_retries_follow_dedicated_env(monkeypatch):
    monkeypatch.setenv("LLM_CLIENT_MAX_RETRIES", "3")

    assert get_llm_runtime_settings()["max_retries"] == 3


def test_outer_policy_env_does_not_reenable_inner_retries(monkeypatch):
    monkeypatch.delenv("LLM_CLIENT_MAX_RETRIES", raising=False)
    monkeypatch.setenv("LLM_MAX_RETRIES", "9")

    assert get_llm_runtime_settings()["max_retries"] == 0


def test_no_backoff_decorators_left_in_pipeline_stages():
    """ThrottledRunnable is the single retry owner; stages must not stack @backoff."""
    from agent.pipeline.stages import critique, evaluation, refinement

    for module in (critique, evaluation, refinement):
        assert not hasattr(module, "backoff"), f"{module.__name__} still imports backoff"
