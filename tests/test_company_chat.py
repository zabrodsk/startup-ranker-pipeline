import sys
import asyncio
from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import web.app as web_app
from agent.company_chat import answer_company_question
from agent.ingest.store import Chunk


class _FakeChatRunnable:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    async def ainvoke(self, _messages):
        content = self._responses.pop(0)
        return type("Resp", (), {"content": content})()


def _login(client: TestClient) -> None:
    response = client.post("/api/login", json={"password": "9876"})
    assert response.status_code == 200
    client.cookies.set("session_id", response.json()["session_id"])


def test_company_chat_endpoints_roundtrip(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "available_chat_models_payload", lambda: [
        {
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "label": "Gemini 3.1 Flash Lite",
            "summary": "Budget speed",
            "tier": "budget",
            "available": True,
            "selectable": True,
            "pricing_available": True,
            "supports_structured_output": True,
            "unavailable_reason": "",
        },
        {
            "provider": "openai",
            "model": "gpt-5-mini",
            "label": "GPT-5 mini",
            "summary": "Balanced pick",
            "tier": "balanced",
            "available": True,
            "selectable": True,
            "pricing_available": True,
            "supports_structured_output": True,
            "unavailable_reason": "",
        },
    ])
    monkeypatch.setattr(web_app, "db", type("FakeDb", (), {
        "is_configured": staticmethod(lambda: True),
        "load_company_chat_context": staticmethod(lambda company_lookup_key: {
            "company_lookup_key": company_lookup_key,
            "company_name": "Apify",
            "runs": [{
                "job_id": "job-1",
                "company_name": "Apify",
                "created_at": "2026-03-10T10:00:00Z",
                "results": {
                    "qa_provenance_rows": [{"question": "Why now?", "answer": "Because growth accelerated."}],
                    "argument_rows": [{"type": "pro", "argument_text": "Strong automation wedge"}],
                },
                "chunks": [{
                    "chunk_id": "chunk_1",
                    "text": "Apify is growing quickly in enterprise automation.",
                    "source_file": "deck.pdf",
                    "page_or_slide": "4",
                }],
            }],
        }),
    })())

    async def fake_answer_company_question(**kwargs):
        assert kwargs["llm_selection"]["provider"] == "openai"
        assert kwargs["llm_selection"]["model"] == "gpt-5-mini"
        return {
            "answer": "The strongest support is enterprise automation growth.",
            "citations": [{
                "kind": "chunk",
                "citation_id": "chunk:job-1:chunk_1",
                "label": "deck.pdf · 4",
                "excerpt": "Apify is growing quickly in enterprise automation.",
                "job_id": "job-1",
                "created_at": "2026-03-10T10:00:00Z",
                "source_file": "deck.pdf",
                "page_or_slide": "4",
            }],
            "used_run_ids": ["job-1"],
            "used_web_search": True,
            "web_search_query": "Apify enterprise automation growth",
            "web_search_results": "Web fallback snippet",
            "llm_selection": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "label": "GPT-5 mini",
            },
            "model_label": "GPT-5 mini",
            "run_costs": {
                "currency": "USD",
                "status": "complete",
                "total_usd": 0.0092,
                "llm_usd": 0.0042,
                "perplexity_usd": 0.005,
                "llm_tokens": {"prompt": 1000, "completion": 200, "total": 1200},
                "perplexity_search": {"requests": 1, "total_usd": 0.005},
                "by_model": [{
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "label": "GPT-5 mini",
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                    "usd": 0.0042,
                    "pricing_available": True,
                    "partial": False,
                }],
            },
            "model_executions": [{
                "service": "llm",
                "provider": "openai",
                "model": "gpt-5-mini",
                "stage": "company_chat",
                "status": "done",
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "estimated_cost_usd": 0.0042,
                "request_count": 1,
                "metadata": {},
            }],
            "source_counts": {"chunks": 1, "qa": 1, "arguments": 1},
            "run_count": 1,
        }

    monkeypatch.setattr(web_app, "answer_company_question", fake_answer_company_question)

    with TestClient(web_app.app) as client:
        _login(client)

        initial = client.get("/api/companies/name:apify/chat")
        assert initial.status_code == 200
        assert initial.json()["transcript"] == []
        assert initial.json()["session_run_costs"]["status"] == "unavailable"

        posted = client.post(
            "/api/companies/name:apify/chat",
            json={
                "message": "What is the strongest evidence?",
                "active_job_id": "job-1",
                "llm_provider": "openai",
                "llm_model": "gpt-5-mini",
            },
        )
        assert posted.status_code == 200
        payload = posted.json()
        assert payload["answer"] == "The strongest support is enterprise automation growth."
        assert payload["used_web_search"] is True
        assert payload["web_search_query"] == "Apify enterprise automation growth"
        assert payload["llm_provider"] == "openai"
        assert payload["llm_model"] == "gpt-5-mini"
        assert payload["model_label"] == "GPT-5 mini"
        assert payload["run_costs"]["total_usd"] == 0.0092
        assert payload["run_costs"]["perplexity_search"]["total_usd"] == 0.005
        assert payload["session_run_costs"]["total_usd"] == 0.0092
        assert payload["session_run_costs"]["perplexity_search"]["total_usd"] == 0.005
        assert len(payload["transcript"]) == 2
        assert payload["transcript"][1]["citations"][0]["kind"] == "chunk"
        assert payload["transcript"][1]["llm_label"] == "GPT-5 mini"
        assert payload["transcript"][1]["run_costs"]["total_usd"] == 0.0092

        fetched = client.get("/api/companies/name:apify/chat")
        assert fetched.status_code == 200
        fetched_payload = fetched.json()
        assert len(fetched_payload["transcript"]) == 2
        assert fetched_payload["llm_provider"] == "openai"
        assert fetched_payload["llm_model"] == "gpt-5-mini"
        assert fetched_payload["session_run_costs"]["total_usd"] == 0.0092
        assert fetched_payload["session_run_costs"]["perplexity_search"]["total_usd"] == 0.005

        cleared = client.delete("/api/companies/name:apify/chat")
        assert cleared.status_code == 200

        after_clear = client.get("/api/companies/name:apify/chat")
        assert after_clear.status_code == 200
        assert after_clear.json()["transcript"] == []
        assert after_clear.json()["llm_provider"] == "openai"
        assert after_clear.json()["llm_model"] == "gpt-5-mini"
        assert after_clear.json()["session_run_costs"]["status"] == "unavailable"


def test_company_chat_persists_shared_history_across_users(monkeypatch) -> None:
    shared_sessions: dict[str, dict] = {}

    monkeypatch.setattr(web_app, "available_chat_models_payload", lambda: [])
    monkeypatch.setattr(web_app, "db", type("FakeDb", (), {
        "is_configured": staticmethod(lambda: True),
        "load_company_chat_context": staticmethod(lambda company_lookup_key: {
            "company_lookup_key": company_lookup_key,
            "company_name": "Apaleo",
            "runs": [{
                "job_id": "job-28",
                "company_name": "Apaleo",
                "created_at": "2026-03-15T22:58:47Z",
                "results": {
                    "qa_provenance_rows": [],
                    "argument_rows": [],
                },
                "chunks": [{
                    "chunk_id": "chunk_1",
                    "text": "Apaleo is API-first hospitality infrastructure.",
                    "source_file": "deck.pdf",
                    "page_or_slide": "8",
                }],
            }],
        }),
        "load_company_chat_session": staticmethod(lambda company_lookup_key: deepcopy(shared_sessions.get(company_lookup_key))),
        "persist_company_chat_session": staticmethod(lambda **payload: shared_sessions.__setitem__(payload["company_lookup_key"], deepcopy(payload)) or True),
        "delete_company_chat_session": staticmethod(lambda company_lookup_key: shared_sessions.pop(company_lookup_key, None) is not None or True),
    })())

    async def fake_answer_company_question(**_kwargs):
        return {
            "answer": "Shared persisted answer with web evidence.",
            "citations": [{
                "kind": "web",
                "citation_id": "web:fallback",
                "label": "Web fallback",
                "excerpt": "Raw web search excerpt",
                "web_search_query": "Apaleo competitors moats comparison alternatives rivals",
                "web_search_results": "Raw search result text with named competitors and moat framing.",
                "web_search_provider": "sonar",
                "web_search_cost_usd": 0.005,
            }],
            "used_run_ids": ["job-28"],
            "used_web_search": True,
            "web_search_query": "Apaleo competitors moats comparison alternatives rivals",
            "web_search_results": "Raw search result text with named competitors and moat framing.",
            "llm_selection": {
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite-preview",
                "label": "Gemini 3.1 Flash Lite",
            },
            "model_label": "Gemini 3.1 Flash Lite",
            "run_costs": {
                "currency": "USD",
                "status": "complete",
                "total_usd": 0.0065,
                "llm_usd": 0.0015,
                "perplexity_usd": 0.005,
                "llm_tokens": {"prompt": 900, "completion": 150, "total": 1050},
                "perplexity_search": {"requests": 1, "total_usd": 0.005},
                "by_model": [{
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite-preview",
                    "label": "Gemini 3.1 Flash Lite",
                    "prompt_tokens": 900,
                    "completion_tokens": 150,
                    "total_tokens": 1050,
                    "usd": 0.0015,
                    "pricing_available": True,
                    "partial": False,
                }],
            },
            "model_executions": [{
                "service": "llm",
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite-preview",
                "stage": "company_chat",
                "status": "done",
                "prompt_tokens": 900,
                "completion_tokens": 150,
                "total_tokens": 1050,
                "estimated_cost_usd": 0.0015,
                "request_count": 1,
                "metadata": {},
            }],
            "source_counts": {"chunks": 1, "qa": 0, "arguments": 0},
            "run_count": 1,
        }

    monkeypatch.setattr(web_app, "answer_company_question", fake_answer_company_question)

    with TestClient(web_app.app) as client_a, TestClient(web_app.app) as client_b:
        _login(client_a)
        _login(client_b)

        post = client_a.post(
            "/api/companies/name:apaleo/chat",
            json={"message": "Who are the competitors?", "active_job_id": "job-28"},
        )
        assert post.status_code == 200
        assert "name:apaleo" in shared_sessions

        fetched = client_b.get("/api/companies/name:apaleo/chat")
        assert fetched.status_code == 200
        payload = fetched.json()
        assert len(payload["transcript"]) == 2
        assert payload["session_run_costs"]["total_usd"] == 0.0065
        assert payload["session_run_costs"]["perplexity_search"]["total_usd"] == 0.005
        assistant = payload["transcript"][1]
        assert assistant["run_costs"]["total_usd"] == 0.0065
        assert assistant["citations"][0]["web_search_query"] == "Apaleo competitors moats comparison alternatives rivals"
        assert assistant["citations"][0]["web_search_results"].startswith("Raw search result text")

        cleared = client_b.delete("/api/companies/name:apaleo/chat")
        assert cleared.status_code == 200
        assert shared_sessions["name:apaleo"]["transcript"] == []
        assert shared_sessions["name:apaleo"]["selection"]["provider"] == "gemini"


def test_company_chat_prefers_web_for_competitor_questions(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.company_chat.retrieve_chunks",
        lambda question, store, k=8: [
            Chunk(
                chunk_id="chunk_1",
                text="Apaleo emphasizes API-first architecture and integrations.",
                source_file="deck.pdf",
                page_or_slide="7",
            )
        ],
    )
    monkeypatch.setattr(
        "agent.company_chat.create_llm",
        lambda temperature=0.0: _FakeChatRunnable([
            (
                "The provided evidence does not explicitly name specific competitors of Apaleo. "
                "However, it characterizes the competitive landscape at a high level."
            ),
            "Cloudbeds, Mews, and Oracle OPERA are frequent comparables; web search added named rivals and contrast points.",
        ]),
    )

    captured: dict[str, object] = {}

    def fake_run_web_search(query: str, domain_filter):
        captured["query"] = query
        captured["domain_filter"] = domain_filter
        return (
            "Apaleo competitors include Mews, Cloudbeds, and Oracle OPERA Cloud. "
            "Their positioning differs on legacy footprint, cloud-native architecture, and ecosystem breadth."
        )

    monkeypatch.setattr("agent.company_chat._run_web_search", fake_run_web_search)

    context = {
        "company_lookup_key": "name:apaleo",
        "company_name": "Apaleo",
        "domain": "https://apaleo.com",
        "runs": [
            {
                "job_id": "job-1",
                "company_name": "Apaleo",
                "created_at": "2026-03-16T10:00:00Z",
                "results": {
                    "summary_rows": [{"industry": "Hospitality software"}],
                    "qa_provenance_rows": [],
                    "argument_rows": [],
                },
                "chunks": [],
            }
        ],
    }

    result = asyncio.run(
        answer_company_question(
            context=context,
            transcript=[],
            conversation_summary="",
            question="Who are competitors of Apaleo and what are their moats compare to apaleo?",
            use_web_search=True,
        )
    )

    assert result["used_web_search"] is True
    assert result["web_search_query"] == "Apaleo competitors moats comparison alternatives rivals"
    assert captured["domain_filter"] is None
    assert result["answer"].startswith("Cloudbeds, Mews, and Oracle OPERA")
    assert result["citations"][0]["kind"] == "web"
