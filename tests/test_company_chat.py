import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import web.app as web_app


def _login(client: TestClient) -> None:
    response = client.post("/api/login", json={"password": "9876"})
    assert response.status_code == 200
    client.cookies.set("session_id", response.json()["session_id"])


def test_company_chat_endpoints_roundtrip(monkeypatch) -> None:
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
            "model_label": "Gemini 3.1 Flash Lite",
            "source_counts": {"chunks": 1, "qa": 1, "arguments": 1},
            "run_count": 1,
        }

    monkeypatch.setattr(web_app, "answer_company_question", fake_answer_company_question)

    with TestClient(web_app.app) as client:
        _login(client)

        initial = client.get("/api/companies/name:apify/chat")
        assert initial.status_code == 200
        assert initial.json()["transcript"] == []

        posted = client.post(
            "/api/companies/name:apify/chat",
            json={"message": "What is the strongest evidence?", "active_job_id": "job-1"},
        )
        assert posted.status_code == 200
        payload = posted.json()
        assert payload["answer"] == "The strongest support is enterprise automation growth."
        assert payload["used_web_search"] is True
        assert payload["web_search_query"] == "Apify enterprise automation growth"
        assert len(payload["transcript"]) == 2
        assert payload["transcript"][1]["citations"][0]["kind"] == "chunk"

        fetched = client.get("/api/companies/name:apify/chat")
        assert fetched.status_code == 200
        assert len(fetched.json()["transcript"]) == 2

        cleared = client.delete("/api/companies/name:apify/chat")
        assert cleared.status_code == 200

        after_clear = client.get("/api/companies/name:apify/chat")
        assert after_clear.status_code == 200
        assert after_clear.json()["transcript"] == []
