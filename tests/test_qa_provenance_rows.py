from agent.batch import build_qa_provenance_rows
from agent.dataclasses.company import Company


def test_build_qa_provenance_rows_includes_web_search_decision() -> None:
    results = [
        {
            "slug": "apify",
            "skipped": False,
            "company": Company(name="Apify"),
            "final_state": {
                "all_qa_pairs": [
                    {
                        "question": "What integrations does Apify support?",
                        "answer": "Unknown from provided documents.",
                        "chunk_ids": ["chunk_1", "chunk_3"],
                        "chunks_preview": "[chunk_1]: ...",
                        "web_search_query": "\"Apify\" integrations",
                        "web_search_results": "Apify integrates with Zapier and Make.",
                        "web_search_used": True,
                        "web_search_decision": "used: web results relevant to company/question",
                    }
                ]
            },
        }
    ]

    rows = build_qa_provenance_rows(results)
    assert len(rows) == 1
    assert rows[0]["web_search_used"] is True
    assert "relevant" in rows[0]["web_search_decision"]
