from agent.batch import (
    build_argument_rows,
    build_qa_provenance_rows,
    build_summary_rows,
)
from agent.dataclasses.argument import Argument
from agent.dataclasses.company import Company
from agent.dataclasses.ranking import CompanyRankingResult, DimensionScore
from agent.ingest.store import Chunk, EvidenceStore


def test_build_qa_provenance_rows_includes_web_search_decision() -> None:
    store = EvidenceStore(
        startup_slug="apify",
        chunks=[
            Chunk(
                chunk_id="chunk_1",
                text="Apify supports many integrations.",
                source_file="leadgen:research_packet",
                page_or_slide="product_software_usp",
                metadata={"evidence_id": "evidence-integrations"},
            ),
            Chunk(
                chunk_id="chunk_3",
                text="Zapier and Make integrations are public.",
                source_file="leadgen:research_packet",
                page_or_slide="product_software_usp",
                metadata={"evidence_id": "evidence-zapier"},
            ),
        ],
    )
    results = [
        {
            "slug": "apify",
            "skipped": False,
            "company": Company(name="Apify"),
            "evidence_store": store,
            "final_state": {
                "all_qa_pairs": [
                    {
                        "question": "What integrations does Apify support?",
                        "answer": "Unknown from provided documents.",
                        "aspect": "general_company",
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
    assert rows[0]["aspect"] == "general_company"
    assert rows[0]["dimension"] == "strategy_fit"
    assert rows[0]["web_search_used"] is True
    assert "relevant" in rows[0]["web_search_decision"]
    assert [item["chunk_id"] for item in rows[0]["chunk_lineage"]] == ["chunk_1", "chunk_3"]


def test_argument_and_summary_rows_include_dimension_metadata() -> None:
    ranking = CompanyRankingResult(
        company_name="Apify",
        slug="apify",
        strategy_fit_score=84.0,
        team_score=71.0,
        upside_score=79.0,
        composite_score=78.0,
        bucket="priority_review",
        dimension_scores=[
            DimensionScore(
                dimension="strategy_fit",
                raw_score=90.0,
                confidence=0.8,
                evidence_count=3,
                evidence_snippets=["Strong ICP match"],
                critical_gaps=["Need deeper pricing proof"],
            ),
            DimensionScore(
                dimension="team",
                raw_score=75.0,
                confidence=0.6,
                evidence_count=2,
                evidence_snippets=["Repeat founder signal"],
                critical_gaps=[],
            ),
        ],
        strategy_fit_summary="Strong fit with the target thesis.",
        team_summary="Founder background is credible.",
        potential_summary="Large upside if expansion continues.",
    )
    argument = Argument(
        content="The company fits the fund thesis and has a credible founder base.",
        argument_type="pro",
        qa_indices=[0, 1],
        qa_pairs=[
            {"question": "Why fit?", "answer": "Clear vertical alignment.", "aspect": "general_company"},
            {"question": "Why team?", "answer": "Founder has repeat experience.", "aspect": "team"},
        ],
        score=9,
        refined_content="The company matches thesis and the founder has repeat execution signals.",
    )
    results = [
        {
            "slug": "apify",
            "skipped": False,
            "company": Company(name="Apify"),
            "final_state": {
                "all_qa_pairs": [],
                "final_arguments": [argument],
                "current_iteration": 2,
                "final_decision": "invest",
                "ranking_result": ranking,
            },
        }
    ]

    argument_rows = build_argument_rows(results)
    summary_rows = build_summary_rows(results)

    assert argument_rows[0]["dimensions"] == ["strategy_fit", "team"]
    assert argument_rows[0]["qa_indices"] == [0, 1]
    assert argument_rows[0]["chunk_ids"] == []
    dimension_scores = summary_rows[0]["dimension_scores"]
    assert dimension_scores[0]["dimension"] == "strategy_fit"
    assert dimension_scores[0]["raw_score"] == 90.0
    assert dimension_scores[0]["adjusted_score"] == 84.6
    assert dimension_scores[0]["confidence"] == 0.8
    assert dimension_scores[0]["evidence_count"] == 3
    assert dimension_scores[0]["evidence_snippets"] == ["Strong ICP match"]
    assert dimension_scores[0]["critical_gaps"] == ["Need deeper pricing proof"]
    assert "sub_scores" in dimension_scores[0]
    assert "adjustment_policy" in dimension_scores[0]

    assert dimension_scores[1]["dimension"] == "team"
    assert dimension_scores[1]["raw_score"] == 75.0
    assert dimension_scores[1]["adjusted_score"] == 66.0
    assert dimension_scores[1]["confidence"] == 0.6
    assert dimension_scores[1]["evidence_count"] == 2
    assert dimension_scores[1]["evidence_snippets"] == ["Repeat founder signal"]
    assert dimension_scores[1]["critical_gaps"] == []
