from agent.dataclasses.company import Company
from agent.ingest.store import Chunk, EvidenceStore
from agent.pipeline.scoring_signals import build_scoring_signals


def test_build_scoring_signals_extracts_specter_clients_investors_and_decay() -> None:
    store = EvidenceStore(
        startup_slug="acme",
        chunks=[
            Chunk(
                chunk_id="chunk_0",
                source_file="specter-mcp",
                page_or_slide="Company Overview",
                text="Company: Acme\nGrowth Stage: Seed\nHighlights: Web Traffic Surge, Top Tier Investors",
            ),
            Chunk(
                chunk_id="chunk_1",
                source_file="specter-mcp",
                page_or_slide="Funding & Investors",
                text="Investors: OpenAI, Inovo.vc, Local Angel",
            ),
            Chunk(
                chunk_id="chunk_2",
                source_file="specter-mcp",
                page_or_slide="Investor Interest & Reported Clients",
                text=(
                    "Investor Interest Signals for Acme:\n"
                    "  - [2026-04-01] score 8/10: fresh investor interest\n"
                    "Reported Clients: BigCo, OtherCo"
                ),
            ),
        ],
    )

    signals = build_scoring_signals(Company(name="Acme"), evidence_store=store)

    assert signals["stage"] == "seed"
    assert signals["reported_clients"] == ["BigCo", "OtherCo"]
    assert signals["investors"] == [
        {"name": "OpenAI", "tier": "tier_1_top_global_or_elite_strategic"},
        {"name": "Inovo.vc", "tier": "tier_2_notable_institutional_or_specialist"},
        {"name": "Local Angel", "tier": "tier_3_regional_angel_accelerator_or_unknown"},
    ]
    assert signals["investor_interest_signals"][0]["recency_weight"] == "full_weight_0_6_months"
    assert signals["specter_highlights"] == [
        {"label": "Web Traffic Surge", "mapped_factor": "demand / market-pull signal"},
        {"label": "Top Tier Investors", "mapped_factor": "investor validation"},
    ]


def test_build_scoring_signals_extracts_young_and_operator_archetype_evidence() -> None:
    company = Company(
        name="BuilderCo",
        about=(
            "Founder has GitHub open source projects, hackathon wins, a Stanford CS background, "
            "a prior failed startup with clear learnings, and was an early employee at a high-growth startup."
        ),
    )

    signals = build_scoring_signals(company)

    assert "GitHub/open-source/project activity" in signals["young_founder_signals"]
    assert "hackathon/competition signal" in signals["young_founder_signals"]
    assert "top-tier school or technical background" in signals["young_founder_signals"]
    assert "serial founder with prior failure/learnings" in signals["founder_archetype_evidence"]
    assert "previous high-growth startup operator" in signals["founder_archetype_evidence"]
