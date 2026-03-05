from agent.person_intel.dedup import build_provenance_index, deduplicate_facts
from agent.person_intel.models import EvidenceRecord, ExtractedFact


def _ev(url: str, snippet: str) -> EvidenceRecord:
    return EvidenceRecord(
        url=url,
        snippet_or_field=snippet,
        source_type="apify_profile",
        retrieved_at="2026-03-04T00:00:00Z",
    )


def test_deduplicate_facts_merges_evidence() -> None:
    f1 = ExtractedFact(
        text="Led growth team",
        section="strengths",
        evidence=[_ev("https://a", "x")],
        confidence=0.6,
        status="supported",
    )
    f2 = ExtractedFact(
        text="led growth team",
        section="strengths",
        evidence=[_ev("https://b", "y")],
        confidence=0.7,
        status="supported",
    )

    merged = deduplicate_facts([f1, f2])
    assert len(merged) == 1
    assert len(merged[0].evidence) == 2
    assert merged[0].confidence == 0.7


def test_build_provenance_index_deduplicates_by_hash() -> None:
    fact = ExtractedFact(
        text="Fact",
        section="key_points",
        evidence=[_ev("https://a", "x"), _ev("https://a", "x")],
        confidence=0.6,
        status="supported",
    )
    prov = build_provenance_index([fact])
    assert len(prov) == 1
