from agent.person_intel.extract import (
    build_unknowns,
    enforce_safe_risk_text,
    extract_facts,
    is_public_profile_url,
    normalize_profile_url,
)
from agent.person_intel.models import EvidenceRecord, PersonIntelSubject


def test_normalize_profile_url_strips_tracking() -> None:
    normalized = normalize_profile_url("https://www.linkedin.com/in/foo/?utm_source=x&utm_campaign=y")
    assert "utm_" not in normalized


def test_is_public_profile_url_blocks_private_hosts() -> None:
    assert not is_public_profile_url("http://localhost/profile")
    assert is_public_profile_url("https://www.linkedin.com/in/example")


def test_extract_facts_unknown_when_no_evidence() -> None:
    subject = PersonIntelSubject(
        primary_profile_url="https://www.linkedin.com/in/example",
        normalized_profile_url="https://www.linkedin.com/in/example",
    )
    facts = extract_facts(subject, [])
    assert len(facts) == 1
    assert facts[0].status == "unknown"


def test_build_unknowns_marks_missing_high_impact_fields() -> None:
    facts = extract_facts(
        PersonIntelSubject(
            primary_profile_url="https://www.linkedin.com/in/example",
            normalized_profile_url="https://www.linkedin.com/in/example",
        ),
        [
            EvidenceRecord(
                url="https://www.linkedin.com/in/example",
                snippet_or_field="skills: Built GTM team",
                source_type="apify_profile",
                retrieved_at="2026-03-04T00:00:00Z",
            )
        ],
    )
    unknowns = build_unknowns(facts)
    assert unknowns


def test_risk_text_validator_redacts_defamatory_terms() -> None:
    text = enforce_safe_risk_text("Possible fraud risk due to unknowns")
    assert "fraud" not in text.lower()


def test_web_snippet_maps_to_multiple_sections() -> None:
    subject = PersonIntelSubject(
        primary_profile_url="https://www.linkedin.com/in/example",
        normalized_profile_url="https://www.linkedin.com/in/example",
    )
    facts = extract_facts(
        subject,
        [
            EvidenceRecord(
                url="https://example.com/article",
                snippet_or_field=(
                    "web: She previously served as COO, scaled revenue, and discussed mission-driven leadership "
                    "while acknowledging execution challenges during transition."
                ),
                source_type="web_fallback",
                retrieved_at="2026-03-04T00:00:00Z",
            )
        ],
    )
    sections = {f.section for f in facts}
    assert "key_points" in sections
    assert "more_details" in sections
    assert "biggest_achievements" in sections
    assert "values_beliefs" in sections
    assert "top_risk" in sections
