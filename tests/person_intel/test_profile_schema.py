from pydantic import ValidationError
import pytest

from agent.pipeline.state.schemas import PersonProfileSections


def test_person_profile_sections_validate_strength_prefix_and_counts() -> None:
    with pytest.raises(ValidationError):
        PersonProfileSections(
            interests_lifestyle="One. Two.",
            strengths=["Not prefixed"] * 5,
            more_details="x",
            biggest_achievements=["a", "b", "c"],
            values_beliefs="One. Two.",
            key_points=["a", "b", "c", "d", "e"],
            coolest_fact="One.",
            top_risk="Risk text.",
        )


def test_person_profile_sections_valid_payload() -> None:
    sections = PersonProfileSections(
        interests_lifestyle="Sentence one. Sentence two.",
        strengths=[f"✅ Strength {i}" for i in range(1, 6)],
        more_details="Short paragraph.",
        biggest_achievements=["A", "B", "C"],
        values_beliefs="Sentence one. Sentence two.",
        key_points=["K1", "K2", "K3", "K4", "K5"],
        coolest_fact="Interesting fact.",
        top_risk="Main uncertainty is evidence depth.",
    )
    assert len(sections.strengths) == 5
