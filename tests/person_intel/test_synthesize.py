import asyncio

from agent.person_intel.models import ExtractedFact
import agent.person_intel.synthesize as synth_mod
from agent.person_intel.synthesize import synthesize_sections


def _sentence_count(text: str) -> int:
    parts = [p.strip() for p in text.replace("\n", " ").split(".")]
    return len([p for p in parts if p])


def test_synthesize_fallback_enforces_sentence_ranges() -> None:
    facts = [
        ExtractedFact(
            text="Completed a personal endurance challenge",
            section="interests_lifestyle",
            evidence=[],
            confidence=0.8,
            status="supported",
        ),
    ]

    sections = asyncio.run(synthesize_sections(facts, unknowns=[]))

    assert 2 <= _sentence_count(sections.interests_lifestyle) <= 5
    assert 2 <= _sentence_count(sections.values_beliefs) <= 5


def test_synthesize_repairs_raw_llm_output(monkeypatch) -> None:
    class FakeStructured:
        async def ainvoke(self, messages):
            return synth_mod._RawPersonProfileSections(
                interests_lifestyle="Only one sentence",
                strengths=["Led teams"],
                more_details="",
                biggest_achievements=["Launched product"],
                values_beliefs="single",
                key_points=["One"],
                coolest_fact="Interesting",
                top_risk="Operational strain due to transition",
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeStructured()

    monkeypatch.setattr(synth_mod, "create_llm", lambda temperature=0.0: FakeLLM())

    facts = [
        ExtractedFact(
            text="Built and scaled an operations team",
            section="strengths",
            evidence=[],
            confidence=0.8,
            status="supported",
        )
    ]
    sections = asyncio.run(synthesize_sections(facts, unknowns=[]))

    assert 2 <= _sentence_count(sections.interests_lifestyle) <= 5
    assert len(sections.strengths) >= 5
    assert len(sections.biggest_achievements) >= 3
    assert 2 <= _sentence_count(sections.values_beliefs) <= 5
