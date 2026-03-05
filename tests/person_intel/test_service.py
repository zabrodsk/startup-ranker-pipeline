import asyncio

from agent.person_intel.models import EvidenceRecord
from agent.person_intel.models import PersonProfileJobRequest
from agent.person_intel.service import PersonIntelService


def test_service_build_profile_has_claim_evidence_fields() -> None:
    service = PersonIntelService()

    async def fake_user_collect(request, subject):
        return [
            EvidenceRecord(
                url=subject.normalized_profile_url,
                snippet_or_field="skills: Led product and growth",
                source_type="user_text",
                retrieved_at="2026-03-04T00:00:00Z",
            )
        ]

    async def fake_web_collect(request, subject):
        return []

    service.user_provider.collect = fake_user_collect  # type: ignore[method-assign]
    service.web_provider.collect = fake_web_collect  # type: ignore[method-assign]

    profile_json, profile_md, _, diagnostics = asyncio.run(
        service.build_profile(
            PersonProfileJobRequest(primary_profile_url="https://www.linkedin.com/in/example")
        )
    )

    assert profile_json.claims
    assert all(claim.timestamp for claim in profile_json.claims)
    assert "## INTERESTS & LIFESTYLE" in profile_md
    assert diagnostics.get("apify_actor_used") == "harvestapi/linkedin-profile-scraper"
    assert diagnostics.get("apify_actor_blocked") is False


def test_service_marks_unknown_with_missing_evidence() -> None:
    service = PersonIntelService()

    async def fake_user_collect(request, subject):
        return []

    async def fake_web_collect(request, subject):
        return []

    service.user_provider.collect = fake_user_collect  # type: ignore[method-assign]
    service.web_provider.collect = fake_web_collect  # type: ignore[method-assign]

    profile_json, _, _, _ = asyncio.run(
        service.build_profile(
            PersonProfileJobRequest(primary_profile_url="https://www.linkedin.com/in/example")
        )
    )

    assert any(c.status == "unknown" for c in profile_json.claims)
    assert profile_json.unknowns
