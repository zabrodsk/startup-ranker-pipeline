"""Orchestrator service for person profile intelligence pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from agent.person_intel.dedup import build_provenance_index, deduplicate_facts
from agent.person_intel.extract import (
    build_subject,
    build_unknowns,
    extract_facts,
    is_public_profile_url,
    order_sections,
)
from agent.person_intel.models import PersonProfileJobRequest
from agent.person_intel.providers.apify_mcp import ApifyMcpPersonProvider
from agent.person_intel.providers.user_inputs import UserInputsProvider
from agent.person_intel.providers.web_fallback import WebFallbackProvider
from agent.person_intel.render_markdown import render_person_profile_markdown
from agent.person_intel.synthesize import synthesize_sections
from agent.pipeline.state.schemas import (
    PersonClaim,
    PersonClaimEvidence,
    PersonProfileOutput,
    PersonSubject,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersonIntelService:
    """Builds structured profile JSON and markdown from public evidence."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self.apify_provider = ApifyMcpPersonProvider()
        self.user_provider = UserInputsProvider()
        self.web_provider = WebFallbackProvider()

    @staticmethod
    def _extract_field_from_evidence(
        evidence: list[Any],
        field_prefixes: list[str],
    ) -> str | None:
        for ev in evidence:
            snippet = getattr(ev, "snippet_or_field", "") or ""
            lower = snippet.lower()
            for prefix in field_prefixes:
                p = prefix.lower() + ":"
                if lower.startswith(p):
                    value = snippet[len(p):].strip()
                    if value:
                        return value
        return None

    def _cache_key(self, request: PersonProfileJobRequest) -> str:
        apify_actor = getattr(self.apify_provider, "profile_actor_id", "") or ""
        key_raw = "|".join(
            [
                request.primary_profile_url.strip().lower(),
                (request.full_name or "").strip().lower(),
                (request.current_company or "").strip().lower(),
                apify_actor.strip().lower(),
            ]
        )
        return hashlib.sha1(key_raw.encode("utf-8")).hexdigest()[:24]

    async def build_profile(
        self,
        request: PersonProfileJobRequest,
    ) -> tuple[PersonProfileOutput, str, str, dict[str, Any]]:
        """Return profile_json, markdown, cache_key, diagnostics."""
        cache_key = self._cache_key(request)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return (
                cached["profile_json"],
                cached["profile_markdown"],
                cache_key,
                cached.get("diagnostics", {}),
            )

        subject = build_subject(request)
        if not is_public_profile_url(subject.normalized_profile_url):
            raise ValueError("Only public http(s) profile URLs are allowed")

        evidence = []
        evidence.extend(await self.apify_provider.collect(request, subject))
        evidence.extend(await self.user_provider.collect(request, subject))
        evidence.extend(await self.web_provider.collect(request, subject))
        diagnostics: dict[str, Any] = {
            "apify_error": self.apify_provider.get_last_error_message(),
            "apify_attempts": self.apify_provider.get_last_attempts(),
            "apify_actor_used": self.apify_provider.get_actor_used(),
            "apify_actor_blocked": self.apify_provider.get_actor_blocked(),
        }

        resolved_name = (
            subject.full_name
            or self._extract_field_from_evidence(
                evidence,
                ["fullName", "firstName", "name"],
            )
        )
        profile_image_url = self._extract_field_from_evidence(
            evidence,
            ["profilePicture", "profilePic", "profilePhoto", "profileImage", "photo"],
        )

        facts = deduplicate_facts(extract_facts(subject, evidence))
        unknowns = build_unknowns(facts)
        sections = await synthesize_sections(facts, unknowns)

        claims: list[PersonClaim] = []
        for idx, fact in enumerate(facts, start=1):
            claims.append(
                PersonClaim(
                    claim_id=f"claim_{idx}",
                    text=fact.text,
                    section=fact.section,
                    evidence=[
                        PersonClaimEvidence(
                            url=e.url,
                            snippet_or_field=e.snippet_or_field,
                            source_type=e.source_type,
                            retrieved_at=e.retrieved_at,
                        )
                        for e in fact.evidence
                    ],
                    confidence=fact.confidence,
                    timestamp=_utc_now_iso(),
                    status=fact.status if fact.evidence else "unknown",
                )
            )

        # Ensure section-level unknown claims are explicit where facts are absent.
        existing_sections = {claim.section for claim in claims}
        for section in order_sections():
            if section in existing_sections:
                continue
            claims.append(
                PersonClaim(
                    claim_id=f"claim_unknown_{section}",
                    text=f"Unknown: insufficient evidence for section '{section}'.",
                    section=section,
                    evidence=[],
                    confidence=0.1,
                    timestamp=_utc_now_iso(),
                    status="unknown",
                )
            )

        profile_json = PersonProfileOutput(
            subject=PersonSubject(
                primary_profile_url=subject.primary_profile_url,
                normalized_profile_url=subject.normalized_profile_url,
                profile_image_url=profile_image_url,
                full_name=resolved_name,
                location=subject.location,
                current_company=subject.current_company,
                role=subject.role,
                known_aliases=subject.known_aliases,
            ),
            sections=sections,
            claims=claims,
            unknowns=unknowns,
            provenance_index=build_provenance_index(facts),
        )

        profile_markdown = render_person_profile_markdown(profile_json)
        self._cache[cache_key] = {
            "profile_json": profile_json,
            "profile_markdown": profile_markdown,
            "diagnostics": diagnostics,
        }
        return profile_json, profile_markdown, cache_key, diagnostics
