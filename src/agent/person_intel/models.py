"""Core models for the person profile intelligence pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PersonProfileJobRequest(BaseModel):
    """API request payload for person profile jobs."""

    primary_profile_url: str
    full_name: str | None = None
    location: str | None = None
    current_company: str | None = None
    role: str | None = None
    known_aliases: list[str] = Field(default_factory=list)
    user_uploaded_text: str | None = None
    user_uploaded_images: list[str] = Field(default_factory=list)
    company_slug: str | None = None
    person_key: str | None = None

    @field_validator("primary_profile_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("primary_profile_url is required")
        if v.startswith("www.") or v.startswith("linkedin.com/") or v.startswith("www.linkedin.com/"):
            v = f"https://{v}"
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("primary_profile_url must be http(s)")
        return v


class FounderCandidate(BaseModel):
    """Founder candidate used by bulk founder enrichment endpoint."""

    primary_profile_url: str | None = None
    full_name: str | None = None
    location: str | None = None
    current_company: str | None = None
    role: str | None = None
    known_aliases: list[str] = Field(default_factory=list)
    person_key: str | None = None


class BulkFounderJobRequest(BaseModel):
    """Request payload for founder bulk job creation."""

    company_slug: str
    founders: list[FounderCandidate]
    user_uploaded_text: str | None = None
    user_uploaded_images: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    """Normalized evidence unit collected from providers."""

    url: str
    snippet_or_field: str
    source_type: Literal[
        "apify_profile",
        "apify_posts",
        "user_text",
        "user_image_ocr",
        "web_fallback",
    ]
    retrieved_at: str


class ExtractedFact(BaseModel):
    """Candidate fact extracted from normalized evidence."""

    text: str
    section: Literal[
        "interests_lifestyle",
        "strengths",
        "more_details",
        "biggest_achievements",
        "values_beliefs",
        "key_points",
        "coolest_fact",
        "top_risk",
    ]
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    status: Literal["supported", "unknown", "conflicted"]


class PersonIntelSubject(BaseModel):
    """Normalized subject identity used internally."""

    primary_profile_url: str
    normalized_profile_url: str
    full_name: str | None = None
    location: str | None = None
    current_company: str | None = None
    role: str | None = None
    known_aliases: list[str] = Field(default_factory=list)
