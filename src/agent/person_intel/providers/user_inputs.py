"""Provider for user-supplied text and images."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.person_intel.models import EvidenceRecord, PersonIntelSubject, PersonProfileJobRequest
from agent.person_intel.providers.base import PersonSourceProvider


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserInputsProvider(PersonSourceProvider):
    """Converts user-provided text/images into normalized evidence records."""

    async def collect(
        self,
        request: PersonProfileJobRequest,
        subject: PersonIntelSubject,
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        url = subject.normalized_profile_url

        if request.user_uploaded_text and request.user_uploaded_text.strip():
            records.append(
                EvidenceRecord(
                    url=url,
                    snippet_or_field=f"user_text: {request.user_uploaded_text.strip()[:1200]}",
                    source_type="user_text",
                    retrieved_at=_utc_now_iso(),
                )
            )

        for image_ref in request.user_uploaded_images:
            if not image_ref:
                continue
            trimmed = image_ref.strip()
            if not trimmed:
                continue
            # OCR is intentionally conservative here: preserve provenance and avoid guessing.
            records.append(
                EvidenceRecord(
                    url=url,
                    snippet_or_field=f"user_image_ref: {trimmed[:400]}",
                    source_type="user_image_ocr",
                    retrieved_at=_utc_now_iso(),
                )
            )

        return records
