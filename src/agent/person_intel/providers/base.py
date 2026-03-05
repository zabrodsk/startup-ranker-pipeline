"""Base provider interface for person intelligence evidence collection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.person_intel.models import EvidenceRecord, PersonProfileJobRequest, PersonIntelSubject


class PersonSourceProvider(ABC):
    """Evidence source provider contract."""

    @abstractmethod
    async def collect(
        self,
        request: PersonProfileJobRequest,
        subject: PersonIntelSubject,
    ) -> list[EvidenceRecord]:
        """Collect normalized evidence records from this provider."""
        raise NotImplementedError
