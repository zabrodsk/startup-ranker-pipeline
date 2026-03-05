"""Dedup helpers for facts and provenance records."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict

from agent.person_intel.models import EvidenceRecord, ExtractedFact
from agent.pipeline.state.schemas import PersonProvenanceRecord


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def deduplicate_facts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Merge near-identical facts and union their evidence arrays."""
    merged: OrderedDict[str, ExtractedFact] = OrderedDict()

    for fact in facts:
        key = f"{fact.section}:{_normalize_text(fact.text)}"
        if key not in merged:
            merged[key] = fact
            continue

        existing = merged[key]
        seen_evidence = {
            (e.url, e.snippet_or_field, e.source_type)
            for e in existing.evidence
        }
        for evidence in fact.evidence:
            evidence_key = (evidence.url, evidence.snippet_or_field, evidence.source_type)
            if evidence_key not in seen_evidence:
                existing.evidence.append(evidence)
                seen_evidence.add(evidence_key)

        existing.confidence = max(existing.confidence, fact.confidence)
        if existing.status != "supported" and fact.status == "supported":
            existing.status = "supported"

    return list(merged.values())


def build_provenance_index(facts: list[ExtractedFact]) -> list[PersonProvenanceRecord]:
    """Build deduplicated provenance table keyed by stable hash."""
    deduped: OrderedDict[str, PersonProvenanceRecord] = OrderedDict()

    for fact in facts:
        for evidence in fact.evidence:
            key_material = f"{evidence.url}|{evidence.snippet_or_field}|{evidence.source_type}"
            key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:20]
            if key in deduped:
                continue
            deduped[key] = PersonProvenanceRecord(
                key=key,
                url=evidence.url,
                snippet_or_field=evidence.snippet_or_field,
                source_type=evidence.source_type,
                retrieved_at=evidence.retrieved_at,
            )

    return list(deduped.values())


def facts_missing_evidence(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Return facts with missing evidence to support unknown markers/tests."""
    return [fact for fact in facts if not fact.evidence]
