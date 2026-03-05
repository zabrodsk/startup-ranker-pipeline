"""Normalization and fact extraction for person intelligence."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlparse, urlunparse

from agent.person_intel.models import EvidenceRecord, ExtractedFact, PersonIntelSubject, PersonProfileJobRequest

_SECTION_ORDER = [
    "interests_lifestyle",
    "strengths",
    "more_details",
    "biggest_achievements",
    "values_beliefs",
    "key_points",
    "coolest_fact",
    "top_risk",
]

_SOURCE_WEIGHT = {
    "apify_profile": 0.85,
    "apify_posts": 0.75,
    "user_text": 0.65,
    "user_image_ocr": 0.5,
    "web_fallback": 0.5,
}

_DEFAMATORY_PATTERNS = [
    r"\bcriminal\b",
    r"\bfraud\b",
    r"\bscam\b",
    r"\billegal\b",
    r"\bcorrupt\b",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_profile_url(url: str) -> str:
    """Canonicalize profile URL and remove tracking params."""
    raw = (url or "").strip()
    if raw.startswith("www.") or raw.startswith("linkedin.com/") or raw.startswith("www.linkedin.com/"):
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if not path:
        path = "/"

    allowed_params = {"id", "profile"}
    query_items = [(k, v) for k, v in parse_qsl(parsed.query) if k in allowed_params]

    return urlunparse((scheme, netloc, path, "", "&".join(f"{k}={v}" for k, v in query_items), ""))


def is_public_profile_url(url: str) -> bool:
    """Reject private/local profiles and non-http(s) URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return False

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        # Not an IP literal.
        pass

    return True


def build_subject(request: PersonProfileJobRequest) -> PersonIntelSubject:
    """Build canonical subject from request."""
    normalized_url = normalize_profile_url(request.primary_profile_url)

    aliases = [a.strip() for a in request.known_aliases if a and a.strip()]
    aliases = list(dict.fromkeys(aliases))

    return PersonIntelSubject(
        primary_profile_url=request.primary_profile_url.strip(),
        normalized_profile_url=normalized_url,
        full_name=(request.full_name or "").strip() or None,
        location=(request.location or "").strip() or None,
        current_company=(request.current_company or "").strip() or None,
        role=(request.role or "").strip() or None,
        known_aliases=aliases,
    )


def _section_from_snippet(snippet: str) -> str:
    lower = snippet.lower()
    if any(k in lower for k in ("accomplishments", "award", "founded", "exit", "raised")):
        return "biggest_achievements"
    if any(k in lower for k in ("skills", "led", "delivered", "managed", "scaled")):
        return "strengths"
    if any(k in lower for k in ("about", "bio", "experience", "education", "currentposition")):
        return "more_details"
    if any(k in lower for k in ("post:", "hobby", "interests", "lifestyle", "outside work")):
        return "interests_lifestyle"
    if any(k in lower for k in ("believe", "mission", "value", "principle")):
        return "values_beliefs"
    if any(k in lower for k in ("risk", "gap", "unknown", "missing")):
        return "top_risk"
    return "key_points"


def _is_noise_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 3:
        return True
    if t.lower().startswith("search results for:"):
        return True
    if re.match(r"^https?://\\S+$", t):
        return True
    if t.startswith("[{") or t.startswith("{\"") or t.startswith("[\""):
        return True
    return False


def _split_field_snippet(snippet: str) -> tuple[str, str]:
    if ":" not in snippet:
        return ("", snippet.strip())
    field, value = snippet.split(":", 1)
    return (field.strip(), value.strip())


def _map_record_to_facts(record: EvidenceRecord) -> list[tuple[str, str]]:
    field, value = _split_field_snippet(record.snippet_or_field)
    field_l = field.lower()
    value_clean = " ".join(value.split())
    if _is_noise_text(value_clean):
        return []

    out: list[tuple[str, str]] = []
    if field_l in {"fullname", "firstname", "lastname", "profilepic", "profilepicture", "profilephoto", "profileimage"}:
        return []
    if field_l == "headline":
        out.append(("strengths", value_clean))
        out.append(("key_points", value_clean))
        return out
    if field_l == "about":
        out.append(("more_details", value_clean))
        return out
    if field_l == "currentposition":
        out.append(("key_points", f"Current role: {value_clean}"))
        return out
    if field_l == "currentcompany":
        out.append(("key_points", f"Current company: {value_clean}"))
        return out
    if field_l == "followers":
        out.append(("key_points", f"LinkedIn followers: {value_clean}"))
        return out
    if field_l == "connections":
        out.append(("key_points", f"LinkedIn connections: {value_clean}"))
        return out
    if field_l == "skills":
        skills = [s.strip() for s in re.split(r",|;", value_clean) if s.strip()]
        for skill in skills[:10]:
            out.append(("strengths", f"Skill: {skill}"))
        return out
    if field_l == "experience":
        parts = [p.strip() for p in value_clean.split(";") if p.strip()]
        for part in parts[:6]:
            out.append(("biggest_achievements", part))
        return out
    if field_l == "education":
        out.append(("more_details", f"Education: {value_clean}"))
        return out
    if field_l == "accomplishments":
        parts = [p.strip() for p in value_clean.split(";") if p.strip()]
        for part in parts[:6]:
            out.append(("biggest_achievements", part))
        return out
    if field_l == "post":
        out.append(("values_beliefs", value_clean))
        out.append(("interests_lifestyle", value_clean))
        return out
    if field_l == "web":
        web_lower = value_clean.lower()
        if any(k in web_lower for k in ("founded", "launched", "raised", "award", "acquired", "grew", "scaled")):
            out.append(("biggest_achievements", value_clean))
        if any(k in web_lower for k in ("experience", "career", "previously", "served", "worked", "education")):
            out.append(("more_details", value_clean))
        if any(k in web_lower for k in ("believe", "mission", "principle", "ethic", "value", "purpose")):
            out.append(("values_beliefs", value_clean))
        if any(k in web_lower for k in ("hobby", "outside work", "sport", "lifestyle", "interests", "fitness")):
            out.append(("interests_lifestyle", value_clean))
        if any(k in web_lower for k in ("risk", "challenge", "uncertain", "transition", "execution", "strain")):
            out.append(("top_risk", value_clean))
        out.append(("key_points", value_clean))
        return out

    # Fallback mapping for unrecognized fields.
    out.append((_section_from_snippet(f"{field}: {value_clean}"), value_clean))
    return out


def _fact_key(text: str, section: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(f"{section}|{normalized}".encode("utf-8")).hexdigest()[:16]


def _fact_key_with_source(text: str, section: str, record: EvidenceRecord) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    source_url = (record.url or "").strip().lower()
    return hashlib.sha1(f"{section}|{normalized}|{source_url}".encode("utf-8")).hexdigest()[:18]


def _compute_confidence(records: list[EvidenceRecord]) -> float:
    if not records:
        return 0.15

    base = 0.0
    for record in records:
        base += _SOURCE_WEIGHT.get(record.source_type, 0.4)

    base = base / len(records)
    agreement_bonus = min(0.15, 0.03 * (len(records) - 1))
    confidence = max(0.05, min(0.98, base + agreement_bonus))
    return round(confidence, 2)


def extract_facts(subject: PersonIntelSubject, evidence: list[EvidenceRecord]) -> list[ExtractedFact]:
    """Extract candidate facts from evidence records with confidence estimates."""
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    text_by_key: dict[str, str] = {}
    section_by_key: dict[str, str] = {}

    for record in evidence:
        pairs = _map_record_to_facts(record)
        for section, raw_text in pairs:
            cleaned = " ".join(raw_text.split()).strip()
            if _is_noise_text(cleaned):
                continue
            if record.source_type == "web_fallback":
                key = _fact_key_with_source(cleaned, section, record)
            else:
                key = _fact_key(cleaned, section)
            grouped[key].append(record)
            text_by_key[key] = cleaned
            section_by_key[key] = section

    facts: list[ExtractedFact] = []
    for key, records in grouped.items():
        facts.append(
            ExtractedFact(
                text=text_by_key[key],
                section=section_by_key[key],
                evidence=records,
                confidence=_compute_confidence(records),
                status="supported",
            )
        )

    if not facts:
        # Ensure strict non-hallucination fallback.
        facts.append(
            ExtractedFact(
                text="Insufficient public evidence to build a reliable profile.",
                section="top_risk",
                evidence=[],
                confidence=0.1,
                status="unknown",
            )
        )

    return facts


def enforce_safe_risk_text(text: str) -> str:
    """Redact defamatory wording and keep risk framing uncertainty-focused."""
    safe_text = text
    for pattern in _DEFAMATORY_PATTERNS:
        safe_text = re.sub(pattern, "[redacted]", safe_text, flags=re.IGNORECASE)

    if "uncert" not in safe_text.lower() and "unknown" not in safe_text.lower():
        safe_text = f"{safe_text.strip()} Main uncertainty: limited public evidence depth."

    return safe_text.strip()


def build_unknowns(facts: list[ExtractedFact]) -> list[str]:
    """Compute missing high-impact fields."""
    lower_text = " \n".join(f.text.lower() for f in facts)
    unknowns: list[str] = []
    checks = [
        ("education", "Education history is incomplete or unavailable."),
        ("experience", "Detailed work timeline is incomplete or unavailable."),
        ("achievement", "Independent evidence for major achievements is limited."),
        ("value", "Values or beliefs are weakly evidenced from public sources."),
    ]
    for token, message in checks:
        if token not in lower_text:
            unknowns.append(message)
    return unknowns


def order_sections() -> list[str]:
    """Expose canonical section ordering."""
    return list(_SECTION_ORDER)
