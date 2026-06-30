"""Shared analysis quality guardrails for company identity and run preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

QUALITY_PREFLIGHT_FAILED_CODE = "quality_preflight_failed"
ALLOWED_INPUT_MODES = ("pitchdeck", "specter", "original")

_DOMAIN_ONLY_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/.*)?$",
    re.IGNORECASE,
)
_HEX_PREFIX_RE = re.compile(r"^(?:[a-f0-9]{8,}|[a-f0-9]{6,})[-_\s]+", re.IGNORECASE)
_HEX_ONLY_RE = re.compile(r"^[a-f0-9]{8,}$", re.IGNORECASE)
_UUID_RE = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    re.IGNORECASE,
)
_FILE_EXT_RE = re.compile(r"\.(?:pdf|pptx?|docx?|xlsx?|csv|txt|md)$", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(?:20\d{2}(?:[01]\d(?:[0-3]\d)?)?|\d{8})\b")
_VERSION_RE = re.compile(r"\b(?:v|ver|version)\s*\d+(?:\.\d+)?\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_BOILERPLATE_PATTERNS = (
    r"\binvestor\s+deck\b",
    r"\bpitch\s*deck\b",
    r"\bpitch\s*presentation\b",
    r"\bcorporate\s*profile\b",
    r"\bcompany\s*profile\b",
    r"\bnon\s*confidential\b",
    r"\bconfidential\b",
    r"\bpresentation\b",
    r"\bprofile\b",
    r"\bdeck\b",
    r"\bteaser\b",
    r"\boverview\b",
    r"\bfinal\b",
    r"\bdraft\b",
)
_GENERIC_TOKENS = {
    "company",
    "startup",
    "pitch",
    "deck",
    "presentation",
    "profile",
    "corporate",
    "confidential",
    "non",
    "unknown",
    "upload",
    "document",
    "documents",
    "file",
    "analysis",
    "run",
}
_ACRONYMS = {"ai", "api", "ar", "b2b", "b2c", "hr", "iot", "pc", "saas", "vc", "vr"}
_TRUSTED_DOMAIN_DISPLAY_NAME_SOURCES = {"specter_url", "persistence_specter_url"}


@dataclass(frozen=True)
class NormalizedCompanyName:
    raw: str | None
    value: str | None
    valid: bool
    source: str = "unknown"
    reason: str | None = None
    changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug_token(value: str | None) -> str:
    return "-".join(_TOKEN_RE.findall((value or "").lower()))


def _domain_root(value: str | None) -> str:
    text = (value or "").strip().lower().lstrip("@").strip("'\"")
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    text = text.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return text.rstrip(".,;:)")


def _brand_stem_from_domain(value: str | None) -> str:
    root = _domain_root(value)
    if not root or "." not in root:
        return ""
    parts = root.split(".")
    while parts and parts[0] in {"app", "go", "my", "secure", "web", "portal"}:
        parts.pop(0)
    return parts[0] if parts else ""


def _looks_domain_only(value: str | None) -> bool:
    text = (value or "").strip()
    return bool(text and " " not in text and _DOMAIN_ONLY_RE.match(text))


def _looks_dotted_brand(value: str | None) -> bool:
    text = (value or "").strip()
    parts = text.split(".")
    return (
        len(parts) == 2
        and len(parts[0]) == 1
        and parts[0].isalpha()
        and parts[1].isalpha()
        and text.upper() == text
    )


def _allows_domain_display_name(source: str) -> bool:
    return source in _TRUSTED_DOMAIN_DISPLAY_NAME_SOURCES


def _trusted_domain_display_name(value: str) -> str:
    text = value.strip()
    if text.startswith(("http://", "https://")) or "/" in text:
        return _domain_root(text)
    return text.removeprefix("www.")


def _strip_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = _DATE_RE.sub(" ", cleaned)
    cleaned = _VERSION_RE.sub(" ", cleaned)
    return cleaned


def _format_word(word: str) -> str:
    stripped = word.strip(" .")
    if not stripped:
        return ""
    lower = stripped.lower()
    if "." in stripped and any(char.isalpha() for char in stripped):
        return stripped.upper() if stripped == lower else stripped
    if lower in _ACRONYMS:
        return lower.upper()
    if stripped.isupper() and len(stripped) <= 5:
        return stripped
    if any(char.isupper() for char in stripped[1:]):
        return stripped[:1].upper() + stripped[1:]
    if lower.endswith("ai") and len(lower) > 2:
        return f"{lower[:-2].capitalize()}AI"
    if lower.endswith("vc") and len(lower) > 2:
        return f"{lower[:-2].capitalize()}VC"
    return lower.capitalize()


def _smart_display_case(text: str) -> str:
    words = [_format_word(word) for word in text.split()]
    return " ".join(word for word in words if word)


def _invalid_reason(display: str, *, raw: str | None, source: str, allow_domain_fallback: bool) -> str | None:
    if not display:
        return "Company name is empty after normalization."
    token = _slug_token(display)
    if not token:
        return "Company name has no usable letters."
    if _HEX_ONLY_RE.match(token.replace("-", "")) or _UUID_RE.match((raw or "").strip()):
        return "Company name looks like a generated job/hash id."
    tokens = set(_TOKEN_RE.findall(display.lower()))
    if tokens and tokens <= _GENERIC_TOKENS:
        return "Company name is generic boilerplate."
    if (
        _looks_domain_only(display)
        and not allow_domain_fallback
        and not _looks_dotted_brand(display)
        and not _allows_domain_display_name(source)
    ):
        return "Company name is only a domain."
    if source in {"csv", "specter_csv"} and _looks_domain_only(raw):
        return "Specter CSV company name is only a domain."
    if len(token.replace("-", "")) < 2:
        return "Company name is too short."
    return None


def normalize_company_display_name(
    raw: Any,
    *,
    source: str = "unknown",
    allow_domain_fallback: bool = False,
) -> NormalizedCompanyName:
    """Normalize a company display name or return a concrete rejection reason."""

    raw_text = "" if raw is None else str(raw).strip()
    if not raw_text:
        return NormalizedCompanyName(
            raw=None,
            value=None,
            valid=False,
            source=source,
            reason="Company name is missing.",
        )

    filename_candidate = Path(raw_text.split("?", 1)[0].split("#", 1)[0]).name
    has_file_extension = bool(_FILE_EXT_RE.search(filename_candidate))
    if (
        _looks_domain_only(raw_text)
        and not has_file_extension
        and not _looks_dotted_brand(raw_text)
    ):
        if _allows_domain_display_name(source):
            value = _trusted_domain_display_name(raw_text)
            return NormalizedCompanyName(
                raw=raw_text,
                value=value,
                valid=True,
                source=source,
                changed=value != raw_text,
            )
        if not allow_domain_fallback:
            return NormalizedCompanyName(
                raw=raw_text,
                value=None,
                valid=False,
                source=source,
                reason="Company name is only a domain.",
            )
        text = _brand_stem_from_domain(raw_text)
    else:
        text = filename_candidate

    text = _FILE_EXT_RE.sub("", text)
    text = _HEX_PREFIX_RE.sub("", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[_\-–—/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ._-")
    text = _strip_boilerplate(text)
    text = re.sub(r"[^A-Za-z0-9.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ._-")

    display = _smart_display_case(text)
    reason = _invalid_reason(
        display,
        raw=raw_text,
        source=source,
        allow_domain_fallback=allow_domain_fallback,
    )
    if reason:
        return NormalizedCompanyName(
            raw=raw_text,
            value=None,
            valid=False,
            source=source,
            reason=reason,
        )

    return NormalizedCompanyName(
        raw=raw_text,
        value=display,
        valid=True,
        source=source,
        changed=display != raw_text,
    )


def normalize_company_from_domain(raw: Any, *, source: str = "domain") -> NormalizedCompanyName:
    return normalize_company_display_name(raw, source=source, allow_domain_fallback=True)


def safe_company_slug_for_key(company_name: str | None, startup_slug: str | None) -> str | None:
    """Return slug only when it is already the same identity as the display name."""

    if not startup_slug or not company_name:
        return None
    name_token = _slug_token(company_name)
    slug_token = _slug_token(startup_slug)
    if name_token and slug_token == name_token:
        return startup_slug
    return None


def is_failed_analysis_status(status: Any) -> bool:
    return str(status or "").strip().lower() in {"error", "failed", "timeout", "interrupted"}
