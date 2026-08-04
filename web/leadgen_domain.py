"""Canonical company-domain policy shared by LeadGen machine lifecycle seams."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_COMPANY_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)

COMPANY_SOURCE_HOSTS = frozenset(
    {
        "angellist.com",
        "bitbucket.org",
        "bloomberg.com",
        "crunchbase.com",
        "dealroom.co",
        "eu-startups.com",
        "facebook.com",
        "forbes.com",
        "github.com",
        "github.io",
        "gitlab.com",
        "gitlab.io",
        "linkedin.com",
        "medium.com",
        "producthunt.com",
        "substack.com",
        "techcrunch.com",
        "twitter.com",
        "wellfound.com",
        "wikipedia.org",
        "x.com",
    }
)


def normalize_company_domain(raw: str) -> str | None:
    """Return one canonical company hostname from a bare domain or HTTP(S) URL."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value or any(
        char.isspace() or ord(char) < 32 or ord(char) == 127
        for char in value
    ):
        return None
    parsed = urlsplit(value if "://" in value else f"//{value}")
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname if _COMPANY_DOMAIN_RE.fullmatch(hostname) else None


def company_source_host(domain: str | None) -> str | None:
    """Return the blocked shared-source suffix for a normalized domain."""
    text = normalize_company_domain(domain or "")
    if not text:
        return None
    parts = text.split(".")
    for index in range(len(parts)):
        suffix = ".".join(parts[index:])
        if suffix in COMPANY_SOURCE_HOSTS:
            return suffix
    return None


__all__ = [
    "COMPANY_SOURCE_HOSTS",
    "company_source_host",
    "normalize_company_domain",
]
