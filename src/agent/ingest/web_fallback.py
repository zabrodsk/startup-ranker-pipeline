"""Strict homepage bootstrap for quota-fenced LeadGen machine analysis.

This module is intentionally narrow: it is not a general replacement for
Specter ingestion. It lets an explicitly enabled LeadGen machine run bootstrap
from the canonical company homepage when Specter's shared daily MCP quota is
exhausted. The normal pipeline must still use web evidence search.
"""

from __future__ import annotations

import ipaddress
import os
import re
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from agent.dataclasses.company import Company
from agent.ingest.specter_ingest import _company_slug
from agent.ingest.store import Chunk, EvidenceStore

_FALLBACK_ENV = "SPECTER_QUOTA_WEB_FALLBACK_ENABLED"
_MAX_RESPONSE_BYTES = 1_500_000
_MAX_EVIDENCE_CHARS = 20_000
_CHUNK_CHARS = 3_500
_GENERIC_BRAND_TOKENS = {
    "ai",
    "company",
    "digital",
    "energy",
    "group",
    "labs",
    "software",
    "systems",
    "technologies",
    "technology",
}


class HomepageFallbackError(RuntimeError):
    """Raised when a canonical homepage cannot provide safe bootstrap evidence."""


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.text: list[str] = []
        self.metadata: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "svg", "noscript", "template"}:
            self._hidden_depth += 1
        if lowered == "meta":
            values = {str(key).lower(): value for key, value in attrs}
            key = str(values.get("name") or values.get("property") or "").lower()
            content = str(values.get("content") or "").strip()
            if key in {"description", "og:description", "og:title", "twitter:title"} and content:
                self.metadata.append(content)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript", "template"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0:
            text = data.strip()
            if text:
                self.text.append(text)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def quota_web_fallback_allowed(
    run_config: dict[str, Any],
    *,
    use_web_search: bool,
    specter_url: str | None,
    expected_name: str | None,
) -> bool:
    """Return whether this run is authorized to use the narrow fallback."""
    del expected_name
    return bool(
        _truthy(os.getenv(_FALLBACK_ENV))
        and run_config.get("source") == "leadgen_machine"
        and use_web_search
        and str(specter_url or "").strip()
    )


def _canonical_host(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise HomepageFallbackError("Company homepage URL is empty.")
    candidate = raw if "://" in raw else f"https://{raw}"
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise HomepageFallbackError("Company homepage must use HTTP or HTTPS.")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host or host == "localhost" or "." not in host:
        raise HomepageFallbackError("Company homepage host is not a public domain.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise HomepageFallbackError("Company homepage host must not be an IP address.")
    return host


def _brand_tokens(expected_name: str, host: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", expected_name.lower()) if len(token) >= 3]
    distinctive = [token for token in tokens if token not in _GENERIC_BRAND_TOKENS]
    domain_stem = host.split(".", 1)[0]
    if len(domain_stem) >= 3:
        distinctive.append(domain_stem)
    return list(dict.fromkeys(distinctive))


def _decode_html(payload: bytes, content_type: str) -> str:
    match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, flags=re.IGNORECASE)
    charset = match.group(1) if match else "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def fetch_company_homepage(
    specter_url: str,
    *,
    expected_name: str | None = None,
    opener: Callable[..., Any] = urlopen,
) -> tuple[Company, EvidenceStore]:
    """Fetch and identity-check one canonical homepage as bootstrap evidence."""
    requested_host = _canonical_host(specter_url)
    request_url = specter_url if "://" in specter_url else f"https://{specter_url}"
    request = Request(
        request_url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "RockawayDealIntelligence/1.0 (+https://rockaway.vc)",
        },
    )
    try:
        response = opener(request, timeout=15)
        final_url = str(response.geturl())
        final_host = _canonical_host(final_url)
        if final_host != requested_host:
            raise HomepageFallbackError("Company homepage redirected to a different registrable host.")
        content_type = str(response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise HomepageFallbackError("Company homepage did not return HTML.")
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    except HomepageFallbackError:
        raise
    except Exception as exc:
        raise HomepageFallbackError(f"Company homepage fetch failed: {type(exc).__name__}.") from exc
    finally:
        close = locals().get("response")
        if close is not None:
            try:
                close.close()
            except Exception:
                pass

    if len(payload) > _MAX_RESPONSE_BYTES:
        raise HomepageFallbackError("Company homepage response exceeded the evidence limit.")
    parser = _VisibleTextParser()
    parser.feed(_decode_html(payload, content_type))
    cleaned = "\n".join(dict.fromkeys([*parser.metadata, *parser.text]))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) < 120:
        raise HomepageFallbackError("Company homepage did not expose enough readable evidence.")

    normalized_name = str(expected_name or "").strip()
    if normalized_name:
        identity_haystack = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())
        brand_tokens = _brand_tokens(normalized_name, requested_host)
        if not brand_tokens or not any(token in identity_haystack for token in brand_tokens):
            raise HomepageFallbackError("Company homepage did not corroborate the expected company identity.")

    evidence_text = cleaned[:_MAX_EVIDENCE_CHARS]
    company_name = normalized_name or requested_host
    slug = _company_slug(company_name) or requested_host.replace(".", "-")
    chunks = [
        Chunk(
            chunk_id=f"homepage-{index + 1}",
            text=part,
            source_file=final_url,
            page_or_slide="homepage",
        )
        for index, start in enumerate(range(0, len(evidence_text), _CHUNK_CHARS))
        if (part := evidence_text[start : start + _CHUNK_CHARS].strip())
    ]
    if not chunks:
        raise HomepageFallbackError("Company homepage produced no evidence chunks.")

    company = Company(
        name=company_name,
        domain=requested_host,
        company_url=final_url,
        about=evidence_text[:1000],
    )
    return company, EvidenceStore(startup_slug=slug, chunks=chunks)


__all__ = [
    "HomepageFallbackError",
    "fetch_company_homepage",
    "quota_web_fallback_allowed",
]
