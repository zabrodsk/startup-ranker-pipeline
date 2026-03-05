"""Web search enrichment provider for person intelligence."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from agent.person_intel.models import EvidenceRecord, PersonIntelSubject, PersonProfileJobRequest
from agent.person_intel.providers.base import PersonSourceProvider
from agent.web_search import get_provider


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _extract_url(line: str) -> str | None:
    match = re.search(r"https?://\S+", line)
    return match.group(0).rstrip(",)") if match else None


def _is_low_quality_line(line: str) -> bool:
    l = (line or "").strip()
    if not l:
        return True
    lower = l.lower()
    if lower.startswith("search results for:"):
        return True
    if "no search results returned" in lower:
        return True
    if re.match(r"^\d+\.\s*", l):
        return True
    if re.match(r"^\d+$", l):
        return True
    if len(l) < 40:
        return True
    if l.count("|") > 5:
        return True
    if lower.startswith("http://") or lower.startswith("https://"):
        return True
    return False


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


class WebFallbackProvider(PersonSourceProvider):
    """Adds web evidence that complements profile evidence."""

    def __init__(self) -> None:
        self.enabled = os.getenv("PERSON_INTEL_WEB_ENRICHMENT", "true").lower() != "false"
        self.max_records = int(os.getenv("PERSON_INTEL_WEB_MAX_RECORDS", "36"))
        self.max_per_query = int(os.getenv("PERSON_INTEL_WEB_MAX_PER_QUERY", "8"))

    async def collect(
        self,
        request: PersonProfileJobRequest,
        subject: PersonIntelSubject,
    ) -> list[EvidenceRecord]:
        if not self.enabled:
            return []

        provider_name = os.getenv("WEB_SEARCH_PROVIDER", "sonar")
        pplx_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not pplx_key and not brave_key:
            return []

        try:
            provider = get_provider(search_end_date=_today(), provider_name=provider_name)
        except Exception:
            return []

        terms = [subject.full_name or "", subject.current_company or "", subject.role or ""]
        query_core = " ".join([t.strip() for t in terms if t and t.strip()])
        if not query_core:
            query_core = subject.normalized_profile_url

        identity_queries = [
            f"{query_core} biography profile timeline leadership background",
            f"{query_core} career history education role transition",
        ]
        achievements_queries = [
            f"{query_core} achievements milestones launches awards growth",
            f"{query_core} interview keynote publication report speaking",
        ]
        risk_queries = [
            f"{query_core} scale transition execution challenges",
            f"{query_core} organizational growth operations risk context",
        ]

        preferred_domains = [
            "linkedin.com",
            "crunchbase.com",
            "forbes.com",
            "techcrunch.com",
            "bloomberg.com",
            "reuters.com",
        ]
        if subject.current_company and subject.current_company.strip():
            company_token = re.sub(r"[^a-z0-9.-]", "", subject.current_company.lower())
            if company_token:
                preferred_domains.append(company_token)

        query_plan: list[tuple[str, list[str] | None]] = []
        for q in identity_queries + achievements_queries + risk_queries:
            # pass 1: focused domains
            query_plan.append((q, preferred_domains))
            # pass 2: broad web if focused did not provide enough quality snippets
            query_plan.append((q, None))

        records: list[EvidenceRecord] = []
        seen_keys: set[str] = set()
        for query, domain_filter in query_plan:
            try:
                raw = provider.search(query, domain_filter=domain_filter)
            except Exception:
                continue

            if not raw or len(raw.strip()) < 40:
                continue

            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            kept = 0
            for line in lines:
                if _is_low_quality_line(line):
                    continue
                url = _extract_url(line) or subject.normalized_profile_url
                domain = _domain_of(url)
                clean = " ".join(line.split())
                dedupe_key = f"{domain}|{clean.lower()}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                if domain and any(noisy in domain for noisy in ("google.", "bing.", "duckduckgo.")):
                    continue
                records.append(
                    EvidenceRecord(
                        url=url,
                        snippet_or_field=f"web: {clean[:500]}",
                        source_type="web_fallback",
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                kept += 1
                if kept >= self.max_per_query or len(records) >= self.max_records:
                    break
            if len(records) >= self.max_records:
                break

        return records
