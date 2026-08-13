"""
Web search provider abstractions supporting Serper, Brave, and Perplexity Sonar.

Both providers expose the same string-based output contract that the rest of the
agent relies on (`Search Results for: <query> ...`). This keeps downstream
processing unchanged while allowing us to swap implementations via an
environment toggle.
"""

from __future__ import annotations

import importlib
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from agent.rate_limit import run_with_sync_retries

DEFAULT_RESULT_COUNT = 3
DEFAULT_COUNTRY = "US"
DEFAULT_MAX_TOKENS_PER_PAGE = 200
_DOCUMENTED_PLACEHOLDER_KEYS = {
    "your_serper_api_key_here",
    "your_perplexity_api_key_here",
}


def _has_live_api_key(value: str | None) -> bool:
    """Return whether a configured key is non-empty and not a documented placeholder."""
    normalized = (value or "").strip()
    return bool(normalized and normalized.lower() not in _DOCUMENTED_PLACEHOLDER_KEYS)


class WebSearchProvider(ABC):
    """Minimal interface for search providers."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        domain_filter: Optional[List[str]] = None,
        deadline_seconds: float | None = None,
    ) -> str:
        """Execute a query and return a formatted string.

        domain_filter: Optional list of domains to limit results (Perplexity only).
        """


class BraveSearchProvider(WebSearchProvider):
    """Wrapper around LangChain's BraveSearch tool."""

    def __init__(self, search_end_date: str, *, country: str = DEFAULT_COUNTRY):
        try:
            BraveSearch = importlib.import_module(
                "langchain_community.tools.brave_search.tool"
            ).BraveSearch
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "langchain_community.tools.brave_search.tool.BraveSearch is required for BraveSearchProvider."
            ) from exc

        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY environment variable is required")

        freshness = self._convert_date_to_freshness(search_end_date)
        self._brave_search = BraveSearch.from_api_key(
            api_key=api_key,
            search_kwargs={
                "count": DEFAULT_RESULT_COUNT,
                "country": country,
                "search_lang": "en",
                "safesearch": "moderate",
                "freshness": freshness,
                "result_filter": "web,news",
            },
        )

    def search(
        self,
        query: str,
        *,
        domain_filter: Optional[List[str]] = None,
        deadline_seconds: float | None = None,
    ) -> str:
        return self._brave_search.run(query)

    @staticmethod
    def _convert_date_to_freshness(search_end_date: str) -> str:
        if "T" in search_end_date:
            end_date = datetime.fromisoformat(search_end_date.replace("Z", "+00:00"))
        else:
            end_date = datetime.strptime(search_end_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=365)
        return f"{start_date.strftime('%Y-%m-%d')}to{end_date.strftime('%Y-%m-%d')}"


class SerperSearchProvider(WebSearchProvider):
    """Client for Serper's Google Search REST API."""

    BASE_URL = "https://google.serper.dev/search"

    def __init__(
        self,
        search_end_date: str,
        *,
        country: str = DEFAULT_COUNTRY,
        max_results: int = 5,
    ):
        api_key = os.getenv("SERPER_API_KEY")
        if not _has_live_api_key(api_key):
            raise ValueError("Serper requires SERPER_API_KEY environment variable.")

        self._api_key = api_key
        self._country = (country or DEFAULT_COUNTRY).lower()
        self._max_results = max(1, min(max_results, 10))
        self._search_end_date = search_end_date
        try:
            self._requests = importlib.import_module("requests")
        except ImportError as exc:
            raise ImportError(
                "The 'requests' package is required for SerperSearchProvider."
            ) from exc

    def search(
        self,
        query: str,
        *,
        domain_filter: Optional[List[str]] = None,
        deadline_seconds: float | None = None,
    ) -> str:
        search_query = self._apply_domain_filter(query, domain_filter)
        search_query = self._apply_date_filter(search_query, self._search_end_date)
        payload = {
            "q": search_query,
            "gl": self._country,
            "hl": "en",
            "num": self._max_results,
        }
        response = run_with_sync_retries(
            self._requests.post,
            self.BASE_URL,
            headers={
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
            max_elapsed_seconds=deadline_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected Serper response format: root is not an object")
        return self._format_results(query, data)

    @staticmethod
    def _apply_domain_filter(query: str, domain_filter: Optional[List[str]]) -> str:
        domains = [
            str(domain).strip().lower()
            for domain in (domain_filter or [])[:10]
            if str(domain).strip()
        ]
        if not domains:
            return query
        clauses = " OR ".join(f"site:{domain}" for domain in domains)
        return f"{query} ({clauses})"

    @staticmethod
    def _apply_date_filter(query: str, search_end_date: str) -> str:
        """Constrain Google results to the year ending on the requested date."""
        if not search_end_date:
            return query
        if "T" in search_end_date:
            end = datetime.fromisoformat(search_end_date.replace("Z", "+00:00"))
        else:
            end = datetime.strptime(search_end_date, "%Y-%m-%d")
        start = end - timedelta(days=365)
        # Google's before operator is exclusive, so use the next day to make
        # the analysis cutoff inclusive.
        before = end + timedelta(days=1)
        return f"{query} after:{start:%Y-%m-%d} before:{before:%Y-%m-%d}"

    @staticmethod
    def _format_results(query: str, data: dict) -> str:
        lines: List[str] = [f"Search Results for: {query}", ""]

        answer_box = data.get("answerBox")
        if isinstance(answer_box, dict):
            title = answer_box.get("title") or answer_box.get("answer") or "Answer"
            answer = answer_box.get("answer") or answer_box.get("snippet") or ""
            link = answer_box.get("link") or ""
            lines.append(f"Answer box: {title}" + (f" — {link}" if link else ""))
            if answer and answer != title:
                lines.append(f"   {answer}")
            lines.append("")

        results = data.get("organic", [])
        if not isinstance(results, list):
            raise ValueError("Unexpected Serper response format: 'organic' is not a list")
        if not results and not isinstance(answer_box, dict):
            lines.append("No search results returned.")
            return "\n".join(lines)

        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "No title"
            url = item.get("link") or "No URL provided"
            snippet = item.get("snippet") or ""
            date = item.get("date")
            lines.append(f"{index}. {title} — {url}")
            if date:
                lines.append(f"   Date: {date}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines).rstrip()



class SonarSearchProvider(WebSearchProvider):
    """Client for Perplexity Sonar Search API."""

    BASE_URL = "https://api.perplexity.ai/search"

    def __init__(
        self,
        search_end_date: str,
        *,
        country: str = DEFAULT_COUNTRY,
        max_results: int = DEFAULT_RESULT_COUNT,
        max_tokens_per_page: int = DEFAULT_MAX_TOKENS_PER_PAGE,
    ):
        api_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError(
                "Perplexity Sonar requires PPLX_API_KEY (or PERPLEXITY_API_KEY) environment variable."
            )

        self._api_key = api_key
        self._country = country
        self._max_results = max(1, min(max_results, 20))
        self._max_tokens_per_page = max_tokens_per_page
        self._search_after, self._search_before = self._derive_date_filters(search_end_date)
        try:
            self._requests = importlib.import_module("requests")
        except ImportError as exc:
            raise ImportError(
                "The 'requests' package is required for SonarSearchProvider."
            ) from exc

    def search(
        self,
        query: str,
        *,
        domain_filter: Optional[List[str]] = None,
        deadline_seconds: float | None = None,
    ) -> str:
        payload = {
            "query": query,
            "max_results": self._max_results,
            "country": self._country,
            "max_tokens_per_page": self._max_tokens_per_page,
        }

        if self._search_after:
            payload["search_after_date_filter"] = self._search_after
        if self._search_before:
            payload["search_before_date_filter"] = self._search_before
        if domain_filter:
            payload["search_domain_filter"] = domain_filter[:20]

        response = run_with_sync_retries(
            self._requests.post,
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
            max_elapsed_seconds=deadline_seconds,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Unexpected Sonar response format: 'results' is not a list")

        return self._format_results(query, results)

    @staticmethod
    def _derive_date_filters(search_end_date: str) -> tuple[Optional[str], Optional[str]]:
        """Return (after, before) tuple formatted as MM/DD/YYYY."""
        if not search_end_date:
            return None, None

        if "T" in search_end_date:
            end = datetime.fromisoformat(search_end_date.replace("Z", "+00:00"))
        else:
            end = datetime.strptime(search_end_date, "%Y-%m-%d")

        start = end - timedelta(days=365)
        return start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y")

    @staticmethod
    def _format_results(query: str, results: List[dict]) -> str:
        lines: List[str] = [f"Search Results for: {query}", ""]

        if not results:
            lines.append("No search results returned.")
            return "\n".join(lines)

        for index, item in enumerate(results, start=1):
            title = item.get("title") or "No title"
            url = item.get("url") or "No URL provided"
            snippet = item.get("snippet") or ""
            date = item.get("date")

            lines.append(f"{index}. {title} — {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines).rstrip()


class HybridSearchProvider(WebSearchProvider):
    """Serper-first search with automatic fallback to another configured provider."""

    def __init__(self, search_end_date: str):
        """Build available providers without letting one broken setup disable others."""
        candidates: list[tuple[str, WebSearchProvider]] = []
        initialization_failures: list[str] = []
        if _has_live_api_key(os.getenv("SERPER_API_KEY")):
            try:
                candidates.append(
                    ("serper", SerperSearchProvider(search_end_date=search_end_date))
                )
            except Exception as exc:
                initialization_failures.append(f"SerperSearchProvider: {exc}")
        pplx_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
        if _has_live_api_key(pplx_key):
            try:
                candidates.append(
                    ("sonar", SonarSearchProvider(search_end_date=search_end_date))
                )
            except Exception as exc:
                initialization_failures.append(f"SonarSearchProvider: {exc}")
        if os.getenv("BRAVE_SEARCH_API_KEY"):
            try:
                candidates.append(
                    ("brave", BraveSearchProvider(search_end_date=search_end_date))
                )
            except Exception as exc:
                initialization_failures.append(f"BraveSearchProvider: {exc}")
        if not candidates:
            detail = "; ".join(initialization_failures)
            raise ValueError(
                "Hybrid web search requires at least one working configured provider."
                + (f" Initialization failures: {detail}" if detail else "")
            )
        self._providers = candidates
        self._initialization_failures = initialization_failures
        self.last_provider_name: str | None = None
        self.attempted_provider_names: list[str] = []

    @staticmethod
    def _is_usable(query: str, result: str) -> bool:
        """Require substantive evidence that also overlaps the requested topic."""
        text = (result or "").strip()
        normalized = text.lower()
        if (
            not normalized
            or "no search results returned" in normalized
            or normalized.startswith(("web search failed", "search failed"))
        ):
            return False
        result_count = len(re.findall(r"(?m)^\d+\.\s+", text))
        substantive_chars = len(re.sub(r"\s+", " ", text))
        substantive = (result_count >= 2 and substantive_chars >= 160) or (
            "answer box:" in normalized and substantive_chars >= 120
        )
        if not substantive:
            return False

        # Formatted providers echo the query in their first line. Exclude that
        # line so it cannot make unrelated results appear relevant.
        result_body = re.sub(
            r"(?im)^search results for:\s*[^\n]*(?:\n|$)", "", text, count=1
        ).lower()
        stop_words = {
            "after",
            "before",
            "company",
            "from",
            "into",
            "market",
            "current",
            "evidence",
            "latest",
            "overview",
            "recent",
            "search",
            "site",
            "software",
            "that",
            "their",
            "this",
            "what",
            "when",
            "where",
            "which",
            "with",
        }
        ordered_query_terms = [
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if (
                (len(token) >= 3 or token in {"ai", "eu", "ml", "uk", "us"})
                and token not in stop_words
                and not token.isdigit()
            )
        ]
        if not ordered_query_terms:
            return True
        result_terms = set(re.findall(r"[a-z0-9]+", result_body))
        overlaps = {term for term in ordered_query_terms if term in result_terms}
        # The first meaningful query term is normally the company, person, or
        # industry anchor. Generic terms such as "funding" cannot substitute
        # for it; multi-term queries also need one corroborating overlap.
        required_overlap = min(2, len(set(ordered_query_terms)))
        return (
            ordered_query_terms[0] in result_terms
            and len(overlaps) >= required_overlap
        )

    def search(
        self,
        query: str,
        *,
        domain_filter: Optional[List[str]] = None,
        max_provider_attempts: int | None = None,
        deadline_seconds: float | None = None,
    ) -> str:
        """Return the first usable result, trying configured fallbacks in order."""
        last_result = ""
        failures = list(self._initialization_failures)
        self.last_provider_name = None
        self.attempted_provider_names = []
        providers = self._providers
        if max_provider_attempts is not None:
            providers = providers[: max(1, max_provider_attempts)]
        deadline = (
            time.monotonic() + max(0.001, deadline_seconds)
            if deadline_seconds is not None
            else None
        )
        for provider_name, provider in providers:
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                break
            self.attempted_provider_names.append(provider_name)
            try:
                last_result = provider.search(
                    query,
                    domain_filter=domain_filter,
                    deadline_seconds=remaining,
                )
                self.last_provider_name = provider_name
                if self._is_usable(query, last_result):
                    return last_result
            except Exception as exc:
                failures.append(f"{type(provider).__name__}: {exc}")
        if last_result:
            return last_result
        raise RuntimeError(
            "All hybrid web search providers failed: " + "; ".join(failures)
        )


def resolve_provider_name(provider_name: str | None = None) -> str | None:
    """Resolve configured search strategy against the API keys actually available."""
    configured = (provider_name or os.getenv("WEB_SEARCH_PROVIDER", "sonar")).strip().lower()
    pplx_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
    has_pplx = _has_live_api_key(pplx_key)
    has_serper = _has_live_api_key(os.getenv("SERPER_API_KEY"))
    has_brave = bool(os.getenv("BRAVE_SEARCH_API_KEY"))

    if configured == "hybrid":
        if has_serper:
            return "hybrid" if has_pplx else "serper"
        if has_pplx:
            return "sonar"
        return "brave" if has_brave else None
    if configured == "serper" and not has_serper:
        return "sonar" if has_pplx else ("brave" if has_brave else None)
    if configured == "sonar" and not has_pplx:
        return "serper" if has_serper else ("brave" if has_brave else None)
    if configured == "brave" and not has_brave:
        return "sonar" if has_pplx else ("serper" if has_serper else None)
    return configured if configured in {"sonar", "serper", "brave"} else None


def get_provider(search_end_date: str, *, provider_name: str) -> WebSearchProvider:
    """Factory to instantiate the requested provider."""
    provider = resolve_provider_name(provider_name)
    if provider is None:
        raise ValueError("No configured web search provider is available.")
    if provider == "sonar":
        return SonarSearchProvider(search_end_date=search_end_date)
    if provider == "serper":
        return SerperSearchProvider(search_end_date=search_end_date)
    if provider == "hybrid":
        return HybridSearchProvider(search_end_date=search_end_date)
    if provider == "brave":
        return BraveSearchProvider(search_end_date=search_end_date)
    raise ValueError(f"Unsupported web search provider '{provider_name}'.")
