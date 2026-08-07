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
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from agent.rate_limit import run_with_sync_retries

DEFAULT_RESULT_COUNT = 3
DEFAULT_COUNTRY = "US"
DEFAULT_MAX_TOKENS_PER_PAGE = 200


class WebSearchProvider(ABC):
    """Minimal interface for search providers."""

    @abstractmethod
    def search(self, query: str, *, domain_filter: Optional[List[str]] = None) -> str:
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

    def search(self, query: str, *, domain_filter: Optional[List[str]] = None) -> str:
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
        if not api_key:
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

    def search(self, query: str, *, domain_filter: Optional[List[str]] = None) -> str:
        search_query = self._apply_domain_filter(query, domain_filter)
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

    def search(self, query: str, *, domain_filter: Optional[List[str]] = None) -> str:
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

            lines.append(f"{index}. {title} — {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines).rstrip()


def get_provider(search_end_date: str, *, provider_name: str) -> WebSearchProvider:
    """Factory to instantiate the requested provider."""
    provider = provider_name.lower()
    if provider == "sonar":
        return SonarSearchProvider(search_end_date=search_end_date)
    if provider in {"serper", "hybrid"}:
        return SerperSearchProvider(search_end_date=search_end_date)
    if provider == "brave":
        return BraveSearchProvider(search_end_date=search_end_date)
    raise ValueError(f"Unsupported web search provider '{provider_name}'.")
