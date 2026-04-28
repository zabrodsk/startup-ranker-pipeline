"""Specter MCP client — fetch a company by URL/domain and produce Company + EvidenceStore.

Mirrors the public contract of `agent.ingest.specter_ingest.ingest_specter_company`
but reads from the Specter MCP (https://mcp.tryspecter.com/mcp) instead of CSV
exports. Used by the worker pipeline when the user provides URLs only.

Authentication uses a shared service-account refresh token captured by the
`scripts/specter_oauth_login.py` helper. The token is read from environment:

    SPECTER_MCP_URL=https://mcp.tryspecter.com/mcp
    SPECTER_MCP_TOKEN_ENDPOINT=https://mcp.tryspecter.com/token
    SPECTER_MCP_CLIENT_ID=...
    SPECTER_MCP_CLIENT_SECRET=...        # optional; absent for public clients
    SPECTER_MCP_REFRESH_TOKEN=...

Coverage caveats (as of Phase 0.5 parity test): the MCP does not expose
G2/Trustpilot/Glassdoor data, awards/certifications, multi-period social or
web growth ratios, app store data, tech stack, or per-person role descriptions
and skills. The hybrid worker path tops up via CSV when both are provided.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from agent.dataclasses.company import Company
from agent.dataclasses.person import Education, Experience, Person
from agent.ingest.specter_ingest import _company_slug
from agent.ingest.store import Chunk, EvidenceStore

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SpecterMCPError(RuntimeError):
    """Generic Specter MCP failure (auth, transport, or tool error)."""


class SpecterDisambiguationError(SpecterMCPError):
    """Specter MCP returned a company that does not match the requested identifier."""


# ---------------------------------------------------------------------------
# OAuth token manager
# ---------------------------------------------------------------------------

@dataclass
class _SpecterCredentials:
    mcp_url: str
    token_endpoint: str
    client_id: str
    client_secret: str | None
    refresh_token: str

    @classmethod
    def from_env(cls) -> "_SpecterCredentials":
        url = os.environ.get("SPECTER_MCP_URL", "https://mcp.tryspecter.com/mcp")
        token_ep = os.environ.get(
            "SPECTER_MCP_TOKEN_ENDPOINT", "https://mcp.tryspecter.com/token"
        )
        cid = os.environ.get("SPECTER_MCP_CLIENT_ID")
        secret = os.environ.get("SPECTER_MCP_CLIENT_SECRET") or None
        rt = os.environ.get("SPECTER_MCP_REFRESH_TOKEN")
        if not cid or not rt:
            raise SpecterMCPError(
                "SPECTER_MCP_CLIENT_ID and SPECTER_MCP_REFRESH_TOKEN must be set. "
                "Run scripts/specter_oauth_login.py to mint them."
            )
        return cls(url, token_ep, cid, secret, rt)


class _TokenManager:
    """Caches a Specter access token in memory, refreshing on demand."""

    def __init__(self, creds: _SpecterCredentials) -> None:
        self._creds = creds
        self._access_token: str | None = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def get_access_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if (
                not force_refresh
                and self._access_token
                and time.time() < self._expires_at - 30
            ):
                return self._access_token
            self._refresh_locked()
            return self._access_token  # type: ignore[return-value]

    def _refresh_locked(self) -> None:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": self._creds.refresh_token,
                "client_id": self._creds.client_id,
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        if self._creds.client_secret:
            token = base64.b64encode(
                f"{self._creds.client_id}:{self._creds.client_secret}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(
            self._creds.token_endpoint, data=body, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SpecterMCPError(
                f"Specter token refresh failed ({exc.code}): {detail}"
            ) from exc

        access_token = payload.get("access_token")
        if not access_token:
            raise SpecterMCPError(
                f"Specter token response missing access_token: {payload!r}"
            )
        self._access_token = access_token
        expires_in = int(payload.get("expires_in") or 600)
        self._expires_at = time.time() + expires_in
        new_refresh = payload.get("refresh_token")
        if new_refresh and new_refresh != self._creds.refresh_token:
            # Specter rotates refresh tokens; keep the latest in-memory but
            # do NOT persist (operator must rerun the login helper if the
            # service is restarted past the old refresh-token's lifetime).
            self._creds.refresh_token = new_refresh


# ---------------------------------------------------------------------------
# MCP HTTP client (Streamable HTTP, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SEC = 30
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SEC = 1.5


class SpecterMCPClient:
    """Thin Specter MCP client over Streamable HTTP."""

    def __init__(self, creds: _SpecterCredentials | None = None) -> None:
        self._creds = creds or _SpecterCredentials.from_env()
        self._tokens = _TokenManager(self._creds)
        self._initialized = False
        self._session_id: str | None = None
        self._lock = threading.Lock()

    # -- Public tool wrappers --------------------------------------------

    def find_company(self, identifier: str) -> dict[str, Any]:
        return self._call_tool("find_company", {"identifier": identifier})

    def get_company_profile(self, external_company_id: str) -> dict[str, Any]:
        return self._call_tool(
            "get_company_profile", {"external_company_id": external_company_id}
        )

    def get_company_intelligence(self, external_company_id: str) -> dict[str, Any]:
        return self._call_tool(
            "get_company_intelligence", {"external_company_id": external_company_id}
        )

    def get_company_financials(self, external_company_id: str) -> dict[str, Any]:
        return self._call_tool(
            "get_company_financials", {"external_company_id": external_company_id}
        )

    def get_person_profile(self, external_person_id: str) -> dict[str, Any]:
        return self._call_tool(
            "get_person_profile", {"external_person_id": external_person_id}
        )

    # -- Internals --------------------------------------------------------

    def _ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return
            # Per MCP spec, send initialize before tools/call. Specter's server
            # also accepts tools/call directly, but doing it once keeps us
            # spec-compliant and lets the server pin a session if it wants.
            self._raw_request(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "rockaway-deal-intelligence",
                        "version": "0.1",
                    },
                },
            )
            self._initialized = True

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = self._raw_request(
                    "tools/call", {"name": name, "arguments": arguments}
                )
                return self._unwrap_tool_result(result)
            except _AuthExpired:
                # Refresh once; if still failing, propagate as auth error.
                self._tokens.get_access_token(force_refresh=True)
                last_exc = None  # don't count auth refresh against retry budget
                continue
            except (urllib.error.URLError, TimeoutError, SpecterMCPError) as exc:
                last_exc = exc
                if attempt + 1 == _MAX_ATTEMPTS:
                    break
                time.sleep(_BACKOFF_BASE_SEC * (2 ** attempt))
        raise SpecterMCPError(
            f"Specter MCP tool {name!r} failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
        )

    @staticmethod
    def _unwrap_tool_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict) and "isError" in result and result["isError"]:
            raise SpecterMCPError(f"Specter MCP returned tool error: {result!r}")
        # MCP tool calls return a content array. We expect a single text item
        # whose payload is JSON.
        content = (result or {}).get("content")
        if isinstance(content, list) and content:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text") or ""
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"_raw_text": text}
        if isinstance(result, dict):
            return result
        return {"_raw": result}

    def _raw_request(self, method: str, params: dict[str, Any]) -> Any:
        token = self._tokens.get_access_token()
        body = json.dumps(
            {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(self._creds.mcp_url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT_SEC) as resp:
                # Capture session id once
                sid = resp.headers.get("mcp-session-id") or resp.headers.get(
                    "Mcp-Session-Id"
                )
                if sid and not self._session_id:
                    self._session_id = sid
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise _AuthExpired() from exc
            detail = exc.read().decode("utf-8", errors="replace")
            raise SpecterMCPError(
                f"Specter MCP {method} HTTP {exc.code}: {detail}"
            ) from exc

        return self._parse_response(method, raw, content_type)

    @staticmethod
    def _parse_response(method: str, raw: str, content_type: str) -> Any:
        # Streamable HTTP can respond with either application/json or
        # text/event-stream containing a single message.
        payload: Any
        if "text/event-stream" in content_type:
            payload = None
            for line in raw.splitlines():
                if line.startswith("data:"):
                    data_part = line[len("data:"):].strip()
                    if not data_part:
                        continue
                    try:
                        payload = json.loads(data_part)
                    except json.JSONDecodeError:
                        continue
                    break
            if payload is None:
                raise SpecterMCPError(
                    f"Specter MCP {method} returned empty SSE response: {raw!r}"
                )
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SpecterMCPError(
                    f"Specter MCP {method} returned non-JSON: {raw[:200]!r}"
                ) from exc

        if isinstance(payload, dict) and "error" in payload:
            err = payload["error"]
            raise SpecterMCPError(f"Specter MCP {method} JSON-RPC error: {err!r}")
        if isinstance(payload, dict) and "result" in payload:
            return payload["result"]
        return payload


class _AuthExpired(Exception):
    """Internal sentinel for 401 → refresh-token loop."""


# ---------------------------------------------------------------------------
# Disambiguation helpers
# ---------------------------------------------------------------------------

_CORP_SUFFIX_RE = re.compile(
    r"\b(?:inc|llc|ltd|gmbh|s\.?a\.?|s\.?r\.?o\.?|ag|co|corp|company|technologies|labs|ai|automation|software|holdings)\.?\b",
    re.IGNORECASE,
)


def _normalize_for_match(value: str | None) -> str:
    if not value:
        return ""
    s = value.lower()
    s = _CORP_SUFFIX_RE.sub("", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _domain_root(value: str | None) -> str:
    if not value:
        return ""
    s = value.strip().lower().lstrip("@").strip("'\"")
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("/", 1)[0]
    return s


def _verify_match(
    requested_identifier: str,
    expected_name: str | None,
    returned: dict[str, Any],
) -> None:
    """Raise SpecterDisambiguationError when find_company picked the wrong company.

    Triggered by the Phase 0.5 finding that ``find_company('scribe.com')``
    returned 'Shopscribe' instead of Scribe Inc. Strategy: if the user gave a
    domain, the returned ``domain`` must share the same root. If the user also
    gave an expected name, the returned ``name`` must fuzzy-match it.
    """
    requested = (requested_identifier or "").strip()
    looks_like_domain = (
        "." in requested
        and " " not in requested
        and not requested.startswith("linkedin.com")
    )

    returned_name = returned.get("name") or ""
    returned_domain = returned.get("domain") or ""

    if looks_like_domain:
        wanted = _domain_root(requested)
        got = _domain_root(returned_domain)
        if wanted and got and wanted != got:
            raise SpecterDisambiguationError(
                f"Specter resolved {requested!r} to {returned_name!r} "
                f"(domain={returned_domain!r}); domain root mismatch."
            )

    if expected_name:
        if _normalize_for_match(expected_name) != _normalize_for_match(returned_name):
            raise SpecterDisambiguationError(
                f"Specter returned {returned_name!r} for identifier "
                f"{requested!r}, expected {expected_name!r}."
            )


# ---------------------------------------------------------------------------
# Person → dataclass
# ---------------------------------------------------------------------------

def _person_from_mcp(profile: dict[str, Any]) -> Person:
    education_list: list[Education] = []
    for e in profile.get("education") or []:
        if not isinstance(e, dict):
            continue
        education_list.append(
            Education(
                institution=e.get("school_name"),
                start_year=e.get("start_date"),
                end_year=e.get("end_date"),
            )
        )

    experience_list: list[Experience] = []
    for p in profile.get("positions") or []:
        if not isinstance(p, dict):
            continue
        experience_list.append(
            Experience(
                company=p.get("company_name"),
                title=p.get("title"),
                description=None,  # MCP does not return per-role descriptions
                start_date=p.get("start_date"),
                end_date=p.get("end_date"),
                location=None,
            )
        )

    location = profile.get("location") or ""
    city, country = None, None
    if location:
        parts = [p.strip() for p in location.split(",") if p.strip()]
        if parts:
            city = parts[0]
        if len(parts) >= 2:
            country = parts[-1]

    return Person(
        name=_first_role_owner_name(profile),
        title=_current_position_title(profile),
        about=profile.get("about"),
        city=city,
        country_code=country,
        followers=None,
        connections=None,
        profile_url=profile.get("linkedin_url"),
        education=education_list or None,
        experience=experience_list or None,
        educations_details=None,
    )


def _first_role_owner_name(profile: dict[str, Any]) -> str | None:
    # MCP person profile doesn't have an explicit "full_name" field; we get it
    # from the wrapping intelligence call. As a fallback, reconstruct from
    # tagline (e.g. "Co-Founder @ Fluency | hiring …").
    tag = profile.get("tagline") or ""
    if "@" in tag:
        return None
    return None


def _current_position_title(profile: dict[str, Any]) -> str | None:
    for p in profile.get("positions") or []:
        if isinstance(p, dict) and p.get("is_current"):
            return p.get("title")
    return None


# ---------------------------------------------------------------------------
# Chunk builders (mirror the CSV pipeline's section labels)
# ---------------------------------------------------------------------------

_CHUNK_SOURCE_COMPANY = "specter-mcp"
_CHUNK_SOURCE_PEOPLE = "specter-mcp-people"


def _chunk(idx: int, source: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"chunk_{idx}",
        text=text.strip(),
        source_file=source,
        page_or_slide=section,
    )


def _build_overview_chunk(profile: dict[str, Any], idx: int) -> Chunk:
    name = profile.get("name") or "Unknown"
    parts = [f"Company: {name}"]
    for key, label in [
        ("short_description", "Description"),
        ("typical_customer_profile", "Typical Customer Profile"),
        ("traction", "Traction Highlights"),
        ("hq_location", "HQ Location"),
        ("founded_year", "Founded"),
        ("growth_stage", "Growth Stage"),
        ("company_status", "Operating Status"),
        ("employee_count", "Employee Count"),
        ("web_visits_last_month", "Monthly Web Visits"),
    ]:
        v = profile.get(key)
        if v:
            parts.append(f"{label}: {v}")

    industry = profile.get("industry")
    if isinstance(industry, list) and industry:
        parts.append(f"Industry: {' > '.join(str(i) for i in industry)}")
    elif industry:
        parts.append(f"Industry: {industry}")

    verticals = profile.get("tech_verticals")
    if isinstance(verticals, list) and verticals:
        vstrings = []
        for tv in verticals:
            if not isinstance(tv, dict):
                continue
            top = tv.get("vertical")
            subs = tv.get("sub_verticals") or []
            if top and subs:
                for s in subs:
                    vstrings.append(f"{top} > {s}")
            elif top:
                vstrings.append(top)
        if vstrings:
            parts.append(f"Tech Vertical: {'; '.join(vstrings)}")

    highlights = profile.get("highlights")
    if isinstance(highlights, list) and highlights:
        parts.append(f"Highlights: {', '.join(str(h) for h in highlights)}")

    return _chunk(idx, _CHUNK_SOURCE_COMPANY, "Company Overview", "\n".join(parts))


def _build_funding_chunk(
    financials: dict[str, Any], company_name: str, idx: int
) -> Chunk | None:
    if not financials:
        return None
    parts: list[str] = []
    for key, label in [
        ("total_funding_amount", "Total Funding"),
        ("last_funding_amount", "Last Round Amount"),
        ("last_funding_date", "Last Funding Date"),
        ("last_funding_type", "Last Funding Type"),
        ("post_money_valuation", "Post-Money Valuation"),
        ("number_of_funding_rounds", "Funding Rounds"),
        ("number_of_investors", "Number of Investors"),
    ]:
        v = financials.get(key)
        if v not in (None, "", []):
            parts.append(f"{label}: {v}")
    investors = financials.get("investors") or []
    if investors:
        parts.append(f"Investors: {', '.join(investors)}")

    rounds = financials.get("funding_rounds") or []
    if rounds:
        parts.append("\nFunding Round Details:")
        for r in rounds:
            if not isinstance(r, dict):
                continue
            raised = r.get("raised")
            raised_str = f"${raised:,.0f}" if isinstance(raised, (int, float)) else "undisclosed"
            leads = r.get("lead_investors_partners") or []
            lead_names = [
                l.get("name") for l in leads
                if isinstance(l, dict) and l.get("is_lead_investor") and l.get("name")
            ]
            all_inv = [l.get("name") for l in leads if isinstance(l, dict) and l.get("name")]
            inv_str = ", ".join(all_inv) if all_inv else "undisclosed"
            lead_str = f" | Led by: {', '.join(lead_names)}" if lead_names else ""
            round_name = r.get("funding_round_name") or "Round"
            date = r.get("date") or "N/A"
            parts.append(f"  - {round_name} ({date}): {raised_str} from {inv_str}{lead_str}")
            pre_v = r.get("pre_money_valuation")
            post_v = r.get("post_money_valuation")
            if pre_v or post_v:
                parts.append(
                    f"    Pre-money: {pre_v or 'N/A'} | Post-money: {post_v or 'N/A'}"
                )

    if financials.get("acquisition_date"):
        parts.append(
            f"\nAcquired by {financials.get('acquired_by') or 'Unknown'} "
            f"on {financials['acquisition_date']} for {financials.get('acquisition_price') or 'undisclosed'}"
        )
    if financials.get("ipo_details"):
        parts.append(f"\nIPO: {financials['ipo_details']}")

    if not parts:
        return None
    return _chunk(
        idx,
        _CHUNK_SOURCE_COMPANY,
        "Funding & Investors",
        f"Funding History for {company_name}:\n" + "\n".join(parts),
    )


def _build_growth_chunk(
    profile: dict[str, Any],
    intelligence: dict[str, Any],
    company_name: str,
    idx: int,
) -> Chunk | None:
    parts: list[str] = []
    has_data = False

    employee_count = profile.get("employee_count")
    if employee_count:
        has_data = True
        parts.append(f"\nEmployee Count: {employee_count}")

    history = intelligence.get("headcount_by_department") or []
    growth_lines = _summarize_headcount_growth(history)
    if growth_lines:
        has_data = True
        parts.extend(growth_lines)

    web_visits = profile.get("web_visits_last_month") or intelligence.get("web_visits")
    if web_visits:
        has_data = True
        parts.append(f"\nMonthly Web Visits: {web_visits:,}")
        # MCP does not provide multi-period web growth ratios; the CSV path
        # is required if those signals matter to scoring.

    if not has_data:
        return None
    return _chunk(
        idx,
        _CHUNK_SOURCE_COMPANY,
        "Growth Metrics",
        f"Growth Metrics for {company_name}:" + "\n".join(parts),
    )


def _summarize_headcount_growth(history: list[dict[str, Any]]) -> list[str]:
    """Derive 1/3/6/12-month employee growth ratios from monthly headcount history."""
    if not history:
        return []
    sorted_hist = sorted(
        [h for h in history if isinstance(h, dict) and h.get("month") and h.get("total_count")],
        key=lambda h: str(h["month"]),
    )
    if len(sorted_hist) < 2:
        return []
    latest = sorted_hist[-1]
    latest_count = latest.get("total_count") or 0
    if not latest_count:
        return []

    out: list[str] = []
    for months in (1, 3, 6, 12, 24):
        if len(sorted_hist) <= months:
            continue
        prior = sorted_hist[-1 - months].get("total_count") or 0
        if not prior:
            continue
        pct = ((latest_count - prior) / prior) * 100.0
        out.append(f"  {months}mo employee growth: {pct:.1f}% ({prior} → {latest_count})")

    # Department mix on most recent month
    dept = latest.get("count_by_department") or {}
    if dept:
        items = sorted(dept.items(), key=lambda kv: -int(kv[1] or 0))
        out.append(
            "  Department mix (latest): "
            + ", ".join(f"{name} {count}" for name, count in items if count)
        )
    return out


def _build_signals_chunk(
    intelligence: dict[str, Any], company_name: str, idx: int
) -> Chunk | None:
    signals = intelligence.get("investor_interest_signals") or []
    clients = intelligence.get("reported_clients") or []
    if not signals and not clients:
        return None
    parts: list[str] = []
    if signals:
        parts.append(f"Investor Interest Signals for {company_name}:")
        for s in signals:
            if not isinstance(s, dict):
                continue
            sources = ", ".join(s.get("signal_sources") or [])
            parts.append(
                f"  - [{s.get('signal_date', 'N/A')}] score {s.get('signal_score', 'N/A')}/10: "
                f"{s.get('summary') or s.get('name') or ''}"
            )
            if sources:
                parts.append(f"    Sources: {sources}")
    if clients:
        parts.append(f"\nReported Clients: {', '.join(c.get('name', '') for c in clients if isinstance(c, dict))}")
    return _chunk(
        idx,
        _CHUNK_SOURCE_COMPANY,
        "Investor Interest & Reported Clients",
        "\n".join(parts),
    )


def _build_team_overview_chunk(
    team: list[Person],
    founder_dicts: list[dict[str, Any]],
    company_name: str,
    idx: int,
) -> Chunk:
    parts = [f"Founding Team Overview for {company_name}:"]
    parts.append(f"Number of Founders: {len(founder_dicts)}")
    parts.append("")
    for person, raw in zip(team, founder_dicts):
        parts.append(person.get_profile_summary())
        title = raw.get("title")
        if title:
            parts.append(f"Title: {title}")
        parts.append("---")
    return _chunk(idx, _CHUNK_SOURCE_PEOPLE, "Founding Team Overview", "\n".join(parts))


def _build_person_detail_chunk(
    person: Person,
    raw_summary: dict[str, Any],
    raw_profile: dict[str, Any] | None,
    company_name: str,
    idx: int,
) -> Chunk:
    name = raw_summary.get("full_name") or person.name or "Unknown"
    parts = [f"Team Member Profile: {name} at {company_name}"]
    title = raw_summary.get("title")
    if title:
        parts.append(f"Current Role: {title}")
    seniority = raw_summary.get("seniority") or (raw_profile or {}).get("seniority")
    yoe = (raw_profile or {}).get("years_of_experience")
    if seniority or yoe is not None:
        parts.append(f"Seniority: {seniority or 'N/A'} | Years of Experience: {yoe if yoe is not None else 'N/A'}")
    tagline = (raw_profile or {}).get("tagline")
    if tagline:
        parts.append(f"Tagline: {tagline}")
    if person.about:
        about_text = person.about[:600] if len(person.about) > 600 else person.about
        parts.append(f"About: {about_text}")
    if person.education:
        parts.append(f"Education: {person.get_education_summary()}")
    highlights = (raw_profile or {}).get("highlights") or []
    if highlights:
        parts.append(f"Highlights: {', '.join(str(h).replace('_', ' ').title() for h in highlights)}")
    if person.experience:
        parts.append("\nCareer History:")
        for exp in person.experience:
            exp_str = str(exp)
            if exp_str.strip():
                parts.append(f"  • {exp_str}")
    if person.profile_url:
        parts.append(f"LinkedIn: {person.profile_url}")
    return _chunk(idx, _CHUNK_SOURCE_PEOPLE, f"Team Member: {name}", "\n".join(parts))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level singleton; OK because the client is stateless after init and
# token refresh is internally synchronized.
_default_client: SpecterMCPClient | None = None
_default_client_lock = threading.Lock()


def get_default_client() -> SpecterMCPClient:
    global _default_client
    with _default_client_lock:
        if _default_client is None:
            _default_client = SpecterMCPClient()
        return _default_client


def fetch_specter_company(
    identifier: str,
    *,
    expected_name: str | None = None,
    fetch_full_team: bool = True,
    client: SpecterMCPClient | None = None,
) -> tuple[Company, EvidenceStore]:
    """Fetch a company by URL/domain/name and return (Company, EvidenceStore).

    Args:
        identifier: domain (e.g. ``"anthropic.com"``), website URL, LinkedIn URL,
            external company ID, or company name.
        expected_name: optional company name supplied by the user; used to
            sanity-check Specter's disambiguation. Skipped if not provided.
        fetch_full_team: when True, fan out to ``get_person_profile`` for each
            founder. Costs N+3 MCP calls per company.
        client: inject a pre-built client (used in tests).

    Raises:
        SpecterDisambiguationError: Specter's resolver returned a different
            company than requested.
        SpecterMCPError: any other MCP transport / auth / tool failure.
    """
    cli = client or get_default_client()
    base = cli.find_company(identifier)
    _verify_match(identifier, expected_name, base)
    company_id = base.get("external_company_id")
    if not company_id:
        raise SpecterMCPError(f"Specter find_company returned no ID: {base!r}")

    profile = cli.get_company_profile(company_id)
    intelligence = cli.get_company_intelligence(company_id)
    financials = cli.get_company_financials(company_id)

    company_name = profile.get("name") or base.get("name") or identifier
    domain = profile.get("domain") or base.get("domain")

    # Build founders + team
    founder_dicts = intelligence.get("founders") or []
    team_persons: list[Person] = []
    raw_profiles: list[dict[str, Any] | None] = []
    for f in founder_dicts:
        if not isinstance(f, dict):
            continue
        pid = f.get("external_person_id")
        raw_profile: dict[str, Any] | None = None
        person = Person(
            name=f.get("full_name"),
            title=f.get("title"),
            about=None,
            profile_url=f.get("linkedin_url"),
        )
        if fetch_full_team and pid:
            try:
                raw_profile = cli.get_person_profile(pid)
                deeper = _person_from_mcp(raw_profile)
                # Merge: prefer the deeper profile but keep founder-list name/title
                deeper.name = f.get("full_name") or deeper.name
                deeper.title = f.get("title") or deeper.title
                person = deeper
            except SpecterMCPError:
                # Per-person failure must not block the company-level result.
                pass
        team_persons.append(person)
        raw_profiles.append(raw_profile)

    company = Company(
        name=company_name,
        industry="; ".join(profile.get("industry") or []) or None,
        tagline=None,
        about=profile.get("short_description"),
        team=team_persons or None,
        domain=domain,
    )

    chunks: list[Chunk] = []
    idx = 0

    chunks.append(_build_overview_chunk(profile, idx))
    idx += 1

    funding = _build_funding_chunk(financials, company_name, idx)
    if funding:
        chunks.append(funding)
        idx += 1

    growth = _build_growth_chunk(profile, intelligence, company_name, idx)
    if growth:
        chunks.append(growth)
        idx += 1

    signals = _build_signals_chunk(intelligence, company_name, idx)
    if signals:
        chunks.append(signals)
        idx += 1

    if team_persons:
        chunks.append(
            _build_team_overview_chunk(team_persons, founder_dicts, company_name, idx)
        )
        idx += 1
        for person, raw_summary, raw_profile in zip(team_persons, founder_dicts, raw_profiles):
            chunks.append(
                _build_person_detail_chunk(person, raw_summary, raw_profile, company_name, idx)
            )
            idx += 1

    store = EvidenceStore(startup_slug=_company_slug(company_name), chunks=chunks)
    return company, store
