"""Tests for the Specter MCP client and chunk builders.

Network-free: every test mocks the SpecterMCPClient. The integration test that
hits the real MCP server is gated on SPECTER_MCP_REFRESH_TOKEN being present.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from agent.dataclasses.company import Company
from agent.ingest.specter_mcp_client import (
    SpecterCompanyNotFoundError,
    SpecterDisambiguationError,
    SpecterMCPClient,
    SpecterMCPError,
    SpecterQuotaAuthorizationError,
    SpecterQuotaLimitError,
    _brand_stem,
    _build_funding_chunk,
    _build_growth_chunk,
    _build_overview_chunk,
    _build_signals_chunk,
    _domain_root,
    _normalize_for_match,
    _person_from_mcp,
    _summarize_headcount_growth,
    _verify_match,
    fetch_specter_company,
    is_specter_quota_limit_message,
    specter_quota_reset_hint,
)
from agent.ingest.store import EvidenceStore
from web.specter_quota_broker import SpecterQuotaBrokerRequest, use_specter_quota_broker


# ---------------------------------------------------------------------------
# Tool result unwrapping — definitive errors fast-fail without retry
# ---------------------------------------------------------------------------

def test_unwrap_tool_result_raises_company_not_found_for_definitive_error():
    """`No company found` is Specter's definitive 'no match' answer; must
    raise the dedicated exception so the retry loop in _call_tool fast-fails."""
    payload = {
        "isError": True,
        "content": [
            {"type": "text", "text": "Error calling tool 'find_company': No company found"}
        ],
    }
    with pytest.raises(SpecterCompanyNotFoundError):
        SpecterMCPClient._unwrap_tool_result(payload)


def test_unwrap_tool_result_raises_generic_for_other_errors():
    """Other tool errors (rate-limit, internal, etc.) keep the generic
    SpecterMCPError type so the retry loop can still back off and retry."""
    payload = {
        "isError": True,
        "content": [{"type": "text", "text": "Internal server error"}],
    }
    with pytest.raises(SpecterMCPError) as exc_info:
        SpecterMCPClient._unwrap_tool_result(payload)
    # Must NOT be the not-found subclass
    assert not isinstance(exc_info.value, SpecterCompanyNotFoundError)


def test_unwrap_tool_result_raises_quota_limit_for_daily_mcp_limit():
    payload = {
        "isError": True,
        "content": [
            {"type": "text", "text": "Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC."}
        ],
    }
    with pytest.raises(SpecterQuotaLimitError) as exc_info:
        SpecterMCPClient._unwrap_tool_result(payload)
    assert exc_info.value.reset_hint == "00:00 UTC"


def test_quota_message_helpers_detect_daily_limit_and_reset_hint():
    message = "Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC."

    assert is_specter_quota_limit_message(message) is True
    assert specter_quota_reset_hint(message) == "00:00 UTC"


@pytest.mark.parametrize(
    "message",
    [
        (
            "You've used today's 100 MCP credits. Your account has no API credits "
            "to cover the overflow, so MCP is paused until the daily limit resets "
            "at 00:00 UTC."
        ),
        (
            "You have used all your 100 MCP credits today. "
            "MCP calls are paused until the daily reset at 00:00 UTC."
        ),
        "All daily MCP credits used; MCP calls paused until the daily reset at 00:00 UTC.",
        "All daily MCP credits have been used; calls resume after reset at 00:00 UTC.",
        "MCP quota exhausted for today. Resets at 00:00 UTC.",
    ],
)
def test_quota_message_helpers_detect_production_credit_wording(message: str):
    assert is_specter_quota_limit_message(message) is True
    assert specter_quota_reset_hint(message) == "00:00 UTC"


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 429: too many requests",
        "Specter authentication is unavailable",
        "No company found",
        "Temporary MCP transport failure",
    ],
)
def test_quota_message_helpers_do_not_classify_ambiguous_failures(message: str):
    assert is_specter_quota_limit_message(message) is False


def test_parse_response_raises_quota_limit_for_json_rpc_error():
    raw = '{"jsonrpc":"2.0","error":{"code":429,"message":"Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC."}}'

    with pytest.raises(SpecterQuotaLimitError) as exc_info:
        SpecterMCPClient._parse_response("tools/call", raw, "application/json")
    assert exc_info.value.reset_hint == "00:00 UTC"


def test_company_not_found_is_subclass_of_mcp_error():
    """Callers that catch SpecterMCPError still see SpecterCompanyNotFoundError
    (subclass relationship preserved for backward compat)."""
    assert issubclass(SpecterCompanyNotFoundError, SpecterMCPError)


def test_quota_limit_is_subclass_of_mcp_error():
    assert issubclass(SpecterQuotaLimitError, SpecterMCPError)


def test_quota_tool_error_receives_exactly_one_attempt(monkeypatch):
    client = object.__new__(SpecterMCPClient)
    calls = 0

    monkeypatch.setattr(client, "_ensure_initialized", lambda: None)

    def raw_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "isError": True,
            "content": [
                {"type": "text", "text": "All daily MCP credits used"}
            ],
        }

    monkeypatch.setattr(client, "_raw_request", raw_request)
    monkeypatch.setattr(
        "agent.ingest.specter_mcp_client.time.sleep",
        lambda _seconds: pytest.fail("quota error entered retry backoff"),
    )

    with pytest.raises(SpecterQuotaLimitError):
        client._call_tool("find_company", {"identifier": "galtea.ai"})
    assert calls == 1


def test_exact_production_quota_tool_error_receives_exactly_one_attempt(monkeypatch):
    client = object.__new__(SpecterMCPClient)
    calls = 0

    monkeypatch.setattr(client, "_ensure_initialized", lambda: None)

    def raw_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You've used today's 100 MCP credits. Your account has no API "
                        "credits to cover the overflow, so MCP is paused until the daily "
                        "limit resets at 00:00 UTC."
                    ),
                }
            ],
        }

    monkeypatch.setattr(client, "_raw_request", raw_request)
    monkeypatch.setattr(
        "agent.ingest.specter_mcp_client.time.sleep",
        lambda _seconds: pytest.fail("quota error entered retry backoff"),
    )

    with pytest.raises(SpecterQuotaLimitError) as exc_info:
        client._call_tool("get_company_profile", {"external_company_id": "company-1"})

    assert calls == 1
    assert exc_info.value.reset_hint == "00:00 UTC"


def test_call_tool_uses_quota_broker_reservation_and_commit(monkeypatch):
    client = object.__new__(SpecterMCPClient)
    client._ensure_initialized = lambda: None
    client._raw_request = lambda *_args, **_kwargs: {
        "content": [{"type": "text", "text": '{"ok": true}'}]
    }
    reservations: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "web.db.reserve_specter_quota_authorization",
        lambda **kwargs: reservations.append(kwargs)
        or {
            "authorization_id": "specter-auth-0001",
            "status": "authorized",
            "circuit_state": "closed",
            "state": "open",
        },
    )
    monkeypatch.setattr(
        "web.db.commit_specter_quota_authorization",
        lambda **kwargs: commits.append(kwargs) or {"status": "committed"},
    )

    with use_specter_quota_broker(
        SpecterQuotaBrokerRequest(
            target_environment="staging",
            business_date="2026-08-24",
            business_timezone="Europe/Prague",
            consumer="rdi",
            operation=None,
            quota_class="autonomous_campaign",
            company_ref="domain:acme.example",
            remaining_rdi_slots=12,
            actor="service:rockaway-leadgen",
            intake_id="rdi-v2-intake-" + "a" * 32,
            metadata={"intake_id": "rdi-v2-intake-" + "a" * 32},
            enforcement_enabled=True,
        )
    ):
        result = client._call_tool(
            "get_company_profile",
            {"external_company_id": "company-1"},
        )

    assert result == {"ok": True}
    assert reservations[0]["operation"] == "get_company_profile"
    assert reservations[0]["quota_class"] == "autonomous_campaign"
    assert commits == [
        {
            "authorization_id": "specter-auth-0001",
            "target_environment": "staging",
            "operation": "get_company_profile",
            "intake_id": "rdi-v2-intake-" + "a" * 32,
            "outcome": "succeeded",
            "provider_quota_error": False,
            "reason_code": None,
            "actor": "service:rockaway-leadgen",
        }
    ]


def test_call_tool_reserves_once_per_actual_dispatch_attempt(monkeypatch):
    client = object.__new__(SpecterMCPClient)
    calls = 0
    reservations: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []

    monkeypatch.setattr(client, "_ensure_initialized", lambda: None)

    def raw_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SpecterMCPError("temporary MCP transport failure")
        return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    monkeypatch.setattr(client, "_raw_request", raw_request)
    monkeypatch.setattr("agent.ingest.specter_mcp_client.time.sleep", lambda *_a: None)
    monkeypatch.setattr(
        "web.db.reserve_specter_quota_authorization",
        lambda **kwargs: reservations.append(kwargs)
        or {
            "authorization_id": f"specter-auth-{len(reservations):04d}",
            "status": "authorized",
            "circuit_state": "closed",
            "state": "open",
        },
    )
    monkeypatch.setattr(
        "web.db.commit_specter_quota_authorization",
        lambda **kwargs: commits.append(kwargs) or {"status": kwargs["outcome"]},
    )

    with use_specter_quota_broker(
        SpecterQuotaBrokerRequest(
            target_environment="staging",
            business_date="2026-08-24",
            business_timezone="Europe/Prague",
            consumer="analysis_preflight",
            operation=None,
            quota_class="flex",
            company_ref="domain:acme.example",
            remaining_rdi_slots=None,
            actor="service:rockaway-leadgen",
            intake_id=None,
            metadata={},
            enforcement_enabled=True,
        )
    ):
        assert client._call_tool("find_company", {"identifier": "acme.example"}) == {"ok": True}

    assert len(reservations) == 2
    assert reservations[0]["idempotency_key"] != reservations[1]["idempotency_key"]
    assert [item["outcome"] for item in commits] == ["failed", "succeeded"]


def test_call_tool_observe_mode_logs_and_bypasses_denied_broker_reservation(monkeypatch, caplog):
    client = object.__new__(SpecterMCPClient)
    client._ensure_initialized = lambda: None
    client._raw_request = lambda *_args, **_kwargs: {
        "content": [{"type": "text", "text": '{"ok": true}'}]
    }

    monkeypatch.setattr(
        "web.db.reserve_specter_quota_authorization",
        lambda **_kwargs: {
            "authorization_id": "specter-auth-0001",
            "status": "deferred",
            "reason": "quota_estimate_exhausted",
            "circuit_state": "closed",
            "state": "open",
        },
    )

    with use_specter_quota_broker(
        SpecterQuotaBrokerRequest(
            target_environment="staging",
            business_date="2026-08-24",
            business_timezone="Europe/Prague",
            consumer="analysis_preflight",
            operation=None,
            quota_class="flex",
            company_ref="domain:acme.example",
            remaining_rdi_slots=None,
            actor="service:rockaway-leadgen",
            intake_id=None,
            metadata={},
            enforcement_enabled=False,
        )
    ):
        with caplog.at_level("WARNING"):
            assert client._call_tool("find_company", {"identifier": "acme.example"}) == {"ok": True}

    assert "observe-mode bypass" in caplog.text


def test_call_tool_enforce_mode_raises_distinct_broker_denial_error(monkeypatch):
    client = object.__new__(SpecterMCPClient)
    client._ensure_initialized = lambda: None
    client._raw_request = lambda *_args, **_kwargs: pytest.fail("provider call should not dispatch")
    commits: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "web.db.reserve_specter_quota_authorization",
        lambda **_kwargs: {
            "authorization_id": "specter-auth-0001",
            "status": "deferred",
            "reason": "company_cap_exhausted",
            "circuit_state": "closed",
            "state": "open",
        },
    )
    monkeypatch.setattr(
        "web.db.commit_specter_quota_authorization",
        lambda **kwargs: commits.append(kwargs) or {"status": kwargs["outcome"]},
    )

    with use_specter_quota_broker(
        SpecterQuotaBrokerRequest(
            target_environment="staging",
            business_date="2026-08-24",
            business_timezone="Europe/Prague",
            consumer="analysis_preflight",
            operation=None,
            quota_class="flexible_pool",
            company_ref="Acme deck / batch 1",
            remaining_rdi_slots=None,
            actor="service:rockaway-leadgen",
            intake_id=None,
            metadata={},
            enforcement_enabled=True,
        )
    ):
        with pytest.raises(SpecterQuotaAuthorizationError):
            client._call_tool("find_company", {"identifier": "acme.example"})

    assert commits == []


def test_transient_tool_error_retains_bounded_retries(monkeypatch):
    client = object.__new__(SpecterMCPClient)
    calls = 0
    sleeps: list[float] = []

    monkeypatch.setattr(client, "_ensure_initialized", lambda: None)

    def raw_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise SpecterMCPError("temporary MCP transport failure")
        return {"content": [{"type": "text", "text": '{"ok":true}'}]}

    monkeypatch.setattr(client, "_raw_request", raw_request)
    monkeypatch.setattr(
        "agent.ingest.specter_mcp_client.time.sleep",
        sleeps.append,
    )

    assert client._call_tool("find_company", {"identifier": "galtea.ai"}) == {
        "ok": True
    }
    assert calls == 3
    assert len(sleeps) == 2


# ---------------------------------------------------------------------------
# Financial tool compatibility
# ---------------------------------------------------------------------------

def test_get_company_financials_adapts_split_funding_rounds_tool(monkeypatch):
    """Specter replaced get_company_financials with split finance tools.

    Keep the client-facing compatibility method stable while normalizing the
    funding-round payload into the schema consumed by the evidence builders.
    """
    client = object.__new__(SpecterMCPClient)
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, arguments))
        assert name == "get_company_funding_rounds"
        return {
            "external_company_id": "61643d92c3c073075bcb8983",
            "total": 2,
            "funding_rounds": [
                {
                    "external_funding_round_id": "fnr_latest",
                    "name": "Series E",
                    "investment_type": "series_e",
                    "announced_on": "2025-01-01",
                    "raised_amount_usd": 500_000_000,
                    "pre_money_valuation_usd": 49_500_000_000,
                    "post_money_valuation_usd": 50_000_000_000,
                    "investors": [
                        {
                            "external_investor_id": "inv_a",
                            "name": "A",
                            "is_lead_investor": True,
                        },
                        {
                            "external_investor_id": "inv_b",
                            "name": "B",
                            "is_lead_investor": False,
                        },
                    ],
                },
                {
                    "external_funding_round_id": "fnr_seed",
                    "name": "Seed",
                    "investment_type": "seed",
                    "announced_on": "2022-01-01",
                    "raised_amount_usd": 100_000_000,
                    "pre_money_valuation_usd": None,
                    "post_money_valuation_usd": None,
                    "investors": [
                        {
                            "external_investor_id": "inv_a",
                            "name": "A",
                            "is_lead_investor": True,
                        }
                    ],
                },
            ],
        }

    monkeypatch.setattr(client, "_call_tool", fake_call_tool)

    financials = client.get_company_financials("61643d92c3c073075bcb8983")

    assert calls == [
        (
            "get_company_funding_rounds",
            {"external_company_id": "61643d92c3c073075bcb8983", "limit": 50},
        )
    ]
    assert financials == {
        "total_funding_amount": 600_000_000,
        "last_funding_amount": 500_000_000,
        "last_funding_date": "2025-01-01",
        "last_funding_type": "Series E",
        "post_money_valuation": 50_000_000_000,
        "number_of_funding_rounds": 2,
        "number_of_investors": 2,
        "investors": ["A", "B"],
        "funding_rounds": [
            {
                "funding_round_name": "Series E",
                "date": "2025-01-01",
                "raised": 500_000_000,
                "lead_investors_partners": [
                    {
                        "external_investor_id": "inv_a",
                        "name": "A",
                        "is_lead_investor": True,
                    },
                    {
                        "external_investor_id": "inv_b",
                        "name": "B",
                        "is_lead_investor": False,
                    },
                ],
                "pre_money_valuation": 49_500_000_000,
                "post_money_valuation": 50_000_000_000,
            },
            {
                "funding_round_name": "Seed",
                "date": "2022-01-01",
                "raised": 100_000_000,
                "lead_investors_partners": [
                    {
                        "external_investor_id": "inv_a",
                        "name": "A",
                        "is_lead_investor": True,
                    }
                ],
                "pre_money_valuation": None,
                "post_money_valuation": None,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Brand stem extraction
# ---------------------------------------------------------------------------


def test_brand_stem_strips_tld_and_subdomains():
    assert _brand_stem("adspawn.com") == "adspawn"
    assert _brand_stem("adspawn.io") == "adspawn"
    assert _brand_stem("www.adspawn.com") == "adspawn"  # _domain_root strips www
    assert _brand_stem("https://www.adspawn.com/contact") == "adspawn"
    assert _brand_stem("acme.co.uk") == "acme"
    assert _brand_stem("app.acme.com") == "acme"  # app subdomain stripped
    assert _brand_stem("go.acme.io") == "acme"
    assert _brand_stem("") == ""
    assert _brand_stem(None) == ""
    # Non-domain inputs return as-is (no period)
    assert _brand_stem("AdSpawn") == "adspawn"


# ---------------------------------------------------------------------------
# Brand-stem fallback in fetch_specter_company
# ---------------------------------------------------------------------------


class _FakeFindCompanyClient:
    """Mock SpecterMCPClient where find_company can fail-then-succeed."""

    def __init__(self, behaviour: dict[str, Any]) -> None:
        self.calls: list[str] = []
        self._behaviour = behaviour

    def find_company(self, identifier: str) -> dict[str, Any]:
        self.calls.append(identifier)
        result = self._behaviour.get(identifier)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise SpecterCompanyNotFoundError(f"No company found: {identifier}")
        return result

    def get_company_profile(self, _id: str) -> dict[str, Any]:
        return {"name": "AdSpawn", "domain": "adspawn.io", "industry": ["AdTech"]}

    def get_company_intelligence(self, _id: str) -> dict[str, Any]:
        return {"founders": []}

    def get_company_financials(self, _id: str) -> dict[str, Any]:
        return {}


class _QuotaProfileClient(_FakeFindCompanyClient):
    def __init__(self) -> None:
        super().__init__({})
        self.find_calls = 0

    def find_company(self, identifier: str) -> dict[str, Any]:
        self.find_calls += 1
        return super().find_company(identifier)

    def get_company_profile(self, _id: str) -> dict[str, Any]:
        raise SpecterQuotaLimitError("Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC.")


class _QuotaFounderClient(_FakeFindCompanyClient):
    def get_company_intelligence(self, _id: str) -> dict[str, Any]:
        return {
            "founders": [
                {
                    "full_name": "Ada Founder",
                    "title": "CEO",
                    "external_person_id": "person-1",
                }
            ]
        }

    def get_person_profile(self, _id: str) -> dict[str, Any]:
        raise SpecterQuotaLimitError("Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC.")


def test_fetch_with_known_company_id_does_not_fallback_on_quota_limit():
    client = _QuotaProfileClient()

    with pytest.raises(SpecterQuotaLimitError):
        fetch_specter_company(
            "adspawn.com",
            fetch_full_team=False,
            client=client,
            known_company_id="known-id",
        )
    assert client.find_calls == 0


def test_fetch_full_team_propagates_quota_limit_from_person_profile():
    client = _QuotaFounderClient(
        {
            "adspawn.io": {
                "external_company_id": "abc123",
                "name": "AdSpawn",
                "domain": "adspawn.io",
            }
        }
    )

    with pytest.raises(SpecterQuotaLimitError):
        fetch_specter_company("adspawn.io", fetch_full_team=True, client=client)


def test_fetch_falls_back_to_brand_stem_when_domain_lookup_fails():
    """The AdSpawn case: deck has adspawn.com, Specter indexes adspawn.io.
    find_company('adspawn.com') returns 'No company found' but
    find_company('adspawn') (the brand stem) finds the company."""
    client = _FakeFindCompanyClient({
        "adspawn.com": SpecterCompanyNotFoundError("No company found"),
        "adspawn": {
            "external_company_id": "abc123",
            "name": "AdSpawn",
            "domain": "adspawn.io",
        },
    })
    company, store = fetch_specter_company(
        "adspawn.com", fetch_full_team=False, client=client
    )
    assert company.name == "AdSpawn"
    assert company.domain == "adspawn.io"
    assert company.company_url == "https://adspawn.io"
    assert company.specter_company_id == "abc123"
    assert company.specter_profile_url == "https://app.tryspecter.com/signals/company/feed/abc123"
    # Two find_company calls: failed domain, then brand stem fallback
    assert client.calls == ["adspawn.com", "adspawn"]


def test_fetch_brand_stem_fallback_rejects_cross_company_match():
    """If brand-stem search returns a company whose domain has a DIFFERENT
    brand stem, that's a cross-company collision (different 'AdSpawn'),
    must raise SpecterDisambiguationError."""
    client = _FakeFindCompanyClient({
        "adspawn.com": SpecterCompanyNotFoundError("No company found"),
        # Brand-stem search returns SOME company called 'AdSpawn' but on
        # an unrelated domain — likely a different company.
        "adspawn": {
            "external_company_id": "xyz",
            "name": "AdSpawn",
            "domain": "bigtech.com",  # ← different brand stem
        },
    })
    with pytest.raises(SpecterDisambiguationError):
        fetch_specter_company("adspawn.com", fetch_full_team=False, client=client)


def test_fetch_does_not_attempt_brand_stem_for_non_domain_identifiers():
    """If the identifier is already a name (no period), don't fall back."""
    client = _FakeFindCompanyClient({
        "AdSpawn": SpecterCompanyNotFoundError("No company found"),
    })
    with pytest.raises(SpecterCompanyNotFoundError):
        fetch_specter_company("AdSpawn", fetch_full_team=False, client=client)
    # Only one attempt — no fallback (already a name)
    assert client.calls == ["AdSpawn"]


def test_fetch_does_not_attempt_brand_stem_for_short_stems():
    """Brand stems shorter than 3 chars are too generic to safely match."""
    client = _FakeFindCompanyClient({
        "ai.com": SpecterCompanyNotFoundError("No company found"),
    })
    with pytest.raises(SpecterCompanyNotFoundError):
        fetch_specter_company("ai.com", fetch_full_team=False, client=client)
    # Only one attempt — stem 'ai' is < 3 chars
    assert client.calls == ["ai.com"]


# ---------------------------------------------------------------------------
# Disambiguation
# ---------------------------------------------------------------------------


def test_normalize_for_match_strips_corp_suffixes():
    assert _normalize_for_match("Mimica Automation") == _normalize_for_match("Mimica")
    assert _normalize_for_match("Skan AI") == _normalize_for_match("skan.ai")
    assert _normalize_for_match("Foo Inc.") == _normalize_for_match("Foo")
    assert _normalize_for_match("") == ""
    assert _normalize_for_match(None) == ""


def test_domain_root_handles_schemes_and_www_and_at():
    assert _domain_root("https://www.foo.com/bar?x=1") == "foo.com"
    assert _domain_root("foo.com") == "foo.com"
    assert _domain_root("@foo.com") == "foo.com"
    assert _domain_root("'foo.com'") == "foo.com"
    assert _domain_root(None) == ""


def test_verify_match_accepts_matching_domain():
    _verify_match(
        "anthropic.com",
        None,
        {"name": "Anthropic", "domain": "anthropic.com"},
    )


def test_verify_match_rejects_scribe_to_shopscribe_regression():
    """Phase 0.5 parity test surfaced this — must keep rejecting it forever."""
    with pytest.raises(SpecterDisambiguationError):
        _verify_match(
            "scribe.com",
            None,
            {"name": "Shopscribe", "domain": "shopscribe.com"},
        )


@pytest.mark.parametrize(
    ("identifier", "expected_name", "returned_name"),
    [
        ("https://amilabs.xyz/", "AMI Labs", "AMI - Advanced Machine Intelligence"),
        ("https://designverse.ai/", "DesignVerse", "FigVision"),
        ("https://www.npco.ai/", "NP Company", "NPco"),
        ("https://www.recurvia.ai/", "Inephany Ltd (Recurvia)", "recurvia"),
        ("https://staer.ai/", "Staer AB", "Staer"),
    ],
)
def test_verify_match_accepts_name_variants_for_exact_verified_domain(
    identifier, expected_name, returned_name
):
    _verify_match(
        identifier,
        expected_name,
        {"name": returned_name, "domain": _domain_root(identifier)},
    )


def test_verify_match_rejects_name_mismatch_for_name_identifier():
    with pytest.raises(SpecterDisambiguationError):
        _verify_match(
            "Scribe",
            "Scribe",
            {"name": "Shopscribe", "domain": "scribe.com"},
        )


def test_verify_match_accepts_known_corp_suffix_variation():
    _verify_match(
        "mimica.ai",
        "Mimica",
        {"name": "Mimica Automation", "domain": "mimica.ai"},
    )


# ---------------------------------------------------------------------------
# Person → Person dataclass
# ---------------------------------------------------------------------------


def test_person_from_mcp_extracts_education_and_positions():
    raw = {
        "linkedin_url": "https://www.linkedin.com/in/jane",
        "tagline": "CEO @ Acme",
        "about": "Building Acme",
        "location": "San Francisco, California",
        "country": "United States",
        "highlights": ["vc_backed_founder"],
        "seniority": "Executive Level",
        "years_of_experience": 10,
        "education": [
            {
                "school_name": "Stanford",
                "start_date": "2010-09-01",
                "end_date": "2014-06-01",
                "field_of_study": "CS",
                "degree": "BS",
                "top_university": True,
            }
        ],
        "positions": [
            {
                "title": "CEO",
                "company_name": "Acme",
                "start_date": "2020-01-01",
                "end_date": None,
                "is_current": True,
            },
            {
                "title": "Engineer",
                "company_name": "BigCo",
                "start_date": "2015-01-01",
                "end_date": "2019-12-01",
                "is_current": False,
            },
        ],
    }
    person = _person_from_mcp(raw)
    assert person.about == "Building Acme"
    assert person.profile_url == "https://www.linkedin.com/in/jane"
    assert person.city == "San Francisco"
    assert person.country_code == "California"
    assert person.education and person.education[0].institution == "Stanford"
    assert person.experience and len(person.experience) == 2
    assert person.experience[0].company == "Acme"


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------


def test_build_overview_chunk_includes_tech_verticals_and_industry():
    profile = {
        "name": "Acme",
        "short_description": "AI for accountants",
        "founded_year": 2022,
        "growth_stage": "seed",
        "employee_count": 12,
        "hq_location": "San Francisco, USA",
        "web_visits_last_month": 12345,
        "industry": ["Software", "Finance"],
        "tech_verticals": [
            {"vertical": "AI & Machine Learning", "sub_verticals": ["LLMs", "Agents"]},
            {"vertical": "FinTech", "sub_verticals": []},
        ],
        "highlights": ["Web Traffic Surge"],
    }
    chunk = _build_overview_chunk(profile, idx=0)
    assert chunk.page_or_slide == "Company Overview"
    assert "Acme" in chunk.text
    assert "AI & Machine Learning > LLMs" in chunk.text
    assert "Software > Finance" in chunk.text
    assert "FinTech" in chunk.text
    assert "Web Traffic Surge" in chunk.text


def test_build_funding_chunk_extracts_lead_investors():
    fin = {
        "total_funding_amount": 5_000_000,
        "last_funding_amount": 5_000_000,
        "last_funding_date": "2025-01-01",
        "last_funding_type": "Seed",
        "post_money_valuation": 25_000_000,
        "number_of_funding_rounds": 1,
        "number_of_investors": 3,
        "investors": ["A", "B", "C"],
        "funding_rounds": [
            {
                "funding_round_name": "Seed",
                "date": "2025-01-01",
                "raised": 5_000_000,
                "lead_investors_partners": [
                    {"name": "A", "is_lead_investor": True},
                    {"name": "B", "is_lead_investor": False},
                ],
                "pre_money_valuation": 20_000_000,
                "post_money_valuation": 25_000_000,
            }
        ],
    }
    chunk = _build_funding_chunk(fin, "Acme", idx=0)
    assert chunk is not None
    assert "Led by: A" in chunk.text
    assert "$5,000,000" in chunk.text


def test_summarize_headcount_growth_computes_period_ratios():
    history = [
        {"month": f"2025-{m:02d}-01", "total_count": c, "count_by_department": {"Eng": c}}
        for m, c in enumerate([10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55], start=1)
    ]
    lines = _summarize_headcount_growth(history)
    assert any("1mo employee growth" in l for l in lines)
    assert any("3mo employee growth" in l for l in lines)
    # Department mix on most recent month
    assert any("Department mix" in l for l in lines)


def test_build_growth_chunk_returns_none_when_no_data():
    chunk = _build_growth_chunk({}, {}, "Acme", idx=0)
    assert chunk is None


def test_build_signals_chunk_includes_summaries_and_clients():
    intel = {
        "investor_interest_signals": [
            {
                "signal_date": "2025-12-01",
                "signal_score": 8,
                "signal_sources": ["a16z"],
                "summary": "Strong interest from a16z",
                "name": "Acme",
            }
        ],
        "reported_clients": [{"name": "BigCo"}, {"name": "Other"}],
    }
    chunk = _build_signals_chunk(intel, "Acme", idx=0)
    assert chunk is not None
    assert "Strong interest" in chunk.text
    assert "BigCo" in chunk.text


# ---------------------------------------------------------------------------
# fetch_specter_company end-to-end with mocked client
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal SpecterMCPClient shim for tests."""

    def __init__(self, fixtures: dict[str, dict[str, Any]]) -> None:
        self.fixtures = fixtures
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def find_company(self, identifier: str) -> dict[str, Any]:
        self.calls.append(("find_company", {"identifier": identifier}))
        return self.fixtures["find_company"]

    def get_company_profile(self, external_company_id: str) -> dict[str, Any]:
        self.calls.append(("get_company_profile", {"id": external_company_id}))
        return self.fixtures["profile"]

    def get_company_intelligence(self, external_company_id: str) -> dict[str, Any]:
        self.calls.append(("get_company_intelligence", {"id": external_company_id}))
        return self.fixtures["intelligence"]

    def get_company_financials(self, external_company_id: str) -> dict[str, Any]:
        self.calls.append(("get_company_financials", {"id": external_company_id}))
        return self.fixtures["financials"]

    def get_person_profile(self, external_person_id: str) -> dict[str, Any]:
        self.calls.append(("get_person_profile", {"id": external_person_id}))
        return self.fixtures["person_profiles"][external_person_id]


def _anthropic_like_fixtures() -> dict[str, Any]:
    return {
        "find_company": {
            "external_company_id": "61643d92c3c073075bcb8983",
            "name": "Anthropic",
            "domain": "anthropic.com",
        },
        "profile": {
            "external_company_id": "61643d92c3c073075bcb8983",
            "name": "Anthropic",
            "domain": "anthropic.com",
            "short_description": "AI safety lab",
            "industry": ["Professional Services", "Research"],
            "tech_verticals": [{"vertical": "AI & Machine Learning", "sub_verticals": ["LLMs"]}],
            "founded_year": 2021,
            "employee_count": 4704,
            "growth_stage": "late",
            "hq_location": "San Francisco, USA",
            "web_visits_last_month": 17_211_098,
        },
        "intelligence": {
            "external_company_id": "61643d92c3c073075bcb8983",
            "headcount_by_department": [
                {"month": "2025-01-01", "total_count": 100, "count_by_department": {"Engineering": 50}},
                {"month": "2025-02-01", "total_count": 110, "count_by_department": {"Engineering": 55}},
                {"month": "2025-03-01", "total_count": 120, "count_by_department": {"Engineering": 60}},
                {"month": "2025-04-01", "total_count": 130, "count_by_department": {"Engineering": 65}},
            ],
            "investor_interest_signals": [
                {"signal_date": "2025-04-01", "signal_score": 9, "summary": "Hot",
                 "name": "Anthropic", "signal_sources": ["X"]}
            ],
            "reported_clients": [{"name": "BigCo"}],
            "founders": [
                {
                    "external_person_id": "per_dario",
                    "full_name": "Dario Amodei",
                    "title": "CEO",
                    "linkedin_url": "https://www.linkedin.com/in/dario",
                    "departments": ["Senior Leadership"],
                    "seniority": "Executive Level",
                    "is_founder": True,
                }
            ],
            "highlights": ["Web Traffic Surge"],
        },
        "financials": {
            "total_funding_amount": 1_000_000_000,
            "number_of_funding_rounds": 5,
            "number_of_investors": 20,
            "investors": ["A", "B"],
            "last_funding_date": "2025-01-01",
            "last_funding_amount": 500_000_000,
            "last_funding_type": "Series E",
            "post_money_valuation": 50_000_000_000,
            "funding_rounds": [
                {
                    "funding_round_name": "Series A",
                    "date": "2022-01-01",
                    "raised": 100_000_000,
                    "lead_investors_partners": [{"name": "A", "is_lead_investor": True}],
                    "pre_money_valuation": 400_000_000,
                    "post_money_valuation": 500_000_000,
                }
            ],
        },
        "person_profiles": {
            "per_dario": {
                "linkedin_url": "https://www.linkedin.com/in/dario",
                "tagline": "AI safety researcher",
                "about": "Building safe AI",
                "location": "San Francisco, California",
                "highlights": ["vc_backed_founder"],
                "seniority": "Executive Level",
                "years_of_experience": 12,
                "education": [
                    {"school_name": "Princeton", "start_date": "2002", "end_date": "2006",
                     "degree": "PhD", "field_of_study": "Physics"}
                ],
                "positions": [
                    {"title": "CEO", "company_name": "Anthropic",
                     "start_date": "2021-01-01", "end_date": None, "is_current": True}
                ],
            }
        },
    }


def test_fetch_specter_company_builds_company_and_chunks():
    fakes = _anthropic_like_fixtures()
    client = _FakeClient(fakes)
    company, store = fetch_specter_company(
        "anthropic.com",
        expected_name="Anthropic",
        client=client,  # type: ignore[arg-type]
    )

    assert isinstance(company, Company)
    assert company.name == "Anthropic"
    assert company.domain == "anthropic.com"
    assert company.team and len(company.team) == 1
    assert company.team[0].name == "Dario Amodei"
    assert company.team[0].profile_url == "https://www.linkedin.com/in/dario"

    assert isinstance(store, EvidenceStore)
    assert store.startup_slug == "anthropic"
    sections = [c.page_or_slide for c in store.chunks]
    assert "Company Overview" in sections
    assert "Funding & Investors" in sections
    assert "Growth Metrics" in sections
    assert "Investor Interest & Reported Clients" in sections
    assert "Founding Team Overview" in sections
    assert any(s.startswith("Team Member: ") for s in sections)


def test_fetch_specter_company_raises_on_disambiguation_failure():
    fakes = _anthropic_like_fixtures()
    fakes["find_company"] = {
        "external_company_id": "shopscribe-id",
        "name": "Shopscribe",
        "domain": "shopscribe.com",
    }
    client = _FakeClient(fakes)
    with pytest.raises(SpecterDisambiguationError):
        fetch_specter_company("scribe.com", client=client)  # type: ignore[arg-type]


def test_fetch_specter_company_continues_when_person_profile_fails():
    fakes = _anthropic_like_fixtures()

    class _BadPerson(_FakeClient):
        def get_person_profile(self, external_person_id: str) -> dict[str, Any]:
            raise SpecterMCPError("simulated person 500")

    client = _BadPerson(fakes)
    company, store = fetch_specter_company(
        "anthropic.com",
        client=client,  # type: ignore[arg-type]
    )
    # Founder still listed (from intelligence call), just without deeper profile data.
    assert company.team and company.team[0].name == "Dario Amodei"
    assert any(c.page_or_slide.startswith("Team Member: Dario") for c in store.chunks)


# ---------------------------------------------------------------------------
# Refresh-token persistence (Supabase-backed rotation chain)
# ---------------------------------------------------------------------------


def test_token_manager_prefers_persisted_refresh_token_over_env(monkeypatch):
    """When Supabase has a rotated refresh token, _TokenManager uses it instead
    of the env-supplied bootstrap value. This is what saves us across deploys."""
    from agent.ingest import specter_mcp_client as mcp_mod

    monkeypatch.setattr(
        mcp_mod, "_load_persisted_refresh_token", lambda: "ROTATED-REFRESH-TOKEN"
    )
    creds = mcp_mod._SpecterCredentials(
        mcp_url="https://example/mcp",
        token_endpoint="https://example/token",
        client_id="cid",
        client_secret="csec",
        refresh_token="ENV-BOOTSTRAP-TOKEN",
    )
    tm = mcp_mod._TokenManager(creds)
    assert tm._creds.refresh_token == "ROTATED-REFRESH-TOKEN"


def test_token_manager_falls_back_to_env_when_supabase_empty(monkeypatch):
    from agent.ingest import specter_mcp_client as mcp_mod

    monkeypatch.setattr(mcp_mod, "_load_persisted_refresh_token", lambda: None)
    creds = mcp_mod._SpecterCredentials(
        mcp_url="https://example/mcp",
        token_endpoint="https://example/token",
        client_id="cid",
        client_secret="csec",
        refresh_token="ENV-BOOTSTRAP-TOKEN",
    )
    tm = mcp_mod._TokenManager(creds)
    assert tm._creds.refresh_token == "ENV-BOOTSTRAP-TOKEN"


def test_refresh_persists_rotated_token(monkeypatch):
    """Successful refresh with a rotated refresh_token persists it via CAS."""
    from agent.ingest import specter_mcp_client as mcp_mod

    monkeypatch.setattr(mcp_mod, "_load_persisted_refresh_token", lambda: None)

    persisted: list[tuple[str, str | None]] = []

    def _fake_cas(new_value, prev_value):
        persisted.append((new_value, prev_value))
        return True

    monkeypatch.setattr(mcp_mod, "_persist_rotated_refresh_token", _fake_cas)

    # Stub urllib.request.urlopen to return a token-endpoint payload with a rotated token.
    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, *exc):  # noqa: D401
            return False

        def read(self) -> bytes:
            return self._body

    def _fake_urlopen(req, timeout=20):
        return _FakeResp(
            b'{"access_token":"new-access","expires_in":600,"refresh_token":"NEW-ROTATED-TOKEN"}'
        )

    monkeypatch.setattr(mcp_mod.urllib.request, "urlopen", _fake_urlopen)

    creds = mcp_mod._SpecterCredentials(
        mcp_url="https://example/mcp",
        token_endpoint="https://example/token",
        client_id="cid",
        client_secret="csec",
        refresh_token="OLD-TOKEN",
    )
    tm = mcp_mod._TokenManager(creds)
    token = tm.get_access_token()
    assert token == "new-access"
    assert tm._creds.refresh_token == "NEW-ROTATED-TOKEN"
    assert persisted == [("NEW-ROTATED-TOKEN", "OLD-TOKEN")]


def test_refresh_does_not_persist_when_token_unchanged(monkeypatch):
    """Some OAuth servers don't rotate. We should NOT call set_mcp_secret in that case."""
    from agent.ingest import specter_mcp_client as mcp_mod

    monkeypatch.setattr(mcp_mod, "_load_persisted_refresh_token", lambda: None)
    persisted: list[str] = []
    monkeypatch.setattr(
        mcp_mod, "_persist_refresh_token", lambda v: persisted.append(v)
    )

    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return self._body

    def _fake_urlopen(req, timeout=20):
        # Same refresh_token as the bootstrap → no rotation
        return _FakeResp(
            b'{"access_token":"new-access","expires_in":600,"refresh_token":"BOOTSTRAP"}'
        )

    monkeypatch.setattr(mcp_mod.urllib.request, "urlopen", _fake_urlopen)

    creds = mcp_mod._SpecterCredentials(
        mcp_url="https://example/mcp",
        token_endpoint="https://example/token",
        client_id="cid",
        client_secret="csec",
        refresh_token="BOOTSTRAP",
    )
    tm = mcp_mod._TokenManager(creds)
    tm.get_access_token()
    assert persisted == []


# ---------------------------------------------------------------------------
# Live integration test (only runs when refresh token is set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("SPECTER_MCP_REFRESH_TOKEN"),
    reason="SPECTER_MCP_REFRESH_TOKEN not set; skipping live MCP test",
)
def test_live_fetch_anthropic():
    """Hits the real Specter MCP. Manual run only."""
    company, store = fetch_specter_company("anthropic.com", expected_name="Anthropic")
    assert company.name and "Anthropic" in company.name
    assert company.domain == "anthropic.com"
    assert any("Funding" in c.page_or_slide for c in store.chunks)


# ---------------------------------------------------------------------------
# Refresh-token rotation: compare-and-swap single-owner semantics (Sprint 2 S2)
# ---------------------------------------------------------------------------

from agent.ingest.specter_mcp_client import (  # noqa: E402
    _REFRESH_LOCK_SECRET_KEY,
    _REFRESH_TOKEN_SECRET_KEY,
    _SpecterCredentials,
    _AuthExpired,
    _TokenManager,
)


class FakeSecretStore:
    """In-memory stand-in for the mcp_secrets table with CAS semantics."""

    def __init__(self):
        self.values: dict[str, str] = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value
        return True

    def cas(self, key, new_value, expected_current):
        current = self.values.get(key)
        if current is None:
            self.values[key] = new_value
            return True
        if expected_current is not None and current == expected_current:
            self.values[key] = new_value
            return True
        return False


@pytest.fixture()
def secret_store(monkeypatch):
    from web import db as web_db

    store = FakeSecretStore()
    monkeypatch.setattr(web_db, "get_mcp_secret", store.get)
    monkeypatch.setattr(web_db, "set_mcp_secret", store.set)
    monkeypatch.setattr(web_db, "compare_and_swap_mcp_secret", store.cas)
    return store


def _creds(refresh_token="env0"):
    return _SpecterCredentials(
        mcp_url="https://mcp.example/mcp",
        token_endpoint="https://mcp.example/token",
        client_id="client",
        client_secret=None,
        refresh_token=refresh_token,
    )


def _token_payload(access, refresh):
    return {"access_token": access, "refresh_token": refresh, "expires_in": 600}


def test_boot_race_late_refresher_adopts_latest_persisted_chain(secret_store):
    """Two processes booting from the same env token: a later refresher should
    adopt the current persisted chain before asking Specter for a token."""
    mgr_a = _TokenManager(_creds())
    mgr_b = _TokenManager(_creds())
    tokens_requested_by_b: list[str] = []
    mgr_a._request_token = lambda rt: _token_payload("at-a", "A1")

    def request_b(rt):
        tokens_requested_by_b.append(rt)
        return _token_payload("at-b", "B1")

    mgr_b._request_token = request_b

    assert mgr_a.get_access_token() == "at-a"
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "A1"

    assert mgr_b.get_access_token() == "at-b"
    assert tokens_requested_by_b == ["A1"]
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "B1"
    assert mgr_b._creds.refresh_token == "B1"


def test_invalid_grant_reloads_persisted_token_and_retries(secret_store):
    """A stale in-memory token must re-read the persisted chain once before
    failing — another process may have rotated while we were idle."""
    mgr = _TokenManager(_creds("env0"))  # boots before any rotation
    secret_store.values[_REFRESH_TOKEN_SECRET_KEY] = "GOOD"  # rotated elsewhere

    def fake_request(rt):
        if rt == "env0":
            raise SpecterMCPError("Specter token refresh failed (400): invalid_grant")
        assert rt == "GOOD"
        return _token_payload("at-new", "GOOD2")

    mgr._request_token = fake_request
    assert mgr.get_access_token() == "at-new"
    # The retried rotation won the CAS from GOOD -> GOOD2.
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "GOOD2"
    assert mgr._creds.refresh_token == "GOOD2"


def test_invalid_grant_reloads_persisted_token_rotated_during_refresh(secret_store):
    """If a sibling process rotates after we choose candidates but before our
    request returns, retry with the newly-persisted winner."""
    mgr = _TokenManager(_creds("env0"))
    requested: list[str] = []

    def fake_request(rt):
        requested.append(rt)
        if rt == "env0":
            secret_store.values[_REFRESH_TOKEN_SECRET_KEY] = "GOOD"
            raise SpecterMCPError("Specter token refresh failed (400): invalid_grant")
        assert rt == "GOOD"
        return _token_payload("at-new", "GOOD2")

    mgr._request_token = fake_request
    assert mgr.get_access_token() == "at-new"
    assert requested == ["env0", "GOOD"]
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "GOOD2"
    assert mgr._creds.refresh_token == "GOOD2"


def test_refresh_uses_current_persisted_token_before_stale_in_memory_token(secret_store):
    """A long-lived process should not burn a known-stale in-memory refresh
    token when Supabase already has a newer chain."""
    mgr = _TokenManager(_creds("env0"))
    secret_store.values[_REFRESH_TOKEN_SECRET_KEY] = "GOOD"
    requested: list[str] = []

    def fake_request(rt):
        requested.append(rt)
        if rt == "env0":
            raise SpecterMCPError("Specter token refresh failed (400): invalid_grant")
        assert rt == "GOOD"
        return _token_payload("at-new", "GOOD2")

    mgr._request_token = fake_request
    assert mgr.get_access_token() == "at-new"
    assert requested == ["GOOD"]
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "GOOD2"
    assert mgr._creds.refresh_token == "GOOD2"


def test_invalid_persisted_token_can_recover_from_env_bootstrap(secret_store):
    """If the Supabase secret row is stale but Railway env was repaired, use
    the env bootstrap token and replace the stale persisted row via CAS."""
    secret_store.values[_REFRESH_TOKEN_SECRET_KEY] = "BAD-DB"
    mgr = _TokenManager(_creds("GOOD-ENV"))
    requested: list[str] = []

    def fake_request(rt):
        requested.append(rt)
        if rt == "BAD-DB":
            raise SpecterMCPError("Specter token refresh failed (400): invalid_grant")
        assert rt == "GOOD-ENV"
        return _token_payload("at-env", "ENV2")

    mgr._request_token = fake_request
    assert mgr.get_access_token() == "at-env"
    assert requested == ["BAD-DB", "GOOD-ENV"]
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "ENV2"
    assert mgr._creds.refresh_token == "ENV2"


def test_invalid_grant_without_alternative_token_raises(secret_store):
    mgr = _TokenManager(_creds("env0"))

    def fake_request(rt):
        raise SpecterMCPError("Specter token refresh failed (400): invalid_grant")

    mgr._request_token = fake_request
    with pytest.raises(SpecterMCPError):
        mgr.get_access_token()


def test_env_bootstrap_without_persisted_row_seeds_chain(secret_store):
    """First-ever rotation (no row in mcp_secrets) must seed the chain."""
    mgr = _TokenManager(_creds("env0"))
    mgr._request_token = lambda rt: _token_payload("at", "FIRST")
    mgr.get_access_token()
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "FIRST"


def test_refresh_lock_blocks_second_process_from_burning_single_use_token(
    secret_store,
    monkeypatch,
):
    """If another process is refreshing, fail closed instead of spending the
    same single-use refresh token and orphaning the chain."""
    from agent.ingest import specter_mcp_client as mcp_mod

    monkeypatch.setenv("SUPABASE_URL", "https://supabase.example")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setattr(mcp_mod, "_REFRESH_LOCK_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(mcp_mod, "_REFRESH_LOCK_POLL_SECONDS", 0.001)
    secret_store.values[_REFRESH_LOCK_SECRET_KEY] = mcp_mod._refresh_lock_payload(
        "other-process",
        mcp_mod.time.time() + 60,
    )

    mgr = _TokenManager(_creds("env0"))
    requested: list[str] = []
    mgr._request_token = lambda rt: requested.append(rt) or _token_payload("at", "NEXT")

    with pytest.raises(SpecterMCPError, match="refresh lock"):
        mgr.get_access_token()

    assert requested == []
    assert _REFRESH_TOKEN_SECRET_KEY not in secret_store.values


def test_refresh_lock_releases_after_success(secret_store, monkeypatch):
    from agent.ingest import specter_mcp_client as mcp_mod

    monkeypatch.setenv("SUPABASE_URL", "https://supabase.example")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")

    mgr = _TokenManager(_creds("env0"))
    mgr._request_token = lambda rt: _token_payload("at", "NEXT")

    assert mgr.get_access_token() == "at"
    assert secret_store.values[_REFRESH_TOKEN_SECRET_KEY] == "NEXT"
    assert mcp_mod._refresh_lock_expired(secret_store.values[_REFRESH_LOCK_SECRET_KEY])


def test_boot_prefers_persisted_token_over_env(secret_store):
    secret_store.values[_REFRESH_TOKEN_SECRET_KEY] = "LIVE"
    mgr = _TokenManager(_creds("env0"))
    assert mgr._creds.refresh_token == "LIVE"


def test_compare_and_swap_returns_false_without_db_client(monkeypatch):
    from web import db as web_db

    monkeypatch.setattr(web_db, "_get_client", lambda: None)
    assert web_db.compare_and_swap_mcp_secret("k", "v", "prev") is False


def test_call_tool_resets_mcp_session_after_auth_expiry(monkeypatch):
    creds = _creds("env0")
    client = SpecterMCPClient(creds)
    client._initialized = True
    client._session_id = "stale-session"
    refresh_calls: list[bool] = []
    raw_calls: list[tuple[str, str | None]] = []

    def fake_refresh(force_refresh=False):
        refresh_calls.append(force_refresh)
        return "fresh-access"

    client._tokens.get_access_token = fake_refresh

    def fake_raw_request(method, params):
        raw_calls.append((method, client._session_id))
        if method == "tools/call" and client._session_id == "stale-session":
            raise _AuthExpired()
        if method == "initialize":
            return {}
        return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    client._raw_request = fake_raw_request

    assert client._call_tool("find_company", {"identifier": "anthropic.com"}) == {"ok": True}
    assert refresh_calls == [True]
    assert raw_calls == [
        ("tools/call", "stale-session"),
        ("initialize", None),
        ("tools/call", None),
    ]
