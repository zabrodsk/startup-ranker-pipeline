"""Sprint 3 (W7): RDI_SPECTER_IDENTITY_REUSE reuses the preflight identity.

Preflight (flag on) writes the resolved Specter company id/name/domain back
onto the intake items; the batch worker plumbs the id to the child via
--specter-company-id; fetch_specter_company(known_company_id=...) skips
find_company + match verification and falls back to full resolution when the
id is stale. Flag off (default): preflight resolves and discards — intake
items stay untouched.
"""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.dataclasses.company import Company
from agent.ingest.specter_mcp_client import (
    SpecterMCPError,
    fetch_specter_company,
)
from agent.specter_batch_worker import _build_company_tasks, _normalize_specter_urls

FLAG_ENV = "RDI_SPECTER_IDENTITY_REUSE"
KNOWN_ID = "spc-12345"


class _FakeClient:
    """Minimal SpecterMCPClient shim recording calls."""

    def __init__(self, stale_ids: set[str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.stale_ids = stale_ids or set()

    def find_company(self, identifier: str) -> dict[str, Any]:
        self.calls.append(("find_company", identifier))
        return {
            "external_company_id": KNOWN_ID,
            "name": "Anthropic",
            "domain": "anthropic.com",
        }

    def get_company_profile(self, company_id: str) -> dict[str, Any]:
        self.calls.append(("get_company_profile", company_id))
        if company_id in self.stale_ids:
            raise SpecterMCPError(f"Unknown company id: {company_id}")
        return {
            "external_company_id": company_id,
            "name": "Anthropic",
            "domain": "anthropic.com",
            "short_description": "AI safety lab",
            "industry": ["Research"],
        }

    def get_company_intelligence(self, company_id: str) -> dict[str, Any]:
        self.calls.append(("get_company_intelligence", company_id))
        return {"founders": []}

    def get_company_financials(self, company_id: str) -> dict[str, Any]:
        self.calls.append(("get_company_financials", company_id))
        return {}


def _calls(client: _FakeClient, name: str) -> list[tuple[str, str]]:
    return [c for c in client.calls if c[0] == name]


# --- fetch_specter_company fast path -----------------------------------------


def test_known_company_id_skips_find_company():
    client = _FakeClient()
    company, _store = fetch_specter_company(
        "anthropic.com",
        known_company_id=KNOWN_ID,
        client=client,  # type: ignore[arg-type]
    )

    assert _calls(client, "find_company") == []
    assert _calls(client, "get_company_profile") == [("get_company_profile", KNOWN_ID)]
    assert company.name == "Anthropic"
    assert company.specter_company_id == KNOWN_ID


def test_stale_known_id_falls_back_to_full_resolution():
    client = _FakeClient(stale_ids={"stale-id"})
    company, _store = fetch_specter_company(
        "anthropic.com",
        known_company_id="stale-id",
        client=client,  # type: ignore[arg-type]
    )

    assert _calls(client, "find_company") == [("find_company", "anthropic.com")]
    assert _calls(client, "get_company_profile") == [
        ("get_company_profile", "stale-id"),
        ("get_company_profile", KNOWN_ID),
    ]
    assert company.name == "Anthropic"
    assert company.specter_company_id == KNOWN_ID


def test_stale_id_without_known_flag_still_raises():
    """Profile failures on the normal path must not be silently re-resolved."""
    client = _FakeClient(stale_ids={KNOWN_ID})
    with pytest.raises(SpecterMCPError):
        fetch_specter_company("anthropic.com", client=client)  # type: ignore[arg-type]
    assert len(_calls(client, "find_company")) == 1


def test_without_known_id_behavior_is_unchanged():
    client = _FakeClient()
    company, _store = fetch_specter_company(
        "anthropic.com",
        client=client,  # type: ignore[arg-type]
    )
    assert _calls(client, "find_company") == [("find_company", "anthropic.com")]
    assert company.specter_company_id == KNOWN_ID


# --- batch-worker task plumbing -----------------------------------------------


def test_normalize_specter_urls_carries_and_tolerates_missing_id():
    run_config = {
        "specter_urls": [
            {"url": "https://anthropic.com", "name": "Anthropic",
             "specter_company_id": KNOWN_ID},
            {"url": "https://example.com", "name": "Example"},
            "https://plain-string.com",
        ]
    }
    tasks = _normalize_specter_urls(run_config)
    assert [t["specter_company_id"] for t in tasks] == [KNOWN_ID, "", ""]


def test_build_company_tasks_keeps_specter_company_id_on_url_tasks():
    run_config = {
        "specter_urls": [
            {"url": "https://anthropic.com", "name": "Anthropic",
             "specter_company_id": KNOWN_ID}
        ]
    }
    tasks = _build_company_tasks(run_config, companies_csv=None)
    assert len(tasks) == 1
    assert tasks[0]["mode"] == "url"
    assert tasks[0]["specter_company_id"] == KNOWN_ID


# --- preflight enrichment (web/app.py) -----------------------------------------


def _run_preflight(monkeypatch, items: list[Any], *, flag_on: bool) -> list[Any]:
    import web.app as app

    if flag_on:
        monkeypatch.setenv(FLAG_ENV, "on")
    else:
        monkeypatch.delenv(FLAG_ENV, raising=False)

    company = Company(
        name="Anthropic",
        domain="anthropic.com",
        specter_company_id=KNOWN_ID,
    )

    async def fake_resolve(item):
        identifier = app._url_value_for_intake_item(item)
        if "unresolvable" in identifier:
            return None, "Specter outage"
        return company, None

    monkeypatch.setattr(app, "_resolve_specter_url_for_preflight", fake_resolve)

    job_id = "test-w7-preflight"
    app._results_cache[job_id] = {
        "upload_dir": "/tmp/does-not-exist",
        "files": [],
        "specter": {},
        "specter_urls": list(items),
    }
    report = app._quality_report(job_id, "specter")
    try:
        asyncio.run(
            app._preflight_specter_inputs(
                job_id, SimpleNamespace(input_mode="specter"), report
            )
        )
        return list(app._results_cache[job_id]["specter_urls"])
    finally:
        app._results_cache.pop(job_id, None)


def test_preflight_flag_off_leaves_items_untouched(monkeypatch):
    items = ["https://anthropic.com", {"url": "https://anthropic.com/x", "name": "Anthropic"}]
    out = _run_preflight(monkeypatch, items, flag_on=False)
    assert out == items


def test_preflight_flag_on_converts_string_items_to_enriched_dicts(monkeypatch):
    out = _run_preflight(monkeypatch, ["https://anthropic.com"], flag_on=True)
    assert out == [
        {
            "url": "https://anthropic.com",
            "specter_company_id": KNOWN_ID,
            "resolved_name": "Anthropic",
            "resolved_domain": "anthropic.com",
        }
    ]


def test_preflight_flag_on_enriches_dict_items_preserving_keys(monkeypatch):
    out = _run_preflight(
        monkeypatch,
        [{"url": "https://anthropic.com", "name": "Anthropic"}],
        flag_on=True,
    )
    assert out == [
        {
            "url": "https://anthropic.com",
            "name": "Anthropic",
            "specter_company_id": KNOWN_ID,
            "resolved_name": "Anthropic",
            "resolved_domain": "anthropic.com",
        }
    ]


def test_preflight_flag_on_keeps_failed_items_unchanged_in_place(monkeypatch):
    items = ["https://unresolvable.example", "https://anthropic.com"]
    out = _run_preflight(monkeypatch, items, flag_on=True)
    assert out[0] == "https://unresolvable.example"
    assert out[1]["specter_company_id"] == KNOWN_ID


def test_identity_reuse_flag_defaults_off_and_warns_once_on_invalid(monkeypatch, caplog):
    import logging

    import web.app as app

    monkeypatch.delenv(FLAG_ENV, raising=False)
    assert app._specter_identity_reuse_enabled() is False

    monkeypatch.setenv(FLAG_ENV, "bogus")
    monkeypatch.setattr(app, "_WARNED_INVALID_IDENTITY_REUSE", set())
    with caplog.at_level(logging.WARNING, logger="web.app"):
        assert app._specter_identity_reuse_enabled() is False
        assert app._specter_identity_reuse_enabled() is False
    warnings = [r for r in caplog.records if FLAG_ENV in r.getMessage()]
    assert len(warnings) == 1
