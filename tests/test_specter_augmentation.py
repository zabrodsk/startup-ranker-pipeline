"""Tests for `agent.ingest.specter_augmentation`.

All tests are network-free. The MCP fetch is mocked via `monkeypatch.setattr`.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent.dataclasses.company import Company
from agent.ingest.specter_augmentation import (
    augment_with_specter,
    extract_company_url,
)
from agent.ingest.specter_mcp_client import (
    SpecterDisambiguationError,
    SpecterMCPError,
)
from agent.ingest.store import Chunk, EvidenceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(idx: int, text: str, source: str = "deck.pdf") -> Chunk:
    return Chunk(chunk_id=f"chunk_{idx}", text=text, source_file=source, page_or_slide=idx)


def _store(slug: str, *texts: str) -> EvidenceStore:
    return EvidenceStore(
        startup_slug=slug,
        chunks=[_chunk(i, t) for i, t in enumerate(texts)],
    )


# ---------------------------------------------------------------------------
# extract_company_url
# ---------------------------------------------------------------------------


def test_extract_url_picks_most_frequent_bare_domain():
    store = _store(
        "acme",
        "Acme is a SaaS for accountants. acme.com is the primary site.",
        "Reach us at acme.com/contact.",
        "Partner: foo.io",
    )
    assert extract_company_url(store) == "acme.com"


def test_extract_url_strips_scheme_and_www():
    store = _store("acme", "Visit https://www.acme.com/about for the team page.")
    assert extract_company_url(store) == "acme.com"


def test_extract_url_skips_blocklisted():
    store = _store(
        "x",
        "Follow us on linkedin.com/company/foo and twitter.com/foo.",
        "Email hello@gmail.com or join slack.com workspace.",
    )
    assert extract_company_url(store) is None


def test_extract_url_handles_labeled_pattern_with_3x_bonus():
    """`Website: foo.com` should beat raw `crunchbase.com` mentions even at higher frequency."""
    store = _store(
        "foo",
        "As covered on crunchbase.com — also crunchbase.com/foo and crunchbase.com/foo/team.",
        "Website: foo.com",
    )
    # crunchbase.com is blocklisted; even if it weren't, the labeled foo.com
    # carries a 3x score multiplier. This test specifically exercises the
    # labeled-pattern path, not the blocklist.
    assert extract_company_url(store) == "foo.com"


def test_extract_url_returns_none_when_no_chunks():
    assert extract_company_url(EvidenceStore(startup_slug="x", chunks=[])) is None
    assert extract_company_url(_store("x")) is None  # no text


def test_extract_url_only_scans_first_n_chunks():
    chunks = [_chunk(i, "irrelevant filler text with no domains") for i in range(10)]
    chunks.append(_chunk(10, "Here is the actual URL: latecomer.com"))
    store = EvidenceStore(startup_slug="x", chunks=chunks)
    # Default max_chunks=10 means index 10 is excluded.
    assert extract_company_url(store, max_chunks=10) is None
    # Bumping max_chunks finds it.
    assert extract_company_url(store, max_chunks=11) == "latecomer.com"


def test_extract_url_position_bonus_applies_to_first_three_chunks():
    """A domain that only appears once but in chunk 0 should beat one that
    appears once in chunk 5 (2x positional bonus)."""
    store = _store(
        "x",
        "Acme is at early.com",
        "filler",
        "filler",
        "filler",
        "filler",
        "Other reference: late.com",
    )
    assert extract_company_url(store) == "early.com"


# ---------------------------------------------------------------------------
# augment_with_specter
# ---------------------------------------------------------------------------


def _mcp_company(name: str = "Acme", domain: str = "acme.com") -> Company:
    return Company(name=name, domain=domain)


def _mcp_store(*texts: str) -> EvidenceStore:
    return EvidenceStore(
        startup_slug="acme",
        chunks=[
            Chunk(chunk_id=f"chunk_{i}", text=t, source_file="specter-mcp", page_or_slide=f"section-{i}")
            for i, t in enumerate(texts)
        ],
    )


def test_augment_returns_unchanged_when_no_url(monkeypatch):
    deck = _store("x", "no domains in this deck text.")
    calls: list[Any] = []
    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company",
        lambda *a, **kw: calls.append((a, kw)) or (_mcp_company(), _mcp_store("x")),
    )
    logs: list[str] = []
    out_store, out_company = augment_with_specter(
        deck, slug="x", on_log=logs.append
    )
    assert out_store is deck  # exact identity — no mutation
    assert out_company is None
    assert calls == []  # MCP fetch never called
    assert any("no company URL" in m for m in logs)


def test_augment_calls_fetch_with_extracted_url_and_kwargs(monkeypatch):
    deck = _store(
        "acme-deck",
        "Acme — pitch deck. Website: acme.com",
        "Founded 2023.",
    )
    captured: dict[str, Any] = {}

    def _fake_fetch(identifier, *, expected_name=None, fetch_full_team=True, client=None):
        captured["identifier"] = identifier
        captured["expected_name"] = expected_name
        captured["fetch_full_team"] = fetch_full_team
        return _mcp_company(), _mcp_store("Company: Acme — SaaS")

    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company", _fake_fetch
    )
    augment_with_specter(
        deck,
        slug="acme",
        expected_name="Acme",
        fetch_full_team=False,
    )
    assert captured["identifier"] == "acme.com"
    assert captured["expected_name"] == "Acme"
    assert captured["fetch_full_team"] is False


def test_augment_merges_chunks_and_renumbers(monkeypatch):
    deck = _store("acme", "deck chunk 0 — acme.com", "deck chunk 1")
    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company",
        lambda *a, **kw: (
            _mcp_company(),
            _mcp_store("mcp chunk A", "mcp chunk B", "mcp chunk C"),
        ),
    )
    out_store, out_company = augment_with_specter(deck, slug="acme")
    assert out_company is not None and out_company.name == "Acme"
    # 2 deck chunks + 3 MCP chunks = 5 total
    assert len(out_store.chunks) == 5
    # IDs are sequential and unique
    ids = [c.chunk_id for c in out_store.chunks]
    assert ids == [f"chunk_{i}" for i in range(5)]
    # Order preserved: deck chunks first, then MCP chunks
    assert "deck chunk 0" in out_store.chunks[0].text
    assert "mcp chunk A" in out_store.chunks[2].text


def test_augment_returns_unchanged_on_disambiguation_error(monkeypatch):
    deck = _store("scribe", "Scribe pitch deck. scribe.com is our domain.")

    def _raises(*a, **kw):
        raise SpecterDisambiguationError(
            "Specter resolved 'scribe.com' to 'Shopscribe' — domain root mismatch."
        )

    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company", _raises
    )
    logs: list[str] = []
    out_store, out_company = augment_with_specter(
        deck, slug="scribe", on_log=logs.append
    )
    assert out_store is deck
    assert out_company is None
    assert any("wrong company" in m.lower() for m in logs)


def test_augment_returns_unchanged_on_mcp_error(monkeypatch):
    deck = _store("acme", "Visit acme.com")

    def _raises(*a, **kw):
        raise SpecterMCPError("simulated 503 from MCP server")

    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company", _raises
    )
    logs: list[str] = []
    out_store, out_company = augment_with_specter(
        deck, slug="acme", on_log=logs.append
    )
    assert out_store is deck
    assert out_company is None
    assert any("MCP failure" in m for m in logs)


def test_augment_swallows_unexpected_exceptions(monkeypatch):
    """Helper must NEVER raise — even on bugs deeper in the stack."""
    deck = _store("acme", "Visit acme.com")

    def _explodes(*a, **kw):
        raise RuntimeError("totally unexpected — should not break the pipeline")

    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company", _explodes
    )
    logs: list[str] = []
    out_store, out_company = augment_with_specter(
        deck, slug="acme", on_log=logs.append
    )
    assert out_store is deck
    assert out_company is None
    assert any("unexpected error" in m for m in logs)


def test_augment_logs_each_branch_at_least_once(monkeypatch):
    """Every successful branch should produce at least one log line for operator visibility."""
    deck = _store("acme", "Acme deck — acme.com")
    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company",
        lambda *a, **kw: (_mcp_company(), _mcp_store("overview chunk")),
    )
    logs: list[str] = []
    out_store, out_company = augment_with_specter(
        deck, slug="acme", on_log=logs.append
    )
    # Expect: extracted-URL line + resolved line
    assert any("extracted URL" in m for m in logs)
    assert any("resolved to" in m for m in logs)
    assert out_company is not None


def test_augment_handles_log_callback_exceptions(monkeypatch):
    """A buggy on_log callback must not crash the helper."""
    deck = _store("acme", "acme.com")
    monkeypatch.setattr(
        "agent.ingest.specter_augmentation.fetch_specter_company",
        lambda *a, **kw: (_mcp_company(), _mcp_store("x")),
    )

    def _broken_log(_msg: str) -> None:
        raise ValueError("logger blew up")

    out_store, out_company = augment_with_specter(
        deck, slug="acme", on_log=_broken_log
    )
    # Helper still completes normally despite the broken logger.
    assert out_company is not None
    assert len(out_store.chunks) == 2  # 1 deck + 1 mcp
