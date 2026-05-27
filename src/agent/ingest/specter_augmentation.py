"""Augment a pitch-deck-derived EvidenceStore with Specter MCP company data.

When a user uploads a pitch deck in `input_mode=pitchdeck`, this module is the
glue between the deck's free-form text and the structured Specter MCP path
already built for URL-only intake. The flow:

1. ``extract_company_url`` scans the first ~10 chunks of the deck for a likely
   company URL (regex + scoring + blocklist; see :func:`extract_company_url`).
2. If a URL is found, ``augment_with_specter`` calls the existing
   :func:`agent.ingest.specter_mcp_client.fetch_specter_company` plumbing to
   fetch profile/intelligence/financials and build MCP-derived chunks.
3. The MCP chunks are merged into the deck's :class:`EvidenceStore` (chunk IDs
   are renumbered globally) and the MCP :class:`Company` is returned so the
   evaluation pipeline can use it as canonical metadata.

The helper never raises: any failure (no URL, MCP error, disambiguation
rejection) silently falls back to the deck-only store + ``None`` company,
preserving the existing pitch-deck behaviour.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from agent.dataclasses.company import Company
from agent.ingest.specter_mcp_client import (
    SpecterCompanyNotFoundError,
    SpecterDisambiguationError,
    SpecterMCPError,
    fetch_specter_company,
)
from agent.ingest.store import Chunk, EvidenceStore

# ---------------------------------------------------------------------------
# Domain blocklist — never resolve these to a "company URL".
# ---------------------------------------------------------------------------

# Common social, file-hosting, productivity, and infra domains that pitch decks
# routinely mention but which never represent THIS company. The blocklist is
# matched against the registered domain (e.g. ``linkedin.com``), so subdomains
# like ``app.linkedin.com`` are caught too.
_BLOCKLIST_DOMAINS: frozenset[str] = frozenset(
    {
        # Social / professional networks
        "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
        "youtube.com", "youtu.be", "tiktok.com", "medium.com", "substack.com",
        "github.com", "gitlab.com", "bitbucket.org",
        # Productivity / docs (often referenced in decks but not the company)
        "slack.com", "notion.so", "notion.site", "calendly.com", "zoom.us",
        "dropbox.com", "drive.google.com", "docs.google.com", "google.com",
        "microsoft.com", "office.com", "sharepoint.com", "onedrive.com",
        # Generic email
        "gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com",
        "proton.me", "protonmail.com",
        # Hosting / build platforms (often the deck's own deploy URL)
        "vercel.app", "vercel.com", "herokuapp.com", "netlify.app", "netlify.com",
        "render.com", "fly.io", "pages.dev", "github.io", "gitlab.io",
        # Payments / referenced infra
        "stripe.com", "paypal.com",
        # Reference / press
        "wikipedia.org", "crunchbase.com", "techcrunch.com", "forbes.com",
        "bloomberg.com", "wsj.com", "nytimes.com", "reuters.com",
    }
)

# File-extension TLDs that commonly slip into bare-domain regex matches.
# (e.g. ``slide_4.png`` looks like a domain).
_FILE_EXTENSION_TLDS: frozenset[str] = frozenset(
    {"png", "jpg", "jpeg", "gif", "svg", "webp", "pdf", "pptx", "ppt", "docx",
     "doc", "xlsx", "xls", "csv", "txt", "md", "mp4", "mov", "avi", "mp3",
     "wav", "zip", "tar", "gz", "exe", "dmg", "pkg"}
)


# ---------------------------------------------------------------------------
# URL regex patterns
# ---------------------------------------------------------------------------

# Scheme-prefixed URLs: catch anything starting with http:// or https://. The
# trailing punctuation/whitespace is trimmed by the consumer.
_SCHEME_URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")

# Bare domains: catch ``acme.com`` / ``app.acme.io`` / ``acme.co.uk``. The TLD
# (last label) is required to be alphabetic-only and at least 2 chars — this
# rejects pseudo-domains like ``9.9m`` (money/metric notation) or ``2.5b``
# that would otherwise match a permissive regex. Negative lookbehind avoids
# matching mid-token (e.g. ``foo.acme.com`` as ``acme.com``); negative
# lookahead avoids dangling alphanumerics. Allows up to 4 labels total.
_BARE_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9._-])"
    r"((?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]{0,62}(?:\.[a-zA-Z0-9-]{1,63}){0,2}\.[a-zA-Z]{2,63})"
    r"(?![A-Za-z0-9-])"
)

# Labeled lines: ``Website: foo.com`` / ``Visit www.foo.com`` / ``URL — foo.io``.
# The captured group goes through the same domain extractor as the bare match.
_LABELED_URL_RE = re.compile(
    r"(?i)\b(?:website|visit|web|url|homepage|site)\b[\s:\-–—]+(\S+)"
)

# Email addresses: ``milan@adspawn.com`` / ``hello@acme.io``. The captured
# domain part is the strongest signal that THIS is the company's own URL —
# pitch decks almost always include the founder/contact email of THIS
# company, not third parties. Treated like labeled (3× score multiplier).
_EMAIL_DOMAIN_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9][A-Za-z0-9.-]*\.[a-zA-Z]{2,63})\b"
)


# ---------------------------------------------------------------------------
# Domain root normalization
# ---------------------------------------------------------------------------

def _domain_root(value: str | None) -> str:
    """Reduce a URL or domain string to its registered hostname.

    Matches the behaviour of ``_domain_root`` in
    :mod:`agent.ingest.specter_mcp_client` — duplicated locally to keep this
    module self-contained.
    """
    if not value:
        return ""
    s = value.strip().lower().lstrip("@").strip("'\"")
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("/", 1)[0]
    s = s.split("?", 1)[0]
    s = s.split("#", 1)[0]
    # Trim trailing punctuation that frequently survives PDF text extraction.
    s = s.rstrip(".,;:)")
    return s


def _is_blocked(domain: str) -> bool:
    """Return True if ``domain`` is on the blocklist or looks like a filename.

    Belt-and-suspenders for the bare-domain regex: also rejects pseudo-domains
    that survive a permissive regex, e.g. money/metric notation like ``9.9m``
    or ``2.5b`` (the regex tightening is the primary defense; this is the
    safety net if a future caller passes us pre-extracted text).
    """
    if not domain or "." not in domain:
        return True
    if domain in _BLOCKLIST_DOMAINS:
        return True
    # Strip subdomains and re-check — ``app.linkedin.com`` should match
    # ``linkedin.com`` in the blocklist.
    parts = domain.split(".")
    for i in range(1, len(parts)):
        if ".".join(parts[i:]) in _BLOCKLIST_DOMAINS:
            return True
    tld = parts[-1]
    if tld in _FILE_EXTENSION_TLDS:
        return True
    # Real TLDs are alphabetic-only and ≥2 chars (.io, .com, .co.uk, etc.).
    # Reject ``9m`` / ``9.9m`` / ``25b`` / single-letter TLDs.
    if len(tld) < 2 or not tld.isalpha():
        return True
    # Real domains have at least one letter in their first label. Money/metric
    # notations like ``$4.8MM`` (4.8mm), ``$2.2Bn`` (2.2bn), or ``9.9m`` always
    # have an all-digit first label. Real domains starting with digits
    # (4chan.org, 99designs.com, 2checkout.com) all contain letters too.
    first_label = parts[0]
    if not any(c.isalpha() for c in first_label):
        return True
    return False


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

def _candidates_from_text(text: str) -> tuple[list[str], list[str]]:
    """Return (general_candidates, high_confidence_candidates) from ``text``.

    High-confidence candidates carry a 3× score multiplier downstream — these
    are domains found inside a ``"Website:" / "Visit:"`` label OR inside an
    email address (``founder@company.com``), both of which are strong human
    signals that this is THIS company's own URL.
    """
    if not text:
        return [], []
    general: list[str] = []
    high_confidence: list[str] = []

    for m in _SCHEME_URL_RE.finditer(text):
        domain = _domain_root(m.group(0))
        if domain and not _is_blocked(domain):
            general.append(domain)
    for m in _BARE_DOMAIN_RE.finditer(text):
        domain = _domain_root(m.group(1))
        if domain and not _is_blocked(domain):
            general.append(domain)
    for m in _LABELED_URL_RE.finditer(text):
        domain = _domain_root(m.group(1))
        if domain and not _is_blocked(domain):
            high_confidence.append(domain)
    for m in _EMAIL_DOMAIN_RE.finditer(text):
        domain = _domain_root(m.group(1))
        if domain and not _is_blocked(domain):
            high_confidence.append(domain)
    return general, high_confidence


def extract_company_url(
    store: EvidenceStore, max_chunks: int = 50
) -> str | None:
    """Pick the most-likely company URL from the first ``max_chunks`` chunks.

    Scoring: ``frequency × (3 if labeled-or-email else 1) × (2 if in first 3
    chunks else 1)``. Returns the highest-scoring registered domain, or
    ``None`` if no eligible candidate is found.

    Default ``max_chunks=50`` covers virtually any pitch deck end-to-end so
    contact-slide emails (``founder@company.com``) on the last slide are
    found. Cost is negligible (regex over text is microseconds per chunk).
    """
    if not store or not store.chunks:
        return None
    chunks = list(store.chunks[:max_chunks])
    scores: dict[str, float] = {}

    for idx, chunk in enumerate(chunks):
        text = chunk.text or ""
        position_weight = 2.0 if idx < 3 else 1.0
        general, labeled = _candidates_from_text(text)
        for d in general:
            scores[d] = scores.get(d, 0.0) + 1.0 * position_weight
        for d in labeled:
            scores[d] = scores.get(d, 0.0) + 3.0 * position_weight

    if not scores:
        return None
    # Pick highest score; tie-break by alpha for determinism.
    best = max(scores.items(), key=lambda kv: (kv[1], -len(kv[0]), kv[0]))
    return best[0]


# ---------------------------------------------------------------------------
# Augmentation — fetch + merge
# ---------------------------------------------------------------------------

def _renumber_chunks(*chunk_lists: Iterable[Chunk]) -> list[Chunk]:
    """Concatenate chunk lists, re-emitting each with a sequential ``chunk_N`` id.

    Both deck-derived and MCP-derived chunks start at ``chunk_0`` independently;
    merging them naively would duplicate IDs. This helper re-numbers globally.
    """
    out: list[Chunk] = []
    for chunks in chunk_lists:
        for chunk in chunks:
            out.append(
                Chunk(
                    chunk_id=f"chunk_{len(out)}",
                    text=chunk.text,
                    source_file=chunk.source_file,
                    page_or_slide=chunk.page_or_slide,
                )
            )
    return out


def _safe_log(on_log: Callable[[str], None] | None, message: str) -> None:
    """Invoke ``on_log`` without ever propagating its exceptions."""
    if on_log is None:
        return
    try:
        on_log(message)
    except Exception:  # noqa: BLE001 — logging must never break the pipeline
        pass


def augment_with_specter(
    deck_store: EvidenceStore,
    *,
    slug: str,
    expected_name: str | None = None,
    fetch_full_team: bool = False,
    on_log: Callable[[str], None] | None = None,
) -> tuple[EvidenceStore, Company | None]:
    """Try to enrich a deck-derived store with Specter MCP company data.

    Args:
        deck_store: The :class:`EvidenceStore` produced by ``ingest_startup_folder``.
        slug: The company slug to use on the merged store.
        expected_name: Optional name to feed the disambiguation safeguard in the
            MCP client. Typically derived from the upload's filename.
        fetch_full_team: When ``False`` (default), founders come only from the
            ``intelligence.founders`` summary list — no per-founder
            ``get_person_profile`` fan-out. Saves ~60% of MCP calls per company
            and is appropriate when the deck already carries founder bios.
        on_log: Optional callback that receives one-line status messages for
            operators (e.g. wired to ``print`` and the run-progress stream).

    Returns:
        A 2-tuple ``(augmented_store, company_or_none)``. On any failure mode
        the returned store is the original ``deck_store`` (unchanged) and the
        company is ``None`` — never raises.
    """
    url = extract_company_url(deck_store)
    if not url:
        _safe_log(on_log, "specter-augment: no company URL found in deck")
        return deck_store, None

    _safe_log(on_log, f"specter-augment: extracted URL {url!r} from deck")

    try:
        company, mcp_store = fetch_specter_company(
            url,
            expected_name=expected_name,
            fetch_full_team=fetch_full_team,
        )
    except SpecterDisambiguationError as exc:
        _safe_log(
            on_log,
            f"specter-augment: {url!r} resolved to wrong company — skipping ({exc})",
        )
        return deck_store, None
    except SpecterCompanyNotFoundError:
        # Specter searched its index and confirmed no record. Common for
        # very early-stage / unfunded companies that aren't tracked. Not an
        # error — just informational.
        _safe_log(
            on_log,
            f"specter-augment: Specter has no record of {url!r} — proceeding with deck only",
        )
        return deck_store, None
    except SpecterMCPError as exc:
        _safe_log(on_log, f"specter-augment: MCP failure for {url!r}: {exc}")
        return deck_store, None
    except Exception as exc:  # noqa: BLE001 — never propagate from this helper
        _safe_log(
            on_log,
            f"specter-augment: unexpected error for {url!r}: {type(exc).__name__}: {exc}",
        )
        return deck_store, None

    mcp_chunks: list[Chunk] = list(getattr(mcp_store, "chunks", []) or [])
    merged = EvidenceStore(
        startup_slug=slug,
        chunks=_renumber_chunks(deck_store.chunks, mcp_chunks),
    )

    company_label = getattr(company, "name", None) or url
    _safe_log(
        on_log,
        f"specter-augment: {url!r} resolved to {company_label!r}, +{len(mcp_chunks)} MCP chunks",
    )
    return merged, company


__all__ = [
    "augment_with_specter",
    "extract_company_url",
]
