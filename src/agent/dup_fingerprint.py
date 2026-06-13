"""Evidence fingerprints for the duplicate-run gate (Sprint 3).

A fingerprint identifies the EVIDENCE INPUTS of one company analysis. Two
runs with the same fingerprint analyzed byte-identical inputs, so the gate
(web/app.py) may reuse the prior result instead of re-running the pipeline.

Shared between the web layer (gate-time lookups) and the worker layer
(persisting fingerprints on result rows) — keep it dependency-light; web and
agent both import it.

Versioned prefixes: bump (doc-v2, ...) whenever the input semantics change.
"""

import hashlib
import json
from typing import Any


def doc_fingerprint(file_sha256s: list[str]) -> str | None:
    """Fingerprint for document modes (pitchdeck single file or multi-file).

    Order-insensitive over the per-file content hashes; a single file yields
    sha256("doc-v1|<sha>").
    """
    hashes = sorted(h for h in (file_sha256s or []) if h)
    if not hashes:
        return None
    raw = "doc-v1|" + "|".join(hashes)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def specter_csv_fingerprint(descriptor: dict[str, Any]) -> str | None:
    """Fingerprint for one Specter CSV company row.

    Hashes the FULL descriptor (false-negative-biased = safe: any changed
    Specter field re-analyzes; revisit a stable-field subset after observing
    hit rates).
    """
    if not descriptor:
        return None
    try:
        canonical = json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError):
        return None
    raw = "specter-csv-v1|" + canonical
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def identity_fingerprint(company_lookup_key: str, input_mode: str) -> str | None:
    """Fingerprint for identity-based modes (Specter URL / LeadGen).

    There is no document evidence to hash — the company identity plus the
    recency window IS the freshness bound.
    """
    key = (company_lookup_key or "").strip()
    if not key:
        return None
    raw = f"identity-v1|{key}|{(input_mode or '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
