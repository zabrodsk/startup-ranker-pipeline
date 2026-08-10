#!/usr/bin/env python3
"""Fail closed when production-protected RDI scoring artifacts drift."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "security/protected-rdi-scoring.json"


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
            raise ValueError("unsupported manifest contract")
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("protected file set is empty")
        failures: list[str] = []
        for relative_name, expected in sorted(files.items()):
            if not isinstance(relative_name, str) or not isinstance(expected, str):
                raise ValueError("protected file entries must be strings")
            candidate = (ROOT / relative_name).resolve()
            if ROOT not in candidate.parents:
                raise ValueError(f"protected path escapes repository: {relative_name}")
            if not candidate.is_file():
                failures.append(f"missing: {relative_name}")
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(
                    f"changed: {relative_name} expected={expected} actual={actual}"
                )
        if failures:
            print("Protected RDI scoring verification FAILED", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Protected RDI scoring verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "Protected RDI scoring verification passed "
        f"({len(files)} files; baseline "
        f"{manifest['authoritative_production_commit']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
