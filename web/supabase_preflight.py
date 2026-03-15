"""Preflight checks for staged Supabase RLS rollout."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

READ_CHECK_TABLES = (
    "jobs",
    "analyses",
    "company_runs",
    "job_controls",
    "person_profile_jobs",
)
PERMISSION_ERROR_MARKERS = (
    "permission denied",
    "row-level security",
    "violates row-level security",
    "not allowed",
    "42501",
    "401",
    "403",
)


class PreflightFailure(RuntimeError):
    """Raised when the RLS preflight detects a production-blocking issue."""


def looks_like_permission_error(exc: Exception) -> bool:
    """Return True when an exception message looks like an auth/RLS failure."""
    text = str(exc).strip().lower()
    return any(marker in text for marker in PERMISSION_ERROR_MARKERS)


def create_supabase_client_from_env() -> Client:
    """Create a Supabase client from the same env vars used by the app backend."""
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise PreflightFailure(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in the environment."
        )
    return create_client(url, key)


def _format_failure(action: str, table: str, exc: Exception) -> str:
    permission_hint = looks_like_permission_error(exc)
    return (
        f"{action} failed for table={table}: {exc} "
        f"(permission_related={permission_hint})"
    )


def run_preflight_checks(
    client: Client,
    *,
    print_fn: Callable[[str], None] | None = None,
) -> str:
    """Verify read/write access expected by the backend before RLS rollout."""
    emit = print_fn or (lambda _message: None)

    for table in READ_CHECK_TABLES:
        emit(f"[check] read {table}")
        try:
            client.table(table).select("*").limit(1).execute()
        except Exception as exc:
            raise PreflightFailure(_format_failure("read", table, exc)) from exc

    job_id_legacy = f"rls-preflight-{uuid.uuid4().hex[:12]}"
    payload: dict[str, Any] = {
        "job_id_legacy": job_id_legacy,
        "input_mode": "pitchdeck",
        "instructions": "Disposable RLS preflight row",
        "use_web_search": False,
        "run_config": {
            "purpose": "rls_preflight",
            "disposable": True,
        },
    }

    emit(f"[check] upsert jobs {job_id_legacy}")
    try:
        client.table("jobs").upsert(payload, on_conflict="job_id_legacy").execute()
    except Exception as exc:
        raise PreflightFailure(_format_failure("upsert", "jobs", exc)) from exc

    emit(f"[check] read back jobs {job_id_legacy}")
    try:
        row = (
            client.table("jobs")
            .select("job_id_legacy")
            .eq("job_id_legacy", job_id_legacy)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise PreflightFailure(_format_failure("read_back", "jobs", exc)) from exc

    if not row.data:
        raise PreflightFailure(
            f"read_back failed for table=jobs: disposable row {job_id_legacy} was not returned"
        )

    emit(f"[check] delete jobs {job_id_legacy}")
    try:
        client.table("jobs").delete().eq("job_id_legacy", job_id_legacy).execute()
    except Exception as exc:
        raise PreflightFailure(_format_failure("delete", "jobs", exc)) from exc

    return job_id_legacy


def main() -> int:
    """Run the preflight checks and print an operator-friendly result."""
    try:
        client = create_supabase_client_from_env()
        job_id = run_preflight_checks(client, print_fn=print)
        print(f"[ok] Supabase RLS preflight passed. Disposable job_id={job_id}")
        return 0
    except PreflightFailure as exc:
        print(f"[fail] {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI wrapper
        print(f"[fail] Unexpected preflight error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
