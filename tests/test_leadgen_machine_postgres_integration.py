from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RDI_RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RDI_RUN_POSTGRES_INTEGRATION=1 to run Docker-backed Postgres tests",
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MIGRATION = ROOT / "supabase/migrations/20260731000000_leadgen_machine_lifecycle.sql"
DAILY_SCOPE_MIGRATION = ROOT / "supabase/migrations/20260807102328_leadgen_machine_daily_start_scope.sql"
PRAGUE = ZoneInfo("Europe/Prague")
CURRENT_BUSINESS_DATE = datetime.now(timezone.utc).astimezone(PRAGUE).date()

_FIXTURE_SCHEMA = """
create role anon nologin;
create role authenticated nologin;
create role service_role nologin;

create table public.company_runs (
  job_id_legacy text,
  company_key text,
  company_id uuid,
  decision text,
  created_at timestamptz,
  run_created_at timestamptz,
  result_payload jsonb,
  composite_score numeric
);
create table public.jobs (
  job_id_legacy text,
  run_config jsonb,
  pipeline_version text
);
create table public.job_status_history (
  job_id_legacy text,
  status text,
  created_at timestamptz
);
create table public.analyses (
  job_id_legacy text,
  company_id uuid,
  status text,
  created_at timestamptz
);
"""


def _digest(material: str, length: int = 32) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass
class _Postgres:
    container: str

    def run(
        self,
        sql: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "PGPASSWORD=postgres",
                self.container,
                "psql",
                "-X",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-Atq",
                "-c",
                sql,
            ],
            check=check,
            text=True,
            capture_output=True,
        )

    def scalar(self, sql: str) -> str:
        return self.run(sql).stdout.strip().splitlines()[-1]

    def json(self, sql: str) -> dict[str, object]:
        return json.loads(self.scalar(sql))

    def file(self, path: Path) -> None:
        container_path = f"/workspace/{path.relative_to(ROOT)}"
        subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "PGPASSWORD=postgres",
                self.container,
                "psql",
                "-X",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-f",
                container_path,
            ],
            check=True,
            text=True,
            capture_output=True,
        )


@pytest.fixture(scope="module")
def postgres() -> _Postgres:
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    if subprocess.run(
        ["docker", "info"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    container = f"rdi-daily-cap-{uuid4().hex[:12]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container,
            "-e",
            "POSTGRES_PASSWORD=postgres",
            "-v",
            f"{ROOT}:/workspace:ro",
            "postgres:17",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    database = _Postgres(container)
    try:
        # During first-time image initialization PostgreSQL briefly accepts
        # connections and then restarts.  Wait for an actual query so the
        # fixture cannot race that restart boundary.
        stable_queries = 0
        for _ in range(120):
            ready = database.run("select 1", check=False)
            if ready.returncode == 0 and ready.stdout.strip() == "1":
                stable_queries += 1
                if stable_queries >= 5:
                    break
            else:
                stable_queries = 0
            time.sleep(0.25)
        else:
            raise RuntimeError("temporary Postgres did not become ready")

        database.run(_FIXTURE_SCHEMA)
        database.file(ORIGINAL_MIGRATION)
        _seed_pre_fix_intake(database, 1, "campaign-b", "2026-08-01T22:30:00Z")
        _seed_pre_fix_intake(database, 2, "campaign-a", "2026-08-01T23:00:00Z")
        database.file(DAILY_SCOPE_MIGRATION)
        yield database
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _record(
    index: int,
    *,
    environment: str = "staging",
    campaign_id: str = "campaign-current",
    business_date: date = CURRENT_BUSINESS_DATE,
    created_at: str | None = None,
    include_scope: bool = True,
) -> dict[str, object]:
    material = f"{environment}:{campaign_id}:{business_date}:{index}"
    digest = _digest(material)
    payload: dict[str, object] = {
        "intake_id": f"rdi-intake-{digest}",
        "contract_version": "rdi.leadgen-machine.v1",
        "idempotency_identity": _digest(f"identity:{material}", 64),
        "payload_hash": _digest(f"payload:{material}", 64),
        "external_company_id": f"company-{index}",
        "canonical_domain": f"company-{index}.example",
        "campaign_id": campaign_id,
        "iteration_id": f"iteration-{index}",
        "source_run_id": f"source-{index}",
        "batch_id": f"batch-{index}",
        "idempotency_key": f"idempotency-{index}",
        "target_environment": environment,
        "provenance_reference": f"test://company-{index}",
        "rdi_company_id": None,
        "rdi_correlation_id": f"rdi-correlation-{_digest(f'correlation:{material}')}",
        "intake_status": "accepted",
        "lifecycle_state": "accepted",
        "approval_required": False,
        "rejection_code": None,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "updated_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    if include_scope:
        payload.update(
            {
                "business_date": business_date.isoformat(),
                "business_timezone": "Europe/Prague",
            }
        )
    return payload


def _create(postgres: _Postgres, record: dict[str, object]) -> dict[str, object]:
    encoded = _literal(json.dumps(record, separators=(",", ":")))
    return postgres.json(
        "set role service_role; "
        f"select public.create_leadgen_machine_intake({encoded}::jsonb)::text;"
    )


def _seed_pre_fix_intake(
    postgres: _Postgres,
    index: int,
    campaign_id: str,
    created_at: str,
) -> None:
    result = _create(
        postgres,
        _record(
            index,
            campaign_id=campaign_id,
            business_date=date(2026, 8, 2),
            created_at=created_at,
            include_scope=False,
        ),
    )
    assert result["action"] == "created"


def _reserve(
    postgres: _Postgres,
    record: dict[str, object],
    *,
    business_date: date | None = None,
) -> dict[str, object]:
    intake_id = str(record["intake_id"])
    environment = str(record["target_environment"])
    scope_date = business_date or date.fromisoformat(str(record["business_date"]))
    job_id = f"rdi-job-{_digest(f'job:{intake_id}')}"
    return postgres.json(
        "set role service_role; "
        "select public.reserve_leadgen_machine_start("
        f"{_literal(intake_id)}, {_literal(environment)}, "
        f"{_literal(scope_date.isoformat())}::date, 'Europe/Prague', "
        f"{_literal(job_id)}, 'service:rockaway-leadgen', 20)::text;"
    )


def _capacity(
    postgres: _Postgres,
    environment: str = "staging",
    business_date: date = CURRENT_BUSINESS_DATE,
) -> dict[str, object]:
    return postgres.json(
        "set role service_role; "
        "select public.get_leadgen_machine_daily_capacity("
        f"{_literal(environment)}, {_literal(business_date.isoformat())}::date, 20)::text;"
    )


def _truncate(postgres: _Postgres) -> None:
    postgres.run(
        "truncate table public.leadgen_machine_events, "
        "public.leadgen_machine_intakes, "
        "public.leadgen_machine_daily_scopes cascade;"
    )


def test_corrective_migration_backfills_applied_staging_shape(
    postgres: _Postgres,
) -> None:
    assert postgres.scalar(
        "select count(*) from public.leadgen_machine_intakes "
        "where business_date = date '2026-08-02' "
        "and business_timezone = 'Europe/Prague';"
    ) == "2"
    assert postgres.scalar(
        "select canonical_campaign_id from public.leadgen_machine_daily_scopes "
        "where target_environment = 'staging' and business_date = date '2026-08-02';"
    ) == "campaign-a"


def test_daily_cap_replay_rotation_reset_and_environment_isolation(
    postgres: _Postgres,
) -> None:
    _truncate(postgres)
    records: list[dict[str, object]] = []
    for index in range(1, 21):
        record = _record(index, campaign_id=f"campaign-{index % 2}")
        assert _create(postgres, record)["action"] == "created"
        reserved = _reserve(postgres, record)
        assert reserved["action"] == "reserved"
        assert reserved["daily_started_count"] == index
        records.append(record)

    replayed_intake = _create(postgres, records[0])
    replayed_start = _reserve(postgres, records[0])
    assert replayed_intake["action"] == "existing"
    assert replayed_intake["intake_id"] == records[0]["intake_id"]
    assert replayed_start["action"] == "existing"
    assert replayed_start["daily_started_count"] == 20

    rotated = _record(21, campaign_id="campaign-rotated")
    assert _create(postgres, rotated)["action"] == "created"
    limited = _reserve(postgres, rotated)
    assert limited["action"] == "rate_limited"
    assert limited["daily_remaining_capacity"] == 0

    production = _record(
        22,
        environment="production",
        campaign_id="campaign-production",
    )
    assert _create(postgres, production)["action"] == "created"
    isolated = _reserve(postgres, production)
    assert isolated["action"] == "reserved"
    assert isolated["daily_started_count"] == 1

    postgres.run(
        "insert into public.jobs(job_id_legacy, run_config, pipeline_version) "
        "values ('human-job', '{}'::jsonb, 'pipeline-v1');"
    )
    assert _capacity(postgres)["daily_started_count"] == 20


def test_previous_day_capacity_does_not_block_current_day(
    postgres: _Postgres,
) -> None:
    _truncate(postgres)
    previous_date = CURRENT_BUSINESS_DATE - timedelta(days=1)
    postgres.run(
        "insert into public.leadgen_machine_daily_scopes "
        "(target_environment, business_date, business_timezone, canonical_campaign_id) "
        f"values ('staging', {_literal(previous_date.isoformat())}::date, "
        "'Europe/Prague', 'campaign-previous');"
        "insert into public.leadgen_machine_intakes ("
        "intake_id, contract_version, idempotency_identity, payload_hash, "
        "external_company_id, canonical_domain, campaign_id, iteration_id, "
        "source_run_id, batch_id, idempotency_key, target_environment, "
        "business_date, business_timezone, provenance_reference, "
        "rdi_correlation_id, lifecycle_state, job_id, start_actor, completed_at"
        ") select "
        "'rdi-intake-' || md5(i::text), 'rdi.leadgen-machine.v1', "
        "md5('identity-' || i) || md5('identity-b-' || i), "
        "md5('payload-' || i) || md5('payload-b-' || i), "
        "'previous-' || i, 'previous-' || i || '.example', "
        "'campaign-previous', 'iteration-' || i, 'source-' || i, "
        "'batch-' || i, 'idempotency-' || i, 'staging', "
        f"{_literal(previous_date.isoformat())}::date, 'Europe/Prague', "
        "'test://previous', 'rdi-correlation-' || md5('correlation-' || i), "
        "'succeeded', 'rdi-job-' || md5('job-' || i), "
        "'service:rockaway-leadgen', now() "
        "from generate_series(1, 20) as i;"
    )
    today = _record(30)
    assert _create(postgres, today)["action"] == "created"
    first_today = _reserve(postgres, today)
    assert first_today["action"] == "reserved"
    assert first_today["daily_started_count"] == 1
    assert _capacity(postgres, business_date=previous_date)["daily_started_count"] == 20


def test_release_restores_capacity_but_uncertain_and_terminal_states_consume(
    postgres: _Postgres,
) -> None:
    _truncate(postgres)
    released_record = _record(40)
    _create(postgres, released_record)
    reserved = _reserve(postgres, released_record)
    released = postgres.json(
        "set role service_role; select public.release_leadgen_machine_start("
        f"{_literal(str(released_record['intake_id']))}, "
        f"{_literal(str(reserved['job_id']))}, 'service:rockaway-leadgen')::text;"
    )
    assert released["lifecycle_state"] == "accepted"
    assert _capacity(postgres)["daily_started_count"] == 0

    uncertain_record = _record(41)
    _create(postgres, uncertain_record)
    uncertain_reservation = _reserve(postgres, uncertain_record)
    postgres.json(
        "set role service_role; select public.finalize_leadgen_machine_start("
        f"{_literal(str(uncertain_record['intake_id']))}, "
        f"{_literal(str(uncertain_reservation['job_id']))}, "
        "'uncertain', 'outcome_uncertain', 'transport_uncertain', "
        "'Remote start outcome is uncertain.', 'service:rockaway-leadgen')::text;"
    )
    assert _capacity(postgres)["daily_started_count"] == 1

    for lifecycle in ("failed", "cancelled", "succeeded"):
        postgres.run(
            "update public.leadgen_machine_intakes "
            f"set lifecycle_state = {_literal(lifecycle)}, completed_at = now() "
            f"where intake_id = {_literal(str(uncertain_record['intake_id']))};"
        )
        assert _capacity(postgres)["daily_started_count"] == 1

    rejected = _record(42)
    _create(postgres, rejected)
    postgres.run(
        "update public.leadgen_machine_intakes set lifecycle_state = 'rejected' "
        f"where intake_id = {_literal(str(rejected['intake_id']))};"
    )
    assert _capacity(postgres)["daily_started_count"] == 1


def test_concurrent_final_slot_has_exactly_one_winner(
    postgres: _Postgres,
) -> None:
    _truncate(postgres)
    for index in range(50, 69):
        record = _record(index)
        _create(postgres, record)
        assert _reserve(postgres, record)["action"] == "reserved"

    contenders = [_record(69), _record(70, campaign_id="campaign-rotated")]
    for record in contenders:
        _create(postgres, record)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda item: _reserve(postgres, item), contenders))

    assert sorted(str(item["action"]) for item in outcomes) == [
        "rate_limited",
        "reserved",
    ]
    assert _capacity(postgres)["daily_started_count"] == 20


def test_scope_forgery_dst_and_privileges_fail_closed(postgres: _Postgres) -> None:
    _truncate(postgres)
    record = _record(80)
    _create(postgres, record)
    mismatch = _reserve(
        postgres,
        record,
        business_date=CURRENT_BUSINESS_DATE - timedelta(days=1),
    )
    assert mismatch["action"] == "scope_mismatch"

    forged = _record(
        81,
        business_date=CURRENT_BUSINESS_DATE + timedelta(days=1),
    )
    encoded = _literal(json.dumps(forged, separators=(",", ":")))
    rejected = postgres.run(
        "set role service_role; "
        f"select public.create_leadgen_machine_intake({encoded}::jsonb);",
        check=False,
    )
    assert rejected.returncode != 0

    assert postgres.scalar(
        "select ((timestamptz '2026-03-29 00:30:00+00' "
        "at time zone 'Europe/Prague')::date = date '2026-03-29') "
        "and ((timestamptz '2026-10-25 01:30:00+00' "
        "at time zone 'Europe/Prague')::date = date '2026-10-25');"
    ) == "t"
    assert postgres.scalar(
        "select not has_table_privilege('service_role', "
        "'public.leadgen_machine_intakes', 'insert') "
        "and not has_table_privilege('service_role', "
        "'public.leadgen_machine_daily_scopes', 'select') "
        "and has_function_privilege('service_role', "
        "'public.reserve_leadgen_machine_start(text,text,date,text,text,text,integer)', "
        "'execute');"
    ) == "t"
