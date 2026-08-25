from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase/migrations/20260823010500_leadgen_v2_reservation_service_boundary.sql"
BROKER_MIGRATION = ROOT / "supabase/migrations/20260824103000_specter_quota_broker.sql"
BROKER_PGCRYPTO_REPAIR = (
    ROOT / "supabase/migrations/20260825062000_specter_quota_broker_pgcrypto_schema.sql"
)
BROKER_RECOVERY_REPAIR = (
    ROOT
    / "supabase/migrations/20260825064000_specter_quota_recovery_probe_closes_circuit.sql"
)
BUNDLE_V2_SCHEMA_MIGRATION = (
    ROOT / "supabase/migrations/20260825135740_allow_leadgen_bundle_v2.sql"
)


def test_v2_reservation_cross_generation_cap_stays_behind_service_rpc() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "SECURITY DEFINER" in sql
    assert "SET search_path TO public, pg_temp" in sql
    assert "FROM PUBLIC, anon, authenticated" in sql
    assert "TO service_role" in sql
    assert "GRANT SELECT ON public.leadgen_machine_intakes" not in sql


def test_bundle_schema_constraint_accepts_only_v1_and_v2() -> None:
    sql = BUNDLE_V2_SCHEMA_MIGRATION.read_text(encoding="utf-8").lower()

    assert "drop constraint leadgen_evidence_bundles_schema_version_check" in sql
    assert "add constraint leadgen_evidence_bundles_schema_version_check" in sql
    assert "'frozen-leadgen-evidence-bundle-v1'" in sql
    assert "'frozen-leadgen-evidence-bundle-v2'" in sql
    assert "not valid" in sql
    assert "validate constraint leadgen_evidence_bundles_schema_version_check" in sql


def test_specter_quota_broker_migration_keeps_policy_and_service_boundary_explicit() -> None:
    sql = BROKER_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create extension if not exists pgcrypto with schema public" in sql
    assert "create table public.specter_quota_authorizations" in sql
    assert "create table public.specter_quota_broker_daily" in sql
    assert "create table public.specter_quota_authorization_events" in sql
    assert "observed_limit integer not null default 250" in sql
    assert "safety_reserve integer not null default 25" in sql
    assert "company_cap integer not null default 8" in sql
    assert "founder_profile_cap integer not null default 3" in sql
    assert "scheduled_import_allowance integer not null default 40" in sql
    assert "recovery_allowance integer not null default 5" in sql
    assert "quota_class text not null" in sql
    assert "quota_class in (" in sql
    assert "'flexible_pool'" in sql
    assert "'manual_batch'" in sql
    assert "'promoted_candidate_refresh'" in sql
    assert "least(160, greatest(coalesce(p_remaining_rdi_slots, 20), 0) * 8)" in sql
    assert "p_business_date <> public._specter_quota_current_prague_date()" in sql
    assert "v_auth.intake_id is not null and p_intake_id is null" in sql
    assert "specter quota authorization intake mismatch" in sql
    assert "circuit_state in ('closed', 'open', 'probing')" in sql
    assert "authorization_id ~ '^specter-auth-[0-9a-f]{64}$'" in sql
    assert "encode(digest(v_authorization_material, 'sha256'), 'hex')" in sql
    assert "set search_path to public, extensions, pg_temp" in sql
    assert "event_type in ('reserve', 'deny', 'commit', 'release')" in sql
    assert "grant execute on function public.reserve_specter_quota_authorization" in sql
    assert "grant execute on function public.commit_specter_quota_authorization" in sql
    assert "grant execute on function public.release_specter_quota_authorization" in sql
    assert "to service_role" in sql


def test_specter_quota_broker_repair_adds_existing_supabase_extension_schema() -> None:
    sql = BROKER_PGCRYPTO_REPAIR.read_text(encoding="utf-8").lower()

    assert "alter function public.reserve_specter_quota_authorization" in sql
    assert "set search_path to public, extensions, pg_temp" in sql


def test_successful_recovery_probe_closes_transient_provider_circuit() -> None:
    original = BROKER_MIGRATION.read_text(encoding="utf-8").lower()
    repair = BROKER_RECOVERY_REPAIR.read_text(encoding="utf-8").lower()

    for sql in (original, repair):
        assert "v_auth.quota_class = 'recovery_probe'" in sql
        assert "v_daily.circuit_state = 'probing'" in sql
        assert "set circuit_state = 'closed'" in sql
        assert "reason_code = null" in sql
        assert "retry_at = null" in sql
