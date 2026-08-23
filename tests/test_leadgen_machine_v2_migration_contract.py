from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase/migrations/20260823010500_leadgen_v2_reservation_service_boundary.sql"


def test_v2_reservation_cross_generation_cap_stays_behind_service_rpc() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "SECURITY DEFINER" in sql
    assert "SET search_path TO public, pg_temp" in sql
    assert "FROM PUBLIC, anon, authenticated" in sql
    assert "TO service_role" in sql
    assert "GRANT SELECT ON public.leadgen_machine_intakes" not in sql
