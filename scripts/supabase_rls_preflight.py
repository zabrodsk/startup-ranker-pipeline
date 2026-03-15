"""CLI entrypoint for the Supabase RLS preflight."""

from web.supabase_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
