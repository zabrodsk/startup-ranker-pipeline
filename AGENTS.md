# Deal Intelligence Agent Guide

This repo powers Rockaway Deal Intelligence: analysis intake, Specter-backed
company identity, company/run persistence, VC portal surfaces, and LeadGen
human-approval intake. Keep changes narrow, evidence-driven, and tested in the
non-production environment before production rollout.

## Operating Principles

- Treat testing and production as separate stacks. Verify Railway project,
  service, environment, `APP_ENV`, and Supabase project before deploying or
  mutating data.
- Prefer repo-local patterns and existing helpers over new abstractions.
- Do not mix LeadGen producer work with Deal Intelligence analysis guardrails.
  LeadGen intake may store pending batches, but analysis should start only after
  explicit approval inside Deal Intelligence.
- Preserve user data. Never run broad deletes, backfills, migrations, or
  production cleanup without a dry run, backup/export, and explicit approval.
- Do not commit local agent artifacts, generated exports, secrets, caches, or
  environment files.

## Analysis Quality Guardrails

- The shared preflight in `web/analysis_quality.py` is the canonical place for
  analysis-start quality checks and display-name normalization.
- `/api/analyze/{job_id}` must run quality preflight before setting a job to
  `running`, queueing a worker, or starting a thread, except for explicitly
  approved internal paths that document why they bypass it.
- Block full runs when any company/input unit cannot produce a trustworthy
  company identity or analyzable evidence.
- Persistence is a second line of defense. `web/db.py` must not create
  `companies` or `company_runs` rows for invalid names, failed URL-only rows,
  timeouts, or error-only analysis results.
- Keep regression coverage for the known bad-run classes:
  `0ddb61a7`, `5199998a`, and `942b82c0`.

## Supabase Work

- Use migrations for schema changes; do not hand-edit production schema as the
  first implementation path.
- For Supabase work, use the `supabase` skill. For SQL, RLS, indexes, query
  performance, or schema review, also use `supabase-postgres-best-practices`.
- Backend code should use service-role access only where appropriate; browser
  clients should authenticate through the app/API rather than querying tables
  directly.
- Before production data changes, create a timestamped export under `exports/`
  and keep mutation reports there. `exports/` is intentionally ignored.

## Deployments

- Testing-first rollout is the default. Validate behavior and smoke tests in the
  testing Railway service before promoting to production.
- Production deploys should include both web and worker services when shared
  behavior or migrations affect background processing.
- After deploys, verify service health, expected route behavior, and logs. For
  auth-protected paths, a `401` can be a healthy smoke result when unauthenticated.
- Railway CLI state can point at the wrong project/environment. Confirm target
  project and service before `railway up` or env-var changes.

## Verification

- Use focused tests for the touched surface. Common commands:
  - `.venv/bin/python -m pytest tests/test_analysis_quality_guardrails.py -q`
  - `.venv/bin/python -m pytest tests/test_leadgen_ingest.py -q`
  - `.venv/bin/python -m pytest tests/test_web_static_companies_sort.py -q`
  - `.venv/bin/python -m py_compile web/app.py web/db.py`
- For frontend/navigation changes, verify the browser-visible behavior when
  feasible, especially first-load and idle-cache cases.
- For production incidents, inspect live logs/data instead of relying on stale
  local assumptions.

## External Review And Package Context

- When work involves an open PR with external review feedback, Greptile,
  GitHub review comments, or failing CI, offer the bounded `pr-review-loop`
  workflow before making broad repair passes.
- When coding against unfamiliar third-party libraries, SDKs, framework APIs, or
  version-sensitive package behavior, inspect existing repo usage first. Use
  `opensrc-context` when deeper package-source inspection is needed.
- For hard bugs, failing behavior, performance regressions, new behavior, or
  architecture cleanup, use `engineering-feedback-loops` to pick a tight
  diagnosis/test/implementation loop.
