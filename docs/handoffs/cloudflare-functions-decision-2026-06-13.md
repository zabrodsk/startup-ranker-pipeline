# Decision: delete the dead Cloudflare `functions/api/` surface (Sprint 4 A3)

**Decision (2026-06-13, Dusan-approved):** DELETE `functions/api/`.

## Evidence the surface is dead
- No `wrangler.toml` / `wrangler.json` / `_routes.json` / `_redirects` / Cloudflare
  Pages config anywhere in the repo.
- The directory was last touched 2026-03-04 (v0.0.2) and was untouched through
  Sprints 1–3.
- CI (`.github/workflows/ci.yml`) never references `functions/`, Cloudflare, Pages,
  or wrangler.
- No file outside `functions/` references `functions/api`.
- The documented live path (`README.md`) is FastAPI on Railway
  (`railway.toml` → `python -m agent.railway_service`) exposed via a `cloudflared`
  **tunnel** — which is not Cloudflare Pages Functions.
- Dusan confirmed there is no out-of-repo Cloudflare Pages project bound to this repo.

## Removed
The 9 files under `functions/api/`: `_proxy.js`, `_utils.js`, `check-session.js`,
`login.js`, `upload.js`, `web-search-available.js`, `analyze/[jobId].js`,
`status/[jobId].js`, `download/[jobId].js`.

## Security note
The removed `_utils.js` contained a spec-invalid wildcard-origin + credentials CORS
combo and a hardcoded `SESSION_SECRET || 'change-me-session-secret'` fallback. Both
are now moot on this dead surface. The live FastAPI equivalents are addressed by
PR-A1 (CORS allowlist) and PR-A5 (`SESSION_SECRET` fail-closed).

## Left untouched
`package.json` is kept — its `react`/`pptxgenjs`/`sharp` devDependencies serve other
purposes. `@supabase/supabase-js` may now be unused; removing it is a candidate
follow-up, not part of this deletion.

## Recovery
Restorable from git history (the parent of this commit) if a Pages surface is ever revived.
