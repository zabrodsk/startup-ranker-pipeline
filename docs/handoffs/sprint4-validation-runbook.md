# Sprint 4 Validation & Enablement Runbook

> Everything here is **Dusan-executed under `railway run`** — the coding agent
> cannot run `railway` and cannot add the permission rule. Standardize every
> command on `.venv/bin/python` (matches the scoped rule), not `uv run python`.

## 0. Prerequisites (hard gate)

1. Add the scoped rule to `.claude/settings.local.json`:
   ```json
   { "permissions": { "allow": ["Bash(railway run --service startup-ranker-web -- .venv/bin/python scripts/:*)"] } }
   ```
2. Railway context: project `rockaway-deal-intelligence-prod`, env **`staging`**,
   service `startup-ranker-web`. Confirm with `railway status` before any run.
3. **All feature flags OFF at session start** — the 6 Sprint 3 flags
   (`PIPELINE_FINAL_SCORE_MODE`, `PIPELINE_EVIDENCE_DIGEST`, `RDI_WEB_SEARCH_CACHE`,
   `RDI_SPECTER_IDENTITY_REUSE`, `RDI_DECOMP_CACHE_STORE`, `RDI_DUP_RUN_GATE`),
   `RDI_WEB_EVIDENCE_PLANNER`, and the Sprint 4 `RDI_SKIP_UNANSWERABLE_SEARCH`.
   One variable per experiment — this is the standing confound guard.

## 1. Planner validation track (job `f20aa510`)

All commands prefixed `railway run --service startup-ranker-web --`:

| Step | Command | Cost | Gate |
|---|---|---|---|
| Tag | `.venv/bin/python scripts/replay_web_planner_benchmark.py --job-id f20aa510 --stage tag` | one-off cheap LLM | — |
| Replay | `... --stage replay` | **free/offline** | asserts 0/35; non-zero exit on fail |
| Report | `... --stage report` | free | **Gate 1 review with Dusan** |
| Live | `... --stage live --sample-size 100` | ~17 min provider calls | — |
| Report (w/ live) | `... --stage report` | free | feeds Gate 1 / Gate 3 |

**GO/NO-GO criteria** (from the Sprint 2 plan):
- **Gate 1 — replay report.** GO iff **0/35** false positives (hard) · route
  distribution sane (`company_specific` < 45% with tags, skip routes < 15%) ·
  spot-checked precision ≥ 80% on 20 newly-accepted results · sampled rescue
  ≥ ~2× the 7.4% route baseline. NO-GO → tune router/queries, re-run tag+replay (free).
- **Gate 2 — shadow** (`RDI_WEB_EVIDENCE_PLANNER=shadow`, one staging batch):
  plans on 100% of web-eligible questions; **zero behavior diff** vs a paired off
  batch; no latency regression > 5%; no new exceptions.
- **Gate 3 — canary `on`** (same companies as a recent off batch): rescue ≥ 2×
  paired baseline · thin-answer share drops on sector/reg/competitor questions ·
  provider calls ≤ 2.2× (check `/api/costs/{job_id}` totals) · zero previously-
  rescued questions skipped · clean 20-answer hallucination read · scoring fields
  shape-unchanged.
- **Gate 4 — default-flip discussion only** (NOT flipped this sprint).

**Confound guard:** during the planner canary keep all 6 Sprint 3 flags AND the
Sprint 4 skip-gate OFF.

## 2. Sprint 3 per-flag staging verification (one flag at a time)

Migrations are already applied + verified on staging. Run a small batch with the
one flag ON (all other new flags off), then assert the signal. Cost-observable
features have a one-command checker:

```
railway run --service startup-ranker-web -- .venv/bin/python \
    scripts/verify_flag_on_staging.py --feature <w3|w9|w11|w13> \
    --job-id <feature_job> [--baseline-job-id <off_job>]
```

| Flag | ON value | Signal | Check |
|---|---|---|---|
| W3 `PIPELINE_EVIDENCE_DIGEST` | `on` | new `evidence_digest` stage + critique/refinement prompt tokens drop | `verify_flag_on_staging.py --feature w3 --job-id <j> --baseline-job-id <off>` + Dusan argument-quality spot-check on 2–3 companies (net-cost is the enable gate) |
| W9 `PIPELINE_FINAL_SCORE_MODE` | `reuse` | fewer `evaluation` LLM calls; record `total_score`/`final_decision` delta | `--feature w9 --job-id <reuse> --baseline-job-id <rescore>` |
| W11 `RDI_DECOMP_CACHE_STORE` | `supabase` | second same-industry company logs `cross_company=true`, zero decomposition LLM calls | `--feature w11 --job-id <second> --baseline-job-id <first>`; also confirm CLI offline run still works |
| W13 `RDI_WEB_SEARCH_CACHE` | `on` | second same-company run shows `cache_hit`, lower Perplexity spend | `--feature w13 --job-id <second> --baseline-job-id <first>` |
| W7 `RDI_SPECTER_IDENTITY_REUSE` | `on` | **MCP logs**: exactly one `find_company` per company; worker config JSON carries `specter_company_id` | manual (not cost-observable) |
| dup-gate `RDI_DUP_RUN_GATE` | `on` | same deck twice → second instant + "Reused from run {date}" badge; `force_reanalyze` overrides | manual (results payload / UI) |

## 3. Cost dashboard before/after

Pick one fixed 5–10 company set (e.g. the planner canary set). Run flag-off
(`job_id_A`), then flag-on (`job_id_B`). Capture from **`GET /api/costs/{job_id}`**
per job (NOT `/api/costs/daily` — it can't isolate one batch):
`totals.total_usd`, `totals.llm_usd`, `totals.perplexity_search.requests`, and the
touched per-stage `prompt_tokens`/`usd`. Record in the sprint-close note.

## 4. Sequencing dependency on Workstream C (skip-unanswerable gate)

> **Enable `RDI_SKIP_UNANSWERABLE_SEARCH` only after the planner reaches Gate 1**
> (free replay 0/35 + report reviewed). The skip-gate and the planner share the
> answering/search path, so enabling C mid-planner-validation confounds both.
> Keep C OFF through any planner canary window. C's own offline merge gate:
> `... scripts/replay_web_planner_benchmark.py --job-id f20aa510 --stage skip`
> (free; hard 0/35; non-zero exit on any rescued-row skip).
