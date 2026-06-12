# Sprint 2 Planning Handoff — Deal Intelligence

Prepared: 2026-06-12 (end of Sprint 1 + A/B + quality deep-dive session)
Audience: a fresh Claude Code session tasked with PLANNING Sprint 2 (plan mode,
no implementation until Dusan approves the plan).

---

## 1. Repo operating facts (verify, don't re-derive)

- Repo: `/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence`
- Branch topology: work branches off **`origin/testing`** (NOT `main`), PRs into
  `testing`. `testing` auto-deploys to Railway staging (project
  `rockaway-deal-intelligence-prod`, env `staging`, services `startup-ranker-web`
  + `startup-ranker-worker`).
- Tooling: `uv sync --group dev`; tests `.venv/bin/python -m pytest tests -q`
  (hermetic — no network; run with `SUPABASE_URL` both unset and set).
  OneDrive gotcha: if imports fail with mmap/timeout errors, `rm -rf .venv && uv sync --group dev`.
- Memory OS: run `coding-memory start-work` (project slug `deal-intelligence`)
  at the top of meaningful work; `end-work` at closeout. See `~/.claude/CLAUDE.md`.
- Guardrails (binding): never read/print/commit `.env` / key material /
  Supabase service-role values. Testing stack first; NO production
  deploys/migrations/deletes without dry-run + explicit Dusan approval.
  Ask before deleting any file.

## 2. What Sprint 1 shipped (all merged to testing, CI green, 455 tests passing)

| PR | Item | Net effect |
|---|---|---|
| #3 | W1/W2 `WEB_SEARCH_HEAVY_OVERRIDE` env gating (always/root_only/never) + `trigger_reason`/`gating_mode` telemetry + `by_reason` cost rollup | default `always` (unchanged behavior) |
| #4 | W6 skip executive summary during matching (`skip_executive_summary` state flag + graph router) | dead LLM call removed in matching |
| #5 | W8 evaluation retry fix (schema `min/max_length=14`, retry 5→2 attempts) | malformed-score retries collapsed |
| #7 | W10 collapse stacked retries (`LLM_CLIENT_MAX_RETRIES` default 0, removed all `@backoff` decorators, dropped `backoff` dep) | single retry owner = ThrottledRunnable (`LLM_MAX_RETRIES`, default 6) |

W5 (thresholds-only rematch) was DROPPED — Dusan doesn't recognize the
VC-portal matching feature; review with him before ever touching it.

## 3. A/B result (job `f20aa510`, 10 LeadGen companies, mode `always`)

- 476 web searches: **98.7% triggered by ungated "documents incomplete"**, only
  6 by the heavy trigger that `root_only` gates.
- **Decision made: do NOT flip the default; `root_only` saves ≤1.3%.** Arm B was
  skipped as provably pointless. `WEB_SEARCH_HEAVY_OVERRIDE` stays `always`.
- Cohort (early-stage, sparse Specter data) is representative of RDI's real usage.

## 4. The quality finding that defines Sprint 2

Spot-check of all 759 persisted QA rows for `f20aa510`:

- **58% of evidence answers are thin** ("Insufficient information available").
- **Web-search rescue rate: 7.4%** — 471 questions got the docs+web hybrid
  re-answer; 92.6% stayed thin. No hallucination (good), but enrichment spend
  is ~wasted.
- Root cause (code-confirmed, 4 compounding company-anchors in
  `src/agent/evidence_answering.py`): company-first query (`_build_web_search_query`
  L338), fallback domain filter `[company_domain, crunchbase, linkedin]`
  (`_web_search_domain_filter` L259), relevance gate rejects results not
  mentioning the company (`_web_results_add_value` L293), hybrid prompt demotes
  web to "ONLY a fallback" (`EVIDENCE_HYBRID_SYSTEM_PROMPT` in
  `src/agent/prompt_library/defaults.py`).

**Key asset:** job `f20aa510`'s 759 persisted `qa_provenance_rows` (question,
query, raw web results, decision, answer) = free offline benchmark. Load via
`web.db.load_job_results("f20aa510")` under `railway run` (staging Supabase).
Note: `model_executions` reads need pagination (Supabase caps at 1000 rows);
`build_run_costs_from_model_executions` does NOT work on persisted rows (no
`service` field) — bucket `stage="perplexity_search"` rows by
`metadata.trigger_reason` instead.

## 5. Sprint 2 centerpiece (agreed direction): WebEvidencePlanner

External research handoff (Dusan via Claude chat):
`~/Downloads/rdi-web-evidence-toolkit-implementation-handoff-20260612.md` —
**read it in full**; it is the implementation spec (routes, domain-filter policy,
relevance-gate policy, prompt update, telemetry, acceptance criteria, test cases).

Verdict from this session's review: **architecture endorsed, v1 router replaced.**
Its §14 keyword router was dry-run on the 759 benchmark questions and failed
(60% → company_specific incl. 37% bare default; collisions: "growth" sends
stage questions to sector_market, "revenue" sends market questions to skip;
~105 VC-fit questions have no route; only 37/471 skips, 1 false positive).

**Agreed amendments (full detail: `docs/handoffs/web-search-quality-deep-dive-20260612.md` addendum):**
1. Routing via **decomposition-time LLM tagging** (extend decomposition output
   schema — zero extra LLM calls) + static tags for templated root questions +
   keyword rules only as fallback biased to `company_specific` (failure mode =
   status quo, never a wrong skip).
2. Add **`internal_fit` route** for VC-thesis questions ("fits the VC's
   thesis/stage/check size", ~14% of questions) → no web search.
3. Planner inputs restricted to `name`/`domain`/`industry` —
   `about`/`tagline` can be document-derived (confidentiality). `Company` model
   has NO geo field; add `geo_hint` from Specter HQ data at ingest or omit in v1.
4. **Shadow mode = plan-only logging** (zero extra searches, or spend doubles).
5. Replay harness two-stage: stage 1 free (routing + gate decisions vs persisted
   results), stage 2 sampled live (~80–100 sector/regulation/competitor
   questions re-searched + re-answered) to measure rescue uplift vs 7.4% baseline.
6. Hybrid-prompt variant selected by the same flag → `off` = byte-identical
   current behavior.
7. Multi-query (≤3/question) counts each provider call against the per-company
   cap (`MAX_PPLX_CALLS_PER_COMPANY`, staging=100) and the 3.5s throttle —
   expected net search cost ≈ flat (skips offset multi-query); the win is
   rescue rate, not search spend.

**Suggested PR breakdown (Sprint-1 discipline: independent PRs w/ tests, CI green):**
- PR1: `src/agent/web_search/planner.py` + unit tests (pure logic; benchmark
  dry-run becomes test fixtures; handoff §18 cases required).
- PR2: integration behind `RDI_WEB_EVIDENCE_PLANNER=off|shadow|on` env flag +
  structured telemetry (handoff §13) — wraps `_build_web_search_query` /
  `_web_search_domain_filter` / `_web_results_add_value` at the `do_search`
  choke point (evidence_answering.py ~L556).
- PR3: replay harness (`tools/replay_web_evidence_planner.py`) + report artifact.
- PR4: hybrid prompt variant + decomposition-schema route tagging.
- Rollout: flag `off` by default → replay report → shadow on one batch → `on`
  for one batch → compare → only then discuss default. Acceptance criteria in
  handoff §12, incl. **0/35 rescued-answer false positives** on the benchmark.

## 6. Other Sprint 2 candidates (tasks #14–16 in session task list)

- **Evidence-coverage badge (display-only, option A)** — compute
  `1 - thin_rate` from `qa_provenance_rows` at results build
  (`build_summary_rows`, batch.py:1077) + UI column. **HARD CONSTRAINT: zero
  changes to scoring computation — Dusan's tuned scoring logic is untouchable;
  needs his explicit sign-off before implementing.** Options B (exec-summary
  mention) and C (confidence-weighted scores) explicitly deferred.
- **Specter single-refresh-owner** — web + worker share one rotating OAuth
  refresh token → boot race orphans the chain (caused a full-day outage
  2026-06-11; recovery runbook now at `.claude/skills/specter-token-recovery/`).
  Fix: one service owns refresh (or DB lock around rotation). Medium effort,
  kills a recurring operational risk.
- **Silent worker-subprocess failure UX** — per-company child crash surfaces as
  generic message, child stdout only in DB `analysis_events`, nothing in UI.
  Validated as real during the outage (was Sprint 4 item 1 in the original
  cost plan). Candidate to pull forward.
- **LeadGen URL-task loading check** — observed: LeadGen jobs loaded only 3 url
  tasks while batches had 8/14 eligible leads. Never investigated. Cheap to
  check whether intended (approved-only?) or a bug.
- **`MAX_PPLX_CALLS_PER_COMPANY=100`** on staging is very permissive — arm A
  peaked at 68/company. Revisit AFTER planner lands (skips change the math).

## 7. Reference docs (read order for the planning session)

1. This file.
2. `~/Downloads/rdi-web-evidence-toolkit-implementation-handoff-20260612.md`
   (implementation spec — consider copying into docs/handoffs/ for permanence).
3. `docs/handoffs/web-search-quality-deep-dive-20260612.md` (root cause,
   threads 2+3, review addendum with dry-run numbers).
4. `docs/handoffs/web-search-gating-ab-finding-20260612.md` (A/B + quality data).
5. `docs/handoffs/web-search-query-tooling-research-spec-20260612.md`
   (the research spec that produced #2 — background only).
6. Original cost plan: `~/.claude/plans/can-you-prepare-implementation-jaunty-bubble.md`
   (Sprint 2/3/4 items as originally drafted — re-evaluate against the new
   quality finding; some items may be superseded).

## 8. Binding constraints for the Sprint 2 plan

- Scoring logic: **untouchable** without explicit Dusan sign-off (display-only
  additions OK to propose).
- Confidentiality: no document chunks / Specter data / private memo text to
  external services; planner queries built only from public fields.
- No model-default/provider/pricing changes; LeadGen flags unchanged.
- `WEB_SEARCH_HEAVY_OVERRIDE` default stays `always` (evidence-backed decision).
- Testing stack only; production untouched.
- Each item = independent PR with tests; plan must include verification
  protocol per PR and a rollout/measurement step (replay report → shadow →
  canary batch), mirroring Sprint 1 discipline.
