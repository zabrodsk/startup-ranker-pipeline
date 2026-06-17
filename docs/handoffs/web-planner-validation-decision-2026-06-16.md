# WebEvidencePlanner + Skip-Unanswerable Gate — Validation Outcome & Decision (2026-06-16)

**Decision: PARK both** the WebEvidencePlanner (`RDI_WEB_EVIDENCE_PLANNER`) and the
skip-unanswerable-search gate (`RDI_SKIP_UNANSWERABLE_SEARCH`). Keep the merged Sprint 2/4
code as-is; **all flags stay OFF by default**. No production enablement, no canary.

## What these were
- **WebEvidencePlanner** (Sprint 2): a route-aware replacement for the naive "documents
  incomplete" web-search fallback — classify the question's route, build targeted queries,
  keep only relevant web evidence. Goal: rescue more thin answers at controlled cost.
- **Skip-unanswerable gate / C1** (Sprint 4): suppress web searches predicted unanswerable;
  delegates its skip judgment to the planner's router (so it inherits the 0/35 safety gate).

## How they were validated (railway-gated replay harness, benchmark job f20aa510)
- **Safety gate — PASS.** `--stage replay` false-positive check = **0/35** after the #29
  routing fix (internal_fit searches company-anchored; skip_public_web skips only on strict
  private metrics).
- **C1 skip-rate — 0%** on this cohort (`skip_rate_documents_incomplete: 0.0`). Early-stage
  "documents incomplete" questions are company-facts, not private metrics → nothing to skip.
- **Live rescue — 35.2%** (88 thin questions sampled, 31 rescued) vs 7.4% baseline. By route:
  geography 7/8, sector_market 8/16, competitors 6/16, customer_need 6/16, regulation 4/16,
  technology_validation 0/16.

## The decisive step: manual quality read (top-15 rescues)
The quantitative bars passed, but a domain-expert read of the actual rescued answers
(`stage2_answers.csv`, top 15 by web-evidence-added) showed the rescue metric **overstates
value**:

1. **The baseline already searches the web.** Pre-planner answers already cite `[web]` /
   "web sources". The planner re-runs a search the pipeline already ran — it is not adding
   web where there was none.
2. **"Rescue" is mostly reclassification.** The old answer leads with "Insufficient
   information available" (which trips the thin-answer detector); the planner rewrites the
   *same generic sector content* as "Company-specific: unavailable; Sector-level: …" (which
   does not). Same information, reclassified thin→rescued. This accounts for most of the
   7.4%→35.2% lift.
3. **Genuine value is narrow (~2/15).** Only the **regulation route on regulated / dual-use
   companies** added real diligence value — e.g. naming EU Regulation 2021/821 and ITAR/EAR
   for a defense-UAV startup. A few others added a single named regulation/stat (EU AI Act;
   "~5% of EU procurement is cross-border"). The majority were generic VC boilerplate with
   zero company specifics.
4. **Mis-grounding risk.** One rescue analyzed *medical-device regulation* (HTA, payer
   coverage, device classification) for a **B2B lead-gen company** — wrong sector entirely.
   In production this is worse than "insufficient info": confident, plausible, and wrong.
   The 0/35 safety gate does **not** catch answer mis-grounding — it only checks that
   searches aren't wrongly *skipped*. The manual read is what caught it.
5. **Formatting note.** The replay's raw `[{'type':'text', ...}]` output is a harness
   artifact (`str(response.content)` in `web/replay_web_planner.py`); the production path
   extracts clean text. The substance problem is independent of it.

## Why park (not enable, not canary)
On the current early-stage deal cohort the planner mostly **reformats web color the baseline
already had**, occasionally adds a useful regulatory fact, and occasionally fabricates the
wrong sector. No broad decision value + a real mis-grounding risk. A clean canary is unlikely
to change the verdict: the replay was a fair-enough quality test — the planner *had* the
documents (note the `[chunk_N]` cites) and produced real web answers; only industry/geo
metadata was absent.

## What remains open (not scheduled)
- **Narrow niche:** regulation / compliance questions for regulated or dual-use sectors
  (defense, aerospace, possibly fintech / medtech) is the one pattern with real value. If
  ever revisited, scope it to that route + a sector gate, and add an **answer-grounding
  check** to catch the mis-grounding failure mode. Not worth enabling now.

## State / action items
- Flags ship **OFF by default**: `RDI_WEB_EVIDENCE_PLANNER`, `RDI_SKIP_UNANSWERABLE_SEARCH`.
  Merged Sprint 4 safety fixes (A1 CORS allowlist, A5 SESSION_SECRET fail-closed, A2 worker
  stop) are unaffected and stay merged.
- **Verified 2026-06-16:** neither flag is set in the Railway **staging** env (`railway status`
  = staging; `railway variables | grep` returned nothing) — both sit at their default **off**.
  Validation used the offline/live replay harness, which calls the router directly regardless of
  the prod flag, so this confirms nothing was left enabled.
- Replay tooling + `scripts/show_rescued_examples.py` (rescued-example inspector, company
  names tokenised) are retained for any future re-evaluation. Confidential replay artifacts
  live under `exports/` (gitignored).
