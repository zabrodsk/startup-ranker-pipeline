# Web-Search Gating A/B — Finding (2026-06-12)

**TL;DR:** The `WEB_SEARCH_HEAVY_OVERRIDE` gating (Sprint 1 W1/W2) is a **negligible cost
lever** for the companies RDI actually analyzes. On a 10-company early-stage cohort,
only **1.3%** of web searches (6 of 476) are addressable by `root_only`. The real
web-search cost is the **"documents incomplete" fallback** (98.7% of searches),
governed by `WEB_SEARCH_TRIGGER` and capped by `MAX_PPLX_CALLS_PER_COMPANY=100`.

## What we tested

Does flipping `WEB_SEARCH_HEAVY_OVERRIDE` from `always` (default) to `root_only`
meaningfully reduce Perplexity/web-search spend?

- The flag gates **only** condition 2 in `evidence_answering.py` (~line 549):
  `_question_prefers_web_search()` — the market/TAM/competitor "heavy" trigger,
  reason string `"question benefits from external web context"`.
- It does **not** gate condition 1 (~line 547): `_answer_indicates_no_evidence()`
  when `WEB_SEARCH_TRIGGER=answer`, reason `"documents incomplete"`. That fires in
  all modes (`always`/`root_only`/`never`). The `elif` chain means condition 1 takes
  precedence — a sparse-data answer is labeled "documents incomplete" and never
  reaches the gated heavy check.

## Method

- **Arm A** (`WEB_SEARCH_HEAVY_OVERRIDE=always`, the default): job `f20aa510`,
  10 Specter-resolvable companies pulled from LeadGen batches
  (panathenea + alex-W23 intakes), web search ON + deep team ON. All 10 completed
  (`completed=10, failed=0`).
- Companies: hardalion.com, mantic.gr, hellomaiko.com, orchestrai.dev, generect.com,
  seriousmind.app, weatherxm.com, passersystems.com, asistent.site, botshift.co.
  (helvia.ai was excluded — not in Specter's intelligence dataset.)
- Telemetry source: persisted `model_executions` rows (paginated past Supabase's
  1000-row cap), bucketed by `metadata.trigger_reason`. NOTE: the live
  `/api/costs/{job_id}` aggregator (`build_run_costs_from_model_executions`) keys on
  an in-memory `service` field that is **not** persisted to the DB, so offline cost
  reads must bucket on `stage=="perplexity_search"` + `metadata.trigger_reason`
  directly.
- **Arm B (`root_only`) was not run.** Arm A is decisive: the max possible delta is
  the 6 heavy searches, and `root_only` only suppresses the *non-root* subset of
  those. Spending ~2x budget to measure a 0–1.3% delta wasn't justified.

## Arm A results (476 web searches, 10 companies, 1950 model executions)

| Trigger reason | Count | Share | Gated by HEAVY_OVERRIDE? |
|---|---|---|---|
| `documents incomplete` | 470 | 98.7% | No (all modes) |
| `question benefits from external web context` | 6 | 1.3% | Yes (root_only cuts non-root subset) |

Per-company web searches (all "documents incomplete"-dominated): 33–68
(passer-systems 68, hellomaiko 56, orchestrai 61, generect 51, weatherxm 43,
realmint-ai 42, botshift 41, hardalion 41, mantic 40, virtuelni-asistent 33).

Heavy-trigger searches were spread across 5 companies (hellomaiko 2; botshift,
weatherxm, mantic, virtuelni-asistent 1 each).

Per-stage LLM volume (calls / prompt-tok / completion-tok), top consumers:
`answering` 1124 / 1.65M / 111k · `refinement` 30 / 268k / 5k ·
`critique` 70 / 494k / 11k · `evaluation` 90 / 75k / 87k. Total 1950 calls,
2.87M prompt + 308k completion tokens.

## Conclusion

- **Do not flip the `WEB_SEARCH_HEAVY_OVERRIDE` default for cost reasons.** It's
  harmless but saves ~nothing for early-stage/sparse-data companies — which *is*
  RDI's real use case (VC deal screening of startups). Keep default `always`.
- **The real web-search lever is the "documents incomplete" fallback** (470/476).
  Candidate follow-ups (future sprint):
  - Lower `MAX_PPLX_CALLS_PER_COMPANY` from 100 (companies are running 33–68; the
    cap is far above usage, so it's not currently binding — but a tighter cap would
    directly bound worst cases).
  - Tighten the `_answer_indicates_no_evidence` heuristic / `WEB_SEARCH_TRIGGER`
    policy so a thin-but-present grounded answer doesn't always fall through to a
    web search.
  - Investigate the `answering` stage (1.65M prompt tokens) — it's the dominant LLM
    token sink and is fed by these searches.

## Quality spot-check (added after Dusan's quality concern, same day)

Pulled all 759 `qa_provenance_rows` for `f20aa510` and checked answers against
`_answer_indicates_no_evidence`:

- **58% (441/759) of all evidence answers are "thin"** ("Insufficient information
  available" style) — the scores for this cohort rest on a majority-thin evidence base.
- **Web-search rescue rate is only 7.4%**: 471 answers got the hybrid docs+web
  re-answer; 436 of them (92.6%) *still* concluded insufficient. The enrichment
  pipeline is wired correctly (web results do reach the hybrid prompt,
  `evidence_answering.py` ~L600), but for these companies the searches rarely
  convert an unanswerable question into an answered one.
- Rescue rate is uniformly bad across question types (private-fact 5.6%,
  public-context 8.5%, other 7.4%) — so it's not just "the web can't know their
  revenue"; even market/sector questions rarely get rescued.
- Positive: the model does NOT hallucinate to fill gaps — thin answers honestly
  say insufficient and cite what the docs do show.

**Implication:** the 470-search spend buys ~35 rescued answers per 10 companies.
The cost lever and the quality lever are the same investigation:
1. Why don't web results rescue answers? (query construction in
   `_build_web_search_query` may be too company-anchored — stealth startups have
   no web footprint; or the hybrid prompt over-anchors on docs; or
   `_web_results_add_value` passes "relevant" results that lack substance.)
2. Should predictably-unanswerable questions (round size / revenue of a
   no_funding 2025 company) skip the search entirely?
3. Should per-company evidence coverage (% answered) surface in the report/score
   so IC readers can calibrate trust in the score?

Per-company thin rates: orchestrai 79.7%, passer-systems 69.5%, hellomaiko 63.6%,
generect 56.9%, mantic 54.9%, realmint 54.7%, botshift 52.7%, hardalion 51.4%,
weatherxm 48.8%, virtuelni-asistent 44.8%.

## State after this run

- Staging env unchanged: `WEB_SEARCH_HEAVY_OVERRIDE` unset (= `always`),
  `WEB_SEARCH_TRIGGER` unset (= `answer`), `MAX_PPLX_CALLS_PER_COMPANY=100`.
  Nothing to restore.
- Specter MCP auth was re-minted earlier today (client `3819f181…`); chain healthy.
  See `.claude/skills/specter-token-recovery`.
