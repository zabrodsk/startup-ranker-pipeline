# LeadGen URL-task loading diagnostic (Sprint 4 A4)

> Read-only investigation. The SQL below is **Dusan-executed** against the
> staging Supabase project (`ykxtuqcfhpauddnbxqyq`) — via the Supabase dashboard
> SQL editor or under the scoped `railway run`. No code change ships by default;
> if the funnel reveals a real bug, propose a separate scoped PR.

## The anomaly

A LeadGen job loaded **3** URL tasks while the intake batches reportedly had
**8 of 14 eligible** leads. We need to reconcile "8 eligible at intake" against
"3 URL tasks in the worker" and identify which narrowing stage accounts for it.

## Code path (verified on `testing`)

There are **four** narrowing stages between "eligible at intake" and "URL task
in the worker":

1. **Intake classification** — `_prepare_leadgen_intake` (`web/app.py:5895-5956`)
   sets `eligible` + `approval_status`. A lead is marked ineligible (`invalid` /
   `ineligible` / `duplicate`) only for: missing/invalid URL, failing
   `_leadgen_is_rockaway_pass`, or in-batch URL duplicate. **It does NOT apply the
   source-domain blocklist** — so `eligible=True` here can still be dropped later.
2. **Approval gate** — `approve_leadgen_intake` (`web/app.py:6166-6178`) keeps a
   lead only if `eligible == True` **AND** `approval_status == "pending"` **AND**
   `not _leadgen_stored_lead_rejection(lead)`. That third condition
   (`web/app.py:6078`) re-runs `_leadgen_source_domain_rejection` (`:5731`,
   blocklist `_LEADGEN_SOURCE_DOMAINS` `:5532`) on the stored URL/domain —
   **the prime suspect**: intake-eligible leads pointing at a source platform
   (not a company site) are silently dropped here. Also, `approve_all_eligible=False`
   with a short `lead_ids` selection narrows further.
3. **URL extraction / intake dedup** — empty `lead.url` is skipped; the URL list
   is de-duplicated by normalized URL (`_normalize_url_items_for_intake`).
4. **Worker domain-ROOT dedup** — `_build_company_tasks`
   (`specter_batch_worker.py:152-159`) collapses two URL tasks that share a domain
   *root* into one. **Prime suspect for the task-count gap specifically** (8 eligible
   → several sharing a root → 3 tasks).

## Read-only query (run for the affected `intake_id`)

```sql
-- (a) funnel by status + eligibility
select approval_status, eligible, count(*)
from leadgen_intake_leads
where intake_id = '<INTAKE_ID>'
group by approval_status, eligible
order by approval_status, eligible;

-- (b) per eligible lead: url, domain, and registrable root (to spot dedup + source-domain hits)
select company_name, url, domain, approval_status, rejection_reason, duplicate_of_url
from leadgen_intake_leads
where intake_id = '<INTAKE_ID>' and eligible = true
order by domain;

-- (c) the approval/queue events for this intake (who approved, which lead_ids)
select * from leadgen_intake_events
where intake_id = '<INTAKE_ID>'
order by created_at;
```

## Funnel reconciliation

Fill in from the query output:

```
eligible (intake, query a)                         = ____
  − leads with approval_status != "pending"        = ____
  − source-domain re-rejections (query b, domain ∈ _LEADGEN_SOURCE_DOMAINS) = ____
  − empty-URL leads                                = ____
  − intake URL-dedup collapses                     = ____
  − worker domain-ROOT dedup collapses (query b, count distinct registrable roots) = ____
  = expected URL-task count                        = ____  (should reconcile to 3)
```

The stage where the count drops most is the answer.

## Findings (fill after running)

- Funnel table: _TBD_
- Stage that accounts for 8 → 3: _TBD_
- Root cause: _TBD_

## Likely conclusion + proposed follow-up

Most probable: intake-time `eligible` **over-counts** because it does not apply
the source-domain rule that approval-time enforces (stage 1 vs stage 2 mismatch),
and/or domain-root dedup legitimately collapses same-root URLs the operator counted
separately. Both are **expectation/UX bugs**, not data loss.

If confirmed a real bug, propose a separate scoped PR (out of this sprint's default
budget): apply `_leadgen_source_domain_rejection` at intake so `eligible`/
`approval_status` reflect the true funnel, and/or surface a pre-approval
"will-queue N of M" count that accounts for domain-root dedup. Each would ship with
its own off-mode test.
