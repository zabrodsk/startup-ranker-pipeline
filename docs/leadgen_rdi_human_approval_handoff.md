# LeadGen to Deal Intelligence Human Approval Handoff

Audience: a new implementation session in the `rockaway-leadgen` project.

## Goal

Move Rockaway-qualified leads from LeadGen into Deal Intelligence without allowing another accidental large Specter run.

The chosen product direction is:

- LeadGen still builds one Rockaway batch.
- Deal Intelligence receives that batch as a pending intake.
- A human approves the batch inside Deal Intelligence before any intelligence job starts.
- The batch remains one batch so companies can be sorted and compared top-to-bottom.
- Within the Deal Intelligence approval UI, it is acceptable to approve or reject individual companies.
- For now, only Rockaway `PASS` leads are eligible.

Do not split one Rockaway batch into multiple user-facing batches.

## Current Unsafe Flow

Today the bridge is too direct:

1. LeadGen selects Rockaway `PASS` companies.
2. LeadGen calls Deal Intelligence `POST /api/leadgen/ingest`.
3. Deal Intelligence validates the API key and URL/domain.
4. Deal Intelligence immediately creates and starts one full Specter analysis job.

The dangerous coupling is:

```text
ingest == execute
```

This is what allowed an accidental agent-initiated LeadGen run to enqueue a large Deal Intelligence job.

Current Deal Intelligence behavior is documented in:

```text
docs/leadgen_connector_claude_handoff.md
```

That older file describes the existing route contract, not the desired guarded future state.

## Desired Future Flow

The desired contract should become:

```text
LeadGen Rockaway PASS batch
  -> Deal Intelligence pending intake
  -> human review inside Deal Intelligence
  -> approved companies start intelligence processing
  -> results remain tied to the original batch
```

In other words:

```text
ingest == store for review
approval == execute
```

## Product Requirements

1. Keep one batch.

   The user needs to see all Rockaway-qualified companies in one batch and sort them from top to bottom. Do not make separate smaller batches as the primary workflow.

2. Approval lives in Deal Intelligence.

   This is the selected Option B. Deal Intelligence owns the expensive Specter execution boundary, so the approval gate should live there.

3. Individual company approval is allowed.

   Because approval will live in Deal Intelligence, the UI may allow selecting individual companies within the pending batch. The original batch identity must still be preserved.

4. Only Rockaway `PASS` is eligible for now.

   Do not send archive, review, pending-human, or other thesis statuses into the automatic Deal Intelligence intake path.

## LeadGen-Side Changes Needed

LeadGen should continue to own qualification and batch construction, but it should stop assuming that a successful HTTP submission means the companies are being analyzed immediately.

Recommended LeadGen changes:

1. Preserve one explicit Rockaway batch id.

   Keep the existing batch id concept, but make it stable and visible enough for Deal Intelligence review.

2. Include explicit thesis metadata per company.

   Deal Intelligence should not infer Rockaway qualification only from a synthetic score bucket. Each lead should carry enough metadata to enforce:

   ```text
   thesis_key = rockaway
   thesis_status = pass
   ```

   If current LeadGen objects use different field names, map them clearly in the RDI payload.

3. Include stable per-company identifiers.

   Each submitted lead should include LeadGen company id, normalized domain, website, company name, source URL, score/rationale, evidence, and any relevant exclusion flags.

4. Treat pending approval as a successful intake delivery, not a completed push.

   The current `company_push_log` has statuses like `queued`, `sent`, `failed`, and `skipped`. With the new flow, LeadGen should avoid using `sent` to mean "analysis started" unless Deal Intelligence confirms that later.

   Good v1 options:

   - record pending approval as `queued`;
   - add a new `pending_approval` status if the enum can be extended safely;
   - keep `sent` only if the team agrees it means "submitted to RDI intake", not "processed by RDI".

5. Keep a dry-run/manifest mode.

   Even though approval lives in Deal Intelligence, the LeadGen runner should still be able to print or export the exact Rockaway batch before submitting it.

## Deal Intelligence-Side Contract To Target

The exact endpoint can be decided during implementation, but the safest shape is either:

```text
POST /api/leadgen/intake
```

or a behavior change to:

```text
POST /api/leadgen/ingest
```

The response should no longer imply that an analysis job started.

Suggested response statuses:

```text
pending_approval
existing_pending
approved
rejected
invalid
```

Minimal response shape:

```json
{
  "batch_id": "alex-2026-W21-...",
  "intake_id": "lg-intake-...",
  "job_id": null,
  "status": "pending_approval",
  "accepted_count": 100,
  "rejected_count": 0,
  "duplicate_count": 8,
  "errors": []
}
```

`job_id` should remain `null` until a human approves execution.

## Deal Intelligence Approval UI Requirements

The pending batch review UI should show:

- batch id;
- generated timestamp;
- scoring version;
- total lead count;
- company name;
- website/domain;
- LeadGen company id;
- Rockaway score/rationale;
- evidence snippets or links;
- duplicate/recently-analyzed warnings;
- current approval status per company;
- estimated analysis cost or at least estimated company count impact.

Actions:

- approve all eligible companies;
- approve selected companies;
- reject selected companies;
- reject whole batch;
- start approved intelligence run;
- show the resulting Deal Intelligence job id after approval.

The UI should keep the batch as the unit of review even if only some companies are approved.

## Guardrails Required In Deal Intelligence

Deal Intelligence should enforce these server-side, not only in the UI:

1. Do not auto-start analysis on raw LeadGen intake.
2. Require Rockaway `PASS` metadata for every eligible lead.
3. Reject or quarantine non-Rockaway and non-PASS leads.
4. Keep one batch id and make retries idempotent.
5. Dedupe by normalized URL/domain within the batch.
6. Warn on companies already analyzed recently or already present in company history.
7. Record approval actor, approval timestamp, and approved company list.
8. Keep a kill switch for LeadGen intake/execution.
9. Keep a worker/concurrency guard so only intentional approved LeadGen work starts.

## Suggested Deal Intelligence Tables

Exact naming can change, but the concept should be:

```text
leadgen_intake_batches
leadgen_intake_leads
leadgen_intake_events
```

`leadgen_intake_batches` should track:

```text
batch_id
source
generated_at
scoring_version
summary
status
created_at
approved_by
approved_at
rejected_by
rejected_at
job_id
```

`leadgen_intake_leads` should track:

```text
batch_id
leadgen_company_id
company_name
website
domain
normalized_url
thesis_key
thesis_status
score
bucket
rationale
evidence
dedupe_status
approval_status
approved_at
rejected_at
rejection_reason
```

`leadgen_intake_events` should provide an audit trail for submission, validation, approval, rejection, and job creation.

## Suggested Implementation Sequence

Do this in small steps.

1. LeadGen payload audit.

   Confirm the current Rockaway `PASS` source of truth and which fields can be sent to Deal Intelligence without ambiguity.

2. Deal Intelligence intake design.

   Agree final endpoint name, response statuses, tables, and idempotency behavior.

3. Deal Intelligence pending intake implementation.

   Implement storage-only intake first. It must not start a worker.

4. LeadGen client update.

   Update the RDI push client to understand `pending_approval` and preserve one batch id.

5. Deal Intelligence approval UI.

   Build the pending batch review page and approval APIs.

6. Approval-to-execution bridge.

   Only after approval, create the standard Specter job for approved companies.

7. End-to-end smoke test.

   Use a tiny Rockaway PASS batch first, then verify the batch appears pending, approval creates exactly one job, and results stay tied to the original batch.

## Acceptance Criteria

- A raw LeadGen submission cannot start Deal Intelligence analysis by itself.
- One Rockaway batch remains one reviewable and sortable batch.
- Deal Intelligence can show pending batches before execution.
- Human approval is required before Specter work starts.
- Only Rockaway `PASS` companies are eligible.
- Approved companies remain linked to the original batch.
- Re-sending the same batch id is idempotent.
- LeadGen no longer treats intake submission as proof that analysis ran.
- There is an audit record of who approved what and when.

## Notes For The New LeadGen Session

The immediate LeadGen-side question is not "how do we call the current RDI ingest endpoint?" The current endpoint works but is unsafe.

The better question is:

```text
What exact Rockaway PASS batch payload should LeadGen send so Deal Intelligence can store, review, sort, dedupe, and approve it before execution?
```

Start by inspecting the current LeadGen RDI push path:

```text
app/jobs/run_push_downstream.py
app/push/rdi.py
supabase/migrations/0005_push_downstream_log.sql
docs/weekly-leadgen-routine.md
```

Then align it with the guarded Deal Intelligence intake contract above.
