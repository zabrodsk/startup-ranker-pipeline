# Leadgen to Deal Intelligence Connector Handoff

Audience: Claude Code working in `rockaway-leadgen`.

## Short Answer

Use:

```text
POST https://rockaway-deal-intelligence.up.railway.app/api/leadgen/ingest
X-API-Key: <shared leadgen API key>
Content-Type: application/json
Accept: application/json
```

The Deal Intelligence side now implements this route. A 404 from the testing host means that host has not been redeployed with this commit, not that leadgen should use a different path.

## Required Configuration

Deal Intelligence testing must have:

```text
LEADGEN_API_KEY=<shared secret>
```

Leadgen should use the same value as its analyzer key, for example:

```text
DEAL_ANALYZER_BASE_URL=https://rockaway-deal-intelligence.up.railway.app
DEAL_ANALYZER_INGEST_PATH=/api/leadgen/ingest
DEAL_ANALYZER_API_KEY=<same shared secret>
```

Do not send the key as a bearer token. Send it as `X-API-Key`.

## Request Contract

Send one leadgen batch per request:

```json
{
  "batch_id": "smoke-rdi-1",
  "generated_at": "2026-05-23T06:30:04+00:00",
  "scoring_version": "phase1-rubric-v1",
  "summary": {
    "lead_count": 2,
    "high_priority_count": 2,
    "review_count": 0,
    "archive_count": 0
  },
  "leads": [
    {
      "lead": {
        "name": "Example Company",
        "website": "https://example.com",
        "domain": "example.com",
        "source": "specter",
        "source_url": "https://source.example/company",
        "raw": {},
        "people": []
      },
      "score": {
        "score": 100,
        "bucket": "high_priority",
        "rationale": "Leadgen rationale",
        "version": "phase1-rubric-v1",
        "exclusion_flags": [],
        "evidence": [],
        "notes": []
      }
    }
  ]
}
```

The `lead` and `score` objects are intentionally permissive. Deal Intelligence currently uses `lead.website`, then `lead.domain`, plus score metadata for review context.

## Response Contract

Successful accepted batch:

```json
{
  "batch_id": "smoke-rdi-1",
  "job_id": "lg-smoke-rdi-1-...",
  "status": "created",
  "accepted_count": 1,
  "rejected_count": 0,
  "accepted_urls": ["https://example.com"],
  "errors": []
}
```

Retrying the same `batch_id` returns the same `job_id` with:

```json
{
  "status": "existing"
}
```

If every lead lacks a usable `website` or `domain`, the endpoint returns HTTP 202 with:

```json
{
  "job_id": null,
  "status": "rejected",
  "accepted_count": 0,
  "rejected_count": 2,
  "accepted_urls": [],
  "errors": [
    {
      "lead_name": "No URL",
      "url": null,
      "reason": "Missing or invalid website/domain."
    }
  ]
}
```

## Error Meanings

- `404 {"detail":"Not Found"}`: testing host is not deployed with this connector route.
- `503 {"detail":"Leadgen ingest is not configured."}`: Deal Intelligence is missing `LEADGEN_API_KEY`.
- `401 {"detail":"Invalid leadgen API key."}`: missing or wrong `X-API-Key`.
- `422`: malformed JSON shape, usually `leads[].lead` is not an object.

## Deal Intelligence Behavior

For each accepted batch, Deal Intelligence creates exactly one standard Specter URL analysis job:

- `input_mode="specter"`
- `use_web_search=true`
- `use_specter_mcp=true`
- `fetch_full_team=true`
- `run_name="leadgen:{batch_id}"`

It accepts all leadgen score buckets, including `archive`. Leadgen owns batch sizing and which leads are included. Deal Intelligence only rejects missing or invalid URL/domain values.

There is no cross-company dedupe. A different `batch_id` creates a new run even if the companies have been analyzed before. A repeated `batch_id` is idempotent and does not enqueue another job.

Do not send leadgen company names as Specter `expected_name`. The connector uses URL/domain as the resolver guardrail.

Leadgen metadata is stored in the Deal Intelligence job `run_config` under:

```text
source = "leadgen"
leadgen.batch_id
leadgen.generated_at
leadgen.scoring_version
leadgen.summary
leadgen.leads[]
```

## Smoke Test

After Deal Intelligence testing is redeployed and both sides have matching keys:

```bash
curl -sS \
  -X POST "https://rockaway-deal-intelligence.up.railway.app/api/leadgen/ingest" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-API-Key: $DEAL_ANALYZER_API_KEY" \
  --data @smoke-leadgen-batch.json
```

Expected:

- HTTP 202.
- Response includes one `job_id`.
- `status` is `created` on first send.
- Repeating the same request returns the same `job_id` with `status="existing"`.
- Deal Intelligence shows the run as `leadgen:{batch_id}`.
- Final results can be opened and sorted by Deal Intelligence composite score.

## Deal Intelligence Verification

The connector implementation is covered by:

```bash
.venv/bin/python -m pytest tests/test_leadgen_ingest.py -q
```

The shared Specter worker start path was also checked with:

```bash
.venv/bin/python -m pytest \
  tests/test_specter_ingest.py::test_start_analysis_queues_worker_backed_specter_job_without_starting_thread \
  tests/test_specter_ingest.py::test_start_analysis_worker_queue_failure_does_not_fallback_to_web \
  -q
```
