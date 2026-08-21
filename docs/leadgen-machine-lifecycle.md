# LeadGen machine lifecycle contract v1

Sprint 6A adds a per-company machine boundary without changing the recovered
LeadGen ingest, human approval, browser status, browser result, or RDI scoring
surfaces.

## Route family

All routes require `X-LeadGen-Service-Key`. The server compares it in constant
time with `RDI_LEADGEN_AUTOSTART_KEY`. A missing server configuration returns
503; a missing or wrong request credential returns 401. The credential is never
accepted in a body or returned in a response. Machine authentication runs before
request-body parsing. Authenticated validation failures return one bounded 422
problem and never echo submitted input.

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/machine/leadgen/v1/intakes` | Create or exactly replay one company intake. |
| `POST` | `/api/machine/leadgen/v1/intakes/{intake_id}/start` | Fence, start, or replay one evaluation. |
| `GET` | `/api/machine/leadgen/v1/intakes/{intake_id}/status` | Read the closed lifecycle state. |
| `GET` | `/api/machine/leadgen/v1/intakes/{intake_id}/result` | Read an authoritative successful terminal result only. |
| `GET` | `/api/machine/leadgen/v1/intakes/{intake_id}/error` | Read a sanitized terminal failure only. |
| `GET` | `/api/machine/leadgen/v1/availability` | Read the shared Specter MCP gate without creating work. |

The contract version is `rdi.leadgen-machine.v1`. Request models reject extra
fields. Identifiers are bounded, and the only target environments are `staging`
and `production`. `RDI_LEADGEN_TARGET_ENVIRONMENT` must configure exactly one of
those values on the server; missing, invalid, or mismatched configuration fails
before intake persistence or start reservation. Machine domains and their
immutable persistence keys use one dedicated canonical policy: lower-case host,
valid HTTP(S) scheme/port, path removed, and only a leading `www.` removed.
Shared source/social hosts and their subdomains are rejected; real company
subdomains are preserved. Existing human LeadGen routes remain unchanged.

## Specter MCP availability

Daily Specter quota exhaustion is a provider condition, not a company failure
or the separate machine daily-start limit. The service-only Supabase gate is
shared by web and worker processes and is keyed by RDI target environment.
Known exhaustion returns HTTP 429 with `Retry-After` and
`machine_specter_mcp_quota_exhausted`; gate-storage or transient provider
failures return 503. Exact intake replays remain readable while genuinely new
intakes and start reservations are rejected. CSV/export and document-only runs
do not consult this gate.

`SPECTER_MCP_QUOTA_GATE_MODE=observe` is the rollout default. In `enforce`, the
backend fails closed if gate storage cannot be read. The first block lasts to
00:05 UTC; afterward one process leases a 60-second harmless recovery probe.
Success or a definitive company-not-found response reopens the gate. Other
responses re-block it for five minutes. There is no manual clear route.

An intake request carries:

- external LeadGen company ID;
- canonical company-owned domain;
- campaign, iteration, source-run, and batch IDs;
- stable per-company idempotency key;
- target environment;
- Prague business date and the fixed `Europe/Prague` business timezone;
- bounded provenance reference.

The caller sends one company per intake. The worker receives only the canonical
company URL: the external company ID is never supplied as an expected display
name.

## Stable identity and replay

The idempotency identity is derived from all stable caller identifiers and the
target environment. The material payload hash also covers canonical domain and
provenance. Exact intake replay returns the same serialized response and durable
intake/correlation references. `rdi_company_id` remains null until authoritative
RDI persistence supplies a company UUID. Reusing the identity with changed
material payload returns 409 and cannot start work.

The start reference is deterministic from the intake ID. Exact and concurrent
start replay return that reference without entering the remote start seam again.
The auditable actor is always `service:rockaway-leadgen`.

## Start gate, ordering, and uncertainty

`RDI_LEADGEN_AUTOSTART_ENABLED` defaults disabled and accepts only the explicit
value `true`. Other unset/false values remain disabled; malformed values fail
closed. `RDI_SCORING_VERSION` must contain an explicit bounded version before a
reservation can be created. `RDI_LEADGEN_DAILY_START_LIMIT` defaults to 20 and
is enforced atomically across all campaigns for the server-configured target
environment and current Prague business date. The deprecated
`RDI_LEADGEN_GLOBAL_START_LIMIT` is accepted only when the preferred variable is
unset; configuring both fails closed. The persisted environment/date scope,
reservation parameter, advisory lock, count, response capacity fields, and
audit event all use or verify that server scope. A caller cannot select another
environment or date partition. Exact replays resolve before the current-date
check and do not consume or bypass capacity; new starts for a closed date fail.

The durable order is:

1. validate service authentication, flag, intake, environment, state, and rate;
2. commit the `start_fenced` reservation and stable job reference;
3. enter the injected existing RDI start adapter once;
4. record `queued`; atomically release the exact fence only for a typed,
   proven no-start; or conservatively record `uncertain` when acceptance may
   have occurred.

An uncertain response keeps the same job and correlation references. Replay
never starts again. If finalizing the remote outcome itself cannot persist, the
committed `start_fenced` state remains an uncertain no-retry fence.
The production wrapper emits typed definite rejection only for proven checks
before a worker-visible write: disabled worker execution, unconfigured worker
storage, and the real provider preflight's quota or unavailable branches. The
provider cases use fixed redacted messages and preserve the human/browser JSON
response behavior. After preflight, exceptions, malformed replies, and
non-typed 400/429/503 responses are ambiguous because persistence or queueing
may already have committed. An injected raw response is never treated as proof
of no start. A definite rejection clears the provisional job, actor, start
timestamp, and safe error fields through the release RPC; only after release
succeeds may an exact retry enter the adapter.

## Lifecycle and terminal projections

The closed states are:

`accepted`, `rejected`, `start_fenced`, `uncertain`, `queued`, `running`,
`succeeded`, `failed`, and `cancelled`.

Nonterminal status returns HTTP 202 with `terminal=false`. Result and error are
separate endpoints and return 409 until their corresponding authoritative
terminal state exists. HTTP acceptance, `start_fenced`, `uncertain`, `queued`,
and `running` are never final success.

A successful result contains the external company identity, the exact persisted
`company_runs.company_id` UUID, stable intake/job/result references, exact
persisted decimal score strings, RDI bucket, authoritative completion timestamp,
pipeline version, explicit persisted `rdi_scoring_version`, and a SHA-256
checksum of the canonical result body excluding the checksum field. Production
correlation selects the persisted company run by exact job ID plus a nullable
immutable `source_company_key` derived from the requested machine domain. The
provider-domain `company_key` remains distinct, so supported aliases such as
`adspawn.com` resolving to `adspawn.io` still return the exact persisted run and
company UUID. Missing or duplicate source matches fail closed; display names
are never correlation keys.

The worker carries that requested `domain:<host>` source key on URL tasks and
prefers it when reconstructing completed company runs after restart. Provider
name, domain, or slug changes therefore cannot repeat an already-persisted
machine evaluation. Legacy and non-machine rows without the new key retain the
existing name/slug fallback.

All runtime version fields are also stored in durable `run_config`, where the
worker reconstructs them. Claim, heartbeat, and finalization partial upserts
omit absent version columns, so they cannot erase an earlier non-null pipeline
version.

A terminal error contains only bounded code, class, safe message, and persisted
terminal timestamp. Raw provider payloads, stack traces, mutable details, and
credential-like text are not projected.

## Persistence and rollout

`supabase/migrations/20260731000000_leadgen_machine_lifecycle.sql` establishes
the machine lifecycle. The forward-only correction
`20260807102328_leadgen_machine_daily_start_scope.sql` backfills the Prague
business date, registers one canonical campaign per environment/date scope,
drops the unsafe environment-only reservation signature, and installs the
daily-scoped reservation/capacity RPCs. RLS, fixed-search-path security-definer
RPCs, row locking, and RPC-only service-role grants remain enforced. Direct
table DML is revoked from all application roles. All mutations, including
definite-no-start fence release and its audit event, go through atomic RPCs; the
adapter does not use read-then-write REST composition. Rollback is operational:
disable autostart and roll back application code; do not reverse the additive
data migration.

Apply the migration to staging before deploying the matching route code. Until
the migration and required environment configuration are present, machine
routes fail closed with storage/configuration errors. Production migration and
production deployment remain separate, explicitly authorized operations.

The schema change is additive. A staging code rollback can leave the unused
tables, functions, nullable column, and index in place; direct application-role
table access remains revoked. Removing those objects is a separate destructive
database operation and is not part of the normal rollback path.
