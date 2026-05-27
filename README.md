# Deal Intelligence

**LLM-based deal evaluation platform for Rockaway Capital**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Work in Progress](https://img.shields.io/badge/status-WIP-orange.svg)](./docs/RECENT_UPDATES.md)

Evaluates pitch decks, metrics, and optional web data to produce ranked deal lists with pro/contra arguments and invest recommendations.

---

## Overview

Deal Intelligence helps investment teams evaluate and prioritize deal flow. Upload pitch decks, metrics, or Specter CSVs — or paste a list of company URLs — and optionally enable web search and Specter MCP augmentation for extra structured signal. The output is ranked deal lists with structured scores, executive summaries, key points, and red flags.

**Pipeline stages:**

1. **Data Collection** — Ingest documents (PDF, PPTX, DOCX, XLSX, CSV) and optional Perplexity/Brave web search
2. **Knowledge Organization** — Build hierarchical question trees
3. **Argument Generation** — Pro and contra investment arguments grounded in evidence
4. **Iterative Refinement** — Devil's advocate critique and refinement
5. **Ranking** — Composite scores (strategy fit, team, upside) with invest/not-invest recommendation

---

## Quick Start

### Install

```bash
pip install -e . "langgraph-cli[inmem]"
cp .env.example .env
```

Add your API keys to `.env` (see [Configuration](#configuration)).

Use separate infrastructure for non-production and production deployments. The recommended layout is:

- local `.env` for development
- one shared Railway + Supabase stack for staging/testing
- one separate Railway + Supabase stack for production

See [`docs/environments.md`](./docs/environments.md) for the deployment checklist and environment matrix.

### Run the Web App (recommended)

```bash
python -m web
```

Then open **http://localhost:8000** in your browser. Set `APP_PASSWORD` in `.env` for login.

- **Upload** PDF, PPTX, Word, Excel, or CSV files
- Choose **Pitch Deck** (each file = one company), **Specter** (company + people CSVs), or **Multi-file** (all files = one company)
- Optionally enable **web search** for richer evidence
- Get ranked results with executive summaries, key points, red flags, and saved analysis history

### Run Batch Mode (CLI)

**From documents (one folder per startup):**

```bash
python -m agent.batch --input ./deals --output results.xlsx
```

**From Specter CSVs (company + people exports):**

```bash
python -m agent.batch --specter-companies companies.csv --specter-people people.csv --output results.xlsx
```

## Collaboration

Use GitHub for shared code and ChatGPT Projects for shared AI context. Do not have two people edit the same iCloud-synced working folder at the same time.

### Recommended setup

1. Add your collaborator to the GitHub repository.
2. Each person clones the repo to their own machine and opens their local clone in Codex.
3. Each person copies `.env.example` to `.env` locally and fills in their own secrets.
4. Work on short-lived branches and merge through GitHub.

### Daily workflow

```bash
git checkout main
git pull origin main
git checkout -b your-branch-name
```

Make changes in Codex, then:

```bash
git add .
git commit -m "Describe the change"
git push -u origin your-branch-name
```

Open a pull request, review, merge, then sync `main` again before starting the next task.

### Shared context

- Share the ChatGPT Project if you want shared project instructions, uploaded files, and related chats.
- Use GitHub as the source of truth for code and assets in this repository.
- Keep `.env` private and out of git. Commit changes to `.env.example` only when setup instructions need to change.

---

## Web App

The web UI lets you upload files, run analyses, and review saved results without touching the CLI.

| Feature | Description |
|---------|-------------|
| **Upload** | PDF, PPTX, DOCX, XLSX, CSV — single or multi-file |
| **Input modes** | Pitch Deck (1 file per company), Specter (2 CSVs **or** URL list), Multi-file (all files = 1 company) |
| **Web search** | Optional Perplexity/Brave search for extra evidence |
| **Specter MCP augmentation** | In Pitch-Deck mode, auto-extracts the company URL from deck text (regex over scheme URLs, bare domains, `Website:` labels, and emails) and pulls structured Specter intelligence — funding, team, growth signals, investor highlights — merging into the deck's evidence store. In Specter URL-list mode, the MCP is the primary intake. Optional "Fetch deep team profiles" sub-toggle adds full LinkedIn-grade career history per founder. |
| **VC strategy** | Optional investment thesis for tailored scoring |
| **Results** | Summary table, executive summaries, key points, red flags, pro/contra arguments |

### Local run

```bash
python -m web
# or
uvicorn web.app:app --reload --port 8000
```

### Local Specter worker

The dedicated Specter worker is designed to run separately from the web app so
long-running Specter batches do not bloat the web process.

```bash
python -m agent.specter_batch_worker
```

For a single polling pass during testing:

```bash
python -m agent.specter_batch_worker --run-once
```

Safe rollout:

- Keep the web app on `python -m web.app`
- Run the worker separately on `python -m agent.specter_batch_worker`
- Enable the worker-backed Specter path with `ENABLE_SPECTER_WORKER_SERVICE=true`
- Leave the flag unset or `false` to keep the current in-web Specter fallback

The worker polls the queue on a configurable interval via
`SPECTER_WORKER_POLL_SECONDS`. The code default is `5` seconds; the current
Railway production worker overrides this to `10` seconds to reduce idle polling
noise and network chatter.

### Specter MCP setup (URL-list intake + pitch-deck augmentation)

Specter MCP unlocks two intake paths that don't require uploading CSVs:

1. **URL-list intake** — In Specter mode, paste one URL per line instead
   of (or alongside) the company/people CSVs.
2. **Pitch-deck augmentation** — In Pitch-deck mode, the company URL is
   auto-extracted from the deck text and Specter chunks are merged in.
   Default ON; toggle off in the UI to skip.

Both paths use the same OAuth-authenticated Specter MCP client.

**One-time auth** — Run the OAuth helper to mint a refresh token:

```bash
python scripts/specter_oauth_login.py
```

Paste the resulting `SPECTER_MCP_CLIENT_ID` and `SPECTER_MCP_REFRESH_TOKEN`
into your `.env` (locally) or Railway env vars (deployed). On every
subsequent token rotation the new refresh token is persisted to the
`mcp_secrets` Supabase table — you do not need to re-run the helper unless
the session reaches its bounded maximum lifetime.

**Cost control** — The deep-team toggle (UI: "Fetch deep team profiles")
controls per-founder profile fan-out. Default OFF: ~3 MCP calls per
company (find + profile + intelligence + financials). Default ON: same
plus one `get_person_profile` per founder/key person — adds ~60% more
calls and yields full LinkedIn-grade career history. Recommended OFF for
pitch-deck mode (deck usually carries founder bios) and ON for URL-list
mode where there is no deck context.

### Railway deployment layout

Railway runs the same image in two service roles:

- `startup-ranker-web` starts `python -m agent.railway_service` with `SERVICE_ROLE=web`
- `startup-ranker-worker` starts `python -m agent.railway_service` with `SERVICE_ROLE=worker`

Recommended shared staging / production behavior:

- `ENABLE_SPECTER_WORKER_SERVICE=true` on web, so Specter runs queue to the dedicated worker
- `RESTART_ON_IDLE_AFTER_ANALYSIS=true` on web, so the web process can recycle after completed analyses and reclaim idle memory
- `SPECTER_WORKER_POLL_SECONDS=10` on worker, to reduce idle queue polling overhead

Use separate Railway projects and separate Supabase projects for staging and production. Do not point production at the current shared non-production backend.

Worker-backed runs only expose saved results once the terminal batch snapshot has
been persisted. While a run is still active, `/api/analyses/<job_id>` returns
`409 Conflict` instead of serving partial per-company results as if the batch
were complete.

The primary live monitoring surface is now the **Analysis overview** page.
New analyses navigate directly there instead of relying on a separate progress
screen. Finished runs always expose an **Open results** action and the server
decides whether the saved report can be opened, which avoids stale browser-side
availability state. Both the **Analysis** and **Companies** headers expose a
manual **Refresh** action for an immediate server sync.

### Deploy via Cloudflare Tunnel

```bash
./deploy.sh
```

Runs the FastAPI server and exposes it via `cloudflared` tunnel. Use the printed URL and `APP_PASSWORD` from your `.env`.

---

## Batch Mode (CLI)

### Document-based (deals folder)

Create a `deals/` folder with one subfolder per startup:

```
deals/
  acme-robotics/
    pitch_deck.pdf
    metrics.xlsx
    notes.txt
  betacorp/
    pitch_deck.pptx
    financials.csv
```

Supported formats: **PDF, PPTX, DOCX, CSV, XLSX, TXT, MD**

```bash
python -m agent.batch --input ./deals --output results.xlsx
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to deals folder |
| `--output` | `results.xlsx` | Output Excel file |
| `--k` | `8` | Evidence chunks per question |
| `--max-startups` | all | Limit number of startups |
| `--web-search` | false | Enable web search (Perplexity/Brave) |

### Specter-based (company + people CSVs)

For Specter data exports:

```bash
python -m agent.batch --specter-companies company-signals.csv --specter-people people-signals.csv --output results.xlsx
```

The system parses company and people CSVs, joins them, and evaluates each company with structured evidence chunks.

### Output

The Excel file has multiple sheets:

- **Summary** — One row per startup, ranked by `total_score`. Top pro/contra arguments, executive summaries, key points, red flags, invest decision.
- **Arguments** — All arguments with type, score, original text, critique, refined text.
- **Evidence** — Document chunks with source file, page/slide, and text.

---

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | ✓ | `gemini`, `openai`, `anthropic`, or `openrouter` |
| `GOOGLE_API_KEY` | if Gemini | For `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | if OpenAI | For `LLM_PROVIDER=openai` |
| `OPENROUTER_API_KEY` | if OpenRouter | For `LLM_PROVIDER=openrouter` |
| `OPENROUTER_BASE_URL` | optional | Defaults to `https://openrouter.ai/api/v1` |
| `ANTHROPIC_API_KEY` | if Anthropic | For `LLM_PROVIDER=anthropic` |
| `APP_ENV` | optional | `development` (default), `staging`, or `production` |
| `APP_PASSWORD` | optional locally, required in production web | Web app login |
| `SESSION_SECRET` | optional locally, required in production web | Cookie/session signing secret |
| `PPLX_API_KEY` | optional | Perplexity for web search |
| `BRAVE_SEARCH_API_KEY` | optional | Brave Search alternative |
| `WEB_SEARCH_PROVIDER` | optional | `sonar` (Perplexity) or `brave` |
| `MAX_PPLX_CALLS_PER_COMPANY` | optional | Per-company web-search cap (default: `100`) |
| `WEB_SEARCH_TRIGGER` | optional | `answer` (default) or `no_chunks` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | optional | Per-request LLM timeout (default: `90`) |
| `LLM_MAX_RETRIES` | optional | Max retries on transient LLM failures (default: `2`) |
| `SUPABASE_URL` | optional | Supabase project URL for persistent storage |
| `SUPABASE_SERVICE_ROLE_KEY` | optional | Supabase service-role key |
| `SUPABASE_SOURCE_FILES_BUCKET` | optional | Shared storage bucket for uploaded source files (default: `analysis-inputs`) |
| `SERVICE_ROLE` | Railway only | `web` or `worker`; selects service behavior inside the shared image |
| `ENABLE_SPECTER_WORKER_SERVICE` | optional | Queue Specter runs for the dedicated worker service instead of executing them in the web process |
| `SPECTER_WORKER_POLL_SECONDS` | optional | Poll interval for the dedicated Specter worker (code default: `5`; current Railway production override: `10`) |
| `SPECTER_MCP_URL` | optional | Specter MCP endpoint (default: `https://mcp.tryspecter.com/mcp`) — used for both URL-list intake and pitch-deck augmentation |
| `SPECTER_MCP_CLIENT_ID` | optional | Specter MCP OAuth client ID minted by `scripts/specter_oauth_login.py` |
| `SPECTER_MCP_REFRESH_TOKEN` | optional | Specter MCP OAuth refresh token (initial value); rotated tokens persist to the `mcp_secrets` Supabase table on subsequent runs |
| `RESTART_ON_IDLE_AFTER_ANALYSIS` | optional | When `true`, the web service can restart after completed analyses have been persisted/served so idle memory is reclaimed |
| `LANGSMITH_API_KEY` | optional | LangSmith tracing |

**Example for Gemini (free tier):**

```
LLM_PROVIDER=gemini
MODEL_NAME=gemini-3.1-flash-lite-preview
GOOGLE_API_KEY=your_google_api_key_here
```

Available Gemini models in the UI: `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-pro-preview`

**Example for Anthropic:**

```
LLM_PROVIDER=anthropic
MODEL_NAME=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Example for OpenAI:**

```
LLM_PROVIDER=openai
MODEL_NAME=gpt-5.4-mini
OPENAI_API_KEY=your_openai_api_key_here
```

Available OpenAI models in the UI: `gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5`, `gpt-4.1-mini`, `o4-mini`, `gpt-5.2`, `gpt-5.4`

**Example for OpenRouter:**

```
LLM_PROVIDER=openrouter
MODEL_NAME=openrouter/hunter-alpha
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Available OpenRouter models in the UI: `openrouter/hunter-alpha`

---

## Supabase (Optional)

When configured, the app persists completed analyses to Supabase so results survive restarts and worker services can share uploaded source files through Supabase Storage.

1. Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env`
2. Apply migrations from `supabase/migrations/` in order
3. The app auto-creates the `analysis-inputs` storage bucket on first source-file upload unless you override `SUPABASE_SOURCE_FILES_BUCKET`

See [`supabase/README.md`](./supabase/README.md) for full setup details.

---

## LangGraph Studio (Interactive)

For step-by-step debugging and graph visualization:

```bash
langgraph dev
```

Opens LangGraph Studio in the browser.

---

## Project Structure

```
src/agent/
├── batch.py                 # CLI entrypoint (documents + Specter)
├── llm.py                   # Multi-provider LLM factory
├── retrieval.py             # TF-IDF chunk retrieval
├── evidence_answering.py    # Document-grounded QA
├── ingest/
│   ├── pdf_ingest.py            # PDF text extraction
│   ├── pptx_ingest.py           # PPTX text extraction
│   ├── tabular_ingest.py        # CSV/XLSX parsing
│   ├── specter_ingest.py        # Specter company + people CSV parsing
│   ├── specter_mcp_client.py    # Specter MCP client (OAuth + tool wrappers + chunk builders)
│   ├── specter_augmentation.py  # Pitch-deck URL extraction + MCP merge
│   ├── chunking.py              # Text chunking
│   └── store.py                 # Evidence store
├── pipeline/
│   ├── graph.py             # Main LangGraph definition
│   ├── stages/              # Decomposition, QA, generation, critique, ranking
│   └── state/               # Pydantic state schemas
├── prompts/                 # LLM prompts
├── prompt_library/          # JSON prompt library
├── dataclasses/             # Company, Person, Argument, QuestionTree, etc.
├── person_intel/            # Team-member profile enrichment (LinkedIn, web)
└── web_search/              # Perplexity, Brave providers

web/
├── app.py                   # FastAPI backend
├── db.py                    # Supabase persistence (optional)
└── static/
    └── index.html           # Single-page UI

supabase/
├── README.md                # Setup instructions
└── migrations/              # SQL schema migrations
```

---

## Recent Updates

See [docs/RECENT_UPDATES.md](./docs/RECENT_UPDATES.md) for recent changes and work in progress.

---

## License

See [LICENSE](./LICENSE) for details.
