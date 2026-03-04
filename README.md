# DIALECTIC

**LLM-Based Multi-Agent System for Startup Evaluation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Work in Progress](https://img.shields.io/badge/status-WIP-orange.svg)](./docs/RECENT_UPDATES.md)
📄 *Accepted at EACL 2026 Industry Track* — [Read the Paper](./paper.pdf)

> **⚠️ Work in progress.** We are actively improving Perplexity web search quality and fallback behavior. See [Recent Updates](./docs/RECENT_UPDATES.md) for the latest changes.

---

## Overview

DIALECTIC helps venture capital investors evaluate and rank startup investment opportunities. Investors face a flood of deals but can only invest in a small fraction. This system automates the analysis: it ingests pitch decks, metrics, and optional web data, generates structured pro and contra arguments, critiques them through a devil's advocate stage, and outputs numeric scores plus an invest/not-invest recommendation.

**Pipeline stages:**

1. **Data Collection** — Gather knowledge from documents (and optionally web search)
2. **Knowledge Organization** — Structure information into hierarchical question trees
3. **Argument Generation** — Synthesize pro and contra investment arguments
4. **Iterative Refinement** — Simulate debate to critique and strengthen arguments
5. **Ranking** — Composite scoring with strategy fit, team, and upside dimensions

---

## Quick Start

### Install

```bash
pip install -e . "langgraph-cli[inmem]"
cp .env.example .env
```

Add your API keys to `.env` (see [Configuration](#configuration)).

### Run the Web App (recommended)

```bash
python -m web
```

Then open **http://localhost:8000** in your browser. Default password: `9876`.

- **Upload** PDF, PPTX, Word, Excel, or CSV files
- Choose **Pitch Deck** (each file = one company), **Specter** (company + people CSVs), or **Original** (all files = one company)
- Optionally enable **web search** for richer evidence
- Get ranked results with pro/contra arguments and Excel export

### Run Batch Mode (CLI)

**From documents (one folder per startup):**

```bash
python -m agent.batch --input ./deals --output results.xlsx
```

**From Specter CSVs (company + people exports):**

```bash
python -m agent.batch --specter-companies companies.csv --specter-people people.csv --output results.xlsx
```

---

## Web App

The web UI lets you upload files, run analyses, and download results without touching the CLI.

| Feature | Description |
|---------|-------------|
| **Upload** | PDF, PPTX, DOCX, XLSX, CSV — single or multi-file |
| **Input modes** | Pitch Deck (1 file per company), Specter (2 CSVs), Original (all files = 1 company) |
| **Web search** | Optional Perplexity/Brave search for extra evidence |
| **VC strategy** | Optional investment thesis for tailored scoring |
| **Results** | Summary table, pro/contra arguments, Excel download |

### Local run

```bash
python -m web
# or
uvicorn web.app:app --reload --port 8000
```

### Deploy via Cloudflare Tunnel

```bash
./deploy.sh
```

Runs the FastAPI server and exposes it via `cloudflared` tunnel. Use the printed URL and `APP_PASSWORD` (default `9876`).

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

The Excel file has three sheets:

- **Summary** — One row per startup, ranked by `total_score`. Top pro/contra arguments, invest decision.
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
| `ANTHROPIC_API_KEY` | if Anthropic | For `LLM_PROVIDER=anthropic` |
| `APP_PASSWORD` | optional | Web app login (default: `9876`) |
| `PPLX_API_KEY` | optional | Perplexity for web search |
| `BRAVE_SEARCH_API_KEY` | optional | Brave Search alternative |
| `WEB_SEARCH_PROVIDER` | optional | `sonar` (Perplexity) or `brave` |
| `MAX_PPLX_CALLS_PER_COMPANY` | optional | Per-company web-search cap (default: `100`) |
| `WEB_SEARCH_TRIGGER` | optional | `answer` (default) or `no_chunks` |
| `LANGSMITH_API_KEY` | optional | LangSmith tracing |

**Example for Gemini (free tier):**

```
LLM_PROVIDER=gemini
MODEL_NAME=gemini-3.1-flash-lite-preview
GOOGLE_API_KEY=your_google_api_key_here
```

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
│   ├── pdf_ingest.py        # PDF text extraction
│   ├── pptx_ingest.py       # PPTX text extraction
│   ├── tabular_ingest.py    # CSV/XLSX parsing
│   ├── specter_ingest.py    # Specter company + people CSV parsing
│   ├── chunking.py         # Text chunking
│   └── store.py            # Evidence store
├── pipeline/
│   ├── graph.py            # Main LangGraph definition
│   ├── stages/             # Decomposition, QA, generation, critique, ranking
│   └── state/              # Pydantic state schemas
├── prompts/                # LLM prompts
├── prompt_library/         # JSON prompt library
├── dataclasses/            # Company, Argument, QuestionTree, etc.
└── web_search/             # Perplexity, Brave providers

web/
├── app.py                  # FastAPI backend
└── static/
    └── index.html          # Single-page UI
```

---

## Citation

```bibtex
@inproceedings{dialectic2026,
  title={DIALECTIC: An LLM-Based Multi-Agent System for Startup Evaluation},
  author={Bae, Jae Yoon and Malberg, Simon and Galang, Joyce and Retterath, Andre and Groh, Georg},
  booktitle={Proceedings of the 2026 Conference of the European Chapter of the Association for Computational Linguistics: Industry Track},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

---

## License

See [LICENSE](./LICENSE) for details.
