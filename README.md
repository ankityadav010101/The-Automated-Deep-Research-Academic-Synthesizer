# LitReview: Research Synthesizer

A local-first workspace for reading academic PDFs, asking questions across a paper library, comparing findings, and drafting literature reviews with page-level citations.

## Run on Windows

From PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001>. Keep the terminal open while using the app. Stop the server with Ctrl+C; leave the environment with `deactivate`.

## Use Ollama

Install and launch Ollama, then download the configured model:

```powershell
ollama pull qwen3:4b-instruct
```

Copy `.env.example` to `.env` if needed and set:

```text
MODEL_PROVIDER=ollama
MODEL_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3:4b-instruct
MODEL_API_KEY=ollama
```

Keep Ollama running while using model-assisted chat, comparison, or review generation. Local model output can be slower and less reliable than hosted models; check claims against their linked passages.

## Features

- Upload and index up to 20 PDFs, with extracted text tied to page numbers.
- Ask questions in Paper Chat across selected papers; conversations are saved locally and can be reopened.
- Compare 2 to 8 papers in a side-by-side evidence matrix, including a study-design preset.
- Build an evidence-traceability review with claim labels, page-linked passages, potential conflicts and gaps, plus saved human verification status, reviewer notes, and reviewer-entered methods/sample details.
- Create a research-question coverage map showing which paper passages match each subquestion.
- Draft literature reviews with source passages, disagreement flags, research gaps, and limitations.
- Open citations at their source pages; add page notes, passage annotations, and bookmarks.
- Export reviews and matrices to PDF, Markdown, or CSV with cited pages and evidence passages. PDF export uses the browser print dialog; choose Save as PDF.
- Manage the local paper library and remove PDFs.

## Local data and limitations

PDFs, extracted text, saved chats, reviews, notes, and bookmarks are stored under `data/`, which Git ignores. Back up that folder to keep your local research library. `.env` is also ignored; never commit API keys.

Retrieval currently uses lexical matching rather than embeddings. A coverage-map match means a passage matched search terms, not that it fully answers the subquestion. Claim labels and comparisons are model suggestions, not final scholarly judgments. Scanned PDFs without extractable text need OCR before they can be searched. This app is designed for a single local user and does not include authentication or deployment hardening.

## Project layout

```text
app.py                 FastAPI endpoints, PDF extraction, retrieval, synthesis
static/index.html      Workspace and feature views
static/app.js          UI interactions, chat, comparison, citations
static/styles.css      Main styling
static/overrides.css   Feature and responsive styling
data/                  Local PDFs, library database, chats, and annotations (ignored)
```
