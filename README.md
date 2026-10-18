 HEAD

 HEAD
# The-Automated-Deep-Research-Academic-Synthesizer
A system where you drop in 10 to 20 complex academic papers or technical PDFs on a niche subject, and it acts as an automated research assistant that can write a comprehensive literature review.

 e885877 (Add academic research synthesizer)
# LitReview — Automated Deep-Research Academic Synthesizer

A local-first web app for organizing academic PDFs, retrieving page-level evidence, comparing findings, and drafting literature-review sections with traceable citations.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

The app stores PDFs and its library database in the ignored `data/` directory beside `app.py`. Keep that directory backed up if you want to preserve your library. The service is intended for a single local user; it has no authentication or remote deployment hardening.

## Model-assisted synthesis

Without an API key, the app uses lexical overlap retrieval and makes an extractive review from the selected source passages. This mode does not claim to detect scholarly contradiction semantically.

By default the app uses OpenAI for model-assisted sub-question expansion, synthesis, and comparison. Copy `.env.example` to `.env`, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`, then restart the server. This uses paid API quota.

### Run a model locally with Ollama

Ollama can serve models on this computer through an OpenAI-compatible endpoint, so this app can use it without sending paper passages to OpenAI or paying per API request. Install Ollama, then download and start a model. For example, in PowerShell:

```powershell
ollama pull qwen3:4b-instruct
ollama run qwen3:4b-instruct
```

In `.env`, select the local provider:

```text
MODEL_PROVIDER=ollama
MODEL_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3:4b-instruct
MODEL_API_KEY=ollama
```

Restart the server. Keep Ollama running; the sidebar shows the selected provider and model. Model size affects download time, memory use, speed, and answer quality. Local models can be less capable than hosted models, so verify every claim against its linked PDF passage.

## Features

- Multiple PDF upload with text extraction and document metadata.
- Page-aware text blocks and page-number source links; scanned PDFs with no text are marked as needing OCR.
- Research questions expanded into focused sub-questions.
- Local lexical retrieval across the selected paper collection.
- Evidence passage panel, draft sections, disagreement flags, limitations, and research gaps.
- Clickable citations open the source PDF at its cited page.
- Export a review to Markdown.
- Remove papers and persist reviews locally.

## Current implementation notes

The MVP uses a lightweight lexical search index computed from extracted passages at request time, suitable for a personal collection of around 20 papers. It does not yet include embeddings, OCR, figure/table understanding, user accounts, or database migrations. Extracted PDF layout can be imperfect, particularly for multi-column articles; inspect source pages before relying on a passage. Automated disagreement flags are prompts for review rather than conclusions.

## Project layout

```text
app.py                 FastAPI endpoints, extraction, retrieval, synthesis
static/index.html      Workspace interface
static/styles.css      Responsive application styling
static/app.js          Upload, library, review, PDF citation interactions
data/                  Runtime PDF and library storage (git-ignored)
```
 HEAD

 2c042df (Add academic research synthesizer)
 e885877 (Add academic research synthesizer)
