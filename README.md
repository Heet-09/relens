# Realty Lens — Brochure Assistant (Gemini + local embeddings)

Upload real-estate project brochure PDFs and ask natural-language questions — factual Q&A,
amenity/unit filters, cross-document comparisons, and summaries — with answers grounded strictly
in the uploaded documents. See [docs/architecture_diagram](docs/architecture_diagram.png) for the full pipeline
diagram and [docs/one-pager.md](docs/one-pager.md) for key decisions.

## Prerequisites
- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey) 

## Setup and run (from a fresh clone)

These steps assume you have the repo checked out locally and nothing else set up yet.

1. **Check your Python version.** You need Python 3.11 or newer.

   ```bash
   python --version
   ```

2. **Create and activate a virtual environment**, from the repo root.

   ```bash
   python -m venv .venv
   ```

   ```bash
   # Windows (PowerShell/cmd):
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies.**

   ```bash
   pip install -r requirements.txt
   # (or separately: pip install -r backend/requirements.txt -r frontend/requirements.txt)
   ```

4. **Get a Gemini API key.** Create one for free at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — it's used for chat
   generation, intent routing, structured extraction, and vision reads of scanned pages.

5. **Create your local `.env` file** and add the key.

   ```bash
   cp .env.example .env
   ```

   Open `.env` and set:

   ```
   GEMINI_API_KEY=your-key-here
   ```

   (The other variables in `.env.example` are optional overrides — leave them commented out
   to use the defaults.)

6. **Start the backend API**, in one terminal (from the repo root, venv activated):

   ```bash
   uvicorn backend.app.main:app --reload
   ```

   It listens on `http://localhost:8000` by default. Leave this terminal running. The first
   time it starts, it will download the local embedding model (~130MB) if it isn't cached yet.

7. **Start the frontend**, in a second terminal (same repo root, same venv activated):

   ```bash
   streamlit run frontend/streamlit_app.py
   ```

8. **Open the app.** Streamlit prints a local URL (default `http://localhost:8501`) — open it
   in your browser. From the sidebar, upload one or more brochure PDFs, wait for ingestion to
   finish, then ask questions in the chat box.

### Troubleshooting
- **Port already in use**: pass `--port <n>` to `uvicorn` or `streamlit run ... --server.port <n>`.
- **`GEMINI_API_KEY` errors**: confirm `.env` is in the repo root (not inside `backend/`) and the
  key has no surrounding quotes or spaces.
- **Slow first request**: the embedding model download and first model load happen lazily on
  first use — subsequent requests are fast.

## Notes
- Uploaded PDFs, the vector index, and extracted structured records are written under `data/`
  (gitignored) — nothing here is committed or shipped with the repo.
- Pages with little extractable text (common in graphics-heavy brochures, including fully-scanned
  PDFs) are read via Gemini's vision input rather than a separate OCR install.
- The `GEMINI_MODEL` and `EMBEDDING_MODEL` env vars can override the defaults if needed (see
  `.env.example`). The default `GEMINI_MODEL` is `gemini-flash-lite-latest` — a self-updating
  alias chosen for reliability (the heavier `gemini-flash-latest` alias was observed returning
  transient 503s under load during development). `llm.py` retries transient 5xx errors automatically.
