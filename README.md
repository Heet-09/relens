# Realty Lens — Brochure Assistant (Gemini + local embeddings)

Upload real-estate project brochure PDFs and ask natural-language questions — factual Q&A,
amenity/unit filters, cross-document comparisons, and summaries — with answers grounded strictly
in the uploaded documents. See [docs/architecture.md](docs/architecture.md) for the full pipeline
diagram and [docs/one-pager.md](docs/one-pager.md) for key decisions.

## Prerequisites
- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey) — used for chat generation, intent
  routing, structured extraction, and vision reads of scanned/low-text pages
- Embeddings run locally via `sentence-transformers` — no key needed, but the first run downloads
  the model (~130MB for the default `BAAI/bge-small-en-v1.5`)

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# (or separately: pip install -r backend/requirements.txt -r frontend/requirements.txt)

cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```

## Run

In one terminal:

```bash
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL it prints (default `http://localhost:8501`). Upload one or more brochure
PDFs from the sidebar, wait for ingestion to finish, then ask questions in the chat box.

## Notes
- Uploaded PDFs, the vector index, and extracted structured records are written under `data/`
  (gitignored) — nothing here is committed or shipped with the repo.
- Pages with little extractable text (common in graphics-heavy brochures, including fully-scanned
  PDFs) are read via Gemini's vision input rather than a separate OCR install.
- The `GEMINI_MODEL` and `EMBEDDING_MODEL` env vars can override the defaults if needed (see
  `.env.example`). The default `GEMINI_MODEL` is `gemini-flash-lite-latest` — a self-updating
  alias chosen for reliability (the heavier `gemini-flash-latest` alias was observed returning
  transient 503s under load during development). `llm.py` retries transient 5xx errors automatically.
