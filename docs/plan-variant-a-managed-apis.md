# Realty Lens — Plan Variant A: Managed APIs (Claude + OpenAI)

## Shared Context

The Realty Lens (real estate developer client) wants a prototype "intelligent assistant" that lets staff upload PDF project brochures and ask natural-language questions (factual Q&A, amenity/unit filters, cross-document comparisons, summaries), with answers grounded strictly in the uploaded documents. The deliverable is a private GitHub repo with working code, a README, an architecture diagram, and a one-pager, submitted within 72 hours. Crucially, **the interviewer will upload brand-new brochures live** to test generalization — so the pipeline must not rely on anything specific to the 18 sample brochures already in [brochure/](../brochure/).

I inspected all 18 sample PDFs directly (via PyMuPDF) to de-risk the design before building:
- 17/18 have a real text layer, but it's sparse — brochures are graphics-heavy, averaging only ~100–1500 extractable chars/page (headlines, amenity labels, unit-spec tables), not prose.
- **1/18 (`JLL_Mumbai_Lodha Regalia`) is a fully scanned/image-only PDF — 0 extractable characters.** This confirms scanned brochures are a real, not hypothetical, case the system must handle, especially given the interview will test unseen files.

This finding shapes both variants (each needs a fallback for image-only pages, not just plain text extraction).

Both variants share the same core architecture idea — an **intent router** that branches between **vector RAG** (for open-ended factual/spec questions) and an **exhaustive structured-JSON-store scan** (for filter/comparison/summarization, where top-k similarity search could silently miss a matching project) — and the same **Streamlit frontend + FastAPI backend** split. They differ only in which models/services perform generation, embeddings, and scanned-page recovery. See the companion plan, `plan-variant-b-open-source.md`, for the fully open-source variant.

## Context
Uses Anthropic Claude for generation, intent routing, structured extraction, and vision-based recovery of scanned pages; OpenAI `text-embedding-3-small` for embeddings. Confirmed with the user as the original stack choice. Trade-off: per-token API cost and brochure content leaves the machine, in exchange for the most reliable structured-output behavior (native forced tool-use) and simplest setup (two API keys, no local model runtime or OS-level OCR dependency).

## Architecture
```
Streamlit UI ──HTTP──▶ FastAPI backend
                          │
     POST /documents/upload (PDF bytes)
                          │
                          ▼
              ┌─────────────────────────┐
              │   Ingestion Pipeline     │
              │ 1. Per-page text extract │
              │    (PyMuPDF)             │
              │ 2. Low-text pages →      │
              │    rasterize (PyMuPDF)   │
              │    + Claude vision call  │
              │ 3. Chunk text (page-     │
              │    aware, w/ overlap)    │
              │ 4. Embed chunks          │
              │    (OpenAI) → Chroma     │
              │    (per doc_id)          │
              │ 5. One Claude call        │
              │    (forced tool-use):     │
              │    extract structured     │
              │    project record (JSON)  │
              │    → JSON store            │
              └─────────────────────────┘
                          │
     POST /query {question}
                          │
                          ▼
              ┌─────────────────────────┐
              │  Intent Router (Claude   │
              │  tool-use, forced choice)│
              │  → factual_qa |          │
              │    amenity_filter |      │
              │    unit_filter |         │
              │    comparison |          │
              │    summarization         │
              │  + extracted slots       │
              │  (project names, filters)│
              └─────────────────────────┘
                    │                │
        factual/spec lookup    filter/compare/summarize
                    │                │
                    ▼                ▼
          Vector search (Chroma,  Scan structured JSON
          filtered by resolved    store across ALL docs
          project_name if named)  (small N → exhaustive,
                    │              not similarity-limited)
                    └───────┬────────┘
                             ▼
                 Claude generation call
                 (strict "answer only from
                 provided context; say not
                 found if absent" prompt)
                             │
                             ▼
                 Structured JSON response
                 {intent, answer_text,
                  table/list data, sources}
                             │
                             ▼
                 Streamlit renders: plain
                 text / dataframe table /
                 per-project cards
```

**Why two retrieval paths?** Filter queries ("list ALL projects with pool AND gym") and comparisons need *completeness* across every ingested doc — top-k similarity search can silently miss a matching project. So at ingestion time, one extra LLM call per doc produces a small structured JSON record (developer, location, handover, amenities[], unit_types[]). Filter/compare/summarize queries scan this compact table exhaustively; open-ended factual/spec questions use chunk-level vector RAG with page citations. This hybrid is the main "AI-native thinking" talking point for the interview.

**Why Claude vision instead of Tesseract OCR for scanned pages?** Avoids a second, OS-dependent toolchain (Tesseract's install differs per OS, complicating the README's "assume reviewer has nothing else" requirement). Reuses the same Anthropic API key already required, and a vision LLM reads marketing layouts (icons, tables, stylized headers) more robustly than raw OCR. Trade-off: higher latency/cost per image-heavy doc — mitigated by only invoking vision on pages under a text-density threshold, capped at a max page count per doc.

## Repo Structure
```
relens/
  backend/
    app/
      main.py         FastAPI app + routes (/documents/upload, /documents, /query)
      ingestion.py     PDF→pages→text/vision extraction, chunking, embedding, structured extraction
      retrieval.py     vector search + structured-store filter/compare/summarize logic
      intent.py        Claude tool-use intent router
      llm.py           Anthropic + OpenAI client wrappers (chat, vision, embeddings)
      schemas.py       Pydantic models (ProjectRecord, UnitType, QueryResponse, etc.)
      store.py         Chroma persistent client + JSON structured-record store
    requirements.txt   fastapi, uvicorn, pymupdf, anthropic, openai, chromadb, pydantic, python-multipart
  frontend/
    streamlit_app.py  upload widget, project list, chat input, renders answer/table/cards
    requirements.txt  streamlit, requests, pandas
  data/                gitignored at runtime: uploads/, chroma/, structured/
  docs/
    architecture.md    Mermaid diagram (renders natively on GitHub)
    one-pager.md        max 1 page: what/how, key decisions + why, improvements given more time
  README.md            setup: set ANTHROPIC_API_KEY + OPENAI_API_KEY in .env, pip install, run backend + frontend
  .env.example          ANTHROPIC_API_KEY=, OPENAI_API_KEY=
  .gitignore            data/, .env, __pycache__, *.pdf (sample brochures not committed)
```
Sample brochures (`brochure/`) stay local only — not committed (large, likely-confidential real developer marketing PDFs; brief already ships them via Drive link).

## Implementation Steps
1. Scaffold repo: folders above, `.gitignore`, `.env.example`, `requirements.txt`.
2. `llm.py`: `chat(messages, system, tools=None, tool_choice=None)` via Anthropic; `vision_extract(image_bytes, prompt)` via Anthropic image content block; `embed(texts) -> list[vector]` via OpenAI.
3. `ingestion.py`:
   - `extract_pages(pdf_bytes) -> list[PageText]`: PyMuPDF text per page; low-text pages rasterized (`page.get_pixmap()`, no Poppler needed) and sent to `vision_extract`, capped at ~20 low-text pages/doc.
   - `chunk_pages(pages) -> list[Chunk]`: page-aware chunking (~800 chars, ~150 overlap), metadata `{doc_id, project_name, page_number, source_file}`.
   - `extract_structured_record(full_text) -> ProjectRecord`: one forced-tool-use Claude call — `project_name, developer, location, expected_handover, amenities[], unit_types[{type, area_sqft, features}], other_facts`; missing fields → `null`, never invented.
   - `ingest_document(pdf_bytes, filename) -> doc_id`: orchestrates the above, embeds chunks into Chroma (collection keyed by doc_id, metadata `project_name`), writes structured record to `data/structured/{doc_id}.json`.
4. `intent.py`: `classify_intent(question, known_projects) -> IntentResult` — single Claude tool-use call with a forced tool schema enumerating the 5 intents + slots (project names resolved against `known_projects`, requested amenities/unit filters, comparison dimensions).
5. `retrieval.py`:
   - `answer_factual(question, resolved_doc_ids|None)`: embed question, Chroma similarity search (filtered by doc_id if a project was named); below-threshold/no hits → "not found in the uploaded documents"; else Claude generation call constrained to retrieved chunks, must cite `project_name (p. N)`.
   - `filter_projects(criteria)`: scan all structured records via one Claude call over the compact JSON table (handles synonyms like "pool" vs "swimming pool").
   - `compare_projects(names, dimensions)`: structured records (+ a few top chunks per project) → structured comparison table.
   - `summarize_projects(names|all)`: structured record + top chunks per project → one-paragraph Claude summary each.
6. `main.py`: FastAPI routes wiring the above; `/documents/upload` (multi-file, returns `{doc_id, project_name, warnings}`); `/documents` (list); `/query` → `QueryResponse {intent, answer_text, table_data|null, sources[]}`.
7. `streamlit_app.py`: sidebar uploader + ingested-project list; chat input; renders `answer_text`/`table_data` appropriately; shows detected intent as a caption; shows `sources`.
8. `docs/architecture.md`, `docs/one-pager.md`, `README.md`: written last, once verified working.

## Verification
- Confirm both API keys are set; start backend (`uvicorn backend.app.main:app --reload`) and frontend (`streamlit run frontend/streamlit_app.py`).
- Ingest a varied set: text-rich (`Oberoi Eternia Towers A to D Brochure.pdf`), sparse (`Mohid Mahakali Hills Brochure.pdf`), and the scanned one (`JLL_Mumbai_Lodha Regalia_8409_BRC_1.pdf`) to confirm the vision fallback engages and produces a usable structured record.
- Run all 7 example query types from the brief, confirming grounded citations, correct "not found" behavior, structured vs. plain output per intent, and correct per-project attribution with multiple docs loaded.
- Ingest one more brochure not touched during dev, ask a fresh question, to rehearse "handling the unknown."
