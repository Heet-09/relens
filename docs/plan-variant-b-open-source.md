# Realty Lens — Plan Variant B: Fully Open Source (Ollama + Local Embeddings + Tesseract)

## Shared Context

The Realty Lens (real estate developer client) wants a prototype "intelligent assistant" that lets staff upload PDF project brochures and ask natural-language questions (factual Q&A, amenity/unit filters, cross-document comparisons, summaries), with answers grounded strictly in the uploaded documents. The deliverable is a private GitHub repo with working code, a README, an architecture diagram, and a one-pager, submitted within 72 hours. Crucially, **the interviewer will upload brand-new brochures live** to test generalization — so the pipeline must not rely on anything specific to the 18 sample brochures already in [brochure/](../brochure/).

I inspected all 18 sample PDFs directly (via PyMuPDF) to de-risk the design before building:
- 17/18 have a real text layer, but it's sparse — brochures are graphics-heavy, averaging only ~100–1500 extractable chars/page (headlines, amenity labels, unit-spec tables), not prose.
- **1/18 (`JLL_Mumbai_Lodha Regalia`) is a fully scanned/image-only PDF — 0 extractable characters.** This confirms scanned brochures are a real, not hypothetical, case the system must handle, especially given the interview will test unseen files.

This finding shapes both variants (each needs a fallback for image-only pages, not just plain text extraction).

Both variants share the same core architecture idea — an **intent router** that branches between **vector RAG** (for open-ended factual/spec questions) and an **exhaustive structured-JSON-store scan** (for filter/comparison/summarization, where top-k similarity search could silently miss a matching project) — and the same **Streamlit frontend + FastAPI backend** split. They differ only in which models/services perform generation, embeddings, and scanned-page recovery. See the companion plan, `plan-variant-a-managed-apis.md`, for the managed-API variant.

## Context
Replaces every paid API with an open-source, locally-run equivalent — no API keys, no per-token cost, no data ever leaves the machine (relevant since these are confidential client brochures). Trade-off: local open-weight models are less reliable at strict structured output than Claude's forced tool-use, so the pipeline adds explicit JSON-schema validation with a retry loop to compensate, and local inference speed depends on the reviewer's hardware.

## Open-Source Stack
| Component | Choice | Why |
|---|---|---|
| LLM serving | **Ollama** (local runtime) | Simplest cross-platform install (single installer, `ollama pull <model>`), MIT-licensed |
| Generation / intent routing / structured extraction | **Qwen2.5:7b-instruct** | Best structured-JSON/instruction-following among open 7B-class models; `qwen2.5:3b-instruct` documented as a low-resource fallback |
| Embeddings | **`BAAI/bge-small-en-v1.5`** via `sentence-transformers` | Strong open embedding benchmark for its size, CPU-friendly, no API cost |
| Vector store | **Chroma** (embedded, local) | Same as Variant A |
| Scanned-page fallback | **Tesseract OCR** via `pytesseract` | No GPU required, fast, deterministic — safer for a live demo than a local vision-LLM on unknown hardware; local vision model (e.g. `qwen2.5vl` via Ollama) documented as an optional GPU upgrade |
| PDF text extraction | **PyMuPDF** | Same as Variant A |
| Backend / Frontend | **FastAPI / Streamlit** | Same as Variant A |

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
              │    + Tesseract OCR       │
              │ 3. Chunk text (page-     │
              │    aware, w/ overlap)    │
              │ 4. Embed chunks          │
              │    (bge-small, local)    │
              │    → Chroma (per doc_id) │
              │ 5. One Ollama call:      │
              │    extract structured    │
              │    project record (JSON, │
              │    schema-validated,     │
              │    retry on failure) →   │
              │    JSON store            │
              └─────────────────────────┘
                          │
     POST /query {question}
                          │
                          ▼
              ┌─────────────────────────┐
              │  Intent Router (Ollama,  │
              │  JSON-mode + Pydantic    │
              │  validation, retry once) │
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
                 Ollama generation call
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

**Why two retrieval paths?** Same reasoning as Variant A: filter/comparison queries need completeness across every doc (top-k similarity can miss matches), so they scan the exhaustive structured-JSON table instead; factual/spec questions use chunk-level vector RAG.

**Why validate-and-retry instead of trusting the model's JSON output?** Open 7B-class models occasionally emit malformed JSON or drift from the schema, unlike Claude's forced tool-use. Every structured call (intent routing, record extraction) is wrapped: request with `format: "json"`, parse into the Pydantic schema, and on validation failure, re-prompt once with the validation error appended. A second failure surfaces as a clear error rather than a guessed/hallucinated structure.

**Why Tesseract instead of a local vision-LLM for scanned pages?** A local multimodal model (llava/qwen2.5vl) is heavier and hardware-dependent — risky to demo live on unknown interview hardware. Tesseract is lightweight, CPU-only, and deterministic. Documented as a config-swappable upgrade if a GPU is available.

## Repo Structure
```
relens/
  backend/
    app/
      main.py         FastAPI app + routes (/documents/upload, /documents, /query)
      ingestion.py     PDF→pages→text/OCR extraction, chunking, embedding, structured extraction
      retrieval.py     vector search + structured-store filter/compare/summarize logic
      intent.py        Ollama JSON-mode intent router (schema validation + retry)
      llm.py           Ollama chat wrapper (JSON mode, retry-on-validation-failure helper)
      embeddings.py    sentence-transformers (bge-small) wrapper
      ocr.py            PyMuPDF rasterize + pytesseract wrapper
      schemas.py       Pydantic models (ProjectRecord, UnitType, IntentResult, QueryResponse, etc.)
      store.py         Chroma persistent client + JSON structured-record store
    requirements.txt   fastapi, uvicorn, pymupdf, pytesseract, sentence-transformers, chromadb, ollama, pydantic, python-multipart
  frontend/
    streamlit_app.py  upload widget, project list, chat input, renders answer/table/cards
    requirements.txt  streamlit, requests, pandas
  data/                gitignored at runtime: uploads/, chroma/, structured/
  docs/
    architecture.md    Mermaid diagram (renders natively on GitHub)
    one-pager.md        max 1 page: what/how, key decisions + why (incl. "fully local, no API cost"), improvements given more time
  README.md            setup: install Ollama, `ollama pull qwen2.5:7b-instruct`, install Tesseract binary, pip install, run backend + frontend
  .env.example          OLLAMA_HOST=http://localhost:11434, OLLAMA_MODEL=qwen2.5:7b-instruct, EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
  .gitignore            data/, .env, __pycache__, *.pdf (sample brochures not committed)
```
Sample brochures (`brochure/`) stay local only — not committed.

## Implementation Steps
1. Scaffold repo: folders above, `.gitignore`, `.env.example`, `requirements.txt`.
2. Prerequisite check: verify Ollama is installable/runnable; document minimum recommended RAM (≥8GB for the 7B model, `qwen2.5:3b-instruct` swap for constrained machines) and that Tesseract must be on PATH.
3. `llm.py`: `chat_json(messages, system, schema: type[BaseModel]) -> BaseModel` — calls Ollama with `format: "json"`, parses into the given Pydantic model, retries once with the validation error appended on failure, raises a clear error on second failure.
4. `embeddings.py`: `embed(texts) -> list[vector]` — loads `BAAI/bge-small-en-v1.5` once at startup via `sentence-transformers`, encodes in batches.
5. `ocr.py`: `ocr_page(page) -> str` — `page.get_pixmap(dpi=200)` → PIL image → `pytesseract.image_to_string`.
6. `ingestion.py`:
   - `extract_pages(pdf_bytes) -> list[PageText]`: PyMuPDF text per page; low-text pages go through `ocr_page`, capped at ~20 low-text pages/doc.
   - `chunk_pages(pages) -> list[Chunk]`: page-aware chunking (~800 chars, ~150 overlap), metadata `{doc_id, project_name, page_number, source_file}`.
   - `extract_structured_record(full_text) -> ProjectRecord`: one `chat_json` call — `project_name, developer, location, expected_handover, amenities[], unit_types[{type, area_sqft, features}], other_facts`; missing fields → `null`, never invented.
   - `ingest_document(pdf_bytes, filename) -> doc_id`: orchestrates the above, embeds chunks into Chroma (collection keyed by doc_id, metadata `project_name`), writes structured record to `data/structured/{doc_id}.json`.
7. `intent.py`: `classify_intent(question, known_projects) -> IntentResult` via `chat_json` — 5 intents + slots (project names resolved against `known_projects`, requested amenities/unit filters, comparison dimensions).
8. `retrieval.py`:
   - `answer_factual(question, resolved_doc_ids|None)`: embed question, Chroma similarity search (filtered by doc_id if named); below-threshold/no hits → "not found in the uploaded documents"; else generation constrained to retrieved chunks, must cite `project_name (p. N)`.
   - `filter_projects(criteria)`: scan all structured records via one `chat_json` call over the compact JSON table (handles synonyms like "pool" vs "swimming pool").
   - `compare_projects(names, dimensions)`: structured records (+ a few top chunks per project) → structured comparison table.
   - `summarize_projects(names|all)`: structured record + top chunks per project → one-paragraph summary each.
9. `main.py`: FastAPI routes wiring the above; `/documents/upload` (multi-file); `/documents` (list); `/query` → `QueryResponse {intent, answer_text, table_data|null, sources[]}`.
10. `streamlit_app.py`: sidebar uploader + ingested-project list; chat input; renders `answer_text`/`table_data` appropriately; shows detected intent as a caption; shows `sources`.
11. `docs/architecture.md`, `docs/one-pager.md`, `README.md`: written last, once verified working — one-pager explicitly calls out "fully open-source/local, zero API cost, data never leaves the machine."

## Verification
- Confirm Ollama is running with `qwen2.5:7b-instruct` pulled (`bge-small` downloads automatically on first use); confirm Tesseract is on PATH.
- Start backend (`uvicorn backend.app.main:app --reload`) and frontend (`streamlit run frontend/streamlit_app.py`).
- Ingest a varied set: text-rich (`Oberoi Eternia Towers A to D Brochure.pdf`), sparse (`Mohid Mahakali Hills Brochure.pdf`), and the scanned one (`JLL_Mumbai_Lodha Regalia_8409_BRC_1.pdf`) to confirm the Tesseract fallback engages and produces a usable structured record.
- Run all 7 example query types, confirming grounded citations, correct "not found" behavior, structured vs. plain output per intent, correct per-project attribution, and that the JSON-validation retry logic recovers from at least one deliberately-triggered malformed-output case.
- Ingest one more brochure not touched during dev, ask a fresh question, to rehearse "handling the unknown."
- Spot-check latency per query/ingestion on this machine's hardware — if the 7B model is too slow, document the `qwen2.5:3b-instruct` fallback in the README and re-test.
