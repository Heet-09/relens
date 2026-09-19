# Architecture — Realty Lens (Gemini + local embeddings)

```mermaid
flowchart TD
    UI[Streamlit UI] -->|HTTP| API[FastAPI backend]

    subgraph Ingestion["POST /documents/upload"]
        direction TB
        P1["1. Per-page text extract (PyMuPDF)"]
        P2["2. Low-text pages -> rasterize + Gemini vision"]
        P3["3. Page-aware chunking (~800 chars, ~150 overlap)"]
        P4["4. Embed chunks (local sentence-transformers) -> Chroma"]
        P5["5. Forced structured-output Gemini call -> ProjectRecord JSON"]
        P1 --> P2 --> P3 --> P4
        P2 --> P5
    end

    API --> Ingestion
    Ingestion --> Chroma[(Chroma: chunks per doc_id)]
    Ingestion --> JSONStore[(JSON structured store: one record per doc_id)]

    subgraph QueryFlow["POST /query"]
        direction TB
        Router["Intent Router (Gemini structured output)\nfactual_qa | amenity_filter | unit_filter | comparison | summarization"]
        VecPath["Vector search (Chroma, local embeddings)\nfiltered by resolved project_name if named"]
        ScanPath["Exhaustive scan of structured JSON store\nacross ALL ingested docs"]
        Gen["Gemini generation call\n'answer only from provided context; say not found if absent'"]
        Router -->|factual/spec| VecPath --> Gen
        Router -->|filter/compare/summarize| ScanPath --> Gen
    end

    API --> QueryFlow
    VecPath --> Chroma
    ScanPath --> JSONStore
    Gen --> Resp["QueryResponse {intent, answer_text, table_data, sources}"]
    Resp --> UI
```

## Who does what

- **Gemini** — everything reasoning-related: chat generation, the intent router (forced
  structured JSON output, not free-text parsing), forced structured extraction of each brochure's
  facts, and vision reads for pages with too little extractable text (graphics-heavy brochures,
  and the one fully-scanned sample PDF with zero extractable characters). One vendor, one API key.
- **Local `sentence-transformers`** (`BAAI/bge-small-en-v1.5` by default) — embeddings for the
  Chroma vector store. Runs on-machine; no embedding API key or network call, so brochure text
  never leaves the machine at that step.

All structured outputs (intent classification, brochure field extraction, filter results,
comparison tables) are defined as Pydantic models in `schemas.py` and passed straight to Gemini
as `response_schema`, so the model is constrained to return valid, schema-shaped JSON rather than
prose that has to be parsed and can drift out of shape.

`llm.py` is the only file that imports the Gemini SDK or the embedding model — every other module
calls `llm.chat_text()`, `llm.generate_structured()`, `llm.vision_extract()`, or `llm.embed()`, so
swapping the underlying provider later only means rewriting `llm.py`.

## Why two retrieval paths?

Filter queries ("list ALL projects with pool AND gym") and comparisons need *completeness*
across every ingested document — top-k similarity search can silently miss a matching
project. So at ingestion time, one extra LLM call per document produces a small structured
JSON record (developer, location, handover, amenities[], unit_types[]). Filter/compare/
summarize queries scan this compact table exhaustively; open-ended factual/spec questions use
chunk-level vector RAG with page citations.

## Why a vision model instead of Tesseract OCR for scanned pages?

Avoids an OS-dependent toolchain (Tesseract's install differs per OS), and a vision LLM reads
marketing layouts (icons, tables, stylized headers) more robustly than raw OCR. Only invoked on
pages under a text-density threshold, capped at a max page count per document, to bound
latency/cost.

## Model choice note

The default Gemini model (`GEMINI_MODEL` env var, default `gemini-flash-lite-latest`) is a
self-updating alias rather than a pinned snapshot, chosen after observing the heavier
`gemini-flash-latest` alias return transient `503 UNAVAILABLE` ("high demand") errors during
development. `llm.py` also retries transient 5xx errors with backoff before giving up.
