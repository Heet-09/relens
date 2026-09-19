# Realty Lens — One-Pager

## What it is
A prototype assistant for Realty Lens staff: upload PDF project brochures, ask natural-language
questions and get answers grounded
strictly in the uploaded documents — with a clear "not found" when the documents don't say.

## How it works
Streamlit frontend + FastAPI backend. **Ingestion:** PyMuPDF extracts per-page text; low-text
pages (graphics-heavy or fully-scanned brochures) are rasterized and read via Gemini vision
instead of an OCR toolchain, with the vision budget split front/back since brochures put specs
on their last pages. Extracted text is cached to disk before anything else touches it, then
chunked page-aware and embedded locally into Chroma — no embedding API
key, nothing leaves the machine. One forced structured-output Gemini call per document also
extracts a compact JSON record (developer, location, amenities, unit types). Multiple documents
ingest concurrently.

**Query:** a Gemini structured-output call routes each question its intents.
Open-ended factual/spec questions use vector similarity search (optionally scoped to a named
project) with page-level citations. Filter, comparison, and summarization questions instead scan
the structured JSON store exhaustively across every document, since top-k similarity can
silently miss a matching project on a filter query. All generation is constrained to answer only
from retrieved context. History is logged to JSONL and replayed on reload.

On startup, a self-heal pass rebuilds any document's Chroma chunks from cached page text if
they're missing (crash, corrupted restart) — no re-spent Gemini vision calls.

## Key decisions and why
- **Two retrieval paths, not one** — a single vector index would silently under-serve "list ALL
  projects with X" queries; splitting by intent trades complexity for correctness there.
- **Gemini vision over OCR** — one sample brochure is fully image-only; vision avoids
  an OS-dependent install and reads marketing layouts better, at the cost of per-page latency
  (bounded by a text-density threshold and page cap).
- **Forced structured output everywhere reasoning needs to be trusted downstream** (extraction,
  intent routing, filtering) — avoids fragile free-text parsing.
- **Single LLM vendor (Gemini)** — one key, one file (`llm.py`) owning the provider integration.
- **Local embeddings, not an embedding API** — brochure text never leaves the machine at that
  step, at the cost of a first-run model download.
- **Cache raw page text separately from the vector index** — vision calls are the expensive,
  hard-to-reproduce part; embeddings are cheap and local, so only the former needs caching to
  make self-healing free.
- **doc_id as the join key** between the vector store and structured store keeps per-document
  provenance and citations simple.

## Given more time

- Confidence/uncertainty on extracted fields, surfaced to a human reviewer for low-confidence
  cases.
- Streamed per-document upload progress instead of one blocking batch call.
- Add chunk-level retrieval recall and an LLM-judge faithfulness score for free-text answers —
  the two eval dimensions not yet covered by the intent-routing and filter-precision evals we
  ran (kept local, happy to share on request).
- Real database instead of flat JSON/JSONL once multi-user/concurrent-writer access is needed.
- **Move Chroma off its local `PersistentClient` (SQLite-backed, single-process) to server mode
  or a managed vector store** (Chroma server, Qdrant, pgvector)
- Auth and per-user document scoping beyond a single-team prototype.
- **Lead-qualification agent** on top of the existing filter path — take a client's stated
  requirements (budget, location, BHK) and auto-run the filter query to hand the sales rep a
  shortlist, instead of the rep typing it themselves.