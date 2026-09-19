# Realty Lens — One-Pager

## What it is
A prototype assistant for Realty Lens staff: upload PDF project brochures, ask natural-language
questions (factual Q&A, amenity/unit filters, cross-document comparisons, summaries), and get
answers grounded strictly in the uploaded documents — including a graceful "not found" when the
documents don't say.

## How it works
Streamlit frontend + FastAPI backend. On upload: PyMuPDF extracts per-page text; pages with too
little extractable text (common in graphics-heavy brochures, and the one fully-scanned sample
PDF) are rasterized and read via Gemini's vision input instead of a separate OCR toolchain. Text
is chunked page-aware and embedded locally (`sentence-transformers`, `BAAI/bge-small-en-v1.5`)
into a per-document Chroma collection — no embedding API key, nothing leaves the machine at this
step; one additional forced structured-output Gemini call extracts a compact structured JSON
record (developer, location, handover, amenities, unit types) per document.

At query time, a Gemini structured-output call routes the question to one of five intents. Open-ended
factual/spec questions go to vector similarity search over chunks (optionally scoped to a named
project) with page-level citations. Filter, comparison, and summarization questions instead scan
the structured JSON store exhaustively across every ingested document, since top-k similarity
search can silently miss a matching project when the query is a filter rather than a lookup. All
generation is constrained to answer only from the retrieved context.

## Key decisions and why
- **Two retrieval paths, not one.** A single vector index would silently under-serve "list ALL
  projects with X" queries. Splitting by intent trades a bit of complexity for correctness on the
  query types the brief explicitly calls out.
- **Gemini vision over Tesseract OCR for scanned pages.** One of the 18 sample brochures is fully
  image-only with zero extractable text — confirmed by direct inspection before building. Vision
  avoids an OS-dependent install step and reads marketing layouts more robustly than raw OCR,
  at the cost of per-page latency, mitigated by a text-density threshold and a page cap.
- **Forced structured output (Pydantic schema -> Gemini `response_schema`) for extraction and
  intent routing.** Both need reliable, schema-shaped output the rest of the pipeline can trust
  without extra parsing/repair logic — free-text prose parsing would be fragile here.
- **Single LLM vendor (Gemini) for everything reasoning-related.** One API key, one SDK, one
  file (`llm.py`) that owns the provider integration — simpler setup and a smaller surface for
  the "assume the reviewer has nothing else" requirement.
- **Local embeddings instead of an embedding API.** Brochure text never leaves the machine for
  the vector-store step, and it removes one of the three external dependencies — at the cost of
  a heavier first-run model download and CPU-bound encode time versus a hosted embedding call.
- **doc_id as the join key** between the vector store and structured store keeps per-document
  provenance simple and lets both paths cite back to a specific source file/page.

## Given more time
- Real hybrid retrieval (BM25 + embeddings) and re-ranking for factual Q&A on longer brochures.
- Persist and expose confidence/uncertainty on extracted structured fields, and surface low-
  confidence extractions to a human reviewer.
- Batch/async ingestion with progress streaming instead of a blocking upload call, for large PDFs.
- Automated eval set (golden Q&A pairs per brochure) to catch retrieval/prompt regressions.
- Auth and per-user document scoping if this moves beyond a single-team prototype.
