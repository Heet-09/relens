# Realty Lens — One-Pager

## What it is
A prototype assistant for Realty Lens staff: upload PDF project brochures, ask natural-language
questions (factual Q&A, amenity/unit filters, cross-document comparisons, summaries), and get
answers grounded strictly in the uploaded documents — including a graceful "not found" when the
documents don't say.

## How it works
Streamlit frontend + FastAPI backend. On upload: PyMuPDF extracts per-page text; pages with too
little extractable text (common in graphics-heavy brochures, and the one fully-scanned sample
PDF) are rasterized and read via Gemini's vision input instead of a separate OCR toolchain. The
extracted page text is cached to disk (`data/pages/<doc_id>.json`) before anything else happens
to it. Text is then chunked page-aware and embedded locally (`sentence-transformers`,
`BAAI/bge-small-en-v1.5`) into a Chroma collection — no embedding API key, nothing leaves the
machine at this step; one additional forced structured-output Gemini call extracts a compact
structured JSON record (developer, location, handover, amenities, unit types) per document.
Multiple documents in one upload batch are ingested concurrently (bounded semaphore + thread
pool), and within a document, vision calls for its low-text pages also run concurrently.

At query time, a Gemini structured-output call routes the question to one of five intents. Open-ended
factual/spec questions go to vector similarity search over chunks (optionally scoped to a named
project) with page-level citations. Filter, comparison, and summarization questions instead scan
the structured JSON store exhaustively across every ingested document, since top-k similarity
search can silently miss a matching project when the query is a filter rather than a lookup. All
generation is constrained to answer only from the retrieved context. Every question and answer is
appended to a JSONL history log; the frontend replays it on load so a page refresh (or a restart
of the Streamlit process) doesn't lose the conversation.

On backend startup, a reconciliation pass compares the structured-record store against Chroma: any
document whose chunks are missing (lost to a crash, a corrupted restart, or manual deletion) is
rebuilt from its cached page text — a local re-embed with zero Gemini vision calls — instead of
silently serving an ingested-but-unsearchable document or requiring a manual re-upload.

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
- **Cache raw page text, not just the vector index.** The Gemini vision calls are the expensive,
  hard-to-reproduce part of ingestion; the embeddings are cheap and local. Caching the former
  separately means the latter can always be rebuilt for free, which is what makes startup
  self-healing possible without re-spending API quota.
- **A single write lock around Chroma/structured-record writes**, once ingestion became
  concurrent across documents. Chroma's SQLite backend isn't guaranteed safe against concurrent
  writers, so the (cheap, fast) commit step is serialized while the (slow, network-bound)
  extraction work ahead of it stays parallel.
- **doc_id as the join key** between the vector store and structured store keeps per-document
  provenance simple and lets both paths cite back to a specific source file/page.

## Given more time
- Real hybrid retrieval (BM25 + embeddings) and re-ranking for factual Q&A on longer brochures.
- Persist and expose confidence/uncertainty on extracted structured fields, and surface low-
  confidence extractions to a human reviewer.
- Streamed upload progress per document instead of one blocking call for the whole batch.
- Automated eval set (golden Q&A pairs per brochure) to catch retrieval/prompt regressions.
- Move structured records and history off flat JSON/JSONL onto a real database once multi-user
  access or concurrent writers from outside a single process become a requirement.
- Auth and per-user document scoping if this moves beyond a single-team prototype.
