from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import os
from datetime import datetime, timezone

from .logging_config import setup_logging

setup_logging()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import llm, store
from .ingestion import chunk_pages, ingest_document
from .intent import classify_intent
from .retrieval import answer_factual, compare_projects, filter_projects, handle_general, summarize_projects
from .schemas import DocumentInfo, HistoryEntry, PageText, QueryRequest, QueryResponse, UploadResponse

log = logging.getLogger(__name__)

app = FastAPI(title="Realty Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Each document's own vision calls already run concurrently (see ingestion.VISION_CONCURRENCY);
# this bounds how many documents' pipelines run at once so a big batch upload doesn't serialize
# end-to-end. Kept conservative by default since it multiplies with per-doc vision concurrency
# against the same Gemini quota — raise it if your quota tier allows more.
MAX_CONCURRENT_DOCUMENTS = int(os.environ.get("MAX_CONCURRENT_DOCUMENTS", "2"))


@app.on_event("startup")
def reconcile_vector_store() -> None:
    """Self-heal: if a structured record exists but its chunks are missing from Chroma (lost to
    a crash, a corrupted restart, or manual deletion), rebuild them from the cached page-text —
    a cheap local re-embed, no Gemini vision calls needed again."""
    healed, unrecoverable = 0, 0
    for record in store.load_all_records():
        doc_id = record["doc_id"]
        if store.count_chunks_for_doc(doc_id) > 0:
            continue

        cached_pages = store.load_pages(doc_id)
        if not cached_pages:
            unrecoverable += 1
            log.warning(
                "Startup check: doc_id=%s (%s) has no chunks and no cached pages — re-upload needed",
                doc_id,
                record.get("project_name"),
            )
            continue

        pages = [PageText(**p) for p in cached_pages]
        chunks = chunk_pages(pages, doc_id, record["source_file"])
        for chunk in chunks:
            chunk.project_name = record.get("project_name")
        if chunks:
            embeddings = llm.embed([c.text for c in chunks])
            store.add_chunks(chunks, embeddings)
            healed += 1
            log.info(
                "Startup check: rebuilt %d chunk(s) for doc_id=%s (%s) from cached pages",
                len(chunks),
                doc_id,
                record.get("project_name"),
            )

    if healed or unrecoverable:
        log.info("Startup reconciliation: healed=%d unrecoverable=%d", healed, unrecoverable)


def _ingest_one_document(filename: str, pdf_bytes: bytes) -> DocumentInfo:
    """Synchronous per-document pipeline, run in a worker thread so several documents can be
    in flight at once without blocking the event loop or each other."""
    try:
        record, chunks = ingest_document(pdf_bytes, filename)
    except Exception as exc:
        log.exception("Ingestion failed for %r", filename)
        return DocumentInfo(doc_id="", source_file=filename, warnings=[f"Ingestion failed: {exc}"])

    warnings = []
    if record.is_scanned:
        warnings.append("Low text density detected — used vision extraction for one or more pages.")
    if not record.project_name:
        warnings.append("Could not confidently determine project name.")

    if chunks:
        embeddings = llm.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)
    else:
        warnings.append("No extractable text found in this document.")

    store.save_record(record)
    return DocumentInfo(
        doc_id=record.doc_id,
        project_name=record.project_name,
        source_file=record.source_file,
        is_scanned=record.is_scanned,
        warnings=warnings,
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    log.info("Upload request: %d file(s)", len(files))

    results: list[DocumentInfo | None] = [None] * len(files)
    pending: list[tuple[int, str, bytes]] = []
    for i, f in enumerate(files):
        if not f.filename.lower().endswith(".pdf"):
            log.warning("Rejected non-PDF upload: %r", f.filename)
            results[i] = DocumentInfo(doc_id="", source_file=f.filename, warnings=["Not a PDF, skipped."])
            continue
        pending.append((i, f.filename, await f.read()))

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOCUMENTS)

    async def _run(index: int, filename: str, pdf_bytes: bytes) -> None:
        async with semaphore:
            results[index] = await asyncio.to_thread(_ingest_one_document, filename, pdf_bytes)

    await asyncio.gather(*(_run(i, filename, pdf_bytes) for i, filename, pdf_bytes in pending))
    return UploadResponse(documents=results)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    records = store.load_all_records()
    return [
        DocumentInfo(
            doc_id=r["doc_id"],
            project_name=r.get("project_name"),
            source_file=r["source_file"],
            is_scanned=r.get("is_scanned", False),
        )
        for r in records
    ]


@app.get("/history", response_model=list[HistoryEntry])
def get_history(limit: int = 200):
    return store.load_history(limit=limit)


def _dispatch_single(question: str, known_projects: list[str], history: list[dict], allow_split: bool) -> tuple[str, dict]:
    """Classify and answer one standalone question. `allow_split` gates sub_questions handling —
    set False on the recursive calls from a compound question so splitting can't recurse forever."""
    try:
        intent_result = classify_intent(question, known_projects, history)
    except Exception as exc:
        log.exception("Intent classification failed")
        raise HTTPException(status_code=502, detail=f"Intent classification failed: {exc}")

    sub_questions = intent_result.get("sub_questions") or []
    if allow_split and len(sub_questions) >= 2:
        log.info("Compound question %r split into %d sub-question(s)", question, len(sub_questions))
        parts, sources = [], []
        for sub_q in sub_questions:
            _sub_intent, sub_result = _dispatch_single(sub_q, known_projects, history, allow_split=False)
            parts.append(sub_result["answer_text"])
            sources.extend(sub_result.get("sources", []))
        return "compound", {"answer_text": "\n\n".join(parts), "table_data": None, "sources": sources}

    resolved_question = intent_result.get("resolved_question") or question
    intent = intent_result.get("intent", "factual_qa")
    project_names = intent_result.get("project_names") or []
    comparison_dimensions = intent_result.get("comparison_dimensions") or []

    try:
        if intent == "general":
            result = handle_general(resolved_question, known_projects)
        elif intent in ("amenity_filter", "unit_filter"):
            # Pass the full (resolved) question, not a template rebuilt from a single slot —
            # the text already carries every condition (amenity, unit type, location, price,
            # ...) the templated version used to silently drop.
            result = filter_projects(resolved_question, project_names or None)
        elif intent == "comparison":
            result = compare_projects(project_names, comparison_dimensions)
        elif intent == "summarization":
            result = summarize_projects(project_names or None)
        else:
            doc_ids = store.resolve_doc_ids(project_names)
            result = answer_factual(resolved_question, doc_ids)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Query failed for intent=%s", intent)
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}")

    return intent, result


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    log.info("Query received: %r", req.question)
    records = store.load_all_records()
    known_projects = [r.get("project_name") for r in records if r.get("project_name")]
    history = store.load_history(limit=6)

    intent, result = _dispatch_single(req.question, known_projects, history, allow_split=True)

    response = QueryResponse(
        intent=intent,
        answer_text=result["answer_text"],
        table_data=result.get("table_data"),
        sources=result.get("sources", []),
    )

    store.add_history_entry(
        HistoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=req.question,
            intent=response.intent,
            answer_text=response.answer_text,
            table_data=response.table_data,
            sources=response.sources,
        )
    )
    log.info("Query resolved: intent=%s answer_len=%d sources=%d", intent, len(response.answer_text), len(response.sources))

    return response
