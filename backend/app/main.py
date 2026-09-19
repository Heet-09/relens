from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime, timezone

from .logging_config import setup_logging

setup_logging()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import llm, store
from .ingestion import ingest_document
from .intent import classify_intent
from .retrieval import answer_factual, compare_projects, filter_projects, summarize_projects
from .schemas import DocumentInfo, HistoryEntry, QueryRequest, QueryResponse, UploadResponse

log = logging.getLogger(__name__)

app = FastAPI(title="Realty Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    log.info("Upload request: %d file(s)", len(files))
    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            log.warning("Rejected non-PDF upload: %r", f.filename)
            results.append(DocumentInfo(doc_id="", source_file=f.filename, warnings=["Not a PDF, skipped."]))
            continue

        pdf_bytes = await f.read()
        try:
            record, chunks = ingest_document(pdf_bytes, f.filename)
        except Exception as exc:
            log.exception("Ingestion failed for %r", f.filename)
            results.append(DocumentInfo(doc_id="", source_file=f.filename, warnings=[f"Ingestion failed: {exc}"]))
            continue

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
        results.append(
            DocumentInfo(
                doc_id=record.doc_id,
                project_name=record.project_name,
                source_file=record.source_file,
                is_scanned=record.is_scanned,
                warnings=warnings,
            )
        )
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


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    log.info("Query received: %r", req.question)
    records = store.load_all_records()
    known_projects = [r.get("project_name") for r in records if r.get("project_name")]

    try:
        intent_result = classify_intent(req.question, known_projects)
    except Exception as exc:
        log.exception("Intent classification failed")
        raise HTTPException(status_code=502, detail=f"Intent classification failed: {exc}")

    intent = intent_result.get("intent", "factual_qa")
    project_names = intent_result.get("project_names") or []
    comparison_dimensions = intent_result.get("comparison_dimensions") or []

    try:
        if intent in ("amenity_filter", "unit_filter"):
            # Pass the full question, not a template rebuilt from a single slot — the
            # original text already carries every condition (amenity, unit type, location,
            # price, ...) the templated version used to silently drop.
            result = filter_projects(req.question, project_names or None)
        elif intent == "comparison":
            result = compare_projects(project_names, comparison_dimensions)
        elif intent == "summarization":
            result = summarize_projects(project_names or None)
        else:
            doc_ids = store.resolve_doc_ids(project_names)
            result = answer_factual(req.question, doc_ids)
    except Exception as exc:
        log.exception("Query failed for intent=%s", intent)
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}")

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
