"""Two retrieval paths: vector RAG for factual/spec Q&A, exhaustive structured-store
scans for filter/comparison/summarization (where top-k similarity could silently miss
a matching project)."""

import json
import logging

from . import llm, store
from .schemas import ComparisonResult, FilterResult, Source

log = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the uploaded documents."

ANSWER_SYSTEM = (
    "You answer questions about real-estate brochures using ONLY the provided context — structured "
    "facts already extracted from a brochure, and/or raw excerpts from it. Never use outside knowledge "
    "or invent facts. If the answer is not present in either source, say clearly that it was not found "
    "in the uploaded documents — do not guess. When you state a fact, mention which project it came "
    "from. If multiple items match a general description (e.g. several unit configurations of the "
    "same bedroom count), list all of them rather than picking just one. Be concise."
)

FILTER_SYSTEM = (
    "You filter a small table of real-estate project records against a natural-language criteria "
    "string. The criteria may combine several conditions at once (amenity, unit type, developer, "
    "location, price, or any other field) — evaluate ALL of them, not just the first one you notice. "
    "Match synonyms and nearby-area references sensibly (e.g. 'pool' matches 'swimming pool', 'near "
    "Andheri West' can match a location text that names Andheri West or an immediately adjacent area). "
    "Only use the data given — never invent amenities or facts not present in the records. A null/missing "
    "field means that fact is unknown for that project, not that it's absent — do not treat 'unknown' as "
    "a non-match if every other condition is satisfied; note the uncertainty in your reason instead."
)

COMPARE_SYSTEM = (
    "You build a side-by-side comparison table for real-estate projects using ONLY the provided data. "
    "For any dimension not present in the data for a project, use the string 'not found'. Never invent facts. "
    "Keep each value concise — a short phrase or comma-separated list, one sentence at most. Never reproduce "
    "a full list (e.g. every amenity) verbatim; summarize it instead."
)

SUMMARY_SYSTEM = (
    "You write a concise, factual one-paragraph summary of a real-estate project for a prospective "
    "buyer, using only the information given. Do not invent details, prices, or availability that "
    "aren't present in the input."
)


def _compact_table(records: list[dict]) -> str:
    compact = [
        {
            "project_name": r.get("project_name"),
            "developer": r.get("developer"),
            "location": r.get("location"),
            "expected_handover": r.get("expected_handover"),
            "starting_price": r.get("starting_price"),
            "amenities": r.get("amenities", []),
            "unit_types": r.get("unit_types", []),
            "other_facts": r.get("other_facts"),
        }
        for r in records
    ]
    return json.dumps(compact, indent=2)


def answer_factual(question: str, doc_ids: list[str] | None = None) -> dict:
    """Vector RAG over chunks, plus the already-extracted structured record for any named
    project — filter/compare/summarize aren't the only paths that can use it. Chunk-only
    retrieval can miss a fact (e.g. a RERA number or amenity list) that the forced structured-
    extraction call already captured cleanly at ingestion time."""
    [query_embedding] = llm.embed([question])

    records = []
    if doc_ids:
        hits = []
        for doc_id in doc_ids:
            hits.extend(store.query_chunks(query_embedding, top_k=6, doc_id=doc_id))
        records = [r for r in store.load_all_records() if r["doc_id"] in doc_ids]
    else:
        hits = store.query_chunks(query_embedding, top_k=8)

    log.info("answer_factual: question=%r doc_ids=%s hits=%d records=%d", question, doc_ids, len(hits), len(records))

    if not hits and not records:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}

    context_parts = []
    if records:
        context_parts.append("Structured facts already extracted for the named project(s):\n" + _compact_table(records))
    if hits:
        chunk_context = "\n\n".join(
            f"[{h['metadata'].get('project_name') or h['metadata']['source_file']}, "
            f"p. {h['metadata']['page_number']}]\n{h['text']}"
            for h in hits
        )
        context_parts.append("Raw excerpts:\n\n" + chunk_context)

    prompt = "\n\n".join(context_parts) + f"\n\nQuestion: {question}"
    answer = llm.chat_text(prompt, system=ANSWER_SYSTEM, max_tokens=1024)

    sources = [
        Source(
            project_name=h["metadata"].get("project_name") or None,
            source_file=h["metadata"]["source_file"],
            page=h["metadata"]["page_number"],
        )
        for h in hits
    ]
    if not sources:
        sources = [Source(project_name=r.get("project_name"), source_file=r["source_file"]) for r in records]

    return {"answer_text": answer, "table_data": None, "sources": sources}


def filter_projects(criteria: str, names: list[str] | None = None) -> dict:
    """`names`, when given, scopes the table to specific named project(s) before filtering —
    e.g. "does Vogue 77 offer 3BR units" should only ever look at Vogue 77, not every project."""
    records = store.load_all_records()
    if names:
        records = [r for r in records if r.get("project_name") in names]
    if not records:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}

    table = _compact_table(records)
    prompt = f"Project data (JSON):\n{table}\n\nFilter request: {criteria}"
    result = llm.generate_structured(prompt, FILTER_SYSTEM, FilterResult, max_tokens=1024)
    log.info("filter_projects: criteria=%r names=%s -> %d match(es)", criteria, names, len(result.matches))

    if not result.matches:
        return {"answer_text": result.summary or NOT_FOUND, "table_data": None, "sources": []}

    by_name = {r.get("project_name"): r for r in records}
    table_data, sources = [], []
    for m in result.matches:
        rec = by_name.get(m.project_name)
        table_data.append(
            {
                "project_name": m.project_name,
                "reason": m.reason,
                "location": rec.get("location") if rec else None,
            }
        )
        sources.append(Source(project_name=m.project_name, source_file=rec["source_file"] if rec else ""))

    return {"answer_text": result.summary, "table_data": table_data, "sources": sources}


def compare_projects(names: list[str], dimensions: list[str]) -> dict:
    records = store.load_all_records()
    selected = [r for r in records if r.get("project_name") in names] if names else records
    if not selected:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}

    table = _compact_table(selected)
    dims = ", ".join(dimensions) if dimensions else "developer, location, expected handover, amenities, unit types"
    prompt = f"Project data (JSON):\n{table}\n\nCompare these projects across: {dims}"
    result = llm.generate_structured(prompt, COMPARE_SYSTEM, ComparisonResult, max_tokens=4000)
    log.info("compare_projects: names=%s dimensions=%s -> %d row(s)", names, dimensions, len(result.rows))

    table_data = [
        {"project_name": row.project_name, **{v.dimension: v.value for v in row.values}}
        for row in result.rows
    ]
    sources = [Source(project_name=r.get("project_name"), source_file=r["source_file"]) for r in selected]

    return {"answer_text": result.summary, "table_data": table_data or None, "sources": sources}


def summarize_projects(names: list[str] | None) -> dict:
    records = store.load_all_records()
    selected = [r for r in records if r.get("project_name") in names] if names else records
    if not selected:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}
    log.info("summarize_projects: names=%s -> %d project(s)", names, len(selected))

    [generic_embedding] = llm.embed(["key amenities, unit types, location, and unique selling points"])

    summaries, sources = [], []
    for r in selected:
        chunks = store.query_chunks(generic_embedding, top_k=3, doc_id=r["doc_id"])
        context = "\n\n".join(c["text"] for c in chunks)
        record_json = json.dumps(r, indent=2)
        prompt = (
            f"Structured record:\n{record_json}\n\nExcerpts:\n{context}\n\n"
            "Write a concise one-paragraph summary of this project for a prospective buyer, "
            "using only the given information."
        )
        summary = llm.chat_text(prompt, system=SUMMARY_SYSTEM, max_tokens=400)
        summaries.append(f"**{r.get('project_name') or r['source_file']}**\n{summary}")
        sources.append(Source(project_name=r.get("project_name"), source_file=r["source_file"]))

    return {"answer_text": "\n\n".join(summaries), "table_data": None, "sources": sources}
