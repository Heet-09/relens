"""Chroma persistent vector store (chunks) + flat JSON structured-record store (per doc_id)
+ an append-only JSONL chat history log."""

import json
import logging
import os
import threading
from pathlib import Path

import chromadb

log = logging.getLogger(__name__)

# Guards the Chroma add() and structured-record write below: cheap, fast operations, but
# multiple documents can now be ingested concurrently, so their final "commit to store" step
# must not race (Chroma's sqlite backend isn't guaranteed safe against concurrent writers).
_write_lock = threading.Lock()

DATA_DIR = Path(os.environ.get("RELENS_DATA_DIR", "data"))
CHROMA_DIR = DATA_DIR / "chroma"
STRUCTURED_DIR = DATA_DIR / "structured"
PAGES_DIR = DATA_DIR / "pages"
HISTORY_FILE = DATA_DIR / "history.jsonl"

_chroma_client = None
_collection = None


def _client():
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        _collection = _client().get_or_create_collection("brochure_chunks")
    return _collection


def add_chunks(chunks, embeddings) -> None:
    if not chunks:
        return
    collection = get_collection()
    ids = [f"{c.doc_id}-{i}" for i, c in enumerate(chunks)]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "project_name": c.project_name or "",
            "source_file": c.source_file,
            "page_number": c.page_number,
        }
        for c in chunks
    ]
    documents = [c.text for c in chunks]
    with _write_lock:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    log.info("Added %d chunks to Chroma for doc_id=%s", len(chunks), chunks[0].doc_id)


def count_chunks_for_doc(doc_id: str) -> int:
    collection = get_collection()
    if collection.count() == 0:
        return 0
    result = collection.get(where={"doc_id": doc_id})
    return len(result["ids"])


def query_chunks(query_embedding, top_k: int = 6, doc_id: str | None = None) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    where = {"doc_id": doc_id} if doc_id else None
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    hits = []
    if result["ids"] and result["ids"][0]:
        for i in range(len(result["ids"][0])):
            hits.append(
                {
                    "text": result["documents"][0][i],
                    "distance": result["distances"][0][i],
                    "metadata": result["metadatas"][0][i],
                }
            )
    log.debug("query_chunks: doc_id=%s top_k=%d -> %d hit(s)", doc_id, top_k, len(hits))
    return hits


def save_record(record) -> None:
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    path = STRUCTURED_DIR / f"{record.doc_id}.json"
    with _write_lock:
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    log.info("Saved structured record doc_id=%s project=%r -> %s", record.doc_id, record.project_name, path)


def load_all_records() -> list[dict]:
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(STRUCTURED_DIR.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def merge_project_records(records: list[dict]) -> dict:
    """Collapse several structured records for the same project (re-uploaded brochure, or a
    project split across multiple source documents) into one canonical view: scalar fields keep
    the first non-null value found, amenities/unit_types are unioned. Without this, a duplicated
    upload shows up as two rows in a comparison table or as two conflicting facts in an answer."""
    if len(records) == 1:
        return records[0]

    merged = dict(records[0])
    for field in ("developer", "location", "expected_handover", "starting_price", "other_facts"):
        if not merged.get(field):
            for r in records[1:]:
                if r.get(field):
                    merged[field] = r[field]
                    break

    amenities: list[str] = []
    for r in records:
        for a in r.get("amenities") or []:
            if a not in amenities:
                amenities.append(a)
    merged["amenities"] = amenities

    unit_types: list[dict] = []
    seen = set()
    for r in records:
        for u in r.get("unit_types") or []:
            key = (u.get("type"), u.get("area_sqft"))
            if key not in seen:
                seen.add(key)
                unit_types.append(u)
    merged["unit_types"] = unit_types

    return merged


def group_and_merge_by_project(records: list[dict]) -> list[dict]:
    """Group records by project_name (case-insensitive) and merge each group into one record via
    merge_project_records. Records with no project_name are kept separate (nothing to key on)."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in records:
        key = (r.get("project_name") or "").strip().lower() or r["doc_id"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [merge_project_records(groups[k]) for k in order]


def load_deduped_records() -> list[dict]:
    """load_all_records collapsed to one merged record per project — the view every
    filter/comparison/summary should use so a duplicated or split upload doesn't produce
    duplicate rows or conflicting facts."""
    return group_and_merge_by_project(load_all_records())


def save_pages(doc_id: str, pages) -> None:
    """Cache extracted (post-vision) page text — the expensive-to-recompute artifact — so the
    vector index can be rebuilt cheaply (local re-embed, no Gemini vision calls) if it's ever
    lost to a crash, a corrupted restart, or manual deletion."""
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = PAGES_DIR / f"{doc_id}.json"
    path.write_text(json.dumps([p.model_dump() for p in pages], indent=2), encoding="utf-8")


def load_pages(doc_id: str) -> list[dict] | None:
    path = PAGES_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_doc_ids(project_names: list[str] | None) -> list[str] | None:
    """Best-effort, case-insensitive substring match against known project_name values."""
    if not project_names:
        return None
    records = load_all_records()
    matched = []
    for name in project_names:
        name_l = name.lower().strip()
        if not name_l:
            continue
        for r in records:
            pn = (r.get("project_name") or "").lower()
            if pn and (name_l in pn or pn in name_l) and r["doc_id"] not in matched:
                matched.append(r["doc_id"])
    log.debug("resolve_doc_ids: names=%s -> doc_ids=%s", project_names, matched)
    return matched or None


def add_history_entry(entry) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
    log.info("History: intent=%s question=%r", entry.intent, entry.question)


def load_history(limit: int = 200) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    lines = [line for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]
