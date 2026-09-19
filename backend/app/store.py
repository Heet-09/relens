"""Chroma persistent vector store (chunks) + flat JSON structured-record store (per doc_id)
+ an append-only JSONL chat history log."""

import json
import logging
import os
from pathlib import Path

import chromadb

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("RELENS_DATA_DIR", "data"))
CHROMA_DIR = DATA_DIR / "chroma"
STRUCTURED_DIR = DATA_DIR / "structured"
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
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    log.info("Added %d chunks to Chroma for doc_id=%s", len(chunks), chunks[0].doc_id)


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
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    log.info("Saved structured record doc_id=%s project=%r -> %s", record.doc_id, record.project_name, path)


def load_all_records() -> list[dict]:
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(STRUCTURED_DIR.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


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
