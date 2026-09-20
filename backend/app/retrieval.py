"""Two retrieval paths: vector RAG for factual/spec Q&A, exhaustive structured-store
scans for filter/comparison/summarization (where top-k similarity could silently miss
a matching project)."""

import json
import logging
import re

from . import llm, store
from .schemas import ComparisonResult, ConditionSplit, FilterMatch, FilterResult, Source, UnitMatchSelection

log = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the uploaded documents."

ANSWER_SYSTEM = (
    "ROLE: Answer a question about a real-estate project using only the given context — structured "
    "facts already extracted from its brochure, and/or raw excerpts from it.\n\n"

    "GROUNDING: never use outside knowledge or invent a fact. If the answer isn't present in either "
    "source, say plainly that it wasn't found in the uploaded documents rather than guessing. When you "
    "state a fact, name the project it came from. If several items match a general description (e.g. "
    "multiple unit configurations of the same bedroom count), list all of them, not just one. But a "
    "fact that has exactly one true value per project (location/address, RERA number, price, "
    "developer, expected handover) gets exactly ONE bullet: if the context contains overlapping or "
    "slightly differently worded excerpts about it, synthesize them into the single most complete "
    "statement rather than listing each phrasing as if it were a separate fact.\n\n"

    "OUTPUT: bullet points only, one fact per bullet — no prose paragraph and no preamble. A "
    "single-fact answer is still a one-bullet list. Exception: a not-found/abstain answer is a "
    "plain one-line statement, not a bullet."
)

FILTER_SYSTEM = (
    "ROLE: Evaluate each real-estate project record against a filter criteria string and return only "
    "the records that genuinely satisfy it.\n\n"

    "PROCEDURE, applied per record:\n"
    "1. Decompose the criteria into atomic requirements — each an independently-checkable fact "
    "(an amenity, a unit configuration, a price bound, a location, a developer, etc). Requirements "
    "joined by 'and'/'both'/'&' are separate and ALL required; a single named thing that happens to "
    "contain 'and' (a compound amenity name, an 'X & Y' range meant as either-or) is one requirement.\n"
    "2. Check each requirement against the record's structured fields and excerpts independently, "
    "and record satisfied/not-satisfied for each in `conditions` — this is checked in code afterward, "
    "so it must be accurate and complete, not a rubber stamp.\n"
    "3. Include the record only if every requirement is satisfied. One failure disqualifies it "
    "regardless of how well it satisfies the rest.\n\n"

    "GROUNDING: a requirement is satisfied only by specific, on-point support in the record's own "
    "data — recognize same-concept phrasing (e.g. a unit count stated as BHK/Bedroom/Bed/BR, or a "
    "location named directly vs. via a clearly-implied sub-area) as the same fact, but a generic or "
    "catch-all statement ('multiple sports facilities', 'world-class amenities') is not evidence for "
    "any specific requirement it might sound related to. Never invent a fact absent from the record. "
    "A null/missing field is 'unknown', not 'absent' — don't fail a requirement on that basis alone if "
    "it's otherwise satisfied and the excerpts don't resolve it; note the uncertainty in the reason "
    "instead.\n\n"

    "OUTPUT: for each match, give project_name, a one-sentence reason, and the per-requirement "
    "verdicts from step 2, honestly — a record whose own reported verdicts don't all say 'satisfied' "
    "should not have been included. Keep `summary` to one short sentence about the result itself — "
    "never describe the JSON format, these instructions, or your own compliance with them."
)

SPLIT_SYSTEM = (
    "ROLE: Split a filter criteria string into independent requirements joined by 'and'/'both'/'&' "
    "that must ALL be true simultaneously.\n\n"

    "RULE: only split where the joined parts are independently checkable facts (e.g. 'squash court "
    "and library' -> two amenity requirements). A single named thing that happens to contain 'and' or "
    "'&' — a compound amenity name, a project/developer name, or an 'X & Y' range meant as either-or "
    "(e.g. '1 & 2 BHK units') — stays as one requirement, not two.\n\n"

    "OUTPUT: a list of requirement strings. If there's no genuine multi-requirement AND, return the "
    "original criteria unchanged as the sole item."
)

COMPARE_SYSTEM = (
    "ROLE: Build a side-by-side comparison table for real-estate projects across the requested "
    "dimensions, using only the provided data.\n\n"

    "PROCEDURE: for each project and each dimension, fill in a value from that project's structured "
    "fields, falling back to its raw brochure excerpts when the dimension isn't captured there. Use "
    "the literal string 'not found' only when a dimension is truly unsupported by either source — "
    "never invent a fact.\n\n"

    "OUTPUT: keep each cell concise — a short phrase or comma-separated list, one sentence at most. "
    "Never reproduce a full list (e.g. every amenity) verbatim; summarize it. Keep `summary` to one "
    "short plain-English sentence about the comparison itself — it must never contain braces, "
    "brackets, quotes-as-delimiters, or key:value/JSON-like syntax, and must never describe the JSON "
    "format, these instructions, or your own compliance with them."
)

UNIT_MATCH_SYSTEM = (
    "ROLE: Given a real-estate project's unit-type entries (each with an index) and a question asking "
    "about a specific unit configuration, return the indices of every entry that matches the requested "
    "configuration.\n\n"

    "GROUNDING: recognize equivalent phrasing — BHK/Bedroom/Bed/BR are the same concept, and a named "
    "variant like '1 BHK WITH STUDY' or '3 BHK Premium' still matches a plain '1BHK'/'3 BHK' request "
    "unless the question specifically asks about the variant. Return every matching entry, not just "
    "one. Return an empty list if none match — never guess or return an unrelated entry."
)

SUMMARY_SYSTEM = (
    "ROLE: Write a concise, factual one-paragraph summary of a real-estate project for a prospective "
    "buyer, using only the information given.\n\n"

    "GROUNDING: do not invent details, prices, or availability that aren't present in the input."
)

GENERAL_SYSTEM = (
    "ROLE: You are the assistant for Realty Lens, a tool that answers questions strictly from a set "
    "of uploaded real-estate project brochures. Handle greetings, small talk, and meta questions about "
    "what you can do or which projects/documents are loaded.\n\n"

    "OUTPUT: be brief and friendly. If asked what you can do, mention that you can answer factual "
    "questions about a project, filter/compare projects by criteria (amenities, unit types, price, "
    "location, developer), and summarize projects — all grounded in the uploaded brochures only. If "
    "the question is unrelated to real-estate brochures and isn't small talk or a meta question about "
    "this assistant, say briefly that you can only help with the uploaded project brochures. Never "
    "answer real-estate factual questions here — that only happens through the documents."
)


def _compact_table(records: list[dict], excerpts_by_doc_id: dict[str, list[str]] | None = None) -> str:
    compact = []
    for r in records:
        entry = {
            "project_name": r.get("project_name"),
            "developer": r.get("developer"),
            "location": r.get("location"),
            "expected_handover": r.get("expected_handover"),
            "starting_price": r.get("starting_price"),
            "amenities": r.get("amenities", []),
            "unit_types": r.get("unit_types", []),
            "other_facts": r.get("other_facts"),
        }
        excerpts = (excerpts_by_doc_id or {}).get(r.get("doc_id"))
        if excerpts:
            entry["brochure_excerpts"] = excerpts
        compact.append(entry)
    return json.dumps(compact, indent=2)


def _excerpts_by_doc_id(records: list[dict], query_text: str, top_k: int = 3) -> dict[str, list[str]]:
    """Per-project raw-chunk excerpts relevant to `query_text`, so filter/comparison can resolve
    criteria that never made it into the structured extraction (loan partners, legal approvals,
    parking ratio, ...) instead of silently treating them as absent. Local embeddings + a local
    Chroma query per project — cheap at this corpus scale, and keeps the exhaustive-scan guarantee
    since every record still gets looked at."""
    if not records or not query_text:
        return {}
    [query_embedding] = llm.embed([query_text])
    out: dict[str, list[str]] = {}
    for r in records:
        doc_id = r.get("doc_id")
        if not doc_id:
            continue
        hits = store.query_chunks(query_embedding, top_k=top_k, doc_id=doc_id)
        if hits:
            out[doc_id] = [h["text"] for h in hits]
    return out


def answer_factual(question: str, doc_ids: list[str] | None = None, highlight_field: str | None = None) -> dict:
    """Vector RAG over chunks, plus the already-extracted structured record for any named
    project — filter/compare/summarize aren't the only paths that can use it. Chunk-only
    retrieval can miss a fact (e.g. a RERA number or amenity list) that the forced structured-
    extraction call already captured cleanly at ingestion time.

    `highlight_field`, when given, surfaces that one structured field (e.g. "developer") in its
    own labeled block ahead of the full record — cheap, and lets developer_info answer straight
    from the field the vast majority of the time without dropping the chunk search, which still
    runs unconditionally so a question asking for more than the field holds (JV partners, ...)
    isn't cut off."""
    [query_embedding] = llm.embed([question])

    records = []
    if doc_ids:
        hits = []
        for doc_id in doc_ids:
            hits.extend(store.query_chunks(query_embedding, top_k=6, doc_id=doc_id))
        # Merge duplicate records for the same project (re-upload, or split across documents) so
        # the structured-facts context doesn't hand the model two conflicting versions of one fact.
        records = store.group_and_merge_by_project([r for r in store.load_all_records() if r["doc_id"] in doc_ids])
    else:
        hits = store.query_chunks(query_embedding, top_k=8)

    log.info("answer_factual: question=%r doc_ids=%s hits=%d records=%d", question, doc_ids, len(hits), len(records))

    if not hits and not records:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}

    context_parts = []
    if records:
        if highlight_field:
            highlighted = [
                {"project_name": r.get("project_name"), highlight_field: r.get(highlight_field)}
                for r in records
                if r.get(highlight_field)
            ]
            if highlighted:
                context_parts.append(
                    f"{highlight_field.capitalize()} field already extracted for the named project(s):\n"
                    + json.dumps(highlighted, indent=2)
                )
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


def answer_developer_info(question: str, doc_ids: list[str] | None = None) -> dict:
    """developer_info: thin wrapper over answer_factual that surfaces the structured `developer`
    field up front. The chunk search still runs unchanged alongside it — cheap enough at this
    corpus scale, and needed whenever the question asks for more than the field holds (JV
    partners, etc.), so there's no need to special-case that detection here."""
    return answer_factual(question, doc_ids, highlight_field="developer")


_BHK_NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*[-\s]?(?:bhk|bed\s*room|bedroom|bed|br)\b", re.IGNORECASE)


def _bhk_number(text: str) -> str | None:
    match = _BHK_NUMBER.search(text or "")
    return match.group(1) if match else None


def _match_unit_types(question: str, unit_types: list[dict]) -> list[dict]:
    """All unit_types entries matching the question's requested configuration — every match, not
    top-k, same "consider everything" principle as the filter path. Simple string matching first
    (handles 'BHK'/'Bedroom'/'Bed'/'BR' synonyms and named variants like '1 BHK WITH STUDY' via
    the shared leading-number extraction), LLM as fallback for names the number regex can't parse
    (e.g. 'penthouse', 'duplex'). No configuration detected in the question at all -> every entry
    matches (a generic "what units does X offer" isn't a filter)."""
    if not unit_types:
        return []

    wanted = _bhk_number(question)
    if wanted is None:
        return unit_types

    matched = [u for u in unit_types if _bhk_number(u.get("type") or "") == wanted]
    if matched:
        return matched

    listing = json.dumps([{"index": i, **u} for i, u in enumerate(unit_types)], indent=2)
    prompt = f"Unit types (JSON, with index):\n{listing}\n\nQuestion: {question}"
    try:
        result = llm.generate_structured(prompt, UNIT_MATCH_SYSTEM, UnitMatchSelection, max_tokens=512)
        return [unit_types[i] for i in result.indices if 0 <= i < len(unit_types)]
    except Exception:
        log.exception("_match_unit_types: LLM fallback failed for question=%r", question)
        return []


def answer_specification(question: str, doc_ids: list[str] | None = None) -> dict:
    """specification_lookup: match the requested unit configuration against each project's
    already-extracted unit_types structurally first, instead of relying on top-k chunk similarity
    to surface the right table row. Falls back to the normal chunk search per-project when that
    project has no unit_types extracted (or none matched), so scanned brochures with a missed
    extraction don't regress versus the old behavior."""
    records = store.group_and_merge_by_project(
        [r for r in store.load_all_records() if not doc_ids or r["doc_id"] in doc_ids]
    )
    if not records:
        return answer_factual(question, doc_ids)

    [query_embedding] = llm.embed([question])

    structural_parts, chunk_hits, cited_hits, structural_used = [], [], [], False
    for r in records:
        matched = _match_unit_types(question, r.get("unit_types") or [])
        if matched:
            structural_used = True
            entry = {"project_name": r.get("project_name"), "matched_unit_types": matched}
            excerpt_hits = store.query_chunks(query_embedding, top_k=6, doc_id=r["doc_id"])
            if excerpt_hits:
                entry["brochure_excerpts"] = [h["text"] for h in excerpt_hits]
                cited_hits.extend(excerpt_hits)
            structural_parts.append(entry)
        else:
            hits = store.query_chunks(query_embedding, top_k=6, doc_id=r["doc_id"])
            chunk_hits.extend(hits)
            cited_hits.extend(hits)

    if not structural_used:
        # No project in scope had a matching (or any) unit_types entry -- identical to the old
        # chunk-vector-search + full-record behavior end to end.
        return answer_factual(question, doc_ids)

    log.info(
        "answer_specification: question=%r doc_ids=%s structural=%d fallback_hits=%d",
        question, doc_ids, len(structural_parts), len(chunk_hits),
    )

    context_parts = []
    if structural_parts:
        context_parts.append(
            "Matched unit-type entries already extracted for the named project(s):\n"
            + json.dumps(structural_parts, indent=2)
        )
    if chunk_hits:
        chunk_context = "\n\n".join(
            f"[{h['metadata'].get('project_name') or h['metadata']['source_file']}, "
            f"p. {h['metadata']['page_number']}]\n{h['text']}"
            for h in chunk_hits
        )
        context_parts.append("Raw excerpts (project(s) with no structural match above):\n\n" + chunk_context)

    prompt = "\n\n".join(context_parts) + f"\n\nQuestion: {question}"
    answer = llm.chat_text(prompt, system=ANSWER_SYSTEM, max_tokens=1024)

    sources = [
        Source(
            project_name=h["metadata"].get("project_name") or None,
            source_file=h["metadata"]["source_file"],
            page=h["metadata"]["page_number"],
        )
        for h in cited_hits
    ]
    if not sources:
        sources = [Source(project_name=r.get("project_name"), source_file=r["source_file"]) for r in records]

    return {"answer_text": answer, "table_data": None, "sources": sources}


def _split_conditions(criteria: str) -> list[str]:
    """Split a compound AND-criteria string into its independent parts. Single LLM call, cheap
    (short output). Falls back to treating the whole string as one condition on any failure."""
    try:
        split = llm.generate_structured(
            f"Filter request: {criteria}", SPLIT_SYSTEM, ConditionSplit, max_tokens=256
        )
        conditions = [c.strip() for c in split.conditions if c.strip()]
        return conditions or [criteria]
    except Exception:
        log.exception("_split_conditions: failed, falling back to single condition (criteria=%r)", criteria)
        return [criteria]


def _scan_records(records: list[dict], condition: str) -> FilterResult:
    """Single-condition LLM scan over the table, with a self-reported per-record condition
    checklist enforced in code afterward — a record only survives if it reports every one of its
    own stated sub-conditions as satisfied (guards against the model listing a record as a match
    while contradicting itself in the checklist)."""
    excerpts = _excerpts_by_doc_id(records, condition)
    table = _compact_table(records, excerpts)
    prompt = f"Project data (JSON):\n{table}\n\nFilter request: {condition}"
    result = llm.generate_structured(prompt, FILTER_SYSTEM, FilterResult, max_tokens=4096)

    kept, dropped = [], []
    for m in result.matches:
        if m.conditions and not all(c.satisfied for c in m.conditions):
            dropped.append(m.project_name)
        else:
            kept.append(m)
    if dropped:
        log.warning(
            "_scan_records: dropped %d match(es) failing self-reported check: %s (condition=%r)",
            len(dropped), dropped, condition,
        )
    result.matches = kept
    return result


def filter_projects(criteria: str, names: list[str] | None = None) -> dict:
    """`names`, when given, scopes the table to specific named project(s) before filtering —
    

    A compound "X and Y" criteria is split into independent conditions and each is scanned
    separately, then the matching project sets are intersected in Python. A single LLM call
    reasoning over several AND-conditions at once has been observed to union matches instead of
    intersecting them (and to self-contradict even when asked to self-check); running one
    condition per call and intersecting deterministically closes that failure mode instead of
    trusting the model to hold several conditions in its head simultaneously."""
    records = store.load_deduped_records()
    if names:
        records = [r for r in records if r.get("project_name") in names]
    if not records:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}

    conditions = _split_conditions(criteria)
    log.info("filter_projects: criteria=%r -> %d condition(s): %s", criteria, len(conditions), conditions)

    if len(conditions) <= 1:
        result = _scan_records(records, criteria)
    else:
        per_condition = [_scan_records(records, cond) for cond in conditions]
        by_name_lists = [{m.project_name: m for m in r.matches} for r in per_condition]
        common_names = set.intersection(*(set(d) for d in by_name_lists)) if by_name_lists else set()
        matches = [
            FilterMatch(
                project_name=name,
                reason=" ".join(d[name].reason for d in by_name_lists),
            )
            for name in common_names
        ]
        summary = (
            f"Found {len(matches)} project(s) matching all {len(conditions)} conditions."
            if matches
            else "No projects match all of the given conditions."
        )
        result = FilterResult(matches=matches, summary=summary)

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


def _looks_malformed(summary: str) -> bool:
    """Guard against the model dumping the structured rows into the free-text summary field
    instead of a plain sentence (observed as raw '{"project_name": ..., "values": [...]}' or
    'rows: [...]' text leaking into the answer). Cheap heuristic: a one-sentence summary should
    never contain JSON/dict punctuation or run unreasonably long."""
    if not summary:
        return True
    markers = ("{", "}", "[", "]", "rows:", "values:")
    return any(m in summary for m in markers) or len(summary) > 400


def compare_projects(names: list[str], dimensions: list[str]) -> dict:
    """`dimensions` empty means the caller didn't ask for anything specific — fall back to the
    handful of fields a buyer would compare by default rather than an empty/useless table."""
    records = store.load_deduped_records()
    selected = [r for r in records if r.get("project_name") in names] if names else records
    if not selected:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}

    dims = ", ".join(dimensions) if dimensions else "developer, location, expected handover, amenities, unit types"
    excerpts = _excerpts_by_doc_id(selected, dims)
    table = _compact_table(selected, excerpts)
    prompt = f"Project data (JSON):\n{table}\n\nCompare these projects across: {dims}"
    result = llm.generate_structured(prompt, COMPARE_SYSTEM, ComparisonResult, max_tokens=8192)
    log.info("compare_projects: names=%s dimensions=%s -> %d row(s)", names, dimensions, len(result.rows))

    if _looks_malformed(result.summary):
        log.warning("compare_projects: malformed summary from model, replacing: %r", result.summary)
        project_list = ", ".join(r.get("project_name") or "" for r in selected)
        result.summary = f"Comparison of {project_list} on {dims}."

    table_data = [
        {"project_name": row.project_name, **{v.dimension: v.value for v in row.values}}
        for row in result.rows
    ]
    sources = [Source(project_name=r.get("project_name"), source_file=r["source_file"]) for r in selected]

    return {"answer_text": result.summary, "table_data": table_data or None, "sources": sources}


def summarize_projects(names: list[str] | None) -> dict:
    records = store.load_deduped_records()
    selected = [r for r in records if r.get("project_name") in names] if names else records
    if not selected:
        return {"answer_text": NOT_FOUND, "table_data": None, "sources": []}
    log.info("summarize_projects: names=%s -> %d project(s)", names, len(selected))

    # A summary has no natural query text of its own, so embed a generic "what a buyer cares
    # about" description instead — pulls the most summary-relevant chunks per project rather
    # than an arbitrary/empty set.
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


def handle_general(question: str, known_projects: list[str]) -> dict:
    """Greetings, "what can you do", and other meta/out-of-scope questions that don't need the
    document pipeline at all — kept out of factual_qa/developer_info/specification_lookup so they don't produce an unhelpful
    "not found in the uploaded documents" for something that was never a document lookup."""
    projects_list = ", ".join(known_projects) if known_projects else "(none ingested yet)"
    prompt = f"Ingested projects: {projects_list}\n\nUser message: {question}"
    answer = llm.chat_text(prompt, system=GENERAL_SYSTEM, max_tokens=300)
    return {"answer_text": answer, "table_data": None, "sources": []}
