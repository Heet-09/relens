"""PDF -> pages -> text/vision extraction -> chunking -> structured record."""

import concurrent.futures
import logging
import os
import re
import uuid

import fitz  # PyMuPDF

from . import llm, store
from .schemas import Chunk, ExtractedRecord, PageText, ProjectRecord, UnitType

log = logging.getLogger(__name__)

LOW_TEXT_THRESHOLD = 200  # chars; pages below this are treated as low-text
MAX_VISION_PAGES = 24  # cap Gemini vision calls per doc
VISION_CONCURRENCY = int(os.environ.get("VISION_CONCURRENCY", "4"))  # concurrent Gemini vision calls per doc
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_STRUCTURED_INPUT_CHARS = 60000

VISION_PROMPT = (
    "ROLE: Transcribe all readable text from this real-estate brochure page — headings, amenity "
    "names, unit specifications, tables, prices, locations, contact info.\n\n"

    "OUTPUT: preserve structure with line breaks; render any tables as plain text rows.\n\n"

    "GROUNDING: do not add commentary or invent text that isn't visible on the page."
)

STRUCTURED_SYSTEM = (
    "ROLE: Extract structured facts from real-estate brochure text for a factual database.\n\n"

    "GROUNDING: use only information explicitly present in the text. If a field isn't stated, use "
    "null (or an empty list for list fields) — never guess or invent a value.\n\n"

    "PROCEDURE for unit_types: record every distinct unit type/configuration label mentioned anywhere "
    "in the text (e.g. floor plan headings like '3 BHK', '2 BHK WITH STUDY'), even if its area, price, "
    "or features aren't stated — an incomplete unit type is still a real one and must appear in the "
    "list with those fields set to null.\n\n"

    "PROCEDURE for amenities: list only concrete, individually named amenities (e.g. 'swimming pool', "
    "'squash court', 'clubhouse'), one per entry. Omit a vague, generic, or catch-all claim ('multiple "
    "sports facilities', 'various recreational amenities', 'world-class amenities') rather than adding "
    "it as an entry — a generic claim with no named specific isn't a fact worth recording, and a "
    "recorded vague phrase can later be mistaken by a filter for a specific amenity match."
)


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    """Per-page text via PyMuPDF; low-text pages are rasterized and sent to Gemini vision.

    Real-estate brochures conventionally put unit floor plans, specs, and RERA/legal details
    on their LAST few pages. If a scanned document has more low-text pages than the vision
    budget allows, spending that budget strictly front-to-back silently drops exactly that
    high-value tail. So when the budget is tight, it's split between the start and the end
    of the low-text page list instead — sacrificing middle "lifestyle" pages first.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        raw_texts = [doc[i].get_text("text").strip() for i in range(len(doc))]
        low_text_indices = [i for i, t in enumerate(raw_texts) if len(t) < LOW_TEXT_THRESHOLD]

        if len(low_text_indices) > MAX_VISION_PAGES:
            head_n = (MAX_VISION_PAGES + 1) // 2
            tail_n = MAX_VISION_PAGES - head_n
            vision_indices = set(low_text_indices[:head_n]) | set(low_text_indices[-tail_n:] if tail_n else [])
        else:
            vision_indices = set(low_text_indices)

        # Rasterizing is fast (CPU, milliseconds) so it stays sequential here — PyMuPDF pages
        # aren't safe to touch from multiple threads. The Gemini vision call is pure network
        # wait per page and dominates total ingestion time, so those calls (and only those)
        # run concurrently instead of one page at a time.
        png_by_index = {i: doc[i].get_pixmap(dpi=150).tobytes("png") for i in vision_indices}
        vision_text_by_index: dict[int, str] = {}
        if png_by_index:
            with concurrent.futures.ThreadPoolExecutor(max_workers=VISION_CONCURRENCY) as pool:
                future_to_index = {
                    pool.submit(llm.vision_extract, png, VISION_PROMPT): i for i, png in png_by_index.items()
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        vision_text_by_index[i] = future.result()
                    except Exception:
                        log.exception("vision_extract failed for page %d", i + 1)
                        vision_text_by_index[i] = ""

        pages: list[PageText] = []
        for i in range(len(doc)):
            text = raw_texts[i]
            if i in vision_indices:
                vision_text = vision_text_by_index.get(i, "")
                combined = f"{text}\n{vision_text}".strip() if text else vision_text
                pages.append(PageText(page_number=i + 1, text=combined, is_vision=True))
            else:
                pages.append(PageText(page_number=i + 1, text=text, is_vision=False))
    finally:
        doc.close()
    log.info(
        "extract_pages: %d page(s), %d used vision fallback (of %d low-text page(s))",
        len(pages),
        len(vision_indices),
        len(low_text_indices),
    )
    return pages


def chunk_pages(pages: list[PageText], doc_id: str, source_file: str) -> list[Chunk]:
    """Page-aware chunking with overlap. project_name is filled in by ingest_document."""
    chunks: list[Chunk] = []
    for page in pages:
        text = re.sub(r"\n{3,}", "\n\n", page.text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        project_name=None,
                        source_file=source_file,
                        page_number=page.page_number,
                        text=piece,
                    )
                )
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
    log.info("chunk_pages: doc_id=%s -> %d chunk(s)", doc_id, len(chunks))
    return chunks


def extract_structured_record(full_text: str) -> dict:
    if not full_text.strip():
        return {}
    truncated = full_text[:MAX_STRUCTURED_INPUT_CHARS]
    prompt = f"Brochure text:\n\n{truncated}"
    record = llm.generate_structured(prompt, STRUCTURED_SYSTEM, ExtractedRecord, max_tokens=4096)
    return record.model_dump()


def ingest_document(pdf_bytes: bytes, filename: str) -> tuple[ProjectRecord, list[Chunk]]:
    """Full per-document pipeline: page extraction -> structured-record extraction -> chunking.
    Structured extraction runs on the full document text rather than per-chunk so amenities/unit
    types scattered across pages get consolidated into one record instead of fragmenting."""
    doc_id = uuid.uuid4().hex[:12]
    log.info("ingest_document: start filename=%r doc_id=%s", filename, doc_id)

    pages = extract_pages(pdf_bytes)
    is_scanned = any(p.is_vision for p in pages)
    store.save_pages(doc_id, pages)

    full_text = "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages if p.text.strip())
    raw = extract_structured_record(full_text)

    unit_types = [UnitType(**u) for u in raw.get("unit_types", [])]
    record = ProjectRecord(
        doc_id=doc_id,
        source_file=filename,
        project_name=raw.get("project_name"),
        developer=raw.get("developer"),
        location=raw.get("location"),
        expected_handover=raw.get("expected_handover"),
        starting_price=raw.get("starting_price"),
        amenities=raw.get("amenities", []),
        unit_types=unit_types,
        other_facts=raw.get("other_facts"),
        is_scanned=is_scanned,
    )
    log.info(
        "ingest_document: extracted project=%r developer=%r location=%r amenities=%d unit_types=%d scanned=%s",
        record.project_name,
        record.developer,
        record.location,
        len(record.amenities),
        len(record.unit_types),
        record.is_scanned,
    )

    chunks = chunk_pages(pages, doc_id, filename)
    for chunk in chunks:
        chunk.project_name = record.project_name

    log.info("ingest_document: done doc_id=%s -> %d chunk(s)", doc_id, len(chunks))
    return record, chunks
