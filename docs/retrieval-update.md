# Retrieval Architecture — One-Pager

## What it is / how it works
Realty Lens routes every question to one of 8 intents, then fetches data for it in one of three
ways depending on what kind of answer that intent needs:

- **Vector-primary** (`factual_qa`) — top-k chunk similarity search is the answer's main source,
  scoped to a named project when one resolves. Still exposed to silent recall risk: a fact whose
  chunk misses top-k just doesn't appear, no error.
- **Exhaustive JSON scan** (`amenity_filter`, `unit_filter`, `comparison`, `summarization`) —
  every project's structured record is considered, never ranked/filtered by similarity first.
  Compound filter criteria are split into independent conditions, scanned separately, and
  intersected in Python rather than trusted to the LLM. Vector search only ever supplements here
  (extra excerpts per project), never decides inclusion.
- **Structural-primary with fallback** (`specification_lookup`, `developer_info`) — both have a
  natural structured field to answer from, so both try that first:
  - `specification_lookup` matches the requested config against the project's *full* `unit_types`
    list (every entry, not top-k) and returns every match, instead of relying on chunk similarity
    to surface the right one. Falls back to chunk search only if the project has no `unit_types`
    at all.
  - `developer_info` highlights the already-extracted `developer` field up front in the prompt so
    the model answers from it reliably, but still runs chunk search alongside it every time —
    kept as a supplement, not a hard skip.

## Key decisions and why
- **Structural match over vector search wherever a field already exists.** `developer_info` and
  `specification_lookup` were routed through the same generic vector-RAG path as `factual_qa`,
  even though their answer is usually already sitting in the structured record. Chunk search
  should be the fallback for facts with no dedicated field, not the default for facts that have
  one.
- **List every matching `unit_types` entry, never pick one.** Same "exhaustive over top-k"
  principle the filter path already uses, applied at the field level: a structured list can have
  real, unresolved ambiguity (e.g. several "1 BHK" entries at different areas across
  floors/wings). Surfacing all of them is honest; picking whichever chunk search happened to
  retrieve was producing a different, contradictory answer on every run.
- **`developer_info` keeps vector search supplementary, not skipped.** A stricter
  zero-vector-search version was built and reverted — the chunk search is cheap at this corpus
  scale, and keeping it unconditional means a compound question (developer + JV partners,
  developer + handover) never silently loses the part the field doesn't cover.

## Given more time
- Fix two comparison bugs found in testing: a requested dimension (e.g. RERA registration)
  sometimes silently dropped from the output table, and duplicate rows for the same project —
  `compare_projects` has no per-dimension self-check, unlike the filter path's per-condition
  checklist.
- Stop rebuilding the full project table on every condition scan in `filter_projects` — an
  N-condition filter re-dumps the whole table N times; build it once, reuse across conditions.
- Head/tail-aware truncation for structured extraction (currently a plain 60k-char cutoff),
  matching how vision extraction already splits its page budget front/back so brochures' end
  pages (unit specs, RERA) aren't the first thing lost on a long document.
- Route cheap/mechanical LLM calls (intent classification, condition splitting) to a smaller
  model; keep the current model where real reasoning is needed.
- Extend the extraction schema with a `connectivity`/`nearby_landmarks` field if that becomes a
  common question type — right now it only survives in raw chunks, so `factual_qa` can't be made
  JSON-first the way the other two intents now are.
