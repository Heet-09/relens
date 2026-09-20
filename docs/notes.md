# Engineering notes

## Redundant table rebuild in filter_projects (retrieval.py)

`filter_projects` splits a compound "X and Y" criteria into N independent conditions and
calls `_scan_records` once per condition. Each `_scan_records` call rebuilds `_compact_table`
from scratch (full JSON per project: amenities, unit_types, other_facts, etc.) and re-embeds
+ re-queries excerpts for that condition.

Cost: an N-condition filter re-dumps the *entire* project table N times into the LLM prompt,
even though the base record fields (everything except the per-condition excerpts) are
identical across all N calls. For a large corpus this is unnecessary token/cost overhead.

Fix (not yet implemented): build the compact base table once per `filter_projects` call and
reuse it across condition scans, only re-fetching the per-condition excerpts each time.

Note: this doesn't affect correctness, only cost — exhaustiveness itself (every record must
be considered per condition) is intentional and shouldn't be replaced with top-k retrieval,
since that reintroduces the recall-miss problem the filter path exists to avoid.

## developer_info always runs vector search, even when the JSON field already has the answer

`developer_info` has no dedicated handler (main.py) — it falls into the same generic
`answer_factual` path as `specification_lookup`/any other single-project question. That path
always does two things regardless of what's actually needed: reads the structured record
(which already has a `developer` field from ingestion) AND runs a vector similarity search
over the document's chunks, then stuffs both into the prompt.

For a plain "who's the developer" question, the JSON field alone would usually be enough —
the vector search is arguably wasted work in the common case.

Why it's there anyway:
- the field can be null if extraction missed it at ingestion time
- it's a single string, so compound questions (JV partners, holding company vs. contractor)
  aren't fully captured by it
- the JSON record has no page numbers — chunk search is the only source of the citations
  shown to the user

Fix (not yet implemented): try the JSON field first; only fall back to the chunk search when
the field is null or the question needs more than the field holds. Keeps the common case
cheap while keeping the null/compound-question safety net.

## Route smaller/simpler queries to a smaller model

Every LLM call in the pipeline (intent classification, condition split, filter scan, compare,
summarize, answer_factual) currently goes through the same model regardless of how simple the
call actually is. Some of these are cheap, low-reasoning tasks that a smaller/cheaper model
would likely handle just as reliably:

- intent classification — fixed taxonomy of 8 labels, mostly pattern matching against the
  disambiguation rules
- condition splitting (`_split_conditions`) — short, mechanical text-splitting task
- single-project factual/spec lookups where the structured field already has the answer
  (ties into the developer_info/specification_lookup notes above)

Bigger-reasoning calls (filter scan across ambiguous/compound criteria, comparison table
synthesis, free-text summarization) likely still need the larger model's judgment.

Fix (not yet implemented): add a size/complexity tier to `llm.py`'s call sites — route
classification + splitting + simple field lookups to a smaller model, keep filter/compare/
summarize on the current model. Would cut cost on the highest-volume calls (every single query
pays for at least one classification call) without touching the paths where reasoning quality
actually matters.


What is the carpet area of a 1BHK unit at 95 WEST?

For the project 95 WEST, the 1 BHK unit has a carpet area of 365 sq. ft.
For the project 95 WEST, the 1 BHK unit has a carpet area of 308 sq. ft.
For the project 95 WEST, the 1 BHK unit has a carpet area of 310 sq. ft.
For the project 95 WEST, the 1 BHK unit has a carpet area of 311 sq. ft.
For the project 95 WEST, the 1 BHK unit has a carpet area of 364 sq. ft.444
