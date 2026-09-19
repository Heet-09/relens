"""Forced structured-output Gemini call that routes a question to one of six intents."""

import logging

from . import llm
from .schemas import IntentClassification

log = logging.getLogger(__name__)

ROUTER_SYSTEM_TEMPLATE = """ROLE: Route a user question for a real-estate brochure assistant to one \
of six intents, and extract the structured fields the downstream handler needs.

INTENTS:
- factual_qa: an open-ended factual/spec question about ONE already-named project ("what is the RERA
  number for X", "what floor is the gym on", "does X have a 3BR unit", "what unit types/amenities does
  X offer" — asking ONE named project to list or describe its own attributes is always factual_qa, even
  if the attribute is "units" or "amenities"; unit_filter/amenity_filter only apply when the question is
  checking a criterion across projects to decide which ones qualify)
- amenity_filter: any question that needs to check a criterion across MULTIPLE or ALL ingested
  projects and return the matching/ranked subset — not just literal amenities. This also covers:
  filtering or counting by developer ("how many projects does Runwal have"), by location ("which
  projects are near Andheri West"), by price ("projects under 1.5 Cr"), combinations of these with
  amenities, and superlative comparisons across every project ("which is nearest to a station",
  "which is cheapest"). If a question requires looking at more than one project to answer correctly,
  and isn't an explicit named-pair comparison, it belongs here even if it never says the word "amenity".
- unit_filter: "which projects have a 3BHK" style filtering specifically by unit type/configuration
  across multiple or all projects — not "what units does X offer" for a single named project, which is
  factual_qa
- comparison: an explicit comparison between two or more named projects
- summarization: "summarize project X" / "give me an overview of all projects"
- general: greetings/small talk, questions about the assistant itself ("what can you do", "how does
  this work"), or meta questions about the corpus ("which projects do you have", "how many documents
  are ingested") that don't need brochure content to answer. Anything clearly unrelated to real-estate
  brochures also belongs here.

CONTEXT:
Known ingested projects:
{projects_list}

Recent conversation (oldest first, may be empty):
{history_block}

PROCEDURE:
1. Extract project names mentioned in the question, matching them to the known project list where
   possible (fuzzy match ok). Leave project_names empty if the question applies to all projects or
   names none specifically. A question naming exactly one project (e.g. "does Vogue 77 offer 3BR
   units") should still set project_names to that project — this scopes the filter/lookup to it.
2. If the question only makes sense in light of the conversation above (it uses a pronoun, says "the
   second one", "that project", omits the subject, etc.), rewrite it as a fully standalone question in
   resolved_question, using the conversation to fill in what's missing. If the question already stands
   on its own, leave resolved_question empty.
3. If the question asks two or more genuinely separate things that need different answers (e.g.
   "summarize X and how does it compare to Y on price"), split it into 2-4 standalone sub_questions
   and leave sub_questions populated — the rest of the fields can be your best-effort guess for the
   question as a whole. If it's a single-part question, leave sub_questions empty."""


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "(none)"
    lines = []
    for entry in history:
        lines.append(f"Q: {entry.get('question', '')}")
        lines.append(f"A: {entry.get('answer_text', '')}")
    return "\n".join(lines)


def classify_intent(question: str, known_projects: list[str], history: list[dict] | None = None) -> dict:
    projects_list = "\n".join(f"- {p}" for p in known_projects) or "(none ingested yet)"
    system = ROUTER_SYSTEM_TEMPLATE.format(projects_list=projects_list, history_block=_format_history(history))
    result = llm.generate_structured(question, system, IntentClassification, max_tokens=768)
    log.info(
        "Intent: %r -> %s (projects=%s, amenities=%s, unit_filter_type=%s, dims=%s, resolved=%r, sub_questions=%s)",
        question,
        result.intent,
        result.project_names,
        result.amenities,
        result.unit_filter_type,
        result.comparison_dimensions,
        result.resolved_question,
        result.sub_questions,
    )
    return result.model_dump()
