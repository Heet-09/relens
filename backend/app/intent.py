"""Forced structured-output Gemini call that routes a question to one of five intents."""

import logging

from . import llm
from .schemas import IntentClassification

log = logging.getLogger(__name__)

ROUTER_SYSTEM_TEMPLATE = """You route user questions for a real-estate brochure assistant to one of five intents:
- factual_qa: an open-ended factual/spec question about ONE already-named project ("what is the RERA
  number for X", "what floor is the gym on", "does X have a 3BR unit")
- amenity_filter: any question that needs to check a criterion across MULTIPLE or ALL ingested
  projects and return the matching/ranked subset — not just literal amenities. This also covers:
  filtering or counting by developer ("how many projects does Runwal have"), by location ("which
  projects are near Andheri West"), by price ("projects under 1.5 Cr"), combinations of these with
  amenities, and superlative comparisons across every project ("which is nearest to a station",
  "which is cheapest"). If a question requires looking at more than one project to answer correctly,
  and isn't an explicit named-pair comparison, it belongs here even if it never says the word "amenity".
- unit_filter: "which projects have a 3BHK" style filtering specifically by unit type/configuration
  across multiple or all projects
- comparison: an explicit comparison between two or more named projects
- summarization: "summarize project X" / "give me an overview of all projects"

Known ingested projects:
{projects_list}

Extract project names mentioned in the question, matching them to the known project list where
possible (fuzzy match ok). Leave project_names empty if the question applies to all projects or
names none specifically. A question naming exactly one project (e.g. "does Vogue 77 offer 3BR
units") should still set project_names to that project — this scopes the filter/lookup to it."""


def classify_intent(question: str, known_projects: list[str]) -> dict:
    projects_list = "\n".join(f"- {p}" for p in known_projects) or "(none ingested yet)"
    system = ROUTER_SYSTEM_TEMPLATE.format(projects_list=projects_list)
    result = llm.generate_structured(question, system, IntentClassification, max_tokens=512)
    log.info(
        "Intent: %r -> %s (projects=%s, amenities=%s, unit_filter_type=%s, dims=%s)",
        question,
        result.intent,
        result.project_names,
        result.amenities,
        result.unit_filter_type,
        result.comparison_dimensions,
    )
    return result.model_dump()
