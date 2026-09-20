"""Forced structured-output Gemini call that routes a question to one of eight intents."""

import logging

from . import llm
from .schemas import IntentClassification

log = logging.getLogger(__name__)

HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 300
MAX_SUB_QUESTIONS = 4

# Placeholders are filled with str.replace (not str.format) so braces are safe anywhere in this text.
ROUTER_SYSTEM_TEMPLATE = """ROLE: You route one user question for a real-estate brochure assistant to exactly one of eight intents \
and fill in the structured fields the downstream handler needs. Every project the assistant knows \
about was ingested from a brochure. Return only the structured object.

<intents>
- developer_info: a question about ONE project that asks about its developer/builder identity (who \
built it, which company, JV partners) — even when compounded with other facts like handover date. The \
presence of a developer-identity question is what triggers this, regardless of what else rides along.
- specification_lookup: a question about ONE project's unit-level physical specs: area/size of a unit, \
room count/configuration, floor plan details, carpet area, which floor a unit type is on.
- factual_qa: a catch-all for any other question about ONE project's own attributes that is neither \
developer-identity nor a unit spec: location, RERA number, expected handover on its own (not paired \
with a developer question), starting price, general/legal/contact facts, floor count of the building, \
amenities list. An open-ended lookup, not a selection across projects.
- amenity_filter: select, rank, or count projects across the corpus by a criterion. Despite the name, \
the criterion can be anything: an amenity, developer, location, price, handover date, or a \
superlative ("nearest to a station", "cheapest"). Also used for any filter that combines several \
conditions ("3BHK under 2 Cr with a pool"). The answer is a subset or a ranking of projects.
- unit_filter: select projects purely by unit type/configuration ("which projects have a 3BHK").
- comparison: the user asks to compare/contrast projects side by side: two or more named projects, or \
all projects, on stated or implied dimensions.
- summarization: summarize / give an overview of / describe one project, several named projects, or \
all projects. No project is excluded, even if the question mentions amenities or unit types.
- general: greetings, questions about the assistant itself, meta questions about the corpus ("which \
projects do you have", "how many documents are ingested"), and anything unrelated to real-estate \
brochures.
</intents>

<disambiguation>
Apply in order. Stop at the first rule that fits.
1. Work out scope first: the projects named, or the projects that "these / those / the second one / \
that project" refer to in the conversation, or all projects.
2. Verb is summarize / overview / describe -> summarization, for every project in scope.
3. Exactly one project in scope and the question asks about that project's developer/builder identity \
(who built it, which company, JV partners) -> developer_info, even when compounded with other facts \
like handover date. The developer-identity question is what triggers this rule; other facts riding \
along with it (handover, location, ...) do not change the intent.
4. Exactly one project in scope and the question asks about a unit's physical spec (area/size of a \
unit, room count/configuration, floor plan, carpet area, which floor a unit type is on) -> \
specification_lookup.
5. Exactly one project in scope and the user wants any other of that project's own attributes \
(location, RERA number, expected handover on its own, starting price, general/legal/contact facts, \
floor count, amenities list) -> factual_qa.
6. The user asks to compare / contrast / show side by side ("X vs Y", "compare all projects on price") \
-> comparison.
7. The user wants to know which projects qualify or win (a pick, a subset, a count, a ranking):
   - selection by unit configuration alone -> unit_filter
   - anything else, or several conditions combined -> amenity_filter
   This includes questions that name two projects but ask which one qualifies ("which of X and Y has \
a pool", "which of X and Y is cheaper").
8. Otherwise -> general.
</disambiguation>

<examples>
Example projects (illustrative only): Vogue 77, Aurum Heights, Skyline Residency

Q: Who is the developer of Vogue 77 and when is the expected handover?
-> intent=developer_info, project_names=["Vogue 77"]

Q: What is the total area of a 3-bedroom unit in Aurum Heights?
-> intent=specification_lookup, project_names=["Aurum Heights"]

Q: Where is Skyline Residency located and what is the expected handover date?
-> intent=factual_qa, project_names=["Skyline Residency"]

Q: What is the RERA number for Vogue 77?
-> intent=factual_qa, project_names=["Vogue 77"]

Q: Which projects have a swimming pool?
-> intent=amenity_filter, project_names=[], amenities=["swimming pool"]

Q: How many projects does Runwal have?
-> intent=amenity_filter, project_names=[]

Q: Which project is the cheapest?
-> intent=amenity_filter, project_names=[]

Q: Which projects are under 1.5 Cr and have a gym?
-> intent=amenity_filter, project_names=[], amenities=["gym"]

Q: Which of Vogue 77 and Aurum Heights has a pool?
-> intent=amenity_filter, project_names=["Vogue 77", "Aurum Heights"], amenities=["swimming pool"]

Q: Which projects have a 3BHK?
-> intent=unit_filter, project_names=[], unit_filter_type="3BHK"

Q: Which projects have a 3BHK under 2 Cr?
-> intent=amenity_filter, project_names=[], unit_filter_type="3BHK"

Q: Compare Vogue 77 and Aurum Heights on price and amenities
-> intent=comparison, project_names=["Vogue 77", "Aurum Heights"], comparison_dimensions=["price", "amenities"]

Q: Compare all projects on price
-> intent=comparison, project_names=[], comparison_dimensions=["price"]

Q: Compare Vogue 77 and Aurum Heights on location and RERA registration
-> intent=comparison, project_names=["Vogue 77", "Aurum Heights"], comparison_dimensions=["location", "RERA registration"]

Q: Summarize the amenities and unit types of all projects
-> intent=summarization, project_names=[]

Q: Give me an overview of Skyline Residency
-> intent=summarization, project_names=["Skyline Residency"]

Q: What can you do?
-> intent=general, project_names=[]

Q: Which projects do you have?
-> intent=general, project_names=[]

Conversation: the previous answer covered Vogue 77, Aurum Heights.
Q: Which of these have a gym?
-> intent=amenity_filter, project_names=["Vogue 77", "Aurum Heights"], amenities=["gym"], resolved_question="Which of Vogue 77 and Aurum Heights have a gym?"

Q: Summarize Vogue 77 and how does it compare to Aurum Heights on price?
-> intent=summarization (the first sub-question), project_names=["Vogue 77", "Aurum Heights"], sub_questions=["Summarize Vogue 77.", "Compare Vogue 77 and Aurum Heights on price."]
</examples>

<projects>
Known ingested projects:
{projects_list}
</projects>

<history>
Recent conversation (oldest first, may be "(none)"). Each answer may be followed by the projects it covered.
{history_block}
</history>

<procedure>
Fill the fields in this order.
1. project_names
   - Match names in the question to the known list (case, spacing, and typos are fine) and return the \
exact spelling from the list.
   - If a named project matches nothing in the list, return it as the user wrote it. Never swap in a \
different project.
   - For "these / those / both / the second one / that project", take the projects from the \
conversation. "The second one" means the second project listed in that answer.
   - Leave it empty only when the question applies to all projects or names none.
   - A single named project still goes in project_names, including for factual_qa, developer_info, \
and specification_lookup.
2. resolved_question: rewrite the question so it stands on its own, replacing pronouns and omitted \
subjects and naming the resolved projects explicitly. This is the exact text handed to the filter/lookup \
step, so it must carry every condition in the question (amenity, unit type, location, price, ...), not \
just the resolved subject. If the question already stands on its own, copy it unchanged.
3. intent: apply the disambiguation rules to the resolved scope, then fill the remaining fields.
4. sub_questions: split only when the user asks two or more genuinely separate things that need \
different answers. One filter with several conditions ("pool and gym under 2 Cr") is ONE question, \
so do not split it. Each sub-question must be fully standalone (name the projects explicitly), at most 4. \
When you split, set the top-level intent to the intent of the first sub-question; the caller routes \
each sub-question separately. For a single-part question leave sub_questions empty.
</procedure>"""


def _projects_from_entry(entry: dict) -> list[str]:
    names: list[str] = []
    for src in entry.get("sources") or []:
        name = src.get("project_name") if isinstance(src, dict) else getattr(src, "project_name", None)
        if name and name not in names:
            names.append(name)
    return names


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "(none)"
    lines = []
    for entry in history[-HISTORY_TURNS:]:
        answer = (entry.get("answer_text") or "").strip()
        if len(answer) > HISTORY_ANSWER_CHARS:
            answer = answer[:HISTORY_ANSWER_CHARS].rstrip() + "..."
        lines.append(f"Q: {entry.get('question', '')}")
        lines.append(f"A: {answer}")
        projects = _projects_from_entry(entry)
        if projects:
            lines.append(f"Projects covered by that answer: {', '.join(projects)}")
    return "\n".join(lines)


def _canonicalize(names: list[str], known_projects: list[str]) -> list[str]:
    """Snap model-returned names to the exact spelling in the known list; keep unknown names as-is.
    Downstream handlers (filter_projects/compare_projects/summarize_projects) match project_name by
    exact string equality, so a case/spacing mismatch here silently drops the project."""
    by_key = {p.strip().lower(): p for p in known_projects}
    out: list[str] = []
    for name in names:
        canonical = by_key.get(name.strip().lower(), name)
        if canonical not in out:
            out.append(canonical)
    return out


def _fallback(question: str) -> IntentClassification:
    # Open-ended lookup is the safest default when routing itself fails.
    return IntentClassification(intent="factual_qa", resolved_question=question)


def classify_intent(question: str, known_projects: list[str], history: list[dict] | None = None) -> dict:
    projects_list = "\n".join(f"- {p}" for p in known_projects) or "(none ingested yet)"
    system = ROUTER_SYSTEM_TEMPLATE.replace("{projects_list}", projects_list).replace(
        "{history_block}", _format_history(history)
    )

    result: IntentClassification | None = None
    for attempt in (1, 2):
        try:
            result = llm.generate_structured(question, system, IntentClassification, max_tokens=1024)
            break
        except Exception:
            log.exception("Intent classification failed (attempt %d) for %r", attempt, question)
    if result is None:
        log.error("Intent classification gave up for %r; falling back to factual_qa", question)
        result = _fallback(question)

    result.project_names = _canonicalize(result.project_names, known_projects)
    result.sub_questions = result.sub_questions[:MAX_SUB_QUESTIONS]
    if not (result.resolved_question or "").strip():
        result.resolved_question = question

    log.info(
        "Intent: %r -> %s (projects=%s, amenities=%s, unit_filter_type=%s, dims=%s, resolved=%r, "
        "sub_questions=%s, reasoning=%r)",
        question,
        result.intent,
        result.project_names,
        result.amenities,
        result.unit_filter_type,
        result.comparison_dimensions,
        result.resolved_question,
        result.sub_questions,
        result.reasoning,
    )
    return result.model_dump()
