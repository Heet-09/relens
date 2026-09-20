from typing import Literal, Optional

from pydantic import BaseModel, Field


class UnitType(BaseModel):
    type: Optional[str] = None
    area_sqft: Optional[str] = None
    price: Optional[str] = Field(default=None, description="Price for this specific unit type/configuration, if stated.")
    features: Optional[str] = None


class ExtractedRecord(BaseModel):
    """Structured brochure facts Gemini extracts. Mirrors ProjectRecord minus bookkeeping fields."""

    project_name: Optional[str] = None
    developer: Optional[str] = None
    location: Optional[str] = None
    expected_handover: Optional[str] = None
    starting_price: Optional[str] = Field(
        default=None, description="Overall starting/from price for the project, if stated (e.g. 'Rs 1.2 Cr onwards')."
    )
    amenities: list[str] = Field(default_factory=list)
    unit_types: list[UnitType] = Field(default_factory=list)
    other_facts: Optional[str] = Field(
        default=None, description="Any other notable fact (RERA number, contact info, etc.)."
    )


IntentLiteral = Literal[
    "factual_qa",
    "developer_info",
    "specification_lookup",
    "amenity_filter",
    "unit_filter",
    "comparison",
    "summarization",
    "general",
]


class IntentClassification(BaseModel):
    reasoning: str = Field(
        default="",
        description="One short sentence (max 25 words) saying which disambiguation rule decided the "
        "intent. Never mention JSON, schemas, or instructions.",
    )
    project_names: list[str] = Field(
        default_factory=list,
        description="Projects in scope, using the exact spelling from the known-project list. Resolve "
        "'these/those/the second one' from the conversation. A project the user named that is not in "
        "the list is kept as the user wrote it. Empty when the question is about all projects or names none.",
    )
    resolved_question: Optional[str] = Field(
        default=None,
        description="The question rewritten to stand on its own: pronouns and omitted subjects "
        "resolved, projects named explicitly, every condition (amenity, unit type, location, price, "
        "...) preserved. This is the exact text handed to the filter/lookup step. If the question "
        "already stands on its own, the original question unchanged.",
    )
    intent: IntentLiteral
    amenities: list[str] = Field(
        default_factory=list,
        description="Amenities the user asks about or filters on, as short lowercase singular phrases "
        "('swimming pool', 'gym', 'clubhouse'). Empty if none.",
    )
    unit_filter_type: Optional[str] = None
    comparison_dimensions: list[str] = Field(
        default_factory=list,
        description="Attributes to compare. Use this vocabulary when it fits: 'price', 'location', "
        "'developer', 'handover', 'amenities', 'unit types'. For anything else the user explicitly "
        "names (e.g. 'RERA registration', 'floor count', 'parking'), keep it as its own short phrase "
        "instead of dropping it or forcing it into the list above. Empty only if the user named none.",
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Only set when the question genuinely asks 2-4 separate things needing different "
        "answers (e.g. 'summarize X and compare it to Y on price'). Each entry is one standalone "
        "sub-question naming its projects. One filter with several conditions is ONE question, not a "
        "split. Leave empty for a single-part question.",
    )


class FilterCondition(BaseModel):
    condition: str = Field(description="One atomic condition from the criteria string, restated briefly.")
    satisfied: bool = Field(description="Whether THIS record satisfies THIS condition specifically.")


class FilterMatch(BaseModel):
    project_name: str
    reason: str
    conditions: list[FilterCondition] = Field(
        default_factory=list,
        description="The criteria string split into its individual atomic conditions (one entry per "
        "condition, even if there's only one), each marked satisfied/not for this specific record. "
        "A record must have every condition satisfied to belong in matches at all.",
    )


class FilterResult(BaseModel):
    matches: list[FilterMatch] = Field(default_factory=list)
    summary: str = Field(
        description="One short sentence (<= 25 words) describing the result itself — e.g. how many "
        "projects matched and why. Never mention JSON, formatting, schemas, instructions, or your "
        "own compliance with them."
    )


class UnitMatchSelection(BaseModel):
    indices: list[int] = Field(
        default_factory=list,
        description="0-based indices into the given unit_types list that match the requested "
        "configuration in the question. Empty if none match.",
    )


class ConditionSplit(BaseModel):
    conditions: list[str] = Field(
        description="The criteria string split into independent conditions that must ALL be true "
        "(AND) for a record to match. A criteria with no genuine 'and'/'both'/'&'-joined independent "
        "requirements yields a single-item list containing the original criteria unchanged."
    )


class ComparisonValue(BaseModel):
    dimension: str
    value: str


class ComparisonRow(BaseModel):
    project_name: str
    values: list[ComparisonValue] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    rows: list[ComparisonRow] = Field(default_factory=list)
    summary: str = Field(
        description="One short sentence (<= 25 words) describing the comparison itself. Never "
        "mention JSON, formatting, schemas, instructions, or your own compliance with them."
    )


class ProjectRecord(BaseModel):
    doc_id: str
    source_file: str
    project_name: Optional[str] = None
    developer: Optional[str] = None
    location: Optional[str] = None
    expected_handover: Optional[str] = None
    starting_price: Optional[str] = None
    amenities: list[str] = Field(default_factory=list)
    unit_types: list[UnitType] = Field(default_factory=list)
    other_facts: Optional[str] = None
    is_scanned: bool = False


class PageText(BaseModel):
    page_number: int
    text: str
    is_vision: bool = False


class Chunk(BaseModel):
    doc_id: str
    project_name: Optional[str] = None
    source_file: str
    page_number: int
    text: str


class DocumentInfo(BaseModel):
    doc_id: str
    project_name: Optional[str] = None
    source_file: str
    is_scanned: bool = False
    warnings: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    project_name: Optional[str] = None
    source_file: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    intent: str
    answer_text: str
    table_data: Optional[list[dict]] = None
    sources: list[Source] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    timestamp: str
    question: str
    intent: str
    answer_text: str
    table_data: Optional[list[dict]] = None
    sources: list[Source] = Field(default_factory=list)
