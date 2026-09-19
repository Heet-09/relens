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


IntentLiteral = Literal["factual_qa", "amenity_filter", "unit_filter", "comparison", "summarization"]


class IntentClassification(BaseModel):
    intent: IntentLiteral
    project_names: list[str] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)
    unit_filter_type: Optional[str] = None
    comparison_dimensions: list[str] = Field(default_factory=list)


class FilterMatch(BaseModel):
    project_name: str
    reason: str


class FilterResult(BaseModel):
    matches: list[FilterMatch] = Field(default_factory=list)
    summary: str


class ComparisonValue(BaseModel):
    dimension: str
    value: str


class ComparisonRow(BaseModel):
    project_name: str
    values: list[ComparisonValue] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    rows: list[ComparisonRow] = Field(default_factory=list)
    summary: str


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
