"""Quick Research mode models.

These are independent of the Strict-mode EvidenceCard/Claim/Verification pipeline.
Quick mode uses Tavily Search + Extract as the primary content source and produces
a readable, cited research report without PDF download, FTS5, or strict verification.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---- Research Mode ----

class ResearchMode(str, Enum):
    QUICK = "quick"
    STRICT = "strict"


# ---- Question Classification ----

class QuestionType(str, Enum):
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    TREND = "trend"
    METHODOLOGICAL = "methodological"
    RESEARCH_LANDSCAPE = "research_landscape"


class AnswerSchema(BaseModel):
    """Operationalised research question — guides all downstream search & extraction."""

    question_type: QuestionType = QuestionType.DESCRIPTIVE
    subject: str = ""
    comparison_target: Optional[str] = None
    outcome: Optional[str] = None
    required_dimensions: list[str] = Field(default_factory=list)
    inclusion_guidance: list[str] = Field(default_factory=list)
    exclusion_guidance: list[str] = Field(default_factory=list)

    @classmethod
    def conservative_default(cls, question: str) -> "AnswerSchema":
        return cls(
            question_type=QuestionType.DESCRIPTIVE,
            subject=question,
            required_dimensions=["key_findings", "methods", "limitations"],
            inclusion_guidance=["academic sources", "empirical studies"],
            exclusion_guidance=[],
        )


# ---- Query Planning ----

class PlannedQuery(BaseModel):
    """A search query with its purpose, used for coverage tracking."""

    query_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    query: str
    purpose: str = ""  # e.g. "empirical_comparison", "benchmark", "review"
    round_index: int = 0


# ---- Source Classification ----

class SourceType(str, Enum):
    PAPER_OFFICIAL = "paper_official"
    PUBLISHER = "publisher"
    PREPRINT = "preprint"
    AUTHOR_PROJECT = "author_project"
    INSTITUTIONAL = "institutional"
    REVIEW = "review"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


# Priority order for source selection (lower = higher priority)
SOURCE_TYPE_PRIORITY: dict[SourceType, int] = {
    SourceType.PAPER_OFFICIAL: 0,
    SourceType.PREPRINT: 1,
    SourceType.PUBLISHER: 2,
    SourceType.AUTHOR_PROJECT: 3,
    SourceType.INSTITUTIONAL: 4,
    SourceType.REVIEW: 5,
    SourceType.SECONDARY: 6,
    SourceType.UNKNOWN: 7,
}


# ---- Web Search Result ----

class WebSearchResult(BaseModel):
    """A single result from Tavily Search, normalised and deduplicated."""

    result_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    query: str
    query_purpose: str = ""
    title: str
    url: str
    snippet: Optional[str] = None
    score: Optional[float] = None
    published_date: Optional[str] = None
    domain: str = ""
    round_index: int = 0
    # Multiple queries may have returned the same URL
    query_purposes: list[str] = Field(default_factory=list)


# ---- Extracted Web Source ----

class ExtractedWebSource(BaseModel):
    """Content extracted from a web source via Tavily Extract."""

    source_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    title: str
    url: str
    domain: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    content: str = ""
    extraction_status: str = "pending"  # pending | success | failed | snippet_only
    extracted_at: str = ""
    query_purposes: list[str] = Field(default_factory=list)
    snippet: Optional[str] = None
    snippet_only: bool = False
    content_length: int = 0
    metadata: dict = Field(default_factory=dict)


# ---- Research Note ----

class ResearchNote(BaseModel):
    """Structured research note extracted from a web source.

    All fields are nullable — only populated when the source text genuinely
    contains the information. Never filled by LLM inference.
    """

    note_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    source_id: str
    title: str = ""
    url: str = ""
    year: Optional[int] = None
    source_type: SourceType = SourceType.UNKNOWN
    research_type: Optional[str] = None
    technique: Optional[str] = None
    baseline: Optional[str] = None
    task: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    reported_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    relevant_quotes: list[str] = Field(default_factory=list)
    relevance_summary: str = ""
    confidence: str = "medium"  # high | medium | low
    extraction_failed: bool = False


# ---- Coverage Assessment ----

class CoverageAssessment(BaseModel):
    """Quick-mode coverage check — driven by AnswerSchema, not ValidatedClaims."""

    sufficient: bool = False
    covered_dimensions: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    covered_techniques: list[str] = Field(default_factory=list)
    underrepresented_areas: list[str] = Field(default_factory=list)
    source_count: int = 0
    high_quality_source_count: int = 0
    new_queries: list[PlannedQuery] = Field(default_factory=list)
    reason: str = ""


# ---- Comparison Matrix ----

class ComparisonRow(BaseModel):
    """A single row in the comparison matrix, synthesised from ResearchNotes."""

    technique: str
    baseline: Optional[str] = None
    task_or_domain: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    reported_result: str = ""
    limitations: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    support_count: int = 1
    confidence: str = "medium"
    note: str = ""  # e.g. "domain-specific evidence", "insufficient_comparative_evidence"


# ---- Quick Report Citation Check Result ----

class QuickCitationCheckResult(BaseModel):
    """Result of lightweight citation check for Quick mode report."""

    valid: bool = True
    issues: list[str] = Field(default_factory=list)
    missing_refs: list[str] = Field(default_factory=list)
    unverifiable_numbers: list[str] = Field(default_factory=list)
    orphan_claims: list[str] = Field(default_factory=list)
    revision_needed: bool = False
    revised_report: Optional[str] = None
