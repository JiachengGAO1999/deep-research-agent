"""Evidence extraction and gap analysis models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedEvidence(BaseModel):
    """Structured evidence extracted from a single paper."""

    paper_id: str = Field(description="internal_id of the Paper")
    research_question: Optional[str] = Field(
        default=None, description="The research question addressed by this paper"
    )
    method: Optional[str] = Field(
        default=None, description="Methodology used in the paper"
    )
    dataset_or_participants: Optional[str] = Field(
        default=None, description="Dataset, corpus, or participant details"
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings or results"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Limitations noted by authors or detected"
    )
    relevance_to_user_question: Optional[str] = Field(
        default=None, description="How this paper relates to the user's question"
    )
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote from the abstract. MUST be null if no abstract is available.",
    )


class EvidenceGap(BaseModel):
    """A gap in the current evidence."""

    sub_question: str = Field(description="The sub-question with insufficient evidence")
    current_coverage: str = Field(description="What evidence we currently have")
    what_is_missing: str = Field(description="What's still needed")
    severity: str = Field(
        default="medium",
        description="How severe is this gap: low, medium, high",
    )


class GapAnalysis(BaseModel):
    """LLM's analysis of evidence gaps."""

    covered_aspects: List[str] = Field(
        default_factory=list,
        description="Sub-questions with adequate evidence support",
    )
    gaps: List[EvidenceGap] = Field(
        default_factory=list,
        description="Identified evidence gaps",
    )
    needs_supplementary_search: bool = Field(
        default=False,
        description="Whether another search round is recommended",
    )
    supplementary_queries: List[str] = Field(
        default_factory=list,
        description="New search queries for the supplementary round",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Overall reasoning for the gap analysis decision",
    )
