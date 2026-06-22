"""Evidence extraction and gap analysis models."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceStance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUAL = "contextual"
    INCONCLUSIVE = "inconclusive"


class EvidenceType(str, Enum):
    DIRECT_QUOTE = "direct_quote"
    CLOSE_PARAPHRASE = "close_paraphrase"
    INFERENCE = "inference"
    ABSTRACT = "abstract"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


class RetrievedPassage(BaseModel):
    """A retrievable source passage independent of the backend implementation."""

    passage_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    paper_id: str
    chunk_id: Optional[str] = None
    text: str
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_url: Optional[str] = None
    retrieval_method: str = "unknown"
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    parser_name: Optional[str] = None
    document_hash: Optional[str] = None
    is_abstract: bool = False


class ExtractedEvidence(BaseModel):
    """Structured evidence extracted from a single paper."""

    evidence_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    paper_id: str = Field(description="internal_id of the Paper")
    passage_id: Optional[str] = None
    sub_question_id: Optional[str] = None
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

    # Full-text provenance (populated when PDF is available)
    chunk_id: Optional[str] = Field(
        default=None, description="DocumentChunk.chunk_id that this evidence comes from"
    )
    section_title: Optional[str] = Field(default=None)
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    evidence_level: str = Field(
        default="paraphrase",
        description="direct_quote | paraphrase | inferred",
    )
    stance: EvidenceStance = EvidenceStance.SUPPORTS
    evidence_type: EvidenceType = EvidenceType.CLOSE_PARAPHRASE
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_reason: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchClaim(BaseModel):
    """A report claim bound to one or more verified evidence records."""

    claim_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    claim_text: str
    claim_type: str = "factual"
    importance: str = "core"
    evidence_ids: List[str] = Field(default_factory=list)
    paper_ids: List[str] = Field(default_factory=list)
    support_status: str = "unsupported"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    section_id: Optional[str] = None


class EvidenceQualitySummary(BaseModel):
    """Task-level evidence and claim quality statistics."""

    evidence_count: int = 0
    verified_evidence_count: int = 0
    direct_quote_count: int = 0
    abstract_evidence_count: int = 0
    claim_count: int = 0
    supported_claim_count: int = 0
    unsupported_important_claim_count: int = 0
    quote_verification_rate: float = 0.0
    citation_completeness: float = 0.0
    passed: bool = False
    issues: List[str] = Field(default_factory=list)


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
