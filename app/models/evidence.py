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
    """Structured evidence extracted from a single paper.

    Structured numeric fields (metric, value, unit, direction, comparison, scope)
    enable deterministic validation. All MUST be null when not directly supported
    by the source chunk — never filled by LLM inference.
    """

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
        description="Verbatim quote from source chunk. MUST be null if no abstract/chunk available.",
    )

    # Normalized statement — must NOT express stronger conclusions than exact_quote
    normalized_statement: Optional[str] = Field(
        default=None,
        description="Conservative restatement; never stronger than the source.",
    )

    # Full-text provenance (copied from structured data, never LLM-generated)
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

    # ---- Structured numeric extraction (all nullable) ----
    subject: Optional[str] = Field(
        default=None, description="What is being measured or claimed about"
    )
    metric: Optional[str] = Field(
        default=None, description="Name of the metric (e.g. accuracy, F1, consistency)"
    )
    value: Optional[str] = Field(
        default=None, description="The numeric or categorical value as it appears in source"
    )
    unit: Optional[str] = Field(
        default=None, description="Unit if any (e.g. %, points, ms)"
    )
    direction: Optional[str] = Field(
        default=None, description="increase | decrease | positive | negative | null"
    )
    comparison: Optional[str] = Field(
        default=None, description="What is being compared against (baseline, prior work, etc.)"
    )
    scope: Optional[str] = Field(
        default=None, description="Domain, task, dataset, or population scope"
    )
    sample_or_dataset: Optional[str] = Field(
        default=None, description="Sample size or dataset name"
    )

    # ---- Categorical classification ----
    evidence_type: EvidenceType = EvidenceType.CLOSE_PARAPHRASE
    paper_role: str = Field(
        default="unknown",
        description=(
            "direct_evidence | mitigation_method | benchmark | conceptual_framework | "
            "personalization_study | safety_study | adjacent_application | unknown"
        ),
    )
    is_inference: bool = Field(
        default=False,
        description="True if this evidence was inferred rather than directly extracted",
    )

    # ---- Verification ----
    stance: EvidenceStance = EvidenceStance.SUPPORTS
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_reason: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ResearchClaim(BaseModel):
    """A report claim bound to one or more verified evidence records.

    validation_status:
    - validated: passed deterministic + LLM checks
    - rejected: failed one or more checks
    - needs_review: borderline, requires human judgment
    """

    claim_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    claim_text: str
    claim_type: str = Field(
        default="factual",
        description="empirical_result | method_description | theoretical_claim | limitation | "
        "benchmark_result | research_question | background | other",
    )
    importance: str = "core"
    evidence_ids: List[str] = Field(default_factory=list)
    paper_ids: List[str] = Field(default_factory=list)
    support_status: str = "unsupported"
    paper_role: str = Field(
        default="unknown",
        description="Same enum as ExtractedEvidence.paper_role",
    )
    # Structured fields matching EvidenceCard
    subject: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    direction: Optional[str] = None
    comparison: Optional[str] = None
    scope: Optional[str] = None
    is_inference: bool = False
    validation_status: str = Field(
        default="unvalidated",
        description="validated | rejected | needs_review | unvalidated",
    )
    validation_reasons: List[str] = Field(default_factory=list)
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


class SentenceAudit(BaseModel):
    """Per-sentence audit trail for factual claims in the final report."""

    sentence_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    text: str
    claim_ids: List[str] = Field(default_factory=list)
    citation_paper_ids: List[str] = Field(default_factory=list)
    audit_status: str = Field(
        default="unchecked",
        description="passed | failed | unchecked",
    )
    audit_reasons: List[str] = Field(default_factory=list)


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
