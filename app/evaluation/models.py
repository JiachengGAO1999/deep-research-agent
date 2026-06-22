from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GoldPaper(BaseModel):
    paper_id: Optional[str] = None
    title: str
    doi: Optional[str] = None


class GoldPassage(BaseModel):
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    doi: Optional[str] = None
    text: str
    page_start: Optional[int] = None
    section_title: Optional[str] = None
    relevance: int = Field(default=1, ge=1, le=3)


class GoldClaim(BaseModel):
    text: str
    direction: Optional[str] = None
    value: Optional[str] = None
    paper_ids: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    domain: str
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    expected_concepts: list[str] = Field(default_factory=list)
    must_cover: list[str] = Field(default_factory=list)
    gold_papers: list[GoldPaper] = Field(default_factory=list)
    gold_passages: list[GoldPassage] = Field(default_factory=list)
    gold_claims: list[GoldClaim] = Field(default_factory=list)
    status: str = "seed"


class CaseScore(BaseModel):
    case_id: str
    discovery_recall_at_50: Optional[float] = None
    selected_paper_recall_at_k: Optional[float] = None
    paper_recall_at_k: Optional[float] = None
    passage_recall_at_10: Optional[float] = None
    passage_ndcg_at_10: Optional[float] = None
    evidence_card_validity: float = 0.0
    claim_entailment_precision: Optional[float] = None
    concept_coverage: float = 0.0
    citation_integrity: bool = False
    unsupported_claim_rate: float = 0.0
    selected_paper_count: int = 0
    verified_evidence_count: int = 0
    elapsed_seconds: Optional[float] = None
    estimated_cost_usd: float = 0.0


class EvaluationSummary(BaseModel):
    cases: list[CaseScore]
    mean_discovery_recall_at_50: Optional[float] = None
    mean_selected_paper_recall_at_k: Optional[float] = None
    mean_paper_recall_at_k: Optional[float] = None
    mean_passage_recall_at_10: Optional[float] = None
    mean_passage_ndcg_at_10: Optional[float] = None
    mean_claim_entailment_precision: Optional[float] = None
    mean_concept_coverage: float = 0.0
    citation_integrity_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    total_estimated_cost_usd: float = 0.0
    gates_passed: bool = False
