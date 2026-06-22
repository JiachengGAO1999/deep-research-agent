from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GoldPaper(BaseModel):
    title: str
    doi: Optional[str] = None


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    domain: str
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    expected_concepts: list[str] = Field(default_factory=list)
    must_cover: list[str] = Field(default_factory=list)
    gold_papers: list[GoldPaper] = Field(default_factory=list)
    status: str = "seed"


class CaseScore(BaseModel):
    case_id: str
    paper_recall_at_k: Optional[float] = None
    concept_coverage: float = 0.0
    citation_integrity: bool = False
    selected_paper_count: int = 0
    verified_evidence_count: int = 0


class EvaluationSummary(BaseModel):
    cases: list[CaseScore]
    mean_paper_recall_at_k: Optional[float] = None
    mean_concept_coverage: float = 0.0
    citation_integrity_rate: float = 0.0
