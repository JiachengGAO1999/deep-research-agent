"""Task state and LangGraph state model."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.paper import Paper
from app.models.search_plan import SearchPlan
from app.models.evidence import ExtractedEvidence, GapAnalysis


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TaskMetrics(BaseModel):
    """Performance and cost metrics."""

    start_time: Optional[str] = None
    end_time: Optional[str] = None
    # Provider request counts
    provider_requests: Dict[str, int] = Field(default_factory=dict)
    provider_results: Dict[str, int] = Field(default_factory=dict)
    # Paper counts
    raw_paper_count: int = 0
    after_dedup_count: int = 0
    after_selection_count: int = 0
    # LLM usage
    llm_call_count: int = 0
    llm_tokens_used: int = 0
    # Retries
    provider_retries: Dict[str, int] = Field(default_factory=dict)
    provider_failures: Dict[str, int] = Field(default_factory=dict)
    # Stage durations in seconds
    stage_durations: Dict[str, float] = Field(default_factory=dict)


class CitationValidation(BaseModel):
    """Result of citation validation."""

    citations_in_text: List[str] = Field(
        default_factory=list, description="Citation markers found in report text"
    )
    papers_in_reference_list: List[str] = Field(
        default_factory=list, description="Paper IDs in the reference section"
    )
    orphan_citations: List[str] = Field(
        default_factory=list,
        description="Citations in text that don't map to any paper",
    )
    unused_papers: List[str] = Field(
        default_factory=list,
        description="Papers in reference list but never cited in text",
    )
    is_valid: bool = False
    issues: List[str] = Field(default_factory=list)
    fixed: bool = False


class TaskState(BaseModel):
    """Central state object for the LangGraph workflow.

    This is the single state object passed between nodes.
    """

    task_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    original_question: str = ""
    status: TaskStatus = TaskStatus.PENDING

    # Search configuration
    year_from: Optional[int] = None
    year_to: Optional[int] = None

    # Current round tracking
    current_round: int = 0
    max_rounds: int = 3

    # Search plan
    search_plan: Optional[SearchPlan] = None

    # Current queries (for this round)
    queries: List[str] = Field(default_factory=list)

    # Raw results from providers (pre-normalization)
    raw_results: List[Dict] = Field(default_factory=list)

    # Normalized and deduplicated papers
    normalized_papers: List[Paper] = Field(default_factory=list)

    # Selected papers after ranking
    selected_papers: List[Paper] = Field(default_factory=list)

    # Extracted evidence
    evidence: List[ExtractedEvidence] = Field(default_factory=list)

    # Gap analysis
    gap_analysis: Optional[GapAnalysis] = None

    # Supplementary search tracking
    supplementary_rounds_done: int = 0
    previous_round_paper_ids: Set[str] = Field(default_factory=set)
    new_papers_this_round: int = 0

    # Final output
    report: Optional[str] = None

    # Citation validation
    citation_validation: Optional[CitationValidation] = None

    # Errors and warnings
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    # Metrics
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)

    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def is_finished(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
