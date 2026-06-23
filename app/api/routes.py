"""FastAPI routes for the Deep Research Agent."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.db.repository import TaskRepository, PaperRepository, EvidenceRepository
from app.db.models import TaskRecord, PaperRecord, EvidenceRecord
from app.models.task import TaskState, TaskStatus, TaskMetrics
from app.workflow.graph import run_research

logger = logging.getLogger(__name__)

router = APIRouter()


# ---- Request/Response models ----

class ResearchRequest(BaseModel):
    research_question: Optional[str] = Field(
        default=None, description="The research question to investigate"
    )
    question: Optional[str] = Field(
        default=None, description="Backward-compatible alias for research_question"
    )
    topic: Optional[str] = Field(
        default=None, description="User-facing alias for question"
    )
    year_from: Optional[int] = Field(default=None, description="Start year filter (inclusive)")
    year_to: Optional[int] = Field(default=None, description="End year filter (inclusive)")
    max_papers: int = Field(
        default=12, ge=3, le=30, description="Maximum papers in the final report"
    )
    num_papers: Optional[int] = Field(
        default=None, ge=3, le=30, description="Alias for max_papers"
    )
    research_depth: str = Field(
        default="standard", pattern="^(quick|standard|deep)$"
    )
    retrieval_profile: str = Field(
        default="quality", pattern="^(quality|balanced|local)$"
    )
    evidence_backend: Optional[str] = Field(
        default=None, pattern="^(abstract|fts|paperqa|hybrid)$",
        description="Administrative/testing override; normal clients should omit it",
        exclude=True,
    )
    enable_full_text: Optional[bool] = None
    full_text_required: bool = False
    report_language: str = Field(default="zh-CN", pattern="^(zh-CN|en)$")
    language: Optional[str] = Field(default=None, pattern="^(zh-CN|en)$")
    max_cost_usd: Optional[float] = Field(default=None, ge=0)
    research_mode: Optional[str] = Field(
        default="quick",
        pattern="^(quick|strict)$",
        description="Research mode: 'quick' (Tavily web search) or 'strict' (academic PDF full-text)",
    )

    @model_validator(mode="after")
    def validate_request(self):
        value = (self.research_question or self.question or self.topic or "").strip()
        if not value:
            raise ValueError("research_question or topic is required")
        self.research_question = value
        self.question = value
        if self.num_papers is not None:
            self.max_papers = self.num_papers
        if self.language is not None:
            self.report_language = self.language
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        from app.core.config import get_settings

        profile_backends = {
            "quality": get_settings().QUALITY_EVIDENCE_BACKEND,
            "balanced": "hybrid",
            "local": "hybrid",
        }
        if self.evidence_backend is None:
            self.evidence_backend = profile_backends[self.retrieval_profile]
        if self.enable_full_text is None:
            self.enable_full_text = self.evidence_backend != "abstract"
        if self.evidence_backend != "abstract" and not self.enable_full_text:
            raise ValueError("full-text retrieval requires enable_full_text=true")
        return self


class ResearchResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    original_question: str
    current_round: int
    papers_found: int
    papers_selected: int
    warnings: list[str]
    errors: list[str]
    is_completed: bool
    created_at: str
    updated_at: str
    current_stage: Optional[str] = None
    progress_percent: int = 0
    research_mode: Optional[str] = "quick"
    retrieval_backend: Optional[str] = None
    retrieved_passages: int = 0
    verified_evidence: int = 0
    # Quick mode stats
    web_results: int = 0
    sources_selected: int = 0
    research_notes_count: int = 0
    # Common
    llm_calls: int = 0
    estimated_cost_usd: float = 0.0


class PaperResponse(BaseModel):
    internal_id: str
    title: str
    abstract: Optional[str]
    authors: list[dict]
    publication_year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    citation_count: Optional[int]
    source_names: list[str]
    open_access: bool
    relevance_score: Optional[int]
    include: Optional[bool]
    relevance_reason: Optional[str]
    matched_aspects: list[str]


class ReportResponse(BaseModel):
    task_id: str
    report: str
    citation_validation: Optional[dict]
    evidence_quality: Optional[dict]
    references: list[dict]


# ---- Background task runner ----

# In-memory registry of active tasks (complementing DB persistence)
_active_tasks: dict[str, asyncio.Task] = {}
_runtime_states: dict[str, dict] = {}


async def _run_research_background(state: TaskState) -> None:
    """Run research workflow in the background."""
    try:
        # Convert TaskState to ResearchState dict for LangGraph
        research_state = {
            "task_id": state.task_id,
            "original_question": state.original_question,
            "status": TaskStatus.RUNNING.value,
            "research_mode": state.research_mode,
            "year_from": state.year_from,
            "year_to": state.year_to,
            "max_papers": state.max_papers,
            "research_depth": state.research_depth,
            "retrieval_profile": state.retrieval_profile,
            "evidence_backend": state.evidence_backend,
            "enable_full_text": state.enable_full_text,
            "full_text_required": state.full_text_required,
            "report_language": state.report_language,
            "max_cost_usd": state.max_cost_usd,
            "current_round": 0,
            "max_rounds": state.max_rounds,
            "queries": [],
            "normalized_papers": [],
            "selected_papers": [],
            "evidence": [],
            "warnings": [],
            "errors": [],
            "metrics": TaskMetrics(),
            "supplementary_rounds_done": 0,
            "previous_round_paper_ids": [],
            "new_papers_this_round": 0,
        }
        _runtime_states[state.task_id] = research_state

        final_state = await run_research(research_state)
        _runtime_states[state.task_id] = final_state

        # Persist to database
        from app.db.database import async_session_factory
        async with async_session_factory() as session:
            # Update task status
            state.status = TaskStatus(final_state["status"])
            state.report = final_state.get("report")
            state.warnings = final_state.get("warnings", [])
            state.errors = final_state.get("errors", [])
            state.metrics = final_state.get("metrics", TaskMetrics())
            state.citation_validation = final_state.get("citation_validation")
            state.retrieved_passages = final_state.get("retrieved_passages", [])
            state.claims = final_state.get("claims", [])
            state.evidence_quality = final_state.get("evidence_quality")
            state.report_paper_ids = final_state.get("report_paper_ids", [])

            await TaskRepository.save_state(session, state)

            # Save papers
            all_papers = final_state.get("normalized_papers", [])
            if all_papers:
                await PaperRepository.save_batch(session, state.task_id, all_papers)

            # Save evidence
            evidence = final_state.get("evidence", [])
            if evidence:
                await EvidenceRepository.save_batch(session, state.task_id, evidence)

        logger.info(f"Task {state.task_id} completed")

    except Exception as e:
        logger.error(f"Background task {state.task_id} failed: {e}", exc_info=True)
        from app.db.database import async_session_factory
        try:
            async with async_session_factory() as session:
                await TaskRepository.update_status(session, state.task_id, TaskStatus.FAILED)
        except Exception:
            pass
    finally:
        _active_tasks.pop(state.task_id, None)


# ---- Routes ----

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@router.post("/api/research", response_model=ResearchResponse)
async def create_research(
    request: ResearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a new research task. Runs asynchronously in the background."""
    from app.core.config import get_settings
    settings = get_settings()

    depth_rounds = {"quick": 1, "standard": 2, "deep": 3}
    state = TaskState(
        original_question=request.question or "",
        research_mode=request.research_mode or "quick",
        year_from=request.year_from,
        year_to=request.year_to,
        max_papers=request.max_papers,
        research_depth=request.research_depth,
        retrieval_profile=request.retrieval_profile,
        evidence_backend=request.evidence_backend,
        enable_full_text=request.enable_full_text,
        full_text_required=request.full_text_required,
        report_language=request.report_language,
        max_cost_usd=request.max_cost_usd,
        status=TaskStatus.PENDING,
        max_rounds=min(
            settings.MAX_SEARCH_ROUNDS,
            depth_rounds[request.research_depth],
        ),
    )

    # Persist initial state
    await TaskRepository.create(session, state)

    # Launch background task
    task = asyncio.create_task(_run_research_background(state))
    _active_tasks[state.task_id] = task

    return ResearchResponse(task_id=state.task_id, status="pending")


@router.get("/api/research/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get the current status of a research task."""
    record = await TaskRepository.get(session, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    # Count papers
    papers = await PaperRepository.get_by_task(session, task_id)
    selected = [p for p in papers if p.selected]

    warnings = json.loads(record.warnings_json or "[]")
    errors = json.loads(record.errors_json or "[]")
    from app.workflow.graph import get_runtime_progress

    runtime = get_runtime_progress(task_id) or _runtime_states.get(task_id, {})
    metrics = runtime.get("metrics", TaskMetrics())
    if isinstance(metrics, dict):
        metrics = TaskMetrics.model_validate(metrics)
    evidence = runtime.get("evidence", [])

    effective_status = runtime.get("status", record.status)
    return TaskStatusResponse(
        task_id=record.task_id,
        status=effective_status,
        original_question=record.original_question,
        current_round=runtime.get("current_round", record.current_round),
        papers_found=len(runtime.get("normalized_papers", [])) or len(papers),
        papers_selected=len(runtime.get("selected_papers", [])) or len(selected),
        warnings=runtime.get("warnings", warnings),
        errors=runtime.get("errors", errors),
        is_completed=effective_status in ("completed", "failed", "interrupted"),
        created_at=record.created_at,
        updated_at=record.updated_at,
        current_stage=runtime.get("current_stage"),
        progress_percent=runtime.get("progress_percent", 100 if record.status == "completed" else 0),
        research_mode=runtime.get("research_mode", "quick"),
        retrieval_backend=runtime.get("evidence_backend", record.evidence_backend),
        retrieved_passages=len(runtime.get("retrieved_passages", [])),
        verified_evidence=sum(
            getattr(item.verification_status, "value", item.verification_status)
            == "verified"
            for item in evidence
        ),
        web_results=len(runtime.get("web_search_results", [])),
        sources_selected=len(runtime.get("selected_web_sources", [])) if runtime.get("selected_web_sources") else len(runtime.get("extracted_sources", [])),
        research_notes_count=len(runtime.get("research_notes", [])),
        llm_calls=metrics.llm_call_count,
        estimated_cost_usd=metrics.estimated_cost_usd,
    )


@router.get("/api/research/{task_id}/report", response_model=ReportResponse)
async def get_report(
    task_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get the final research report."""
    record = await TaskRepository.get(session, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    if record.status not in ("completed",):
        raise HTTPException(
            status_code=409,
            detail=f"Task not completed (current status: {record.status})",
        )

    # Get selected papers for reference list
    papers = await PaperRepository.get_by_task(session, task_id, selected_only=True)
    report_paper_ids = json.loads(record.report_paper_ids_json or "[]")
    if report_paper_ids:
        positions = {paper_id: index for index, paper_id in enumerate(report_paper_ids)}
        papers = sorted(
            (paper for paper in papers if paper.internal_id in positions),
            key=lambda paper: positions[paper.internal_id],
        )
    references = []
    for p in papers:
        ref = {
            "internal_id": p.internal_id,
            "title": p.title,
            "authors": json.loads(p.authors_json or "[]"),
            "publication_year": p.publication_year,
            "venue": p.venue,
            "doi": p.doi,
            "url": p.url,
        }
        references.append(ref)

    citation_validation = json.loads(record.citation_validation_json or "null")
    evidence_quality = json.loads(record.evidence_quality_json or "null")

    return ReportResponse(
        task_id=record.task_id,
        report=record.report or "No report generated",
        citation_validation=citation_validation,
        evidence_quality=evidence_quality,
        references=references,
    )


@router.get("/api/research/{task_id}/papers")
async def get_papers(
    task_id: str,
    session: AsyncSession = Depends(get_session),
    selected_only: bool = False,
):
    """Get papers for a research task."""
    record = await TaskRepository.get(session, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    papers = await PaperRepository.get_by_task(session, task_id, selected_only=selected_only)

    result = []
    for p in papers:
        result.append(PaperResponse(
            internal_id=p.internal_id,
            title=p.title,
            abstract=p.abstract,
            authors=json.loads(p.authors_json or "[]"),
            publication_year=p.publication_year,
            venue=p.venue,
            doi=p.doi,
            citation_count=p.citation_count,
            source_names=json.loads(p.source_names_json or "[]"),
            open_access=p.open_access,
            relevance_score=p.relevance_score,
            include=p.include,
            relevance_reason=p.relevance_reason,
            matched_aspects=json.loads(p.matched_aspects_json or "[]"),
        ))

    return {"task_id": task_id, "count": len(result), "papers": result}


@router.get("/api/research/{task_id}/evidence")
async def get_evidence(
    task_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get extracted evidence for a research task."""
    record = await TaskRepository.get(session, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    evidence_records = await EvidenceRepository.get_by_task(session, task_id)

    result = []
    for ev in evidence_records:
        result.append({
            "paper_id": ev.paper_id,
            "evidence_id": ev.evidence_id,
            "passage_id": ev.passage_id,
            "sub_question_id": ev.sub_question_id,
            "research_question": ev.research_question,
            "method": ev.method,
            "dataset_or_participants": ev.dataset_or_participants,
            "key_findings": json.loads(ev.key_findings_json or "[]"),
            "limitations": json.loads(ev.limitations_json or "[]"),
            "relevance_to_user_question": ev.relevance_to_user_question,
            "evidence_quote": ev.evidence_quote,
            "chunk_id": ev.chunk_id,
            "section_title": ev.section_title,
            "page_start": ev.page_start,
            "page_end": ev.page_end,
            "source_url": ev.source_url,
            "evidence_level": ev.evidence_level,
            "stance": ev.stance,
            "evidence_type": ev.evidence_type,
            "verification_status": ev.verification_status,
            "verification_reason": ev.verification_reason,
            "confidence": ev.confidence,
        })

    return {"task_id": task_id, "count": len(result), "evidence": result}


@router.get("/api/research/{task_id}/claims")
async def get_claims(
    task_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get the claim-evidence layer used to compose the report."""
    record = await TaskRepository.get(session, task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    claims = json.loads(record.claims_json or "[]")
    quality = json.loads(record.evidence_quality_json or "null")
    return {
        "task_id": task_id,
        "count": len(claims),
        "claims": claims,
        "evidence_quality": quality,
    }


@router.get("/api/research")
async def list_tasks(session: AsyncSession = Depends(get_session)):
    """List all research tasks."""
    tasks = await TaskRepository.list_all(session)
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status,
                "question": t.original_question[:100],
                "created_at": t.created_at,
            }
            for t in tasks
        ]
    }
