"""FastAPI routes for the Deep Research Agent."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
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
    question: str = Field(description="The research question to investigate")
    year_from: Optional[int] = Field(default=None, description="Start year filter (inclusive)")
    year_to: Optional[int] = Field(default=None, description="End year filter (inclusive)")


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


async def _run_research_background(state: TaskState) -> None:
    """Run research workflow in the background."""
    try:
        # Convert TaskState to ResearchState dict for LangGraph
        research_state = {
            "task_id": state.task_id,
            "original_question": state.original_question,
            "status": TaskStatus.RUNNING.value,
            "year_from": state.year_from,
            "year_to": state.year_to,
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

        final_state = await run_research(research_state)

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

    state = TaskState(
        original_question=request.question,
        year_from=request.year_from,
        year_to=request.year_to,
        status=TaskStatus.PENDING,
        max_rounds=settings.MAX_SEARCH_ROUNDS,
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

    return TaskStatusResponse(
        task_id=record.task_id,
        status=record.status,
        original_question=record.original_question,
        current_round=record.current_round,
        papers_found=len(papers),
        papers_selected=len(selected),
        warnings=warnings,
        errors=errors,
        is_completed=record.status in ("completed", "failed", "interrupted"),
        created_at=record.created_at,
        updated_at=record.updated_at,
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
