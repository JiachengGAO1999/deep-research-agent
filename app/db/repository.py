"""Repository layer — CRUD operations for tasks, papers, and evidence."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TaskRecord, PaperRecord, EvidenceRecord
from app.models.paper import Paper, PaperSource, AuthorInfo
from app.models.task import TaskState, TaskStatus, TaskMetrics, CitationValidation
from app.models.search_plan import SearchPlan, SearchQuery, InclusionExclusionCriteria
from app.models.evidence import ExtractedEvidence, GapAnalysis, EvidenceGap


def _json_dump(obj) -> str:
    """Serialize an object to JSON string."""
    if obj is None:
        return "[]" if isinstance(obj, list) else "{}"
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False, default=str)


class TaskRepository:
    """CRUD operations for research tasks."""

    @staticmethod
    async def create(session: AsyncSession, state: TaskState) -> TaskRecord:
        """Create a new task record from state."""
        record = TaskRecord(
            task_id=state.task_id,
            original_question=state.original_question,
            status=state.status.value,
            year_from=state.year_from,
            year_to=state.year_to,
            current_round=state.current_round,
            max_rounds=state.max_rounds,
            search_plan_json=_json_dump(state.search_plan),
            queries_json=_json_dump(state.queries),
            warnings_json=_json_dump(state.warnings),
            errors_json=_json_dump(state.errors),
            metrics_json=_json_dump(state.metrics),
            created_at=state.created_at,
            updated_at=state.updated_at,
        )
        session.add(record)
        await session.commit()
        return record

    @staticmethod
    async def get(session: AsyncSession, task_id: str) -> Optional[TaskRecord]:
        """Get a task by ID."""
        result = await session.execute(
            select(TaskRecord).where(TaskRecord.task_id == task_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_status(
        session: AsyncSession, task_id: str, status: TaskStatus
    ) -> None:
        """Update task status."""
        await session.execute(
            update(TaskRecord)
            .where(TaskRecord.task_id == task_id)
            .values(status=status.value, updated_at=TaskState.model_fields["updated_at"].default)
        )
        await session.commit()

    @staticmethod
    async def save_state(session: AsyncSession, state: TaskState) -> None:
        """Persist the full task state."""
        import datetime

        values = {
            "status": state.status.value,
            "current_round": state.current_round,
            "search_plan_json": _json_dump(state.search_plan),
            "queries_json": _json_dump(state.queries),
            "gap_analysis_json": _json_dump(state.gap_analysis),
            "evidence_json": _json_dump(state.evidence),
            "retrieved_passages_json": _json_dump(state.retrieved_passages),
            "claims_json": _json_dump(state.claims),
            "evidence_quality_json": _json_dump(state.evidence_quality),
            "report_paper_ids_json": _json_dump(state.report_paper_ids),
            "warnings_json": _json_dump(state.warnings),
            "errors_json": _json_dump(state.errors),
            "metrics_json": _json_dump(state.metrics),
            "citation_validation_json": _json_dump(state.citation_validation),
            "report": state.report,
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }
        await session.execute(
            update(TaskRecord)
            .where(TaskRecord.task_id == state.task_id)
            .values(**values)
        )
        await session.commit()

    @staticmethod
    async def list_all(session: AsyncSession) -> list[TaskRecord]:
        """List all tasks."""
        result = await session.execute(select(TaskRecord))
        return list(result.scalars().all())


class PaperRepository:
    """CRUD operations for papers."""

    @staticmethod
    async def save_batch(
        session: AsyncSession, task_id: str, papers: list[Paper]
    ) -> list[PaperRecord]:
        """Save multiple papers for a task."""
        records = []
        for paper in papers:
            record = PaperRecord(
                task_id=task_id,
                internal_id=paper.internal_id,
                title=paper.title,
                normalized_title=paper.normalized_title,
                abstract=paper.abstract,
                authors_json=_json_dump(
                    [a.model_dump() if hasattr(a, "model_dump") else a for a in paper.authors]
                ),
                publication_year=paper.publication_year,
                venue=paper.venue,
                doi=paper.doi,
                url=paper.url,
                full_text_url=paper.full_text_url,
                citation_count=paper.citation_count,
                source_names_json=_json_dump(paper.source_names),
                source_ids_json=_json_dump(
                    [s.model_dump() if hasattr(s, "model_dump") else s for s in paper.source_ids]
                ),
                open_access=paper.open_access,
                relevance_score=paper.relevance_score,
                include=paper.include,
                relevance_reason=paper.relevance_reason,
                matched_aspects_json=_json_dump(paper.matched_aspects),
                search_round=paper.search_round,
                selected=(paper.include is True),
                created_at=paper.created_at,
            )
            records.append(record)
        session.add_all(records)
        await session.commit()
        return records

    @staticmethod
    async def get_by_task(
        session: AsyncSession, task_id: str, selected_only: bool = False
    ) -> list[PaperRecord]:
        """Get all papers for a task."""
        stmt = select(PaperRecord).where(PaperRecord.task_id == task_id)
        if selected_only:
            stmt = stmt.where(PaperRecord.selected == True)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_by_internal_id(
        session: AsyncSession, task_id: str, internal_id: str
    ) -> Optional[PaperRecord]:
        """Get a specific paper by internal_id."""
        result = await session.execute(
            select(PaperRecord).where(
                PaperRecord.task_id == task_id,
                PaperRecord.internal_id == internal_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_selection(
        session: AsyncSession,
        task_id: str,
        paper_ids: list[str],
        scores: dict[str, int],
        reasons: dict[str, str],
    ) -> None:
        """Update selection status for papers."""
        for pid in paper_ids:
            await session.execute(
                update(PaperRecord)
                .where(
                    PaperRecord.task_id == task_id,
                    PaperRecord.internal_id == pid,
                )
                .values(
                    selected=True,
                    relevance_score=scores.get(pid),
                    relevance_reason=reasons.get(pid),
                )
            )
        await session.commit()


class EvidenceRepository:
    """CRUD operations for evidence records."""

    @staticmethod
    async def save_batch(
        session: AsyncSession, task_id: str, evidence_list: list[ExtractedEvidence]
    ) -> list[EvidenceRecord]:
        """Save multiple evidence records."""
        records = []
        for ev in evidence_list:
            record = EvidenceRecord(
                task_id=task_id,
                paper_id=ev.paper_id,
                evidence_id=ev.evidence_id,
                passage_id=ev.passage_id,
                sub_question_id=ev.sub_question_id,
                research_question=ev.research_question,
                method=ev.method,
                dataset_or_participants=ev.dataset_or_participants,
                key_findings_json=_json_dump(ev.key_findings),
                limitations_json=_json_dump(ev.limitations),
                relevance_to_user_question=ev.relevance_to_user_question,
                evidence_quote=ev.evidence_quote,
                chunk_id=ev.chunk_id,
                section_title=ev.section_title,
                page_start=ev.page_start,
                page_end=ev.page_end,
                source_url=ev.source_url,
                evidence_level=ev.evidence_level,
                stance=ev.stance.value,
                evidence_type=ev.evidence_type.value,
                verification_status=ev.verification_status.value,
                verification_reason=ev.verification_reason,
                confidence=ev.confidence,
            )
            records.append(record)
        session.add_all(records)
        await session.commit()
        return records

    @staticmethod
    async def get_by_task(
        session: AsyncSession, task_id: str
    ) -> list[EvidenceRecord]:
        """Get all evidence for a task."""
        result = await session.execute(
            select(EvidenceRecord).where(EvidenceRecord.task_id == task_id)
        )
        return list(result.scalars().all())
