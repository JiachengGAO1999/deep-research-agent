"""Backend-independent evidence engine contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional

from pydantic import BaseModel, Field

from app.models.evidence import ExtractedEvidence, RetrievedPassage
from app.models.paper import Paper


class IngestionResult(BaseModel):
    backend: str
    attempted: int = 0
    ingested: int = 0
    failed_paper_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceEngine(ABC):
    """Stable boundary between orchestration and retrieval implementations."""

    name: str = "base"

    async def is_available(self) -> bool:
        return True

    async def ingest(
        self,
        papers: list[Paper],
        document_paths: Optional[Mapping[str, str]] = None,
        task_id: Optional[str] = None,
    ) -> IngestionResult:
        return IngestionResult(
            backend=self.name,
            attempted=len(papers),
            ingested=0,
        )

    @abstractmethod
    async def retrieve(
        self,
        question: str,
        sub_question: str,
        paper_ids: Optional[list[str]] = None,
        limit: int = 8,
        task_id: Optional[str] = None,
    ) -> list[RetrievedPassage]:
        """Return source passages; never return synthesized prose."""
        ...

    async def extract(
        self,
        sub_question: str,
        passages: list[RetrievedPassage],
    ) -> list[ExtractedEvidence]:
        """Create conservative evidence records directly from retrieved text."""
        from app.services.quote_verification import verify_quote

        evidence: list[ExtractedEvidence] = []
        for passage in passages:
            quote = passage.text.strip()[:800]
            if not quote:
                continue
            verification = verify_quote(quote, passage.text)
            evidence.append(
                ExtractedEvidence(
                    paper_id=passage.paper_id,
                    passage_id=passage.passage_id,
                    chunk_id=passage.chunk_id,
                    research_question=sub_question,
                    relevance_to_user_question=(
                        "Retrieved source passage relevant to the sub-question."
                    ),
                    evidence_quote=quote,
                    key_findings=[quote],
                    section_title=passage.section_title,
                    page_start=passage.page_start,
                    page_end=passage.page_end,
                    source_url=passage.source_url,
                    evidence_level=(
                        "abstract" if passage.is_abstract else "direct_quote"
                    ),
                    evidence_type=(
                        "abstract" if passage.is_abstract else "direct_quote"
                    ),
                    verification_status=verification.status,
                    verification_reason=verification.reason,
                    confidence=0.65 if passage.is_abstract else 0.85,
                )
            )
        return evidence

    async def retrieve_many(
        self,
        question: str,
        subquestions: list[str],
        papers: list[Paper],
        limit: int,
        *,
        task_id: str | None = None,
    ) -> list[RetrievedPassage]:
        """Batch-friendly project contract with stable de-duplication."""
        seen: set[str] = set()
        results: list[RetrievedPassage] = []
        paper_ids = [paper.internal_id for paper in papers]
        for subquestion in subquestions or [question]:
            passages = await self.retrieve(
                question=question,
                sub_question=subquestion,
                paper_ids=paper_ids,
                limit=limit,
                task_id=task_id,
            )
            for passage in passages:
                if passage.passage_id in seen:
                    continue
                seen.add(passage.passage_id)
                results.append(passage)
        return results
