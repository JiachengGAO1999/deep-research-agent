"""Abstract-only evidence backend used as the stable low-dependency baseline."""

from __future__ import annotations

from typing import Optional

from app.models.evidence import RetrievedPassage
from app.models.paper import Paper
from app.services.evidence_engine.base import EvidenceEngine, IngestionResult


class AbstractEvidenceEngine(EvidenceEngine):
    name = "abstract"

    def __init__(self):
        self._papers: dict[str, Paper] = {}

    async def ingest(self, papers, document_paths=None, task_id=None) -> IngestionResult:
        self._papers = {paper.internal_id: paper for paper in papers}
        ingested = sum(1 for paper in papers if paper.abstract)
        return IngestionResult(
            backend=self.name,
            attempted=len(papers),
            ingested=ingested,
            failed_paper_ids=[
                paper.internal_id for paper in papers if not paper.abstract
            ],
        )

    async def retrieve(
        self,
        question: str,
        sub_question: str,
        paper_ids: Optional[list[str]] = None,
        limit: int = 8,
        task_id: Optional[str] = None,
    ) -> list[RetrievedPassage]:
        allowed = set(paper_ids or self._papers)
        terms = {
            term.lower()
            for term in (question + " " + sub_question).split()
            if len(term) > 3
        }
        scored = []
        for paper_id, paper in self._papers.items():
            if paper_id not in allowed or not paper.abstract:
                continue
            text = f"{paper.title} {paper.abstract}".lower()
            score = sum(1 for term in terms if term in text)
            scored.append((score, paper))
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].relevance_score or 0,
                item[1].citation_count or 0,
            ),
            reverse=True,
        )
        return [
            RetrievedPassage(
                passage_id=f"abstract:{paper.internal_id}",
                paper_id=paper.internal_id,
                chunk_id=None,
                text=paper.abstract or "",
                section_title="Abstract",
                source_url=paper.url,
                retrieval_method=self.name,
                retrieval_score=float(score),
                is_abstract=True,
            )
            for score, paper in scored[:limit]
        ]
