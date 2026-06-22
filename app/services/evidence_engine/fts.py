"""SQLite FTS5 evidence backend retained as an experimental adapter."""

from __future__ import annotations

from typing import Mapping, Optional

from app.models.evidence import RetrievedPassage
from app.models.paper import Paper
from app.services.evidence_engine.base import EvidenceEngine, IngestionResult


class FTSEvidenceEngine(EvidenceEngine):
    name = "fts"

    async def ingest(
        self,
        papers: list[Paper],
        document_paths: Optional[Mapping[str, str]] = None,
        task_id: Optional[str] = None,
    ) -> IngestionResult:
        # Parsing and chunk persistence remain a separate acquisition concern.
        paths = document_paths or {}
        return IngestionResult(
            backend=self.name,
            attempted=len(papers),
            ingested=sum(1 for paper in papers if paper.internal_id in paths),
            failed_paper_ids=[
                paper.internal_id
                for paper in papers
                if paper.internal_id not in paths
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
        if not task_id:
            return []
        from app.services.fts_search import search_chunks

        rows = await search_chunks(
            task_id=task_id,
            query=sub_question or question,
            paper_ids=paper_ids,
            limit=limit,
        )
        return [
            RetrievedPassage(
                passage_id=f"fts:{row.chunk_id}",
                paper_id=row.paper_id,
                chunk_id=row.chunk_id,
                text=row.text,
                section_title=row.section_title,
                page_start=row.page_start,
                page_end=row.page_end,
                source_url=row.source_url,
                retrieval_method=self.name,
                retrieval_score=row.fts_score,
                parser_name=row.parser_name,
                document_hash=row.document_hash,
            )
            for row in rows
        ]
