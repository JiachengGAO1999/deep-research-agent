"""SQLite FTS5 evidence backend — multi-query OR retrieval with per-paper coverage."""

from __future__ import annotations

import logging
from typing import List, Mapping, Optional

from app.models.evidence import RetrievedPassage
from app.models.paper import Paper
from app.services.evidence_engine.base import EvidenceEngine, IngestionResult

logger = logging.getLogger(__name__)


class FTSEvidenceEngine(EvidenceEngine):
    name = "fts"

    async def ingest(
        self,
        papers: list[Paper],
        document_paths: Optional[Mapping[str, str]] = None,
        task_id: Optional[str] = None,
    ) -> IngestionResult:
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

        from app.services.fts_search import search_by_keywords
        from app.services.query_builder import _extract_terms

        # Build keyword list from sub_question: extract meaningful terms
        keywords = _extract_terms(sub_question, min_len=3)
        if not keywords:
            keywords = _extract_terms(question, min_len=3)
        if not keywords:
            return []

        # Multi-word key phrases from sub_question (preserve quoted boundaries)
        import re
        phrases = re.findall(r'"([^"]+)"', sub_question)
        for p in phrases[:3]:
            keywords.append(p)

        # Per-paper retrieval to ensure coverage
        all_passages: list[RetrievedPassage] = []
        seen_ids: set[str] = set()

        if paper_ids:
            for paper_id in paper_ids:
                rows = await search_by_keywords(
                    task_id=task_id,
                    keywords=keywords,
                    paper_ids=[paper_id],
                    limit=max(2, limit // max(len(paper_ids), 1)),
                )
                for row in rows:
                    if row.chunk_id in seen_ids:
                        continue
                    seen_ids.add(row.chunk_id)
                    all_passages.append(
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
                    )
        else:
            rows = await search_by_keywords(
                task_id=task_id,
                keywords=keywords,
                paper_ids=None,
                limit=limit,
            )
            for row in rows:
                if row.chunk_id in seen_ids:
                    continue
                seen_ids.add(row.chunk_id)
                all_passages.append(
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
                )

        logger.info(
            f"FTS retrieve: {len(keywords)} keywords, {len(paper_ids or [])} papers "
            f"→ {len(all_passages)} passages"
        )
        return all_passages
