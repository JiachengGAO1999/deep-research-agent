"""Optional PaperQA2 adapter.

The adapter intentionally uses only the public async Docs methods documented by
PaperQA2 and converts returned contexts into project-owned passage models.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from typing import Mapping, Optional

from app.models.evidence import RetrievedPassage
from app.models.paper import Paper
from app.services.evidence_engine.base import EvidenceEngine, IngestionResult


class PaperQAEvidenceEngine(EvidenceEngine):
    name = "paperqa"

    def __init__(self, settings=None):
        self._settings = settings
        self._docs = None
        self._paper_by_docname: dict[str, str] = {}

    async def is_available(self) -> bool:
        return importlib.util.find_spec("paperqa") is not None

    def _get_docs(self):
        if self._docs is None:
            from paperqa import Docs

            self._docs = Docs()
        return self._docs

    async def ingest(
        self,
        papers: list[Paper],
        document_paths: Optional[Mapping[str, str]] = None,
        task_id: Optional[str] = None,
    ) -> IngestionResult:
        if not await self.is_available():
            return IngestionResult(
                backend=self.name,
                attempted=len(papers),
                failed_paper_ids=[paper.internal_id for paper in papers],
                warnings=[
                    "PaperQA2 is not installed; install the 'paperqa' optional dependency."
                ],
            )
        paths = document_paths or {}
        result = IngestionResult(backend=self.name, attempted=len(papers))
        docs = self._get_docs()
        for paper in papers:
            raw_path = paths.get(paper.internal_id)
            if not raw_path or not Path(raw_path).is_file():
                result.failed_paper_ids.append(paper.internal_id)
                continue
            try:
                await docs.aadd(
                    raw_path,
                    docname=paper.internal_id,
                    title=paper.title,
                    doi=paper.doi,
                    authors=[author.name for author in paper.authors],
                    settings=self._build_settings(),
                )
                self._paper_by_docname[paper.internal_id] = paper.internal_id
                result.ingested += 1
            except Exception as exc:
                result.failed_paper_ids.append(paper.internal_id)
                result.warnings.append(f"{paper.internal_id}: {exc}")
        return result

    async def retrieve(
        self,
        question: str,
        sub_question: str,
        paper_ids: Optional[list[str]] = None,
        limit: int = 8,
        task_id: Optional[str] = None,
    ) -> list[RetrievedPassage]:
        if not await self.is_available() or self._docs is None:
            return []
        query = sub_question or question
        try:
            session = await self._docs.aget_evidence(
                query, settings=self._build_settings()
            )
        except TypeError:
            session = await self._docs.aget_evidence(query)

        contexts = getattr(session, "contexts", None)
        if contexts is None:
            contexts = getattr(session, "context", None)
        if contexts is None and isinstance(session, list):
            contexts = session
        contexts = contexts or []

        passages: list[RetrievedPassage] = []
        allowed = set(paper_ids or [])
        for index, context in enumerate(contexts):
            text_record = getattr(context, "text", None)
            text = getattr(text_record, "text", None) or ""
            if not text:
                continue
            doc = getattr(text_record, "doc", None)
            docname = getattr(doc, "docname", None)
            paper_id = self._paper_by_docname.get(
                docname or "",
                docname or f"paperqa:{index}",
            )
            if allowed and paper_id not in allowed:
                continue
            text_name = getattr(text_record, "name", "") or ""
            page_numbers = [
                int(value)
                for value in re.findall(r"(?:page|pages|p\.)\s*(\d+)", text_name, re.I)
            ]
            page_start = min(page_numbers) if page_numbers else None
            page_end = max(page_numbers) if page_numbers else None
            passages.append(
                RetrievedPassage(
                    passage_id=f"paperqa:{paper_id}:{index}",
                    paper_id=paper_id,
                    chunk_id=getattr(context, "id", None),
                    text=str(text),
                    section_title=getattr(context, "section", None),
                    page_start=page_start,
                    page_end=page_end,
                    source_url=None,
                    retrieval_method=self.name,
                    retrieval_score=getattr(context, "score", None),
                    rerank_score=getattr(context, "score", None),
                    parser_name="paperqa",
                )
            )
            if len(passages) >= limit:
                break
        return passages

    def _build_settings(self):
        from paperqa.settings import get_settings

        overrides = {
            "embedding": self._settings.PAPERQA_EMBEDDING,
        }
        if self._settings.PAPERQA_LLM:
            overrides["llm"] = self._settings.PAPERQA_LLM
        if self._settings.PAPERQA_SUMMARY_LLM:
            overrides["summary_llm"] = self._settings.PAPERQA_SUMMARY_LLM
        api_config = {
            "api_base": self._settings.LLM_BASE_URL,
            "api_key": self._settings.LLM_API_KEY,
        }
        if self._settings.PAPERQA_LLM:
            overrides["llm_config"] = api_config
        if self._settings.PAPERQA_SUMMARY_LLM:
            overrides["summary_llm_config"] = api_config
        base = get_settings(self._settings.PAPERQA_SETTINGS)
        return base.model_copy(update=overrides, deep=True)
