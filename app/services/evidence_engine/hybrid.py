"""Hybrid passage retrieval: FTS5 + dense + RRF + CrossEncoder."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from collections import defaultdict
from typing import Mapping, Optional

from app.models.evidence import RetrievedPassage
from app.models.paper import Paper
from app.services.evidence_engine.base import EvidenceEngine, IngestionResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridEvidenceEngine(EvidenceEngine):
    name = "hybrid"

    def __init__(self, settings=None):
        from app.core.config import get_settings

        self._settings = settings or get_settings()
        self._embedder = None
        self._reranker = None

    async def ingest(
        self,
        papers: list[Paper],
        document_paths: Optional[Mapping[str, str]] = None,
        task_id: Optional[str] = None,
    ) -> IngestionResult:
        return IngestionResult(
            backend=self.name,
            attempted=len(papers),
            ingested=len(document_paths or {}),
        )

    def _load_models(self):
        if not importlib.util.find_spec("sentence_transformers"):
            return None, None
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if self._embedder is None:
            self._embedder = SentenceTransformer(self._settings.HYBRID_DENSE_MODEL)
        if self._reranker is None:
            self._reranker = CrossEncoder(self._settings.HYBRID_RERANK_MODEL)
        return self._embedder, self._reranker

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
        from app.services.fts_search import list_chunks, search_by_keywords
        from app.services.query_builder import _extract_terms

        query = (sub_question or question).strip()
        candidate_limit = max(
            limit * self._settings.HYBRID_CANDIDATE_MULTIPLIER, limit
        )
        keywords = _extract_terms(query, min_len=3) or _extract_terms(
            question, min_len=3
        )
        lexical_rows = await search_by_keywords(
            task_id=task_id,
            keywords=keywords,
            paper_ids=paper_ids,
            limit=candidate_limit,
        )
        chunks = await list_chunks(task_id, paper_ids=paper_ids)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        lexical_ids = [row.chunk_id for row in lexical_rows]

        dense_ids: list[str] = []
        dense_scores: dict[str, float] = {}
        embedder, reranker = await asyncio.to_thread(self._load_models)
        if embedder and chunks:
            texts = [chunk.text for chunk in chunks]
            query_vector = await asyncio.to_thread(
                embedder.encode, query, normalize_embeddings=True
            )
            vectors = await asyncio.to_thread(
                embedder.encode, texts, normalize_embeddings=True
            )
            similarities = vectors @ query_vector
            ranked_indices = similarities.argsort()[::-1][:candidate_limit]
            dense_ids = [chunks[int(i)].chunk_id for i in ranked_indices]
            dense_scores = {
                chunks[int(i)].chunk_id: float(similarities[int(i)])
                for i in ranked_indices
            }
        else:
            logger.warning(
                "sentence-transformers unavailable; hybrid uses lexical retrieval only"
            )

        fused = reciprocal_rank_fusion(
            [ids for ids in (lexical_ids, dense_ids) if ids],
            self._settings.HYBRID_RRF_K,
        )
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:candidate_limit]
        rerank_scores: dict[str, float] = {}
        if reranker and candidate_ids:
            valid_ids = [cid for cid in candidate_ids if cid in by_id]
            predictions = await asyncio.to_thread(
                reranker.predict,
                [(query, by_id[cid].text) for cid in valid_ids],
            )
            rerank_scores = {
                cid: float(score) for cid, score in zip(valid_ids, predictions)
            }
            candidate_ids.sort(
                key=lambda cid: rerank_scores.get(cid, float("-inf")),
                reverse=True,
            )

        per_paper: dict[str, int] = defaultdict(int)
        passages: list[RetrievedPassage] = []
        for chunk_id in candidate_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            if per_paper[chunk.paper_id] >= self._settings.EVIDENCE_MAX_PER_PAPER:
                continue
            per_paper[chunk.paper_id] += 1
            passages.append(
                RetrievedPassage(
                    passage_id=f"hybrid:{chunk.chunk_id}",
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    section_title=chunk.section_title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_url=chunk.source_url,
                    retrieval_method=self.name,
                    retrieval_score=dense_scores.get(chunk_id, fused.get(chunk_id)),
                    rerank_score=rerank_scores.get(chunk_id),
                    parser_name=chunk.parser_name,
                    document_hash=chunk.pdf_sha256,
                )
            )
            if len(passages) >= limit:
                break
        return passages
