"""Project-owned retrieval contract."""

from __future__ import annotations

from typing import Protocol

from app.models.evidence import RetrievedPassage
from app.models.paper import Paper

CandidatePassage = RetrievedPassage


class EvidenceRetriever(Protocol):
    async def retrieve(
        self,
        question: str,
        subquestions: list[str],
        papers: list[Paper],
        limit: int,
        *,
        task_id: str | None = None,
    ) -> list[CandidatePassage]:
        """Return source passages only; never synthesized claims or prose."""
