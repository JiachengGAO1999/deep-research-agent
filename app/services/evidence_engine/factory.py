"""Evidence backend factory with explicit, observable fallback behavior."""

from __future__ import annotations

from app.core.config import get_settings
from app.services.evidence_engine.abstract import AbstractEvidenceEngine
from app.services.evidence_engine.base import EvidenceEngine
from app.services.evidence_engine.fts import FTSEvidenceEngine
from app.services.evidence_engine.paperqa import PaperQAEvidenceEngine


def get_evidence_engine(name: str | None = None) -> EvidenceEngine:
    backend = (name or get_settings().EVIDENCE_BACKEND).strip().lower()
    if backend == "abstract":
        return AbstractEvidenceEngine()
    if backend == "fts":
        return FTSEvidenceEngine()
    if backend == "paperqa":
        return PaperQAEvidenceEngine(settings=get_settings())
    raise ValueError(
        f"Unknown EVIDENCE_BACKEND={backend!r}; expected abstract, fts, or paperqa"
    )
