"""Evidence retrieval backends behind a stable project-owned interface."""

from app.services.evidence_engine.base import EvidenceEngine, IngestionResult
from app.services.evidence_engine.factory import get_evidence_engine

__all__ = ["EvidenceEngine", "IngestionResult", "get_evidence_engine"]
