"""SQLAlchemy ORM models for persistence."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, Text, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    """Persisted research task record."""

    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    original_question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    year_from: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    year_to: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_round: Mapped[int] = mapped_column(Integer, default=0)
    max_rounds: Mapped[int] = mapped_column(Integer, default=3)

    # JSON-serialized fields
    search_plan_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    queries_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gap_analysis_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_passages_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    claims_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_quality_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    report_paper_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    errors_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citation_validation_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Final report
    report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[str] = mapped_column(String(30), default=datetime.utcnow().isoformat)
    updated_at: Mapped[str] = mapped_column(String(30), default=datetime.utcnow().isoformat)

    # Relations
    papers: Mapped[list["PaperRecord"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class PaperRecord(Base):
    """Persisted paper record."""

    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(32), ForeignKey("tasks.task_id"), index=True)
    internal_id: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authors_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publication_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    venue: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    full_text_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citation_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_names_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    open_access: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relevance and selection
    relevance_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    include: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    relevance_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    matched_aspects_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Round info
    search_round: Mapped[int] = mapped_column(Integer, default=1)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamp
    created_at: Mapped[str] = mapped_column(String(30), default=datetime.utcnow().isoformat)

    # Relation
    task: Mapped["TaskRecord"] = relationship(back_populates="papers")


class EvidenceRecord(Base):
    """Persisted evidence extraction with full-text provenance."""

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(32), ForeignKey("tasks.task_id"), index=True)
    paper_id: Mapped[str] = mapped_column(String(20), index=True)
    evidence_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    passage_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sub_question_id: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    research_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dataset_or_participants: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_findings_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    limitations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    relevance_to_user_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full-text provenance (populated when PDF is available)
    chunk_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_level: Mapped[Optional[str]] = mapped_column(String(20), default="paraphrase")
    stance: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    evidence_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    verification_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    verification_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class PDFCacheRecord(Base):
    """Metadata for cached PDF files."""

    __tablename__ = "pdf_cache"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    open_access_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    active_task_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    acquired_at: Mapped[str] = mapped_column(String(30))
    last_accessed_at: Mapped[str] = mapped_column(String(30))


class DocumentChunkRecord(Base):
    """Persisted document chunk with full provenance.

    An FTS5 virtual table is created alongside this for full-text search.
    See app/services/fts_search.py for FTS5 setup.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(20), index=True, unique=True)
    paper_id: Mapped[str] = mapped_column(String(20), index=True)
    task_id: Mapped[str] = mapped_column(String(32), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    section_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    child_chunk_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pdf_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    parser_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    parser_version: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    created_at: Mapped[str] = mapped_column(String(30))
