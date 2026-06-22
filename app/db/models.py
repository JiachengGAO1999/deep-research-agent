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
    """Persisted evidence extraction."""

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(32), ForeignKey("tasks.task_id"), index=True)
    paper_id: Mapped[str] = mapped_column(String(20), index=True)
    research_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dataset_or_participants: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_findings_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    limitations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    relevance_to_user_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
