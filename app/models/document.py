"""Document chunk and PDF metadata models for full-text evidence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ParseStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceLevel(str, Enum):
    DIRECT_QUOTE = "direct_quote"  # Verbatim from source
    PARAPHRASE = "paraphrase"  # Close paraphrase with source
    INFERRED = "inferred"  # Inferred from multiple sources


class DocumentChunk(BaseModel):
    """A chunk of text from a parsed document with full provenance."""

    chunk_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    paper_id: str  # references Paper.internal_id
    task_id: str  # references TaskState.task_id

    # Position in document
    chunk_index: int = 0
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Content
    text: str = ""

    # Parent-child relationship
    parent_chunk_id: Optional[str] = None  # None = parent; set = child
    child_chunk_ids: List[str] = Field(default_factory=list)

    # Source provenance
    source_url: Optional[str] = None
    pdf_sha256: Optional[str] = None

    # Parser metadata
    parser_name: Optional[str] = None  # "docling" or "pymupdf"
    parser_version: Optional[str] = None

    # FTS5 search score (populated at query time)
    fts_score: Optional[float] = None

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PDFCacheEntry(BaseModel):
    """Metadata for a cached PDF file."""

    sha256: str  # Primary key — content-addressed
    source_url: str
    file_path: str  # Relative to PDF_CACHE_DIR
    file_size_bytes: int = 0
    content_type: Optional[str] = None  # e.g. "application/pdf"

    # Access metadata
    open_access_status: Optional[str] = None  # "gold", "green", "hybrid", "bronze", "closed"
    license_info: Optional[str] = None

    # Lifecycle
    acquired_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ttl_days: int = 7

    # References — prevent deletion while active
    active_task_ids: List[str] = Field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        try:
            acquired = datetime.fromisoformat(self.acquired_at)
            age = (datetime.utcnow() - acquired).days
            return age > self.ttl_days and not self.active_task_ids
        except Exception:
            return False


class ParseResult(BaseModel):
    """Result of parsing a single PDF document."""

    paper_id: str
    pdf_sha256: str
    parser_name: str  # "docling" or "pymupdf"
    parser_version: str
    status: ParseStatus = ParseStatus.COMPLETED
    error_message: Optional[str] = None

    # Document-level metadata extracted by parser
    doc_title: Optional[str] = None
    doc_authors: List[str] = Field(default_factory=list)
    num_pages: int = 0
    num_sections: int = 0

    # Chunks produced
    parent_chunks: List[DocumentChunk] = Field(default_factory=list)
    child_chunks: List[DocumentChunk] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
