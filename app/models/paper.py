"""Unified Paper data model used across all providers."""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def normalize_title(title: str) -> str:
    """Normalize a paper title for dedup comparison.

    Lowercases, strips punctuation, collapses whitespace.
    """
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class AuthorInfo(BaseModel):
    """Author with optional affiliation and ORCID."""

    name: str
    affiliation: Optional[str] = None
    orcid: Optional[str] = None


class PaperSource(BaseModel):
    """Record of which provider returned this paper."""

    provider: str  # e.g. "openalex", "semantic_scholar", "arxiv", "crossref"
    provider_id: str  # the ID used by that provider
    raw_score: Optional[float] = None  # relevance / similarity score from provider


class Paper(BaseModel):
    """Normalized paper record — the single model used across the system."""

    # Internal ID
    internal_id: str = Field(default_factory=lambda: uuid4().hex[:12])

    # Core bibliographic fields
    title: str
    abstract: Optional[str] = None
    authors: List[AuthorInfo] = Field(default_factory=list)
    publication_year: Optional[int] = None
    venue: Optional[str] = None  # journal / conference name
    doi: Optional[str] = None

    # URLs
    url: Optional[str] = None
    full_text_url: Optional[str] = None

    # Metrics
    citation_count: Optional[int] = None

    # References (optional — DOI or title strings)
    referenced_works: List[str] = Field(default_factory=list)

    # Provenance
    source_names: List[str] = Field(default_factory=list)  # which providers
    source_ids: List[PaperSource] = Field(default_factory=list)

    # Access
    open_access: bool = False

    # Normalized title for dedup
    normalized_title: Optional[str] = None

    # Relevance (populated during ranking)
    relevance_score: Optional[int] = None  # 0-100
    include: Optional[bool] = None
    relevance_reason: Optional[str] = None
    matched_aspects: List[str] = Field(default_factory=list)

    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    # Which search round this paper was found in
    search_round: int = 1

    def model_post_init(self, __context) -> None:
        if self.normalized_title is None:
            self.normalized_title = normalize_title(self.title)
        if self.source_names and not self.source_ids:
            # Backward compat: ensure source_ids populated
            pass

    def merge_from(self, other: "Paper") -> None:
        """Merge richer fields from another record of the same paper."""
        if not self.abstract and other.abstract:
            self.abstract = other.abstract
        if not self.doi and other.doi:
            self.doi = other.doi
        if not self.publication_year and other.publication_year:
            self.publication_year = other.publication_year
        if not self.venue and other.venue:
            self.venue = other.venue
        if not self.citation_count and other.citation_count:
            self.citation_count = other.citation_count
        if not self.url and other.url:
            self.url = other.url
        if not self.full_text_url and other.full_text_url:
            self.full_text_url = other.full_text_url
        if not self.open_access and other.open_access:
            self.open_access = other.open_access
        if other.authors and len(other.authors) > len(self.authors):
            self.authors = other.authors
        # Merge source info
        for sn in other.source_names:
            if sn not in self.source_names:
                self.source_names.append(sn)
        for si in other.source_ids:
            if si.provider not in [s.provider for s in self.source_ids]:
                self.source_ids.append(si)
        if other.referenced_works:
            existing = set(self.referenced_works)
            for rw in other.referenced_works:
                if rw not in existing:
                    self.referenced_works.append(rw)
