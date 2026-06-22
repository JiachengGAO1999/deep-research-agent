"""SearchIntent — the provider-neutral query model compiled per-provider."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryFamily(BaseModel):
    """A family of related queries targeting the same sub-question from different angles."""

    sub_question: str = ""
    broad: List[str] = Field(default_factory=list)  # High-recall queries
    narrow: List[str] = Field(default_factory=list)  # High-precision (exact phrase, field-specific)
    synonyms: List[str] = Field(default_factory=list)  # Terminology variants / aliases


class SearchIntent(BaseModel):
    """Provider-neutral description of what to search for.

    Each provider's query compiler converts this into provider-specific syntax.
    """

    sub_questions: List[str] = Field(default_factory=list)
    core_concepts: List[str] = Field(default_factory=list)
    aliases: Dict[str, List[str]] = Field(default_factory=dict)  # concept → synonyms
    must_include: List[str] = Field(default_factory=list)
    optional_concepts: List[str] = Field(default_factory=list)
    exclude_concepts: List[str] = Field(default_factory=list)
    year_constraint: Optional[dict] = None  # e.g. {"from": 2020, "to": 2026}
    venue_or_domain_hints: List[str] = Field(default_factory=list)  # e.g. ["cs.CL", "cs.AI"]
    query_families: List[QueryFamily] = Field(default_factory=list)

    def all_query_strings(self) -> List[str]:
        """Flatten to a deduplicated, ordered list."""
        seen = set()
        out = []
        for qf in self.query_families:
            for q in qf.broad + qf.narrow + qf.synonyms:
                if q and q not in seen:
                    seen.add(q)
                    out.append(q)
        return out
