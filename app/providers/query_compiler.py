"""Compile provider-neutral SearchIntent into provider-specific query syntax.

OpenAlex:  search= param + structured filters (publication_year, type, topics)
S2:        query= param + year= modifier
arXiv:     search_query= with field prefixes (ti:, abs:, au:, cat:)
"""

from __future__ import annotations

import re
from typing import Optional

from app.models.search_intent import SearchIntent

_QUESTION_PREFIX = re.compile(
    r"^(to what extent|how does|how do|how|what|which|why|does|do|is|are|can|could)\s+",
    re.IGNORECASE,
)


def compile_query(provider: str, query: str, search_intent: Optional[SearchIntent] = None) -> str:
    """Compile a single query string for the given provider.

    If a SearchIntent is provided, uses provider-specific syntax.
    Otherwise falls back to basic cleaning.
    """
    if search_intent is not None:
        compiled = _compile_from_intent(provider, search_intent)
        if compiled:
            return compiled

    # Fallback: basic cleaning
    cleaned = re.sub(r"[?！？，,;；:：(){}\[\]]", " ", query or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _QUESTION_PREFIX.sub("", cleaned)

    if provider == "arxiv":
        tokens = [
            token for token in cleaned.split()
            if token.upper() not in {"AND", "OR", "NOT"}
        ]
        return " AND ".join(tokens[:12])
    return cleaned[:500]


def _compile_from_intent(provider: str, intent: SearchIntent) -> str:
    """Build a provider-specific query from the search intent."""
    concepts = intent.core_concepts or []
    aliases = intent.aliases or {}

    if provider == "openalex":
        # OpenAlex search= does full-text over title/abstract.
        # Combine core concepts + broad synonyms for high recall.
        terms = list(concepts)
        for alias_list in aliases.values():
            terms.extend(alias_list[:3])
        # Deduplicate, keep order
        seen = set()
        unique = []
        for t in terms:
            t = t.lower().strip()
            if t and t not in seen:
                seen.add(t)
                unique.append(t)
        return " ".join(unique[:15])

    elif provider == "arxiv":
        # arXiv: use field prefixes for precision.
        # Build ti: + abs: queries from core concepts.
        core_terms = " OR ".join(f'ti:"{c}"' for c in concepts[:4] if len(c) > 3)
        abs_terms = " OR ".join(f'abs:"{c}"' for c in concepts[:4] if len(c) > 3)
        parts = []
        if core_terms:
            parts.append(f"({core_terms})")
        if abs_terms:
            parts.append(f"({abs_terms})")
        # Add domain hints as OR categories
        if intent.venue_or_domain_hints:
            cats = " OR ".join(
                f"cat:{d}" for d in intent.venue_or_domain_hints[:4]
            )
            parts.append(f"({cats})")
        if parts:
            return " AND ".join(parts)
        return " AND ".join(concepts[:12])

    elif provider == "semantic_scholar":
        # S2: plain keyword query, but we can use more precise terms.
        # S2 handles natural language well; use concepts + must_include.
        terms = []
        if intent.must_include:
            terms.extend(intent.must_include[:5])
        terms.extend(concepts[:5])
        return " ".join(terms)[:500]

    # Fallback
    return " ".join(concepts)[:500]
