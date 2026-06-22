"""Build structured FTS5 queries from concept groups, paper metadata, and synonyms."""

from __future__ import annotations

import re
from typing import List, Optional, Set

from app.models.paper import Paper
from app.models.search_plan import SearchPlan

# Words that add no retrieval value in academic FTS5 search
STOP_WORDS: Set[str] = {
    "model", "models", "method", "methods", "result", "results",
    "performance", "approach", "approaches", "study", "studies",
    "paper", "papers", "research", "based", "using", "used",
    "proposed", "show", "shown", "found", "find", "can", "may",
    "also", "well", "however", "thus", "therefore", "one", "two",
    "three", "first", "second", "new", "different", "various",
    "including", "within", "across", "among", "analysis",
    "experiment", "experiments", "evaluation", "data", "dataset",
    "task", "tasks", "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "about", "such",
    "this", "that", "these", "those", "from", "with", "for",
    "and", "not", "but", "or", "its", "it", "to", "of", "in",
}


def _sanitize_fts5_term(term: str) -> str:
    """Escape FTS5 special characters and quote multi-word phrases."""
    term = term.strip()
    # Remove FTS5 special chars except spaces within phrases
    term = re.sub(r'[()\[\]{}*^$]', '', term)
    # Collapse whitespace
    term = re.sub(r'\s+', ' ', term).strip()
    if not term:
        return ""
    # Multi-word phrase → quote it
    if ' ' in term:
        return f'"{term}"'
    return term


def _extract_terms(text: str, min_len: int = 3) -> List[str]:
    """Extract meaningful terms from text, filtering stop words and short tokens."""
    # Split on non-alpha boundaries
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9\-]{2,}', text.lower())
    terms = []
    for t in tokens:
        t = t.strip('-').strip()
        if len(t) >= min_len and t not in STOP_WORDS:
            terms.append(t)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def build_concept_groups(
    search_plan: Optional[SearchPlan] = None,
    papers: Optional[List[Paper]] = None,
    extra_terms: Optional[List[str]] = None,
) -> List[List[str]]:
    """Build query groups for multi-query FTS5 retrieval.

    Returns a list of term groups. Each inner list is a group of related terms
    that will be OR-ed together in a single FTS5 query.
    """
    groups: List[List[str]] = []

    # Group 1: Core concepts from search plan
    if search_plan:
        core_terms = []
        for concept in search_plan.core_concepts:
            phrase = _sanitize_fts5_term(concept)
            if phrase:
                core_terms.append(phrase)
        if core_terms:
            groups.append(core_terms)

        # Group 2+: One group per core concept with its synonyms
        for concept in search_plan.core_concepts:
            syns = search_plan.synonyms.get(concept, [])
            terms = [_sanitize_fts5_term(concept)]
            for syn in syns[:3]:
                s = _sanitize_fts5_term(syn)
                if s and s not in terms:
                    terms.append(s)
            if len(terms) > 1:  # Only if there are synonyms
                groups.append(terms)

    # Paper-specific groups: per-paper distinctive terms
    if papers:
        for paper in papers:
            paper_terms: List[str] = []
            if paper.title:
                paper_terms.extend(_extract_terms(paper.title, min_len=4)[:5])
            if paper.abstract:
                # Extract key phrases from abstract (capitalized or quoted)
                abstract_terms = _extract_key_phrases(paper.abstract)
                paper_terms.extend(abstract_terms[:3])
            # Deduplicate and filter
            paper_terms = _deduplicate_preserve_order(paper_terms)
            paper_terms = [t for t in paper_terms if t.lower() not in STOP_WORDS]
            if len(paper_terms) >= 2:
                groups.append(paper_terms)

    # Extra terms group
    if extra_terms:
        terms = [_sanitize_fts5_term(t) for t in extra_terms if _sanitize_fts5_term(t)]
        if terms:
            groups.append(terms)

    # Deduplicate: remove identical groups
    seen = set()
    unique_groups = []
    for g in groups:
        key = tuple(sorted(g))
        if key not in seen:
            seen.add(key)
            unique_groups.append(g)

    return unique_groups


def _extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract distinctive key phrases from text.

    Prioritizes:
    - Capitalized multi-word terms (e.g., "Dialogue History")
    - Terms with numeric modifiers (e.g., "14.66% improvement")
    - Longer unique words
    """
    phrases = []
    # Capitalized sequences
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
        phrase = m.group(1).strip()
        if len(phrase) > 10 and phrase.lower() not in STOP_WORDS:
            phrases.append(phrase.lower())
    # Add distinctive long words
    words = re.findall(r'\b[a-z]{6,}\b', text.lower())
    for w in words:
        if w not in STOP_WORDS and w not in phrases:
            phrases.append(w)
    return phrases[:max_phrases]


def _deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower().strip()
        if key not in seen and len(key) > 2:
            seen.add(key)
            result.append(item)
    return result


def build_fts5_query(terms: List[str]) -> str:
    """Build a single FTS5 OR query from a list of terms.

    Multi-word terms are quoted. Single words used as-is.
    Returns empty string if no valid terms.
    """
    sanitized = []
    for t in terms:
        s = _sanitize_fts5_term(t)
        if s:
            sanitized.append(s)
    if not sanitized:
        return ""
    return " OR ".join(sanitized)
