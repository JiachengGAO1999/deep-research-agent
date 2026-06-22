"""Paper deduplication service.

Dedup priority:
1. DOI match
2. Provider ID match (OpenAlex ID or Semantic Scholar ID)
3. Normalized title exact match
4. Normalized title high similarity + year/author match
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from app.models.paper import Paper, normalize_title

logger = logging.getLogger(__name__)

# Threshold for normalized title similarity (0-1)
TITLE_SIMILARITY_THRESHOLD = 0.90


def titles_are_similar(title1: str, title2: str, threshold: float = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """Check if two normalized titles are highly similar."""
    if not title1 or not title2:
        return False
    n1 = normalize_title(title1)
    n2 = normalize_title(title2)
    if n1 == n2:
        return True
    similarity = SequenceMatcher(None, n1, n2).ratio()
    return similarity >= threshold


def _find_doi_match(papers: list[Paper]) -> dict[str, list[int]]:
    """Group papers by DOI. Returns DOI -> list of indices."""
    groups: dict[str, list[int]] = {}
    for i, paper in enumerate(papers):
        if paper.doi:
            doi = paper.doi.lower().strip()
            if doi not in groups:
                groups[doi] = []
            groups[doi].append(i)
    return {doi: idxs for doi, idxs in groups.items() if len(idxs) > 1}


def _find_provider_id_match(papers: list[Paper]) -> list[tuple[int, int]]:
    """Find papers that share a provider ID (from different sources)."""
    # Build index: provider -> provider_id -> list of paper indices
    pid_index: dict[str, dict[str, list[int]]] = {}
    for i, paper in enumerate(papers):
        for src in paper.source_ids:
            if src.provider not in pid_index:
                pid_index[src.provider] = {}
            pid = src.provider_id.strip()
            if pid not in pid_index[src.provider]:
                pid_index[src.provider][pid] = []
            pid_index[src.provider][pid].append(i)

    # Cross-provider: OpenAlex ID can appear in Semantic Scholar's externalIds
    pairs = []
    # Check if any paper shares IDs across providers
    # This is heuristic — collect all provider IDs and look for matches in other providers
    all_ids: dict[str, list[int]] = {}
    for i, paper in enumerate(papers):
        for src in paper.source_ids:
            pid = f"{src.provider}:{src.provider_id}"
            if pid not in all_ids:
                all_ids[pid] = []
            all_ids[pid].append(i)

    return [(idxs[0], idxs[1]) for pid, idxs in all_ids.items() if len(idxs) > 1]


def _find_title_match(papers: list[Paper]) -> list[tuple[int, int]]:
    """Find papers with matching normalized titles."""
    pairs = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if titles_are_similar(papers[i].title, papers[j].title):
                pairs.append((i, j))
    return pairs


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate a list of papers, merging duplicates.

    Priority order: DOI > Provider ID > Title match.
    When merging, the richer record is kept and the other's fields are merged in.
    """
    if len(papers) <= 1:
        return papers

    n = len(papers)
    # Union-find for grouping duplicates
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    # Step 1: DOI matches
    doi_groups = _find_doi_match(papers)
    for doi, indices in doi_groups.items():
        for idx in indices[1:]:
            union(indices[0], idx)
        logger.debug(f"DOI match: {doi} -> {len(indices)} papers")

    # Step 2: Provider ID matches
    pid_pairs = _find_provider_id_match(papers)
    for i, j in pid_pairs:
        union(i, j)
        logger.debug(f"Provider ID match: papers {i} and {j}")

    # Step 3: Title match (with year/author verification)
    title_pairs = _find_title_match(papers)
    for i, j in title_pairs:
        p1, p2 = papers[i], papers[j]
        # Additional verification: year match or author overlap
        year_match = (
            p1.publication_year is not None
            and p2.publication_year is not None
            and p1.publication_year == p2.publication_year
        )
        author_overlap = bool(
            set(a.name.lower() for a in p1.authors if a.name)
            & set(a.name.lower() for a in p2.authors if a.name)
        )
        if year_match or author_overlap:
            union(i, j)
            logger.debug(f"Title match (verified): {p1.title[:50]}...")
        else:
            logger.debug(f"Title match (SKIPPED - no year/author verification): {p1.title[:50]}...")

    # Group papers by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge each group
    result = []
    for root, indices in groups.items():
        if len(indices) == 1:
            result.append(papers[indices[0]])
        else:
            # Merge: keep the most information-rich paper
            merged = _merge_papers([papers[i] for i in indices])
            result.append(merged)

    before = len(papers)
    after = len(result)
    if before != after:
        logger.info(f"Dedup: {before} -> {after} papers ({before - after} duplicates removed)")

    return result


def _merge_papers(papers: list[Paper]) -> Paper:
    """Merge multiple Paper records into one, keeping the richest fields."""
    # Sort by information completeness (has abstract, has doi, has venue, citation count)
    def _score(p: Paper) -> int:
        s = 0
        if p.abstract:
            s += 3
        if p.doi:
            s += 2
        if p.venue:
            s += 1
        if p.citation_count is not None:
            s += 1
        if len(p.authors) > 0:
            s += 1
        return s

    papers_sorted = sorted(papers, key=_score, reverse=True)
    base = papers_sorted[0]

    for other in papers_sorted[1:]:
        base.merge_from(other)

    return base
