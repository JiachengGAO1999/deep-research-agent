"""Citation validation — ensures all citations in the report map to real papers."""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.models.paper import Paper
from app.models.task import CitationValidation

logger = logging.getLogger(__name__)

# Pattern for citation markers like [P1], [P2], [P1,P2,P3], [P1-P3]
CITATION_PATTERN = re.compile(r"\[P(\d+(?:[,-]\d+)*)\]")


def validate_citations(report: str, selected_papers: list[Paper]) -> CitationValidation:
    """Validate that all citations in the report map to real selected papers.

    Args:
        report: The generated Markdown report text.
        selected_papers: The list of selected papers (with internal_ids).

    Returns:
        CitationValidation with validation results.
    """
    if not report or not selected_papers:
        return CitationValidation(
            is_valid=not report,  # Valid if empty
            issues=["Empty report or no selected papers"] if report else [],
        )

    # Map internal_id -> index in selected_papers
    paper_id_to_index: dict[str, int] = {}
    for i, paper in enumerate(selected_papers):
        paper_id_to_index[paper.internal_id] = i + 1  # 1-indexed

    # Also map by index for papers that are referenced by position
    paper_by_position: dict[int, Paper] = {}
    for i, paper in enumerate(selected_papers):
        paper_by_position[i + 1] = paper

    # Extract all citation markers from text
    citations_in_text = _extract_citation_markers(report)

    # Determine which papers are actually cited
    cited_paper_indices: set[int] = set()
    orphan_citations: list[str] = []
    issues: list[str] = []

    for marker in citations_in_text:
        for idx in _parse_marker_indices(marker):
            if 1 <= idx <= len(selected_papers):
                cited_paper_indices.add(idx)
            else:
                orphan_citations.append(marker)
                issues.append(f"Citation [{marker}] references non-existent paper index {idx} (max: {len(selected_papers)})")

    # Papers in reference list that are never cited
    unused_papers: list[str] = []
    for i, paper in enumerate(selected_papers):
        if (i + 1) not in cited_paper_indices:
            unused_papers.append(paper.internal_id)

    # Check reference list matches
    papers_in_reference_list: list[str] = []
    # Try to extract what's in the reference section
    ref_section_match = re.search(
        r"##\s*参考文献.*?\n(.*?)(?=\n##|\Z)", report, re.DOTALL
    )
    if ref_section_match:
        ref_text = ref_section_match.group(1)
        ref_indices = set()
        for m in re.finditer(r"\[P(\d+)\]", ref_text):
            ref_indices.add(int(m.group(1)))
        papers_in_reference_list = [f"P{i}" for i in sorted(ref_indices)]

        # Check for references not in selected papers
        for idx in ref_indices:
            if idx not in paper_by_position:
                issues.append(f"Reference [P{idx}] in reference list does not correspond to any selected paper")

    # Determine validity
    is_valid = len(orphan_citations) == 0 and len(issues) == 0

    result = CitationValidation(
        citations_in_text=list(citations_in_text),
        papers_in_reference_list=papers_in_reference_list,
        orphan_citations=orphan_citations,
        unused_papers=unused_papers,
        is_valid=is_valid,
        issues=issues,
    )

    if not is_valid:
        logger.warning(f"Citation validation found {len(issues)} issues: {issues}")

    return result


def _extract_citation_markers(text: str) -> set[str]:
    """Extract all citation markers like [P1], [P2,P3] from text."""
    markers = set()
    for match in CITATION_PATTERN.finditer(text):
        markers.add(match.group(1))
    return markers


def _parse_marker_indices(marker: str) -> list[int]:
    """Parse a citation marker like '1,2,3' or '1-3' into individual indices."""
    indices = []
    parts = marker.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                indices.extend(range(int(start), int(end) + 1))
            except ValueError:
                pass
        else:
            try:
                indices.append(int(part))
            except ValueError:
                pass
    return indices


def auto_fix_citations(report: str, selected_papers: list[Paper]) -> str:
    """Attempt to automatically fix citation issues in the report.

    This is a best-effort fixer:
    - Removes citations to non-existent papers
    - Ensures all selected papers appear in the reference list
    """
    if not report or not selected_papers:
        return report or ""

    max_idx = len(selected_papers)
    fixed_report = report

    # Replace citations with indices beyond the range
    def _fix_marker(match):
        marker = match.group(1)
        valid_parts = []
        for part in marker.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    s, e = int(start), int(end)
                    if s <= max_idx and e <= max_idx:
                        valid_parts.append(part)
                    elif s <= max_idx:
                        valid_parts.append(f"{s}-{max_idx}")
                except ValueError:
                    pass
            else:
                try:
                    idx = int(part)
                    if idx <= max_idx:
                        valid_parts.append(part)
                except ValueError:
                    pass
        if valid_parts:
            return f"[P{','.join(valid_parts)}]"
        else:
            return ""  # Remove orphan citation entirely

    fixed_report = CITATION_PATTERN.sub(_fix_marker, fixed_report)

    # Ensure reference list exists and has correct entries
    if "## 参考文献" not in fixed_report:
        fixed_report += "\n\n## 参考文献\n\n"
        for i, paper in enumerate(selected_papers):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            year = paper.publication_year or "n.d."
            fixed_report += (
                f"- [P{i + 1}] {authors}. \"{paper.title}.\" "
                f"{paper.venue or 'Unknown venue'}, {year}."
            )
            if paper.doi:
                fixed_report += f" DOI: {paper.doi}"
            fixed_report += "\n"

    return fixed_report


def build_reference_entries(selected_papers: list[Paper]) -> str:
    """Build authentic reference entries directly from paper records (not LLM)."""
    entries = []
    for i, paper in enumerate(selected_papers):
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."

        year = paper.publication_year or "n.d."
        venue = paper.venue or "Unknown venue"

        entry = f"[P{i + 1}] {authors} ({year}). \"{paper.title}.\" *{venue}*."
        if paper.doi:
            entry += f" DOI: [{paper.doi}](https://doi.org/{paper.doi})"
        if paper.url and not paper.doi:
            entry += f" URL: {paper.url}"

        entries.append(entry)

    return "\n\n".join(entries)
