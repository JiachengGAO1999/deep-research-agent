"""Test citation validation and no-fabrication guarantees."""

import pytest
from app.models.paper import Paper, AuthorInfo, PaperSource
from app.services.citation_validation import (
    validate_citations,
    auto_fix_citations,
    build_reference_entries,
    _extract_citation_markers,
    _parse_marker_indices,
)
from app.models.task import CitationValidation


class TestCitationMarkers:
    def test_extract_single_marker(self):
        markers = _extract_citation_markers("This is a claim [P1] and another [P2].")
        assert markers == {"1", "2"}

    def test_extract_range_marker(self):
        markers = _extract_citation_markers("Multiple papers [P1-3] support this.")
        assert "1-3" in markers

    def test_extract_comma_marker(self):
        markers = _extract_citation_markers("Key works [P1,2,3] show...")
        assert "1,2,3" in markers

    def test_extract_complex_marker(self):
        markers = _extract_citation_markers("Both [P1-3,5] and [P4] support this.")
        assert "1-3,5" in markers or "1-3, 5" in markers
        assert "4" in markers

    def test_no_citations(self):
        markers = _extract_citation_markers("No citations here.")
        assert markers == set()


class TestParseIndices:
    def test_single(self):
        assert _parse_marker_indices("1") == [1]

    def test_range(self):
        assert _parse_marker_indices("1-3") == [1, 2, 3]

    def test_comma(self):
        assert _parse_marker_indices("1,3,5") == [1, 3, 5]

    def test_complex(self):
        result = _parse_marker_indices("1-2,4")
        assert 1 in result
        assert 2 in result
        assert 4 in result


class TestCitationValidation:
    def test_valid_citations(self):
        """Report with valid citations should pass validation."""
        papers = [
            Paper(
                internal_id="p1",
                title="Paper One",
                authors=[AuthorInfo(name="Author A")],
                venue="Test Venue",
                publication_year=2024,
            ),
            Paper(
                internal_id="p2",
                title="Paper Two",
                authors=[AuthorInfo(name="Author B")],
                venue="Test Venue",
                publication_year=2023,
            ),
        ]

        report = """# Test Report

Key findings are supported [P1][P2].

## 参考文献

- [P1] Author A. "Paper One." *Test Venue*, 2024.
- [P2] Author B. "Paper Two." *Test Venue*, 2023.
"""

        result = validate_citations(report, papers)
        assert result.is_valid
        assert len(result.orphan_citations) == 0

    def test_orphan_citation(self):
        """Citations to non-existent papers should be flagged."""
        papers = [
            Paper(
                internal_id="p1",
                title="Paper One",
                authors=[AuthorInfo(name="Author A")],
                venue="Test Venue",
                publication_year=2024,
            ),
        ]

        report = """# Test Report

This claim is supported [P1][P5].  <!-- P5 doesn't exist -->

## 参考文献

- [P1] Author A. "Paper One." *Test Venue*, 2024.
"""
        result = validate_citations(report, papers)
        assert not result.is_valid
        assert len(result.orphan_citations) > 0

    def test_empty_report(self):
        papers = [Paper(title="Test", authors=[AuthorInfo(name="Author")], internal_id="p1")]
        result = validate_citations("", papers)
        assert result.is_valid  # Empty report is vacuously valid

    def test_no_papers(self):
        report = "Some text without real references [P1]."
        result = validate_citations(report, [])
        assert not result.is_valid  # References papers that don't exist


class TestNoFabricatedReferences:
    """Test that we cannot fabricate references."""

    def test_build_reference_entries_from_real_data(self):
        """Reference entries must come from actual paper records."""
        papers = [
            Paper(
                internal_id="p1",
                title="Real Paper Title",
                authors=[AuthorInfo(name="Real Author")],
                venue="Real Venue",
                publication_year=2024,
                doi="10.1234/real.001",
            ),
        ]

        entries = build_reference_entries(papers)
        assert "Real Paper Title" in entries
        assert "Real Author" in entries
        assert "10.1234/real.001" in entries
        assert "Fake Paper" not in entries  # Obviously shouldn't be there

    def test_auto_fix_removes_orphan_citations(self):
        """Auto-fix should remove citations to papers that don't exist."""
        papers = [
            Paper(
                internal_id="p1",
                title="Paper One",
                authors=[AuthorInfo(name="Author A")],
                venue="Test Venue",
                publication_year=2024,
            ),
        ]

        report = "Claim [P1] and also [P999] which doesn't exist."
        fixed = auto_fix_citations(report, papers)
        assert "[P999]" not in fixed
        assert "[P1]" in fixed  # Valid citation should remain


class TestCitationStability:
    """Test that citation IDs remain stable through the workflow."""

    def test_internal_id_preserved(self):
        """Paper internal_id must not change after creation."""
        paper = Paper(
            internal_id="fixed_id_123",
            title="Test Paper",
            authors=[AuthorInfo(name="Author")],
        )
        assert paper.internal_id == "fixed_id_123"

    def test_merged_paper_preserves_id(self):
        """After merging, the surviving paper keeps its internal_id."""
        from app.services.dedup import _merge_papers

        p1 = Paper(
            internal_id="keep_me",
            title="Same Title",
            authors=[AuthorInfo(name="Author A")],
            publication_year=2024,
            abstract="Longer abstract with more information about the study.",
        )
        p2 = Paper(
            internal_id="discard_me",
            title="Same Title",
            authors=[AuthorInfo(name="Author A")],
            publication_year=2024,
            abstract="Short.",
        )

        merged = _merge_papers([p1, p2])
        # The richer paper (p1 with longer abstract) should survive
        assert merged.internal_id == "keep_me"
