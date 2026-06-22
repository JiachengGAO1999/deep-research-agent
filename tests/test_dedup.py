"""Test paper deduplication logic."""

import pytest
from app.models.paper import Paper, PaperSource, AuthorInfo, normalize_title
from app.services.dedup import (
    deduplicate_papers,
    titles_are_similar,
    _find_doi_match,
    _find_title_match,
)


class TestNormalizeTitle:
    def test_lowercase(self):
        assert normalize_title("Hello World") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_title("Multi-turn Reasoning: A Study.") == "multiturn reasoning a study"

    def test_collapse_whitespace(self):
        assert normalize_title("  Multiple   Spaces  ") == "multiple spaces"

    def test_empty(self):
        assert normalize_title("") == ""
        assert normalize_title(None) == ""


class TestTitleSimilarity:
    def test_exact_match(self):
        assert titles_are_similar("Multi-turn Reasoning", "Multi-turn Reasoning")

    def test_case_insensitive(self):
        assert titles_are_similar("Multi-turn Reasoning", "multi-turn reasoning")

    def test_extra_whitespace(self):
        assert titles_are_similar(
            "Context Accumulation in Conversational AI",
            "Context   Accumulation  in  Conversational  AI",
        )

    def test_highly_similar(self):
        # These should be similar after normalization
        assert titles_are_similar(
            "Context Accumulation in Conversational AI",
            "Context Accumulation in Conversational AI Systems",
        )

    def test_different_titles(self):
        assert not titles_are_similar(
            "Multi-turn Reasoning in LLMs",
            "Computer Vision Applications",
        )


class TestDOIDedup:
    def test_same_doi_merges(self, sample_papers):
        """Papers with the same DOI should be merged."""
        # p1 and p3 have the same DOI (10.1234/test.001)
        result = deduplicate_papers(sample_papers)
        dois_in_result = {p.doi for p in result if p.doi}
        assert "10.1234/test.001" in dois_in_result
        # Should have fewer papers than input
        assert len(result) < len(sample_papers)

    def test_merged_paper_has_both_sources(self, sample_papers):
        """Merged paper should retain source info from both."""
        result = deduplicate_papers(sample_papers)
        # Find the merged paper with DOI 10.1234/test.001
        merged = next(p for p in result if p.doi == "10.1234/test.001")
        source_providers = {s.provider for s in merged.source_ids}
        assert "openalex" in source_providers
        assert "crossref" in source_providers

    def test_merged_paper_keeps_richer_abstract(self, sample_papers):
        """Merged paper should keep the longer abstract."""
        result = deduplicate_papers(sample_papers)
        merged = next(p for p in result if p.doi == "10.1234/test.001")
        # p1 has a longer abstract than p3
        assert "study of multi-turn reasoning" in (merged.abstract or "").lower()


class TestTitleDedup:
    def test_similar_title_same_author_merges(self, sample_papers):
        """Papers with similar titles, same author, same year should merge."""
        # p2 and p5 have similar titles, same author (Bob Jones), same year (2023)
        result = deduplicate_papers(sample_papers)
        # Count papers related to "Context Accumulation"
        context_papers = [
            p for p in result
            if "context accumulation" in p.title.lower()
        ]
        assert len(context_papers) == 1  # Should be merged

    def test_no_doi_papers_still_checked(self):
        """Papers without DOIs should still be checked for title similarity."""
        p_a = Paper(
            internal_id="a1",
            title="A Study of Reasoning in Dialogue Systems",
            authors=[AuthorInfo(name="John Doe")],
            publication_year=2023,
        )
        p_b = Paper(
            internal_id="a2",
            title="A Study of Reasoning in Dialogue  Systems",  # Extra space
            authors=[AuthorInfo(name="John Doe")],
            publication_year=2023,
        )
        result = deduplicate_papers([p_a, p_b])
        assert len(result) == 1

    def test_similar_title_different_author_keeps_separate(self):
        """Similar titles but different authors and no year match → keep separate."""
        p_a = Paper(
            internal_id="b1",
            title="Machine Learning for NLP",
            authors=[AuthorInfo(name="Alice")],
            publication_year=2020,
        )
        p_b = Paper(
            internal_id="b2",
            title="Machine Learning for NLP Applications",
            authors=[AuthorInfo(name="Bob")],  # Different author
            publication_year=2022,  # Different year
        )
        result = deduplicate_papers([p_a, p_b])
        assert len(result) == 2  # Should stay separate


class TestDedupEdgeCases:
    def test_single_paper(self):
        paper = Paper(title="Single Paper", authors=[AuthorInfo(name="Test")])
        result = deduplicate_papers([paper])
        assert len(result) == 1

    def test_empty_list(self):
        result = deduplicate_papers([])
        assert result == []

    def test_all_unique(self):
        papers = [
            Paper(internal_id="u0", title="Neural Networks for Image Classification", authors=[AuthorInfo(name="Alice")]),
            Paper(internal_id="u1", title="Transformer Models for Machine Translation", authors=[AuthorInfo(name="Bob")]),
            Paper(internal_id="u2", title="Reinforcement Learning in Robotics Control", authors=[AuthorInfo(name="Charlie")]),
            Paper(internal_id="u3", title="Graph Neural Networks for Drug Discovery", authors=[AuthorInfo(name="Diana")]),
            Paper(internal_id="u4", title="Federated Learning with Differential Privacy", authors=[AuthorInfo(name="Eve")]),
        ]
        result = deduplicate_papers(papers)
        assert len(result) == 5
