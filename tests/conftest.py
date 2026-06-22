"""Pytest fixtures for testing."""

import os
import pytest

# Force mock mode for all tests
os.environ["MOCK_MODE"] = "1"
os.environ["LLM_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from app.models.paper import Paper, PaperSource, AuthorInfo, normalize_title
from app.models.search_plan import SearchPlan, SearchQuery, InclusionExclusionCriteria
from app.models.evidence import ExtractedEvidence, GapAnalysis
from app.models.task import TaskState, TaskMetrics


@pytest.fixture
def sample_papers():
    """Create a set of sample papers for testing."""
    papers = [
        Paper(
            internal_id="p1",
            title="Multi-turn Reasoning in LLMs",
            abstract="A study of multi-turn reasoning in large language models.",
            authors=[AuthorInfo(name="Alice Smith")],
            publication_year=2024,
            doi="10.1234/test.001",
            source_names=["openalex"],
            source_ids=[PaperSource(provider="openalex", provider_id="W001")],
        ),
        Paper(
            internal_id="p2",
            title="Context Accumulation in Conversational AI",
            abstract="How context affects reasoning quality.",
            authors=[AuthorInfo(name="Bob Jones")],
            publication_year=2023,
            doi="10.1234/test.002",
            source_names=["semantic_scholar"],
            source_ids=[PaperSource(provider="semantic_scholar", provider_id="paper2")],
        ),
        Paper(
            internal_id="p3",
            title="Multi-turn  Reasoning in LLMs",  # Extra space - near duplicate of p1
            abstract="A shorter abstract.",
            authors=[AuthorInfo(name="A. Smith")],  # Different author format
            publication_year=2024,
            doi="10.1234/test.001",  # Same DOI as p1
            source_names=["crossref"],
            source_ids=[PaperSource(provider="crossref", provider_id="10.1234/test.001")],
        ),
        Paper(
            internal_id="p4",
            title="Completely Different Topic",
            abstract="Something about computer vision.",
            authors=[AuthorInfo(name="Charlie")],
            publication_year=2022,
            source_names=["arxiv"],
            source_ids=[PaperSource(provider="arxiv", provider_id="2201.00001")],
        ),
        Paper(
            internal_id="p5",
            title="Context Accumulation in Conversational AI Systems",  # Similar title to p2
            abstract="Expanded study on context accumulation.",
            authors=[AuthorInfo(name="Bob Jones")],  # Same author as p2
            publication_year=2023,  # Same year as p2
            doi=None,  # No DOI
            source_names=["openalex"],
            source_ids=[PaperSource(provider="openalex", provider_id="W002")],
        ),
    ]
    return papers


@pytest.fixture
def sample_search_plan():
    """Create a sample search plan."""
    return SearchPlan(
        research_topic="Multi-turn reasoning in LLMs",
        core_concepts=["multi-turn reasoning", "LLM", "context"],
        synonyms={
            "multi-turn reasoning": ["conversational reasoning", "dialogue reasoning"],
            "LLM": ["large language model", "transformer"],
        },
        queries=[
            SearchQuery(
                query_string="multi-turn reasoning large language models",
                rationale="Core topic search",
                keywords=["multi-turn", "reasoning", "LLM"],
            ),
            SearchQuery(
                query_string="context accumulation conversational AI reasoning quality",
                rationale="Context-specific search",
                keywords=["context", "conversational", "reasoning"],
            ),
        ],
        year_from=2020,
        year_to=2025,
        criteria=InclusionExclusionCriteria(
            include=["multi-turn reasoning", "LLM evaluation"],
            exclude=["single-turn only"],
        ),
    )


@pytest.fixture
def sample_task_state(sample_search_plan):
    """Create a sample task state."""
    return TaskState(
        task_id="test_task_001",
        original_question="How does dialogue history affect reasoning in LLMs?",
        search_plan=sample_search_plan,
        year_from=2020,
        year_to=2025,
        max_rounds=3,
    )
