import pytest
from pydantic import ValidationError

from app.api.routes import ResearchRequest


def test_topic_alias_and_research_parameters():
    request = ResearchRequest(
        topic="How does retrieval affect factuality?",
        year_from=2020,
        year_to=2026,
        max_papers=10,
        research_depth="deep",
        report_language="en",
    )
    assert request.question == "How does retrieval affect factuality?"
    assert request.max_papers == 10


def test_product_profile_selects_backend_and_aliases():
    request = ResearchRequest(
        research_question="What evidence supports hybrid retrieval?",
        num_papers=8,
        retrieval_profile="balanced",
        language="en",
    )
    assert request.question == "What evidence supports hybrid retrieval?"
    assert request.max_papers == 8
    assert request.evidence_backend == "hybrid"
    assert request.enable_full_text is True
    assert request.report_language == "en"


def test_invalid_year_range_rejected():
    with pytest.raises(ValidationError):
        ResearchRequest(topic="test", year_from=2026, year_to=2020)


def test_full_text_backend_requires_full_text_flag():
    with pytest.raises(ValidationError):
        ResearchRequest(
            topic="test",
            evidence_backend="paperqa",
            enable_full_text=False,
        )
