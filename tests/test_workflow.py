"""Integration tests for the LangGraph workflow with mock providers."""

import os
import pytest
import sys

# Ensure mock mode before any imports
os.environ["MOCK_MODE"] = "1"
os.environ["LLM_API_KEY"] = ""


@pytest.fixture(autouse=True)
def reset_workflow_cache():
    """Reset cached provider/LLM instances between tests."""
    from app.workflow import graph as wf
    wf._reset_cached_instances()
    yield
    wf._reset_cached_instances()


class TestSearchRoundStopConditions:
    """Test that the workflow doesn't loop infinitely."""

    @pytest.mark.asyncio
    async def test_max_rounds_enforced(self):
        """The gap analysis should enforce max supplementary rounds."""
        from app.workflow.graph import assess_gaps_node

        state = {
            "task_id": "test_stop",
            "original_question": "Test question?",
            "supplementary_rounds_done": 2,
            "max_rounds": 3,
            "evidence": [],
            "selected_papers": [],
            "new_papers_this_round": 0,
            "current_round": 2,
        }

        result = await assess_gaps_node(state)
        gap = result.get("gap_analysis")
        assert gap is not None
        assert not gap.needs_supplementary_search

    @pytest.mark.asyncio
    async def test_low_new_papers_stops(self):
        """When too few new papers are found, should stop supplementary search."""
        from app.workflow.graph import assess_gaps_node

        state = {
            "task_id": "test_low_new",
            "original_question": "Test question?",
            "supplementary_rounds_done": 1,
            "max_rounds": 3,
            "new_papers_this_round": 1,
            "evidence": [],
            "selected_papers": [],
            "current_round": 1,
        }

        result = await assess_gaps_node(state)
        gap = result.get("gap_analysis")
        assert gap is not None
        assert not gap.needs_supplementary_search


class TestWorkflowNodes:
    """Test individual workflow nodes in isolation."""

    @pytest.mark.asyncio
    async def test_initialize_node(self):
        from app.workflow.graph import initialize_node

        state = {
            "task_id": "test_init",
            "original_question": "Test question?",
        }

        result = await initialize_node(state)
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_plan_queries_node(self):
        from app.workflow.graph import plan_queries_node, initialize_node

        state = {
            "task_id": "test_plan",
            "original_question": "How does dialogue history affect LLM reasoning?",
            "year_from": 2020,
            "year_to": 2025,
        }
        state = await initialize_node(state)
        result = await plan_queries_node(state)

        assert result["search_plan"] is not None
        assert len(result["queries"]) > 0

    @pytest.mark.asyncio
    async def test_deduplicate_node(self):
        from app.workflow.graph import normalize_and_deduplicate_node
        from app.models.paper import Paper, PaperSource, AuthorInfo

        papers = [
            Paper(
                internal_id="a1",
                title="Same Title Paper",
                authors=[AuthorInfo(name="Author One")],
                publication_year=2024,
                doi="10.1234/same.001",
                source_names=["openalex"],
                source_ids=[PaperSource(provider="openalex", provider_id="W1")],
            ),
            Paper(
                internal_id="a2",
                title="Same Title Paper",
                authors=[AuthorInfo(name="Author One")],
                publication_year=2024,
                doi="10.1234/same.001",
                source_names=["crossref"],
                source_ids=[PaperSource(provider="crossref", provider_id="10.1234/same.001")],
            ),
        ]

        state = {
            "task_id": "test_dedup",
            "original_question": "Test",
            "normalized_papers": papers,
        }

        result = await normalize_and_deduplicate_node(state)
        assert len(result["normalized_papers"]) == 1


class TestFullWorkflow:
    """End-to-end workflow test with mock providers and mock LLM."""

    @pytest.mark.asyncio
    async def test_full_workflow_completes(self):
        """A full research run should complete without errors."""
        from app.workflow.graph import run_research
        from app.models.task import TaskMetrics

        state = {
            "task_id": "integration_test_001",
            "original_question": "How does dialogue history affect reasoning reliability in large language models?",
            "research_mode": "strict",
            "year_from": 2020,
            "year_to": 2026,
            "max_rounds": 3,
            "status": "pending",
            "current_round": 0,
            "queries": [],
            "normalized_papers": [],
            "selected_papers": [],
            "evidence": [],
            "metrics": TaskMetrics(),
            "supplementary_rounds_done": 0,
            "previous_round_paper_ids": [],
            "new_papers_this_round": 0,
        }

        try:
            final_state = await run_research(state)
        except Exception as e:
            pytest.fail(f"Workflow raised an exception: {e}")

        assert final_state["status"] == "completed"
        assert final_state.get("report"), "Report should not be empty"
        assert len(final_state.get("selected_papers", [])) > 0
        assert len(final_state.get("evidence", [])) > 0

        errors = final_state.get("errors", [])
        if isinstance(errors, list):
            assert not errors, f"Workflow had errors: {errors}"

    @pytest.mark.asyncio
    async def test_report_has_no_fake_citations(self):
        """The final report should not contain fabricated citations."""
        from app.workflow.graph import run_research
        from app.models.task import TaskMetrics
        from app.services.citation_validation import validate_citations

        state = {
            "task_id": "integration_test_002",
            "original_question": "What are the effects of context length on LLM reasoning?",
            "research_mode": "strict",
            "year_from": 2022,
            "year_to": 2025,
            "max_rounds": 3,
            "status": "pending",
            "current_round": 0,
            "queries": [],
            "normalized_papers": [],
            "selected_papers": [],
            "evidence": [],
            "metrics": TaskMetrics(),
            "supplementary_rounds_done": 0,
            "previous_round_paper_ids": [],
            "new_papers_this_round": 0,
        }

        final_state = await run_research(state)
        report = final_state.get("report", "")
        selected = final_state.get("selected_papers", [])

        validation = validate_citations(report, selected)
        assert len(validation.orphan_citations) == 0, f"Orphan citations: {validation.orphan_citations}"
        assert validation.is_valid, f"Citation issues: {validation.issues}"

    @pytest.mark.asyncio
    async def test_workflow_does_not_loop_infinitely(self):
        """Workflow should finish within reasonable constraints."""
        import time
        from app.workflow.graph import run_research
        from app.models.task import TaskMetrics

        state = {
            "task_id": "integration_test_003",
            "original_question": "Test question for loop prevention",
            "research_mode": "strict",
            "max_rounds": 2,
            "status": "pending",
            "current_round": 0,
            "queries": [],
            "normalized_papers": [],
            "selected_papers": [],
            "evidence": [],
            "metrics": TaskMetrics(),
            "supplementary_rounds_done": 0,
            "previous_round_paper_ids": [],
            "new_papers_this_round": 0,
        }

        start = time.time()
        final_state = await run_research(state)
        elapsed = time.time() - start

        assert elapsed < 60, f"Workflow took {elapsed:.1f}s"
        assert final_state["status"] in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_single_provider_failure_does_not_crash(self):
        """If one provider fails, the workflow should continue."""
        import app.workflow.graph as wf
        from app.workflow.graph import run_research
        from app.models.task import TaskMetrics
        from app.providers.mock_provider import MockProvider

        # Override the module-level provider cache with a mixed set
        class AlwaysFailingProvider(MockProvider):
            name = "always_failing"

            async def search(self, query, **kwargs):
                raise Exception("Simulated provider failure")

            async def is_available(self):
                return True

        # Set module-level provider cache
        wf._cached_providers = [
            AlwaysFailingProvider(),
            MockProvider(),
        ]

        try:
            state = {
                "task_id": "integration_test_004",
                "original_question": "Test provider failure resilience",
                "research_mode": "strict",
                "max_rounds": 2,
                "status": "pending",
                "current_round": 0,
                "queries": [],
                "normalized_papers": [],
                "selected_papers": [],
                "evidence": [],
                "metrics": TaskMetrics(),
                "supplementary_rounds_done": 0,
                "previous_round_paper_ids": [],
                "new_papers_this_round": 0,
            }

            final_state = await run_research(state)
            assert final_state["status"] == "completed"
        finally:
            wf._reset_cached_instances()
