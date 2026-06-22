"""FTS5 retrieval tests — structured queries, OR semantics, per-paper coverage."""

import os
import pytest

os.environ["MOCK_MODE"] = "1"


@pytest.fixture(autouse=True)
def _init_db():
    import asyncio
    async def _init():
        from app.db.database import init_db
        await init_db()
    asyncio.get_event_loop().run_until_complete(_init())


class TestFTSSaveAndSearch:
    def test_save_and_search_low_freq_word(self):
        """A rare word inserted into a chunk should be findable via MATCH."""
        import asyncio

        async def _test():
            from app.services.fts_search import save_chunks, search_chunks, init_fts5
            from app.models.document import DocumentChunk

            await init_fts5()

            chunk = DocumentChunk(
                chunk_id="test_rare_c1",
                paper_id="p_test",
                task_id="t_test",
                section_title="Results",
                text="The xylophone algorithm achieves state-of-the-art on zephyr benchmark.",
                parser_name="test",
                parser_version="1.0",
            )
            await save_chunks([chunk])

            # Search for the rare word
            results = await search_chunks(
                task_id="t_test", query="xylophone", limit=5
            )
            assert len(results) >= 1, f"Expected >=1 result for 'xylophone', got {len(results)}"
            assert any("xylophone" in r.text.lower() for r in results)

        asyncio.get_event_loop().run_until_complete(_test())

    def test_implicit_and_returns_zero(self):
        """Original 5-term implicit AND should return 0 on sparse data."""
        import asyncio

        async def _test():
            from app.services.fts_search import save_chunks, search_chunks, init_fts5
            from app.models.document import DocumentChunk

            await init_fts5()

            chunk = DocumentChunk(
                chunk_id="test_and_c1",
                paper_id="p_and",
                task_id="t_and",
                section_title="Intro",
                text="Dialogue systems can reason about context but reliability varies across multi-turn interactions.",
                parser_name="test",
                parser_version="1.0",
            )
            await save_chunks([chunk])

            # 5-term AND: "dialogue history reasoning reliability LLM"
            results = await search_chunks(
                task_id="t_and",
                query="dialogue history reasoning reliability LLM",
                limit=10,
            )
            assert len(results) == 0, (
                f"5-term AND should return 0, got {len(results)}"
            )

        asyncio.get_event_loop().run_until_complete(_test())

    def test_or_query_recovers_recall(self):
        """OR-joined terms should find chunks that AND misses."""
        import asyncio

        async def _test():
            from app.services.fts_search import search_by_keywords, save_chunks, init_fts5
            from app.models.document import DocumentChunk

            await init_fts5()

            chunk = DocumentChunk(
                chunk_id="test_or_c1",
                paper_id="p_or",
                task_id="t_or",
                section_title="Results",
                text="Multi-turn dialogue exhibits reasoning degradation over extended conversations.",
                parser_name="test",
                parser_version="1.0",
            )
            await save_chunks([chunk])

            # Use search_by_keywords (OR semantics)
            results = await search_by_keywords(
                task_id="t_or",
                keywords=["dialogue", "reasoning", "degradation", "multi-turn"],
                limit=10,
            )
            assert len(results) >= 1, f"OR query should find chunk, got {len(results)}"

        asyncio.get_event_loop().run_until_complete(_test())

    def test_paper_id_filter(self):
        """Only chunks from specified paper_ids should be returned."""
        import asyncio

        async def _test():
            from app.services.fts_search import search_by_keywords, save_chunks, init_fts5
            from app.models.document import DocumentChunk

            await init_fts5()

            c1 = DocumentChunk(
                chunk_id="test_filter_c1", paper_id="p_target", task_id="t_filter",
                section_title="Results", text="deep learning transformer architecture",
                parser_name="test", parser_version="1.0",
            )
            c2 = DocumentChunk(
                chunk_id="test_filter_c2", paper_id="p_other", task_id="t_filter",
                section_title="Results", text="deep learning transformer architecture",
                parser_name="test", parser_version="1.0",
            )
            await save_chunks([c1, c2])

            results = await search_by_keywords(
                task_id="t_filter",
                keywords=["deep", "learning", "transformer"],
                paper_ids=["p_target"],
                limit=10,
            )
            assert len(results) >= 1
            assert all(r.paper_id == "p_target" for r in results)

        asyncio.get_event_loop().run_until_complete(_test())

    def test_multi_word_phrase_quoted(self):
        """Multi-word terms should be quoted for FTS5 phrase search."""
        from app.services.query_builder import _sanitize_fts5_term, build_fts5_query

        # Multi-word → quoted
        assert _sanitize_fts5_term("dialogue history") == '"dialogue history"'
        # Single word → not quoted
        assert _sanitize_fts5_term("reasoning") == "reasoning"
        # Build query
        query = build_fts5_query(["dialogue history", "reasoning reliability", "LLM"])
        assert '"dialogue history"' in query
        assert '"reasoning reliability"' in query
        assert "LLM" in query
        assert " OR " in query

    def test_stop_words_filtered(self):
        """Generic academic words should not dominate query terms."""
        from app.services.query_builder import _extract_terms, STOP_WORDS

        text = "The model achieves good performance on various methods and tasks using our proposed approach"
        terms = _extract_terms(text, min_len=3)
        # "model", "performance", "methods", "tasks", "proposed", "approach" are stop words
        bad = [t for t in terms if t in STOP_WORDS]
        assert len(bad) == 0, f"Stop words leaked: {bad}"

    def test_empty_result_no_crash(self):
        """No-match query returns empty list, not exception."""
        import asyncio

        async def _test():
            from app.services.fts_search import search_by_keywords
            results = await search_by_keywords(
                task_id="t_nonexistent",
                keywords=["xyznonexistentword"],
                limit=10,
            )
            assert results == []

        asyncio.get_event_loop().run_until_complete(_test())


class TestQueryBuilder:
    def test_concept_groups_with_synonyms(self):
        from app.services.query_builder import build_concept_groups
        from app.models.search_plan import SearchPlan, InclusionExclusionCriteria

        plan = SearchPlan(
            research_topic="Test",
            core_concepts=["dialogue history", "reasoning reliability"],
            synonyms={
                "dialogue history": ["conversation history", "interaction history"],
                "reasoning reliability": ["consistency", "robustness"],
            },
            criteria=InclusionExclusionCriteria(),
        )
        groups = build_concept_groups(search_plan=plan)
        assert len(groups) >= 2  # core group + at least one synonym group
        # First group should have core concepts
        flat = [t for g in groups for t in g]
        assert any("dialogue history" in t for t in flat)

    def test_paper_terms_from_title_abstract(self):
        from app.services.query_builder import build_concept_groups
        from app.models.paper import Paper, AuthorInfo

        paper = Paper(
            internal_id="p1",
            title="Lost in Multi-Turn Conversation: LLM Reasoning Degradation",
            abstract="We find that extended dialogue leads to 14.66% reasoning accuracy drop in multi-turn settings.",
            authors=[AuthorInfo(name="Test")],
        )
        groups = build_concept_groups(papers=[paper])
        assert len(groups) >= 1
        paper_group = groups[0]
        flat = [t.lower() for t in paper_group]
        assert any("multi-turn" in t for t in flat) or any("conversation" in t for t in flat)
