"""LangGraph workflow — the core research state machine."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import get_settings
from app.models.paper import Paper
from app.models.search_plan import SearchPlan
from app.models.evidence import ExtractedEvidence, GapAnalysis
from app.models.task import TaskState, TaskStatus, TaskMetrics, CitationValidation
from app.services.dedup import deduplicate_papers
from app.services.ranking import deterministic_prefilter, llm_rank_papers, _fallback_selection
from app.services.citation_validation import (
    validate_citations,
    auto_fix_citations,
    build_reference_entries,
)

logger = logging.getLogger(__name__)

# Module-level cached singletons (NOT stored in LangGraph state to avoid serialization issues)
_cached_providers: Optional[List[Any]] = None
_cached_llm_client: Any = None
_cached_evidence_engine: Any = None
_runtime_progress: dict[str, dict] = {}

STAGE_PROGRESS = {
    "initialize": 3,
    "plan_queries": 10,
    "search_sources": 22,
    "normalize_and_deduplicate": 30,
    "rank_and_select": 40,
    "download_pdfs": 50,
    "parse_and_chunk": 58,
    "extract_evidence": 68,
    "validate_evidence": 72,
    "assess_gaps": 76,
    "supplementary_search": 55,
    "build_claims": 84,
    "build_literature_relations": 88,
    "synthesize_report": 92,
    "validate_citations": 97,
    "finalize": 100,
}


def _mark_stage(state: dict, stage: str) -> None:
    state["current_stage"] = stage
    state["progress_percent"] = STAGE_PROGRESS[stage]
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    task_id = state.get("task_id")
    if task_id:
        _runtime_progress[task_id] = state


def get_runtime_progress(task_id: str) -> dict:
    return _runtime_progress.get(task_id, {})


def _record_model_usage(state: dict, usage: dict, purpose: str) -> None:
    if not usage:
        return
    metrics = state.get("metrics", TaskMetrics())
    metrics.model_calls.append(
        {
            "purpose": purpose,
            "model": usage.get("model", "unknown"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_seconds": usage.get("latency_seconds"),
            "input_source": "validated_claims" if purpose == "report" else purpose,
        }
    )
    settings = get_settings()
    metrics.estimated_cost_usd += (
        float(usage.get("prompt_tokens", 0) or 0)
        * settings.LLM_INPUT_COST_PER_1M
        + float(usage.get("completion_tokens", 0) or 0)
        * settings.LLM_OUTPUT_COST_PER_1M
    ) / 1_000_000
    state["metrics"] = metrics


def _budget_exceeded(state: dict) -> bool:
    budget = state.get("max_cost_usd")
    if budget is None or budget <= 0:
        return False
    metrics = state.get("metrics", TaskMetrics())
    return metrics.estimated_cost_usd >= budget


class ResearchState(Dict):
    """LangGraph state for the research workflow. Uses plain dict for serialization."""
    pass


def _get_llm_client():
    """Get LLM client with mock fallback. Cached at module level."""
    global _cached_llm_client
    if _cached_llm_client is not None:
        return _cached_llm_client
    settings = get_settings()
    if settings.MOCK_MODE or not settings.has_llm_key:
        from app.llm.mock_client import get_mock_llm_client
        _cached_llm_client = get_mock_llm_client()
    else:
        from app.llm.client import get_llm_client
        _cached_llm_client = get_llm_client()
    return _cached_llm_client


def _get_providers(settings=None):
    """Get provider instances based on config. Cached at module level."""
    global _cached_providers
    if _cached_providers is not None:
        return _cached_providers
    settings = settings or get_settings()
    if settings.MOCK_MODE or not settings.has_llm_key:
        from app.providers.mock_provider import MockProvider
        logger.info("Using mock providers (mock mode or no API keys)")
        _cached_providers = [MockProvider(settings=settings)]
    else:
        from app.providers.openalex import OpenAlexProvider
        from app.providers.semantic_scholar import SemanticScholarProvider
        _cached_providers = [
            OpenAlexProvider(settings=settings),
            SemanticScholarProvider(settings=settings),
        ]
        # arXiv is NOT used as a search provider: SJTU outbound IP is
        # hard-rate-limited by export.arxiv.org. arXiv PDFs are still
        # fetched via arxiv.org/pdf/{id} in download_pdfs_node.
    return _cached_providers


def _get_evidence_engine():
    global _cached_evidence_engine
    if _cached_evidence_engine is None:
        from app.services.evidence_engine import get_evidence_engine

        _cached_evidence_engine = get_evidence_engine()
    return _cached_evidence_engine


def _reset_cached_instances():
    """Reset cached instances (for testing)."""
    global _cached_providers, _cached_llm_client, _cached_evidence_engine
    _cached_providers = None
    _cached_llm_client = None
    _cached_evidence_engine = None


async def _check_availability(providers) -> list:
    """Check which providers are available, log warnings for unavailable ones."""
    available = []
    for p in providers:
        try:
            is_avail = await p.is_available()
            if is_avail:
                available.append(p)
            else:
                logger.warning(f"Provider {p.name} is not available, skipping")
        except Exception as e:
            logger.warning(f"Provider {p.name} availability check failed: {e}, will try to use it anyway")
            available.append(p)
    return available


# ---- Node functions ----

async def initialize_node(state: dict) -> dict:
    """Initialize the research task."""
    _mark_stage(state, "initialize")
    logger.info(f"[{state.get('task_id', 'unknown')}] Initializing research task")
    state["status"] = TaskStatus.RUNNING.value
    state["current_round"] = 0
    state["warnings"] = []
    state["errors"] = []
    mt = TaskMetrics()
    mt.start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["metrics"] = mt
    state["created_at"] = state.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Initialize providers and LLM client (cached at module level, not stored in state)
    _get_providers()
    _get_llm_client()

    return state


async def plan_queries_node(state: dict) -> dict:
    """Generate a structured search plan from the research question."""
    _mark_stage(state, "plan_queries")
    logger.info(f"[{state['task_id']}] Planning queries")
    t0 = time.time()

    llm = _get_llm_client()
    settings = get_settings()

    # Step 1: Generate SearchPlan (simpler, more reliable)
    system_prompt = """You are an expert academic research strategist. Given a research question, create a structured search plan.

Analyze the research question and produce:
1. A refined research topic statement
2. Core concepts (3-6), each with 2-4 synonyms
3. 3-6 search queries optimized for academic APIs
4. Relevant academic domains (e.g., cs.CL, cs.AI, cs.LG)
5. Inclusion and exclusion criteria

IMPORTANT: Use DIVERSE terminology across queries. Don't repeat the same keywords.
If one query uses "dialogue history", another should use "conversation history" or "multi-turn interaction".
Gold papers may use unexpected synonyms — diversity is critical for recall.

Each query should be a concise string suitable for API search (not natural language questions)."""

    user_prompt = f"""Research Question: {state['original_question']}

Year constraints: from {state.get('year_from') or 'any'} to {state.get('year_to') or 'any'}

Generate a structured search plan with diverse terminology in JSON format."""

    from app.models.search_plan import SearchPlan as SP
    sp_result, usage1 = await llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=SP,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=settings.LLM_FAST_ENABLE_THINKING,
    )
    _record_model_usage(state, usage1, "search_plan")

    search_plan = None
    if sp_result:
        search_plan = sp_result
        state["search_plan"] = search_plan
    else:
        logger.warning("SearchPlan generation failed, using fallback")
        from app.models.search_plan import SearchQuery, InclusionExclusionCriteria
        search_plan = SP(
            research_topic=state["original_question"],
            core_concepts=state["original_question"].split(),
            queries=[SearchQuery(query_string=state["original_question"], rationale="Fallback query", keywords=[])],
            criteria=InclusionExclusionCriteria(),
        )
        state["search_plan"] = search_plan
        state.setdefault("warnings", []).append("Search plan generation failed, using basic fallback")

    # Step 2: Generate SearchIntent from SearchPlan (richer, with query families)
    si_system = """You are a query expansion specialist. Given a search plan, produce a SearchIntent with diverse query families.

For each core concept, generate query families with:
- broad queries: high-recall keyword combinations
- narrow queries: precise phrases, field-specific terms
- synonym variants: use ALL the synonyms from the search plan

CRITICAL RULE: Each family MUST use different synonyms and phrasings.
If concept A has synonyms [X, Y, Z], family 1 uses X, family 2 uses Y, family 3 uses Z.
Maximize lexical diversity — this is essential for finding papers that use unexpected terminology.

Respond with a valid SearchIntent JSON."""

    core = search_plan.core_concepts if search_plan else state["original_question"].split()
    synonyms_text = ""
    if search_plan and search_plan.synonyms:
        synonyms_text = "\n".join(
            f"  {k}: {v}" for k, v in list(search_plan.synonyms.items())[:6]
        )
    si_user = f"""Research Question: {state['original_question']}
Core concepts: {core}
Synonyms:
{synonyms_text}

Generate a SearchIntent with >= 4 query families using DIVERSE terminology.
Return a valid SearchIntent JSON object."""

    from app.models.search_intent import SearchIntent as SI
    si_result, usage2 = await llm.generate_structured(
        system_prompt=si_system,
        user_prompt=si_user,
        output_model=SI,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=False,  # Force off for structured output
    )
    _record_model_usage(state, usage2, "search_intent")

    if si_result:
        state["search_intent"] = si_result
        queries = si_result.all_query_strings()
        if not queries:
            queries = search_plan.all_query_strings()
    else:
        logger.warning("SearchIntent generation failed, using SearchPlan queries")
        queries = search_plan.all_query_strings()
        state.setdefault("warnings", []).append("SearchIntent generation failed, using SearchPlan queries only")

    # Augment with concept + evaluation queries
    concept_query = " ".join(core[:5]).strip()
    evaluation_query = f"{concept_query} benchmark empirical evaluation".strip()
    for query in (concept_query, evaluation_query):
        if query and query not in queries:
            queries.append(query)
    depth_limits = {"quick": 2, "standard": 5, "deep": 8}
    query_limit = depth_limits.get(state.get("research_depth", "standard"), 5)
    state["queries"] = queries[:query_limit]

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 2
    mt.llm_tokens_used = (
        getattr(mt, "llm_tokens_used", 0)
        + usage1.get("total_tokens", 0)
        + usage2.get("total_tokens", 0)
    )
    sd = getattr(mt, "stage_durations", {})
    sd["plan_queries"] = time.time() - t0
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Generated {len(state['queries'])} queries")
    return state


async def search_sources_node(state: dict) -> dict:
    """Execute searches across all available providers in parallel."""
    _mark_stage(state, "search_sources")
    logger.info(f"[{state['task_id']}] Searching sources (round {state['current_round']})")
    t0 = time.time()

    providers = _get_providers()
    available_providers = await _check_availability(providers)
    if not available_providers:
        state.setdefault("errors", []).append("No providers available for search")
        return state

    queries = state.get("queries", [])
    search_intent = state.get("search_intent")
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    mt = state.get("metrics", TaskMetrics())

    all_papers: List[Paper] = []
    ranked_identity_lists: list[list[str]] = []

    async def search_provider(provider, query: str):
        try:
            # Pass SearchIntent for provider-specific query compilation
            papers = await provider.search(
                query, year_from=year_from, year_to=year_to,
                search_intent=search_intent,
            )
            total_hits = getattr(provider, "last_total_hits", 0)
            return provider.name, papers, total_hits, True
        except Exception as e:
            logger.error(f"Provider {provider.name} failed for query '{query[:50]}...': {e}")
            state.setdefault("warnings", []).append(f"Provider {provider.name} failed: {str(e)[:200]}")
            return provider.name, [], 0, False

    for query in queries:
        tasks = [search_provider(p, query) for p in available_providers]
        results = await asyncio.gather(*tasks)

        for provider_name, papers, total_hits, success in results:
            reqs = getattr(mt, "provider_requests", {})
            reqs[provider_name] = reqs.get(provider_name, 0) + 1
            mt.provider_requests = reqs

            results_count = getattr(mt, "provider_results", {})
            results_count[provider_name] = results_count.get(provider_name, 0) + len(papers)
            mt.provider_results = results_count

            # Distinguish total_hits from returned count
            hits = getattr(mt, "provider_total_hits", {})
            hits[provider_name] = max(hits.get(provider_name, 0), total_hits)
            mt.provider_total_hits = hits

            if not success:
                fails = getattr(mt, "provider_failures", {})
                fails[provider_name] = fails.get(provider_name, 0) + 1
                mt.provider_failures = fails

            for p in papers:
                p.search_round = state.get("current_round", 1)
            ranked_identity_lists.append(
                [
                    (paper.doi or paper.normalized_title or paper.title).casefold()
                    for paper in papers
                ]
            )
            all_papers.extend(papers)

    state["normalized_papers"] = all_papers
    state["_paper_ranked_identity_lists"] = ranked_identity_lists
    mt.raw_paper_count = len(all_papers)
    sd = getattr(mt, "stage_durations", {})
    sd["search_sources"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Found {len(all_papers)} raw papers")
    return state


async def normalize_and_deduplicate_node(state: dict) -> dict:
    """Normalize and deduplicate papers."""
    _mark_stage(state, "normalize_and_deduplicate")
    logger.info(f"[{state['task_id']}] Normalizing and deduplicating")
    t0 = time.time()

    papers = state.get("normalized_papers", [])
    if not papers:
        state.setdefault("warnings", []).append("No papers to deduplicate")
        return state

    before = len(papers)
    deduped = deduplicate_papers(papers)
    ranked_lists = state.pop("_paper_ranked_identity_lists", [])
    if ranked_lists:
        from app.services.ranking import reciprocal_rank_fusion

        fusion = reciprocal_rank_fusion(ranked_lists)
        deduped.sort(
            key=lambda paper: fusion.get(
                (paper.doi or paper.normalized_title or paper.title).casefold(),
                0.0,
            ),
            reverse=True,
        )
    after = len(deduped)

    state["normalized_papers"] = deduped
    state["discovery_candidates"] = list(deduped[:50])
    state.setdefault("retrieval_diagnostics", {})["discovery_count"] = len(deduped)
    mt = state.get("metrics", TaskMetrics())
    mt.after_dedup_count = after
    sd = getattr(mt, "stage_durations", {})
    sd["normalize_and_deduplicate"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Dedup: {before} -> {after}")
    return state


async def rank_and_select_node(state: dict) -> dict:
    """Rank papers and select the most relevant ones."""
    _mark_stage(state, "rank_and_select")
    logger.info(f"[{state['task_id']}] Ranking and selecting papers")
    t0 = time.time()

    papers = state.get("normalized_papers", [])
    search_plan = state.get("search_plan")
    settings = get_settings()

    if not papers:
        state.setdefault("warnings", []).append("No papers to rank")
        return state

    if not search_plan:
        state["selected_papers"] = papers[:settings.MAX_SELECTED_PAPERS]
        return state

    final_count = state.get("max_papers", settings.MAX_SELECTED_PAPERS)
    candidate_pool_size = max(
        final_count * settings.CANDIDATE_POOL_MULTIPLIER,
        settings.MAX_CANDIDATES_FOR_RERANK,
    )
    candidate_pool_size = min(candidate_pool_size, settings.MAX_CANDIDATES_AFTER_DEDUP)
    prefiltered = deterministic_prefilter(
        papers,
        search_plan,
        max_after_prefilter=candidate_pool_size,
    )
    state["rerank_candidates"] = list(prefiltered)
    state.setdefault("retrieval_diagnostics", {}).update(
        {
            "rerank_candidate_count": len(prefiltered),
            "candidate_pool_multiplier": settings.CANDIDATE_POOL_MULTIPLIER,
        }
    )

    llm = _get_llm_client()
    selected, usage = await llm_rank_papers(
        prefiltered,
        state["original_question"],
        search_plan,
        max_selected=min(
            state.get("max_papers", settings.MAX_SELECTED_PAPERS),
            settings.MAX_SELECTED_PAPERS,
        ),
        llm_client=llm,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=settings.LLM_FAST_ENABLE_THINKING,
    )
    _record_model_usage(state, usage, "paper_rerank")

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = (
        getattr(mt, "llm_call_count", 0) + usage.get("call_count", 1)
    )
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)

    if not selected:
        selected = _fallback_selection(papers, settings.MAX_SELECTED_PAPERS)
        state.setdefault("warnings", []).append("LLM ranking returned no selections, using fallback")

    # Track new papers
    prev_ids = state.get("previous_round_paper_ids", [])
    if prev_ids:
        prev_set = set(prev_ids)
        new_ids = {p.internal_id for p in selected}
        state["new_papers_this_round"] = len(new_ids - prev_set)
    else:
        state["new_papers_this_round"] = len(selected)

    state["selected_papers"] = selected
    state.setdefault("retrieval_diagnostics", {})["selected_count"] = len(selected)
    mt.after_selection_count = len(selected)
    sd = getattr(mt, "stage_durations", {})
    sd["rank_and_select"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Selected {len(selected)} papers")
    return state


async def download_pdfs_node(state: dict) -> dict:
    """Download PDFs for selected papers from open-access sources."""
    _mark_stage(state, "download_pdfs")
    logger.info(f"[{state['task_id']}] Downloading PDFs")
    t0 = time.time()
    settings = get_settings()
    full_text = state.get("enable_full_text", settings.ENABLE_FULL_TEXT)
    backend = state.get("evidence_backend", settings.EVIDENCE_BACKEND)
    if not full_text or backend == "abstract":
        state["_downloaded_pdfs"] = {}
        return state

    papers = state.get("selected_papers", [])
    if not papers:
        return state

    from app.services.pdf_downloader import PDFDownloader
    from app.services.pdf_lifecycle import PDFLifecycleManager

    downloader = PDFDownloader(settings=settings)
    lifecycle = PDFLifecycleManager(settings=settings)

    downloaded = {}
    for paper in papers:
        pdf_url = paper.full_text_url or paper.url
        if not pdf_url:
            state.setdefault("warnings", []).append(
                f"No PDF URL for {paper.internal_id}: {paper.title[:60]}"
            )
            continue

        sha256, file_path, error = await downloader.download(pdf_url)
        if sha256 and file_path:
            downloaded[paper.internal_id] = {
                "sha256": sha256,
                "file_path": file_path,
                "source_url": pdf_url,
            }
            # Register in lifecycle DB
            import os
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            from app.db.database import async_session_factory
            async with async_session_factory() as session:
                await lifecycle.register_download(
                    session=session,
                    sha256=sha256,
                    source_url=pdf_url,
                    file_path=file_path,
                    file_size_bytes=file_size,
                    task_id=state["task_id"],
                    open_access_status="gold" if paper.open_access else "unknown",
                )
        else:
            state.setdefault("warnings", []).append(
                f"PDF download failed for {paper.internal_id}: {error}"
            )

    state["_downloaded_pdfs"] = downloaded
    if state.get("full_text_required"):
        available_ids = set(downloaded)
        unavailable = [
            paper for paper in papers if paper.internal_id not in available_ids
        ]
        if unavailable:
            state.setdefault("warnings", []).append(
                f"full_text_required excluded {len(unavailable)} papers without downloadable PDFs"
            )
        state["selected_papers"] = [
            paper for paper in papers if paper.internal_id in available_ids
        ]
    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["download_pdfs"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(
        f"[{state['task_id']}] Downloaded {len(downloaded)}/{len(papers)} PDFs"
    )
    return state


async def parse_and_chunk_node(state: dict) -> dict:
    """Parse downloaded PDFs and create document chunks with FTS5 index."""
    _mark_stage(state, "parse_and_chunk")
    logger.info(f"[{state['task_id']}] Parsing PDFs and chunking")
    t0 = time.time()
    settings = get_settings()
    full_text = state.get("enable_full_text", settings.ENABLE_FULL_TEXT)
    backend = state.get("evidence_backend", settings.EVIDENCE_BACKEND)
    if not full_text or backend == "abstract":
        state["_parse_results"] = {}
        state["_all_chunks"] = []
        return state

    downloaded = state.get("_downloaded_pdfs", {})
    papers = state.get("selected_papers", [])
    if not downloaded:
        state.setdefault("warnings", []).append("No PDFs to parse")
        return state

    from app.services.pdf_parser import PDFParser
    from app.services.fts_search import save_chunks, init_fts5

    # Ensure FTS5 is initialized
    await init_fts5()

    parser = PDFParser(backend=settings.PDF_PARSER_BACKEND)
    all_chunks = []
    parse_results = {}

    for paper in papers:
        pdf_info = downloaded.get(paper.internal_id)
        if not pdf_info:
            continue

        try:
            result = await parser.parse(
                pdf_path=pdf_info["file_path"],
                paper=paper,
                task_id=state["task_id"],
            )
            parse_results[paper.internal_id] = result

            if result.status.value == "completed":
                # Save all chunks to DB
                await save_chunks(result.parent_chunks)
                await save_chunks(result.child_chunks)
                all_chunks.extend(result.parent_chunks)
                all_chunks.extend(result.child_chunks)
            else:
                state.setdefault("warnings", []).append(
                    f"Parse failed for {paper.internal_id}: {result.error_message}"
                )
        except Exception as e:
            logger.error(f"Parse error for {paper.internal_id}: {e}")
            state.setdefault("warnings", []).append(
                f"Parse error for {paper.internal_id}: {str(e)[:200]}"
            )

    state["_parse_results"] = parse_results
    state["_all_chunks"] = all_chunks
    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["parse_and_chunk"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(
        f"[{state['task_id']}] Parsed {len(parse_results)} PDFs → {len(all_chunks)} chunks"
    )
    return state


async def extract_evidence_node(state: dict) -> dict:
    """Retrieve and verify evidence through the configured EvidenceEngine."""
    _mark_stage(state, "extract_evidence")
    logger.info(f"[{state['task_id']}] Extracting evidence")
    t0 = time.time()

    papers = state.get("selected_papers", [])
    if not papers:
        state.setdefault("warnings", []).append("No papers for evidence extraction")
        return state

    settings = get_settings()
    from app.services.evidence_engine import get_evidence_engine

    engine = get_evidence_engine(
        state.get("evidence_backend", settings.EVIDENCE_BACKEND)
    )
    available = await engine.is_available()
    if not available:
        state.setdefault("errors", []).append(
            f"Configured evidence backend '{engine.name}' is unavailable"
        )
        state["evidence"] = []
        return state

    downloaded = state.get("_downloaded_pdfs", {})
    document_paths = {
        paper_id: info["file_path"]
        for paper_id, info in downloaded.items()
        if info.get("file_path")
    }
    ingestion = await engine.ingest(
        papers,
        document_paths=document_paths,
        task_id=state["task_id"],
    )
    state["evidence_ingestion"] = ingestion
    state.setdefault("warnings", []).extend(ingestion.warnings)

    search_plan = state.get("search_plan")
    sub_questions = [state["original_question"]]
    if search_plan:
        for query in search_plan.queries[:3]:
            candidate = query.rationale.strip() or query.query_string.strip()
            if candidate and candidate not in sub_questions:
                sub_questions.append(candidate)

    passages = []
    seen_passages = set()
    selected_ids = [paper.internal_id for paper in papers]
    for index, sub_question in enumerate(sub_questions):
        retrieved = await engine.retrieve(
            question=state["original_question"],
            sub_question=sub_question,
            paper_ids=selected_ids,
            limit=settings.EVIDENCE_TOP_K,
            task_id=state["task_id"],
        )
        per_paper: dict[str, int] = {}
        filtered = []
        for passage in retrieved:
            count = per_paper.get(passage.paper_id, 0)
            if count >= settings.EVIDENCE_MAX_PER_PAPER:
                continue
            per_paper[passage.paper_id] = count + 1
            filtered.append(passage)
            if passage.passage_id not in seen_passages:
                seen_passages.add(passage.passage_id)
                passages.append(passage)
        extracted = await engine.extract(sub_question, filtered)
        for item in extracted:
            item.sub_question_id = f"sq{index + 1}"
        state.setdefault("_evidence_batches", []).extend(extracted)

    evidence_list = state.pop("_evidence_batches", [])
    state["retrieved_passages"] = passages
    state.setdefault("retrieval_diagnostics", {}).update(
        {
            "passage_count": len(passages),
            "retrieval_backend": engine.name,
        }
    )

    state["evidence"] = evidence_list
    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["extract_evidence"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(
        "[%s] Evidence backend=%s passages=%d evidence=%d",
        state["task_id"],
        engine.name,
        len(passages),
        len(evidence_list),
    )
    return state


async def validate_evidence_node(state: dict) -> dict:
    """Deterministic evidence validation using structured EvidenceCard checks."""
    _mark_stage(state, "validate_evidence")
    logger.info(f"[{state['task_id']}] Validating evidence")
    t0 = time.time()

    from app.services.evidence_validator import validate_all_evidence

    evidence = state.get("evidence", [])
    chunks = state.get("_all_chunks", [])
    chunk_map = {c.chunk_id: c for c in chunks}

    validated, stats = validate_all_evidence(evidence, chunk_map)

    state["evidence"] = validated
    state["_evidence_validated"] = stats["passed"]
    state["_evidence_rejected"] = stats["failed"]
    state["_evidence_validation_details"] = stats["reasons"]

    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["validate_evidence"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if stats["failed"] > 0:
        state.setdefault("warnings", []).append(
            f"Evidence validation: {stats['passed']}/{stats['total']} passed, "
            f"{stats['failed']} rejected"
        )
    logger.info(
        f"[{state['task_id']}] Evidence validated: {stats['passed']} ok, {stats['failed']} rejected"
    )
    return state


async def build_claims_node(state: dict) -> dict:
    """Build claims from verified evidence and run two-phase verification."""
    _mark_stage(state, "build_claims")
    from app.services.claim_evidence import build_claims, evaluate_evidence_quality
    from app.services.claim_verifier import verify_claim

    evidence = state.get("evidence", [])
    chunks = state.get("_all_chunks", [])
    chunk_map = {c.chunk_id: c for c in chunks}

    # Only build claims from VERIFIED evidence
    verified_evidence = [
        ev for ev in evidence
        if ev.verification_status.value == "verified"
    ]
    claims = build_claims(verified_evidence)

    # Run two-phase verification on each claim
    llm = _get_llm_client()
    settings = get_settings()
    validated_claims = []
    for claim in claims:
        bound_evidence = [ev for ev in verified_evidence if ev.evidence_id in claim.evidence_ids]
        claim = await verify_claim(
            claim, bound_evidence, chunk_map=chunk_map,
            llm_client=llm, model=settings.model_fast,
            skip_llm=settings.MOCK_MODE,
        )
        validated_claims.append(claim)

    accepted_claims = [
        claim
        for claim in validated_claims
        if claim.validation_status == "validated"
        and claim.support_status == "supported"
    ]
    rejected_claims = [
        claim for claim in validated_claims if claim not in accepted_claims
    ]
    quality = evaluate_evidence_quality(
        evidence, accepted_claims,
        max_unsupported_important_claims=get_settings().MAX_UNSUPPORTED_IMPORTANT_CLAIMS,
    )
    state["claim_candidates"] = validated_claims
    state["claims"] = accepted_claims
    state["rejected_claims"] = rejected_claims
    state["evidence_quality"] = quality
    if not quality.passed:
        state.setdefault("warnings", []).extend(
            f"Evidence quality: {issue}" for issue in quality.issues
        )
    logger.info(
        f"[{state['task_id']}] Claims: {len(validated_claims)} candidates, "
        f"{len(accepted_claims)} validated, {len(rejected_claims)} withheld"
    )
    return state


async def build_literature_relations_node(state: dict) -> dict:
    """Pre-compute paper-to-paper relations: consensus, contradiction, complementary.

    This runs BEFORE the report node, so the report only reads validated relations,
    never invents its own consensus/disagreement.
    """
    _mark_stage(state, "build_literature_relations")
    logger.info(f"[{state['task_id']}] Building literature relations")
    t0 = time.time()

    evidence = state.get("evidence", [])
    claims = state.get("claims", [])
    papers = state.get("selected_papers", [])
    paper_map = {p.internal_id: p for p in papers}

    # Collect claims per paper with their key topics
    paper_claims: dict[str, list] = {}
    for claim in claims:
        if claim.validation_status != "validated":
            continue
        for pid in getattr(claim, "paper_ids", []):
            paper_claims.setdefault(pid, []).append(claim)

    # Collect evidence per paper with actual numbers
    paper_numbers: dict[str, list] = {}
    import re
    num_pattern = re.compile(r"(\d+(?:\.\d+)?\s*%?)")
    for ev in evidence:
        pid = getattr(ev, "paper_id", "")
        quote = getattr(ev, "evidence_quote", "") or ""
        numbers = num_pattern.findall(quote)
        for n in numbers:
            paper_numbers.setdefault(pid, []).append(
                {"value": n, "context": quote[:200], "page": getattr(ev, "page_start", None)}
            )

    # Build pairwise relations
    relations = []
    paper_ids = list(paper_claims.keys())
    seen_pairs = set()

    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            pi, pj = paper_ids[i], paper_ids[j]
            pair = tuple(sorted([pi, pj]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            ci = paper_claims.get(pi, [])
            cj = paper_claims.get(pj, [])

            # Simple overlap heuristic: shared key terms in claim text
            terms_i = set()
            terms_j = set()
            for c in ci:
                for word in getattr(c, "claim_text", "").lower().split():
                    if len(word) > 3:
                        terms_i.add(word)
            for c in cj:
                for word in getattr(c, "claim_text", "").lower().split():
                    if len(word) > 3:
                        terms_j.add(word)

            overlap = len(terms_i & terms_j)
            union = len(terms_i | terms_j) or 1
            jaccard = overlap / union

            # Determine relation type
            if jaccard > 0.4:
                # Check for same direction or opposite
                relation_type = "complementary"
                # Simple check: if key findings mention similar improvements, it's consensus
                shared = terms_i & terms_j
                if any(w in shared for w in ["improve", "enhance", "increase", "reduce", "degradation", "decline"]):
                    relation_type = "consensus"
                relations.append({
                    "relation_type": relation_type,
                    "paper_ids": [pi, pj],
                    "shared_concepts": list(shared)[:8],
                    "jaccard": round(jaccard, 2),
                    "validated": True,
                    "rationale": f"Shared concepts: {', '.join(list(shared)[:5])}",
                })
            elif jaccard > 0.15:
                relations.append({
                    "relation_type": "complementary",
                    "paper_ids": [pi, pj],
                    "shared_concepts": list(terms_i & terms_j)[:5],
                    "jaccard": round(jaccard, 2),
                    "validated": True,
                    "rationale": "Partial concept overlap — different perspectives",
                })

    state["literature_relations"] = relations
    state["_paper_numbers"] = paper_numbers

    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["build_literature_relations"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt

    logger.info(
        f"[{state['task_id']}] Literature relations: {len(relations)} pairs "
        f"({sum(1 for r in relations if r['relation_type']=='consensus')} consensus)"
    )
    return state


async def assess_gaps_node(state: dict) -> dict:
    """Assess evidence gaps and decide whether supplementary search is needed."""
    _mark_stage(state, "assess_gaps")
    logger.info(f"[{state['task_id']}] Assessing evidence gaps")
    t0 = time.time()

    papers = state.get("selected_papers", [])
    evidence = state.get("evidence", [])
    search_intent = state.get("search_intent")
    used_queries = state.get("queries", [])
    settings = get_settings()

    if not papers:
        state["gap_analysis"] = GapAnalysis(
            needs_supplementary_search=False,
            rationale="No papers to analyze",
        )
        return state

    # Build a terminology-aware view: what we searched vs what we found
    found_titles = "\n".join(
        f"  - {p.title}" for p in papers[:15]
    )
    used_queries_text = "\n".join(
        f"  - {q}" for q in used_queries[:8]
    )

    # Extract key terms from found papers to help LLM spot vocabulary gaps
    from collections import Counter
    import re
    all_titles = " ".join(p.title.lower() for p in papers)
    title_words = re.findall(r"[a-z]{4,}", all_titles)
    top_terms = Counter(title_words).most_common(30)
    found_terms = ", ".join(w for w, _ in top_terms[:30])

    llm = _get_llm_client()

    system_prompt = """You are a research gap analyst with a focus on TERMINOLOGY DIVERSITY.

Your job: look at the papers we FOUND and identify terminology that was MISSING from our search queries.

CRITICAL: The same concept can be expressed with very different words:
- "dialogue history" = "conversation history" = "multi-turn interaction" = "chat context"
- "reasoning reliability" = "logical consistency" = "inference quality" = "reasoning faithfulness"
- "degradation" = "decay" = "decline" = "deterioration" = "gets lost"

If the found papers all use similar terminology (e.g., all mention "memory systems" but none mention "consistency"), generate supplementary queries that use the MISSING terms.

For each gap you identify:
1. What terminology did the current queries use?
2. What terminology appears in related literature but is absent from our results?
3. Generate 2-4 supplementary queries using the MISSING vocabulary.

Respond with a JSON GapAnalysis object. Make supplementary_queries use terms NOT already in the used queries."""

    user_prompt = f"""Research Question: {state['original_question']}
Supplementary rounds done: {state.get('supplementary_rounds_done', 0)}
Max supplementary rounds: {settings.MAX_SEARCH_ROUNDS - 1}

=== Queries we already used ===
{used_queries_text}

=== Top terms in found papers ===
{found_terms}

=== Papers we found (titles only) ===
{found_titles}

=== Task ===
1. What terminology is OVERUSED in the queries (all papers sound similar)?
2. What terminology is MISSING (related concepts not captured)?
3. Generate supplementary queries using DIFFERENT vocabulary.
4. Return a GapAnalysis JSON object."""

    result, usage = await llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=GapAnalysis,
        model=settings.model_strong,
        max_tokens=min(settings.LLM_STRONG_MAX_TOKENS, 2048),
        enable_thinking=settings.LLM_STRONG_ENABLE_THINKING,
    )
    _record_model_usage(state, usage, "gap_analysis")

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)

    if result:
        if state.get("supplementary_rounds_done", 0) >= settings.MAX_SEARCH_ROUNDS - 1:
            result.needs_supplementary_search = False
            result.rationale = (result.rationale or "") + " [Max supplementary rounds reached]"
        if state.get("new_papers_this_round", 0) < 2 and state.get("supplementary_rounds_done", 0) > 0:
            if result.needs_supplementary_search:
                result.needs_supplementary_search = False
                result.rationale = (result.rationale or "") + " [Too few new papers]"
        state["gap_analysis"] = result
    else:
        state["gap_analysis"] = GapAnalysis(
            needs_supplementary_search=False,
            rationale="Gap analysis failed, proceeding to report",
        )
        state.setdefault("warnings", []).append("Gap analysis LLM call failed")

    sd = getattr(mt, "stage_durations", {})
    sd["assess_gaps"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Gap analysis: supplementary={state['gap_analysis'].needs_supplementary_search}")
    return state


async def supplementary_search_node(state: dict) -> dict:
    """Execute supplementary search with new queries."""
    _mark_stage(state, "supplementary_search")
    logger.info(f"[{state['task_id']}] Supplementary search round {state.get('supplementary_rounds_done', 0) + 1}")

    gap_analysis = state.get("gap_analysis")
    if not gap_analysis or not gap_analysis.supplementary_queries:
        return state

    state["previous_round_paper_ids"] = [p.internal_id for p in state.get("selected_papers", [])]
    state["queries"] = gap_analysis.supplementary_queries
    state["supplementary_rounds_done"] = state.get("supplementary_rounds_done", 0) + 1
    state["current_round"] = state.get("current_round", 0) + 1

    providers = _get_providers()
    available_providers = await _check_availability(providers)

    all_papers: List[Paper] = []
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    mt = state.get("metrics", TaskMetrics())

    async def search_provider(provider, query: str):
        try:
            papers = await provider.search(
                query, year_from=year_from, year_to=year_to,
                search_intent=state.get("search_intent"),
            )
            total_hits = getattr(provider, "last_total_hits", 0)
            return provider.name, papers, total_hits
        except Exception as e:
            logger.error(f"Provider {provider.name} failed: {e}")
            return provider.name, [], 0

    for query in state["queries"]:
        tasks = [search_provider(p, query) for p in available_providers]
        results = await asyncio.gather(*tasks)
        for provider_name, papers, total_hits in results:
            reqs = getattr(mt, "provider_requests", {})
            reqs[provider_name] = reqs.get(provider_name, 0) + 1
            mt.provider_requests = reqs

            res = getattr(mt, "provider_results", {})
            res[provider_name] = res.get(provider_name, 0) + len(papers)
            mt.provider_results = res

            hits = getattr(mt, "provider_total_hits", {})
            hits[provider_name] = max(hits.get(provider_name, 0), total_hits)
            mt.provider_total_hits = hits

            for p in papers:
                p.search_round = state.get("current_round", 2)
            all_papers.extend(papers)

    existing = state.get("normalized_papers", [])
    combined = existing + all_papers
    state["normalized_papers"] = deduplicate_papers(combined)
    mt.after_dedup_count = len(state["normalized_papers"])
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Supplementary search found {len(all_papers)} new raw papers")
    return state


async def synthesize_report_node(state: dict) -> dict:
    """Generate a Deep Research style report with literature landscape and research gaps."""
    _mark_stage(state, "synthesize_report")
    logger.info(f"[{state['task_id']}] Synthesizing report")
    t0 = time.time()

    papers = state.get("selected_papers", [])
    validated_claims = [
        claim
        for claim in state.get("claims", [])
        if claim.validation_status == "validated"
        and claim.support_status == "supported"
    ]
    search_plan = state.get("search_plan")
    settings = get_settings()
    target_language = state.get("report_language", "zh-CN")
    language_name = "English" if target_language == "en" else "Simplified Chinese"

    if not papers:
        state["report"] = "# 研究报告\n\n未找到相关文献，无法生成报告。"
        state.setdefault("warnings", []).append("No papers available for report generation")
        return state

    # 8K context safeguard
    MAX_PAPERS_IN_PROMPT = 12
    supported_paper_ids = {
        paper_id
        for claim in validated_claims
        for paper_id in claim.paper_ids
    }
    supported_papers = [
        paper for paper in papers if paper.internal_id in supported_paper_ids
    ]
    papers_for_report = supported_papers[:MAX_PAPERS_IN_PROMPT]
    if len(supported_papers) > MAX_PAPERS_IN_PROMPT:
        state.setdefault("warnings", []).append(
            f"Truncated supported papers from {len(supported_papers)} "
            f"to {MAX_PAPERS_IN_PROMPT} for report"
        )
    state["report_paper_ids"] = [p.internal_id for p in papers_for_report]

    if _budget_exceeded(state):
        state["report"] = _build_fallback_report(state)
        state.setdefault("warnings", []).append(
            "Cost budget reached; used deterministic ValidatedClaims-only report"
        )
        return state

    # Metadata is only used for labels and references. Factual prose comes from
    # ValidatedClaims exclusively.
    paper_summaries = []
    for i, paper in enumerate(papers_for_report):
        venue_year = f"{paper.venue or 'Unknown venue'}, {paper.publication_year or 'n.d.'}"
        summary = (
            f"[P{i + 1}] {paper.title}\n"
            f"    {venue_year}"
        )
        paper_summaries.append(summary)

    paper_list_text = "\n\n".join(paper_summaries)
    paper_markers = {
        paper.internal_id: f"P{index + 1}"
        for index, paper in enumerate(papers_for_report)
    }
    claims_text = "\n".join(
        (
            f"- claim_id={claim.claim_id}; claim={claim.claim_text}; "
            f"citations={[paper_markers[pid] for pid in claim.paper_ids if pid in paper_markers]}; "
            f"type={claim.claim_type}; confidence={claim.confidence}; "
            f"scope={claim.scope or 'unspecified'}"
        )
        for claim in validated_claims
        if any(pid in paper_markers for pid in claim.paper_ids)
    )
    if not claims_text:
        state["report"] = _build_fallback_report(state)
        state.setdefault("warnings", []).append(
            "No validated claims were available; emitted a metadata-only report"
        )
        return state

    if settings.REPORT_GENERATION_MODE != "llm":
        state["report"] = _build_fallback_report(state)
        mt = state.get("metrics", TaskMetrics())
        sd = getattr(mt, "stage_durations", {})
        sd["synthesize_report"] = time.time() - t0
        mt.stage_durations = sd
        state["metrics"] = mt
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return state

    # Search info
    search_info = f"检索轮次: {state.get('current_round', 0) + 1}"
    if search_plan:
        search_info += f"\n核心概念: {', '.join(search_plan.core_concepts[:6])}"
        if search_plan.year_from or search_plan.year_to:
            search_info += f"\n年份范围: {search_plan.year_from or 'any'} – {search_plan.year_to or 'any'}"

    llm = _get_llm_client()

    system_prompt = f"""
<role>
You are an evidence-grounded academic research synthesizer.
Write a {language_name} research report for domain experts.
Your purpose is faithful synthesis, not producing a complete-looking report.
</role>

<source_of_truth>
You may use validated_claims as the only source of factual content.
paper_metadata may be used only for titles, years, venues and citations.

Do not use outside knowledge.
Do not use abstracts, raw passages or unvalidated evidence.
A paper citation proves provenance, not entailment.
</source_of_truth>

<report_structure>
Generate the following sections:

## Executive Summary
3-5 concise bullet points. Each cites at least one paper.

## 1. 研究问题与范围
Refined research question, search scope, data sources, year range, key concepts.

## 2. 文献全景
- 研究脉络: How has this area evolved?
- 主题聚类: 3-5 thematic clusters with key papers and contributions
- 关键论文表: Columns: Paper, Year, Core Contribution, Evidence Strength

## 3. 核心发现
Organize by theme. For each: what we know, evidence strength, key results.
Include quantitative results only when an exact supporting excerpt is available.

## 4. 共识与争议
- 共识: Requires at least 2 independent papers
- 分歧: Incompatible findings, assumptions, or definitions
- 方法学比较: Different approaches and trade-offs

## 5. 研究缺口与未来方向
- 证据缺口, 方法学缺口, 新兴方向, 具体建议
- Historical trends and future directions must be explicitly supported,
  or labeled "基于当前检索结果的推断"

## 6. 参考文献
Auto-generated — use [P#] markers only.
</report_structure>

<claim_policy>
Every factual statement must be directly entailed by its cited evidence.

The evidence must support the same:
- subject
- direction
- scope
- metric
- comparison
- conclusion

If support is incomplete, omit the statement.
Do not rescue unsupported claims using words such as:
"可能", "一定程度上", "表明", "显著", "主要", "普遍".

Quantitative claims are allowed only when an exact validated evidence
record contains the number, metric, comparison and paper ID.
Copy numbers exactly.
</claim_policy>

<paper_role_policy>
Respect the supplied paper_role field.

Do not treat:
- mitigation methods as direct evidence of a phenomenon
- personalization studies as reasoning-reliability studies
- safety studies as general reasoning studies
- benchmarks as mitigation methods
- conceptual frameworks as validated empirical findings
- mixed memory as multimodal memory
</paper_role_policy>

<meta_claim_policy>
Claims about the literature as a whole require explicit support.

Consensus requires at least two independent directly supporting papers.
Disagreement requires incompatible findings, assumptions or definitions.
Different methods are not automatically a disagreement.

Historical trends, research gaps and future directions must either:
1. be explicitly supported by cited evidence, or
2. be labeled "基于当前检索结果的推断".

Never invent thresholds, missing experiments, computational costs,
causal mechanisms or field-wide trends.
</meta_claim_policy>

<metadata_policy>
Copy years, venues and paper roles exactly from structured metadata.
Do not generate or correct them.

Use the supplied evidence_strength value.
If unavailable, write "未评估".
</metadata_policy>

<writing_policy>
Write in {language_name}. Translate the prescribed section headings when needed.
Prefer precise, narrow statements over broad summaries.
If evidence is insufficient, explicitly state the limitation.
Do not repeat the reference list because it is generated separately.
</writing_policy>

<final_check>
Before returning the report, silently inspect every factual sentence:

1. Is it supported by supplied evidence?
2. Does the citation support the entire sentence?
3. Is the paper role correct?
4. Is every number copied exactly?
5. Is a single-paper result incorrectly called consensus?
6. Is an inference clearly labeled?

Delete any sentence that fails a check.
</final_check>"""

    # Build literature relations text for §4 (共识与争议) — pre-computed, not LLM-invented
    relations = state.get("literature_relations", [])
    consensus_pairs = [r for r in relations if r.get("relation_type") == "consensus"]
    complementary_pairs = [r for r in relations if r.get("relation_type") == "complementary"]
    relations_text = ""
    if consensus_pairs:
        relations_text += "=== PRE-COMPUTED CONSENSUS (use these, do not invent others) ===\n"
        for r in consensus_pairs[:10]:
            relations_text += (
                f"- {r['paper_ids']}: {r.get('rationale','')} "
                f"(shared: {r.get('shared_concepts',[])}, jaccard={r.get('jaccard',0)})\n"
            )
    if complementary_pairs:
        relations_text += "\n=== PRE-COMPUTED COMPLEMENTARY PERSPECTIVES ===\n"
        for r in complementary_pairs[:10]:
            relations_text += (
                f"- {r['paper_ids']}: {r.get('rationale','')} "
                f"(shared: {r.get('shared_concepts',[])}, jaccard={r.get('jaccard',0)})\n"
            )
    if not relations:
        relations_text = (
            "§4 共识与争议 MUST be: '当前证据不足以确定系统性共识或争议。'\n"
            "Do NOT list any paper pairs as consensus or disagreement."
        )

    user_prompt = f"""Research Question: {state['original_question']}

{search_info}

=== SELECTED PAPERS (only these can be cited) ===
{paper_list_text}

=== VALIDATED CLAIMS (the only factual source) ===
{claims_text}

{relations_text}

=== TASK ===
Generate a comprehensive Deep Research report following the specified structure.
- Use the PRE-COMPUTED CONSENSUS and COMPLEMENTARY PERSPECTIVES for §4.
  If consensus_text is empty, write "当前证据不足以确定共识".
- Quantitative statements are permitted only when already present in a validated claim.
- Do NOT invent your own consensus, disagreement, or numbers.
- Do NOT generate a reference list placeholder like [P1]...[P12].
  Only use [P#] in citations within the text body.
Cite papers using [P#] markers ONLY from the list above."""

    # 8K budget with large prompt: limit completion to 2048
    report_max_tokens = min(settings.LLM_STRONG_MAX_TOKENS, 2048)
    report_text, usage = await llm.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=settings.model_strong,
        temperature=0.3,
        max_tokens=report_max_tokens,
        enable_thinking=settings.LLM_STRONG_ENABLE_THINKING,
    )
    _record_model_usage(state, usage, "report")

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)

    if report_text:
        # Only include papers actually shown to the LLM in the reference list
        ref_entries = build_reference_entries(papers_for_report)
        import re
        ref_pattern = re.compile(
            r"(#{1,3}\s*(?:参考文献|References|参考文獻).*?\n).*",
            re.DOTALL | re.IGNORECASE,
        )
        # Strip model-generated reference placeholder like "[P1] [P2] ... [P12]"
        # Also handle it appearing after "## 6. 参考文献" heading
        report_text = re.sub(
            r"\n\s*(\[P\d+\][,\s]*){2,}\n", "\n", report_text
        )
        # Also strip the entire "## 6. 参考文献" section (model-generated, we replace it)
        report_text = re.sub(
            r"\n#{1,3}\s*\d*\.?\s*参考文献\s*\n\s*(\[P\d+\][,\s]*)*\s*\n",
            "\n", report_text
        )
        if ref_pattern.search(report_text):
            report_text = ref_pattern.sub(r"\1\n" + ref_entries, report_text)
        else:
            report_text += f"\n\n## 参考文献\n\n{ref_entries}"

        # Post-generation number audit: flag any dubious percentages/metrics
        _audit_numbers_in_report(report_text, [], state)

        state["report"] = report_text
    else:
        state["report"] = _build_fallback_report(state)
        state.setdefault("warnings", []).append("Report generation LLM call failed")

    sd = getattr(mt, "stage_durations", {})
    sd["synthesize_report"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Report generated")
    return state


def _build_fallback_report(state: dict) -> str:
    """Build a conservative ValidatedClaims-only report without an LLM."""
    papers_by_id = {
        paper.internal_id: paper for paper in state.get("selected_papers", [])
    }
    report_ids = state.get("report_paper_ids", [])
    papers = [papers_by_id[paper_id] for paper_id in report_ids if paper_id in papers_by_id]
    if not papers:
        papers = state.get("selected_papers", [])
    claims = [
        claim
        for claim in state.get("claims", [])
        if claim.validation_status == "validated"
        and claim.support_status == "supported"
    ]
    marker_by_id = {
        paper.internal_id: f"P{index + 1}" for index, paper in enumerate(papers)
    }
    section_priority = (
        ("result", "conclusion", "finding", "discussion", "abstract"),
        ("evaluation", "experiment", "analysis", "method"),
        ("introduction", "background"),
        ("related work", "reference", "dataset"),
    )

    def claim_priority(claim):
        section = (claim.section_id or "").casefold()
        for priority, keywords in enumerate(section_priority):
            if any(keyword in section for keyword in keywords):
                return priority
        return 2

    claims.sort(key=lambda claim: (claim_priority(claim), -claim.confidence))
    selected_claims = []
    per_paper: dict[str, int] = {}
    for claim in claims:
        primary_paper = next(
            (paper_id for paper_id in claim.paper_ids if paper_id in marker_by_id),
            None,
        )
        if not primary_paper or per_paper.get(primary_paper, 0) >= 4:
            continue
        per_paper[primary_paper] = per_paper.get(primary_paper, 0) + 1
        selected_claims.append(claim)
        if len(selected_claims) >= 12:
            break

    if state.get("report_language") == "en":
        report = f"""# Research Report: {state['original_question']}

## Research Question

{state['original_question']}

## Search Scope and Method

- Search rounds: {state.get('current_round', 0) + 1}
- Year range: {state.get('year_from', 'any')} - {state.get('year_to', 'any')}
- Included papers: {len(papers)}

## Core Findings

"""
        if selected_claims:
            for claim in selected_claims:
                markers = [
                    marker_by_id[paper_id]
                    for paper_id in claim.paper_ids
                    if paper_id in marker_by_id
                ]
                if markers:
                    report += f"- {claim.claim_text} [{' '.join(markers)}]\n"
        else:
            report += "No validated claims are available; no factual synthesis was generated.\n"
        report += (
            "\n## Limitations\n\nOnly claims passing EvidenceCard and Claim "
            "validation are shown; unvalidated material was omitted.\n\n"
            "## References\n\n"
        )
        report += build_reference_entries(papers)
        return report

    report = f"""# 研究报告: {state['original_question']}

## 研究问题

{state['original_question']}

## 检索范围与方法

- 检索轮次: {state.get('current_round', 0) + 1}
- 年份范围: {state.get('year_from', 'any')} - {state.get('year_to', 'any')}
- 纳入文献: {len(papers)} 篇

## 核心发现

"""
    if selected_claims:
        for claim in selected_claims:
            markers = [
                marker_by_id[paper_id]
                for paper_id in claim.paper_ids
                if paper_id in marker_by_id
            ]
            if markers:
                report += f"- {claim.claim_text} [{' '.join(markers)}]\n"
    else:
        report += "当前没有通过验证的 Claim，因而不生成事实性总结。\n"

    report += "\n## 研究局限\n\n"
    report += "本报告仅呈现通过 EvidenceCard 与 Claim 验证的内容；未验证信息已省略。\n\n"

    report += "## 参考文献\n\n"
    report += build_reference_entries(papers)
    return report


async def validate_citations_node(state: dict) -> dict:
    """Validate all citations in the report."""
    _mark_stage(state, "validate_citations")
    logger.info(f"[{state['task_id']}] Validating citations")
    t0 = time.time()

    report = state.get("report", "")
    papers_by_id = {
        paper.internal_id: paper for paper in state.get("selected_papers", [])
    }
    report_ids = state.get("report_paper_ids", [])
    papers = [
        papers_by_id[paper_id]
        for paper_id in report_ids
        if paper_id in papers_by_id
    ]
    if not papers:
        papers = state.get("selected_papers", [])

    validation = validate_citations(report, papers)

    # Deeper check: for each cited paper, verify claim→evidence→chunk chain
    evidence = state.get("evidence", [])
    claims = state.get("claims", [])
    paper_has_verified_evidence = set()
    for ev in evidence:
        if getattr(ev, "verification_status", None) == "verified":
            paper_has_verified_evidence.add(ev.paper_id)

    # Build claim→paper mapping from claims
    claim_paper_map = {}
    for claim in claims:
        for pid in getattr(claim, "paper_ids", []):
            claim_paper_map.setdefault(pid, []).append(claim.claim_id)

    # Check each cited paper has evidence backing
    for marker_idx in range(1, len(papers) + 1):
        if marker_idx <= len(papers):
            paper = papers[marker_idx - 1]
            pid = paper.internal_id
            if pid not in paper_has_verified_evidence:
                validation.issues.append(
                    f"[P{marker_idx}] cited but has no verified evidence"
                )
            if pid not in claim_paper_map:
                validation.issues.append(
                    f"[P{marker_idx}] cited but has no associated claim"
                )

    # Re-evaluate validity
    validation.is_valid = (
        len(validation.orphan_citations) == 0 and len(validation.issues) == 0
    )

    if not validation.is_valid:
        logger.warning(f"Citation validation found {len(validation.issues)} issues")
        # Auto-fix only basic orphan citations; deeper issues are reported
        if validation.orphan_citations:
            fixed_report = auto_fix_citations(report, papers)
            validation = validate_citations(fixed_report, papers)
            if validation.is_valid:
                state["report"] = fixed_report
                validation.fixed = True
        if not validation.is_valid:
            state.setdefault("warnings", []).extend(
                f"Citation: {issue}" for issue in validation.issues
            )

    state["citation_validation"] = validation
    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["validate_citations"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Citation validation: valid={validation.is_valid}")
    return state


def _audit_numbers_in_report(report: str, evidence: list, state: dict) -> None:
    """Post-hoc audit: flag numbers in the report that don't appear in any evidence quote."""
    import re

    # Collect all verifiable text from evidence
    evidence_texts = set()
    for ev in evidence:
        if getattr(ev, "evidence_quote", None):
            evidence_texts.add(ev.evidence_quote.strip())
        for f in (getattr(ev, "key_findings", None) or []):
            evidence_texts.add(f.strip())

    # Find numbers with context in the report (e.g., "12%", "15-20%", "48%", "0.85")
    number_pattern = re.compile(
        r"(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*%?|\d+(?:\.\d+)?\s*%)"
    )
    found_numbers = number_pattern.findall(report)

    # Check each number against evidence texts
    suspicious = []
    for num in found_numbers:
        num_clean = num.replace(" ", "")
        found_in_evidence = any(num_clean in t for t in evidence_texts)
        if not found_in_evidence:
            # Also try without %
            num_no_pct = num_clean.replace("%", "")
            found_in_evidence = any(num_no_pct in t for t in evidence_texts)
        if not found_in_evidence:
            suspicious.append(num)

    if suspicious:
        state.setdefault("warnings", []).append(
            f"Number audit: {len(suspicious)} metrics ({', '.join(suspicious[:8])}) "
            f"not found in evidence quotes — may be fabricated"
        )
        logger.warning(
            f"[{state.get('task_id', '?')}] Suspicious numbers: {suspicious[:8]}"
        )


async def finalize_node(state: dict) -> dict:
    """Finalize the research task and save a complete run record."""
    _mark_stage(state, "finalize")
    logger.info(f"[{state['task_id']}] Finalizing")
    state["status"] = TaskStatus.COMPLETED.value
    mt = state.get("metrics", TaskMetrics())
    mt.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _runtime_progress[state["task_id"]] = state

    # Save complete run record for audit and debugging
    await _save_run_record(state)

    # Clean up cached instances
    _reset_cached_instances()
    return state


async def _save_run_record(state: dict) -> None:
    """Export full pipeline data to storage/tasks/{task_id}/run_record.json."""
    import json
    import os
    from pathlib import Path

    task_id = state["task_id"]
    run_dir = Path("storage/tasks") / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Serialize key artifacts (skip non-serializable objects)
    def _safe(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return str(obj)
        return obj

    record = {
        "task_id": task_id,
        "original_question": state.get("original_question", ""),
        "status": state.get("status", ""),
        "research_depth": state.get("research_depth", "standard"),
        "search_rounds": state.get("current_round", 0) + 1,
        "created_at": state.get("created_at", ""),
        "updated_at": state.get("updated_at", ""),
        # Search plan
        "search_plan": _safe(state.get("search_plan")),
        "queries": state.get("queries", []),
        # Paper counts
        "total_papers_found": len(state.get("normalized_papers", [])),
        "papers_selected": len(state.get("selected_papers", [])),
        # Selected papers with provenance
        "selected_papers": [
            {
                "internal_id": p.internal_id,
                "title": p.title,
                "year": p.publication_year,
                "venue": p.venue,
                "doi": p.doi,
                "sources": p.source_names,
                "relevance_score": p.relevance_score,
                "relevance_reason": getattr(p, "relevance_reason", None),
            }
            for p in state.get("selected_papers", [])
        ],
        # Downloaded PDFs
        "downloaded_pdfs": {
            pid: {"sha256": info["sha256"][:16] + "...", "source_url": info.get("source_url", "")}
            for pid, info in state.get("_downloaded_pdfs", {}).items()
        },
        # Parse results
        "parse_results": {
            pid: {
                "parser": r.parser_name,
                "pages": r.num_pages,
                "sections": r.num_sections,
                "parent_chunks": len(r.parent_chunks),
                "child_chunks": len(r.child_chunks),
            }
            for pid, r in state.get("_parse_results", {}).items()
        },
        # Evidence with verification status
        "evidence": [
            {
                "paper_id": getattr(ev, "paper_id", ""),
                "chunk_id": getattr(ev, "chunk_id", None),
                "section": getattr(ev, "section_title", None),
                "page": getattr(ev, "page_start", None),
                "evidence_quote": getattr(ev, "evidence_quote", None),
                "verification_status": getattr(ev, "verification_status", "unchecked"),
                "verification_reason": getattr(ev, "verification_reason", None),
                "evidence_level": getattr(ev, "evidence_level", "paraphrase"),
            }
            for ev in state.get("evidence", [])
        ],
        # Claims
        "claims": [_safe(c) for c in state.get("claims", [])],
        "rejected_claims": [_safe(c) for c in state.get("rejected_claims", [])],
        # Literature relations (consensus/contradiction/complementary)
        "literature_relations": state.get("literature_relations", []),
        # Gap analysis
        "gap_analysis": _safe(state.get("gap_analysis")),
        # Metrics
        "metrics": _safe(state.get("metrics", {})),
        # Citation validation
        "citation_validation": _safe(state.get("citation_validation")),
        # Warnings & errors
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
        # Report
        "report_length": len(state.get("report", "")),
    }

    record_path = run_dir / "run_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    logger.info(f"[{task_id}] Run record saved to {record_path}")

    # Also save the full report as a standalone Markdown file
    report_text = state.get("report", "")
    if report_text:
        report_path = run_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")
        logger.info(f"[{task_id}] Report saved to {report_path}")


# ---- Routing functions ----

def should_supplement(state: dict) -> str:
    """Decide whether to perform supplementary search or proceed to report."""
    gap = state.get("gap_analysis")
    if gap and gap.needs_supplementary_search:
        supp_done = state.get("supplementary_rounds_done", 0)
        max_rounds = state.get("max_rounds", 3)
        if supp_done < max_rounds - 1:
            return "supplementary_search"
    return "build_claims"


# ---- Build the graph ----

def build_research_graph() -> StateGraph:
    """Build and compile the LangGraph research workflow."""
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("plan_queries", plan_queries_node)
    workflow.add_node("search_sources", search_sources_node)
    workflow.add_node("normalize_and_deduplicate", normalize_and_deduplicate_node)
    workflow.add_node("rank_and_select", rank_and_select_node)
    workflow.add_node("download_pdfs", download_pdfs_node)
    workflow.add_node("parse_and_chunk", parse_and_chunk_node)
    workflow.add_node("extract_evidence", extract_evidence_node)
    workflow.add_node("validate_evidence", validate_evidence_node)
    workflow.add_node("assess_gaps", assess_gaps_node)
    workflow.add_node("supplementary_search", supplementary_search_node)
    workflow.add_node("build_claims", build_claims_node)
    workflow.add_node("build_literature_relations", build_literature_relations_node)
    workflow.add_node("synthesize_report", synthesize_report_node)
    workflow.add_node("validate_citations", validate_citations_node)
    workflow.add_node("finalize", finalize_node)

    # Define the flow
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "plan_queries")
    workflow.add_edge("plan_queries", "search_sources")
    workflow.add_edge("search_sources", "normalize_and_deduplicate")
    workflow.add_edge("normalize_and_deduplicate", "rank_and_select")
    workflow.add_edge("rank_and_select", "download_pdfs")
    workflow.add_edge("download_pdfs", "parse_and_chunk")
    workflow.add_edge("parse_and_chunk", "extract_evidence")
    workflow.add_edge("extract_evidence", "validate_evidence")
    workflow.add_edge("validate_evidence", "assess_gaps")

    # Conditional branching
    workflow.add_conditional_edges(
        "assess_gaps",
        should_supplement,
        {
            "supplementary_search": "supplementary_search",
            "build_claims": "build_claims",
        },
    )

    # After supplementary search, normalize + dedup before re-ranking
    workflow.add_edge("supplementary_search", "normalize_and_deduplicate")

    # Final steps
    workflow.add_edge("build_claims", "build_literature_relations")
    workflow.add_edge("build_literature_relations", "synthesize_report")
    workflow.add_edge("synthesize_report", "validate_citations")
    workflow.add_edge("validate_citations", "finalize")
    workflow.add_edge("finalize", END)

    # Compile without checkpointer to avoid serialization issues
    graph = workflow.compile()

    return graph


# ---- Running the workflow ----


async def run_research(state: dict) -> dict:
    """Run the complete research workflow.

    Args:
        state: Initial state dict with at least task_id and original_question.

    Returns:
        Final state dict with report, papers, evidence, etc.
    """
    _reset_cached_instances()

    # Ensure DB tables + FTS5 exist (needed for PDF cache, chunks)
    from app.db.database import init_db
    await init_db()

    graph = build_research_graph()

    config = {"configurable": {"thread_id": state["task_id"]}}

    try:
        final_state = await graph.ainvoke(state, config)
        for key, value in final_state.items():
            state[key] = value
        state["status"] = TaskStatus.COMPLETED.value
    except Exception as e:
        logger.error(f"Research workflow failed: {e}", exc_info=True)
        state["status"] = TaskStatus.FAILED.value
        state.setdefault("errors", []).append(f"Workflow error: {str(e)}")

    _reset_cached_instances()
    return state
