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
    "assess_gaps": 76,
    "supplementary_search": 55,
    "build_claims": 84,
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
        from app.providers.arxiv import ArxivProvider
        _cached_providers = [
            OpenAlexProvider(settings=settings),
            SemanticScholarProvider(settings=settings),
            ArxivProvider(settings=settings),
        ]
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
            all_papers.extend(papers)

    state["normalized_papers"] = all_papers
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
    after = len(deduped)

    state["normalized_papers"] = deduped
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

    prefiltered = deterministic_prefilter(papers, search_plan)

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


async def build_claims_node(state: dict) -> dict:
    """Build report claims only from verified evidence and apply quality gates."""
    _mark_stage(state, "build_claims")
    from app.services.claim_evidence import build_claims, evaluate_evidence_quality

    evidence = state.get("evidence", [])
    claims = build_claims(evidence)
    quality = evaluate_evidence_quality(
        evidence,
        claims,
        max_unsupported_important_claims=get_settings().MAX_UNSUPPORTED_IMPORTANT_CLAIMS,
    )
    state["claims"] = claims
    state["evidence_quality"] = quality
    if not quality.passed:
        state.setdefault("warnings", []).extend(
            f"Evidence quality: {issue}" for issue in quality.issues
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
        max_tokens=settings.LLM_STRONG_MAX_TOKENS,
        enable_thinking=settings.LLM_STRONG_ENABLE_THINKING,
    )

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
    """Generate the final Chinese Markdown research report."""
    _mark_stage(state, "synthesize_report")
    logger.info(f"[{state['task_id']}] Synthesizing report")
    t0 = time.time()

    papers = state.get("selected_papers", [])
    evidence = state.get("evidence", [])
    claims = state.get("claims", [])
    gap_analysis = state.get("gap_analysis")
    settings = get_settings()

    if not papers:
        state["report"] = "# 研究报告\n\n未找到相关文献，无法生成报告。"
        state.setdefault("warnings", []).append("No papers available for report generation")
        return state

    # Safeguard for 8K context: report generation receives verified claims only.
    MAX_PAPERS_IN_PROMPT = 12

    supported_paper_ids = {
        paper_id
        for claim in claims
        if claim.support_status == "supported"
        for paper_id in claim.paper_ids
    }
    eligible_papers = [
        paper for paper in papers if paper.internal_id in supported_paper_ids
    ]
    papers_for_report = eligible_papers[:MAX_PAPERS_IN_PROMPT]
    if not papers_for_report:
        state["report"] = _build_fallback_report(state)
        state["report_paper_ids"] = [
            paper.internal_id for paper in papers[:MAX_PAPERS_IN_PROMPT]
        ]
        state.setdefault("warnings", []).append(
            "No verified claims available; generated fallback report"
        )
        return state
    state["report_paper_ids"] = [
        paper.internal_id for paper in papers_for_report
    ]
    if len(eligible_papers) > MAX_PAPERS_IN_PROMPT:
        state.setdefault("warnings", []).append(
            f"Truncated supported papers from {len(eligible_papers)} to {MAX_PAPERS_IN_PROMPT} for report prompt"
        )

    paper_summaries = []
    for i, paper in enumerate(papers_for_report):
        paper_claims = [
            claim for claim in claims if paper.internal_id in claim.paper_ids
        ][:3]
        claim_text = "\n".join(
            f"    - Claim {claim.claim_id}: {claim.claim_text}"
            for claim in paper_claims
        ) or "    - No verified claim available"
        source_locations = []
        for ev in evidence:
            if ev.paper_id != paper.internal_id:
                continue
            location = ev.section_title or "Abstract"
            if ev.page_start:
                location += f", p.{ev.page_start}"
            source_locations.append(location)
        summary = (
            f"[P{i + 1}] {paper.title} ({paper.publication_year or 'n.d.'})\n"
            f"    Verified claims:\n{claim_text}\n"
            f"    Source locations: {', '.join(source_locations[:3]) or 'N/A'}\n"
        )
        paper_summaries.append(summary)

    paper_list_text = "\n\n".join(paper_summaries)

    gap_text = ""
    if gap_analysis and gap_analysis.gaps:
        gap_text = "\n".join(
            f"- {g.sub_question} (severity: {g.severity})\n  Missing: {g.what_is_missing}"
            for g in gap_analysis.gaps
        )

    llm = _get_llm_client()

    language = state.get("report_language", "zh-CN")
    language_instruction = (
        "Write in Simplified Chinese (zh-CN)."
        if language == "zh-CN"
        else "Write in English."
    )
    system_prompt = f"""You are a senior research analyst writing a comprehensive research report.

Requirements:
1. {language_instruction}
2. Structure the report with clear sections
3. Use [P1], [P2], etc. for in-text citations - ONLY cite papers provided to you
4. NEVER add factual claims, references, findings, or data not present in the verified claims
5. Be honest about evidence strength and gaps
6. If a requested aspect has no verified claim, state that evidence is insufficient

Sections:
- 研究问题 (Research Question)
- 检索范围与方法 (Search Scope and Method)
- 核心发现 (Core Findings)
- 研究之间的一致与分歧 (Agreements and Disagreements)
- 研究局限与证据缺口 (Limitations and Evidence Gaps)
- 对后续研究的建议 (Recommendations)
- 参考文献 (References) - use exact title, venue, year from provided data

CRITICAL: Only cite papers from the provided list. Each [P#] must correspond to a real paper."""

    user_prompt = f"""Research Question: {state['original_question']}

Search rounds: {state.get('current_round', 0) + 1}
Papers selected: {len(papers)}

Papers and verified claims:
{paper_list_text}

Evidence Gaps:
{gap_text if gap_text else 'None'}

Generate a comprehensive Chinese research report. Cite papers using [P#] markers."""

    report_text, usage = await llm.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=settings.model_strong,
        temperature=0.3,
        max_tokens=settings.LLM_STRONG_MAX_TOKENS,
        enable_thinking=settings.LLM_STRONG_ENABLE_THINKING,
    )

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)

    if report_text:
        # Only include papers actually shown to the LLM in the reference list
        ref_entries = build_reference_entries(papers_for_report)
        import re
        ref_pattern = re.compile(r"(##\s*参考文献\s*\n).*", re.DOTALL)
        if ref_pattern.search(report_text):
            report_text = ref_pattern.sub(r"\1\n" + ref_entries, report_text)
        else:
            report_text += f"\n\n## 参考文献\n\n{ref_entries}"
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
    """Build a basic report without LLM."""
    papers_by_id = {
        paper.internal_id: paper for paper in state.get("selected_papers", [])
    }
    report_ids = state.get("report_paper_ids", [])
    papers = [papers_by_id[paper_id] for paper_id in report_ids if paper_id in papers_by_id]
    if not papers:
        papers = state.get("selected_papers", [])
    evidence = state.get("evidence", [])
    gap_analysis = state.get("gap_analysis")

    report = f"""# 研究报告: {state['original_question']}

## 研究问题

{state['original_question']}

## 检索范围与方法

- 检索轮次: {state.get('current_round', 0) + 1}
- 年份范围: {state.get('year_from', 'any')} - {state.get('year_to', 'any')}
- 纳入文献: {len(papers)} 篇

## 核心发现

"""
    for i, paper in enumerate(papers):
        ev = next((e for e in evidence if e.paper_id == paper.internal_id), None)
        report += f"### [{i + 1}] {paper.title}\n\n"
        if paper.abstract:
            report += f"{paper.abstract[:500]}...\n\n"
        if ev and ev.key_findings:
            report += f"关键发现: {'; '.join(ev.key_findings)}\n\n"

    if gap_analysis and gap_analysis.gaps:
        report += "## 研究局限与证据缺口\n\n"
        for gap in gap_analysis.gaps:
            report += f"- **{gap.sub_question}** ({gap.severity}): {gap.what_is_missing}\n"
        report += "\n"

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

    if not validation.is_valid:
        logger.warning(f"Citation validation failed, attempting auto-fix")
        fixed_report = auto_fix_citations(report, papers)
        validation = validate_citations(fixed_report, papers)
        if validation.is_valid:
            state["report"] = fixed_report
            validation.fixed = True
        else:
            state.setdefault("warnings", []).append(f"Citation issues could not be auto-fixed: {validation.issues}")

    state["citation_validation"] = validation
    mt = state.get("metrics", TaskMetrics())
    sd = getattr(mt, "stage_durations", {})
    sd["validate_citations"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Citation validation: valid={validation.is_valid}")
    return state


async def finalize_node(state: dict) -> dict:
    """Finalize the research task."""
    _mark_stage(state, "finalize")
    logger.info(f"[{state['task_id']}] Finalizing")
    state["status"] = TaskStatus.COMPLETED.value
    mt = state.get("metrics", TaskMetrics())
    mt.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _runtime_progress[state["task_id"]] = state

    # Clean up cached instances
    _reset_cached_instances()
    return state


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
    workflow.add_node("assess_gaps", assess_gaps_node)
    workflow.add_node("supplementary_search", supplementary_search_node)
    workflow.add_node("build_claims", build_claims_node)
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
    workflow.add_edge("extract_evidence", "assess_gaps")

    # Conditional branching
    workflow.add_conditional_edges(
        "assess_gaps",
        should_supplement,
        {
            "supplementary_search": "supplementary_search",
            "build_claims": "build_claims",
        },
    )

    # After supplementary search, re-rank
    workflow.add_edge("supplementary_search", "rank_and_select")

    # Final steps
    workflow.add_edge("build_claims", "synthesize_report")
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
