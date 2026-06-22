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


def _get_providers():
    """Get provider instances based on config. Cached at module level."""
    global _cached_providers
    if _cached_providers is not None:
        return _cached_providers
    settings = get_settings()
    if settings.MOCK_MODE or not settings.has_llm_key:
        from app.providers.mock_provider import MockProvider
        logger.info("Using mock providers (mock mode or no API keys)")
        _cached_providers = [MockProvider(settings=settings)]
    else:
        from app.providers.openalex import OpenAlexProvider
        from app.providers.semantic_scholar import SemanticScholarProvider
        from app.providers.arxiv import ArxivProvider
        from app.providers.crossref import CrossrefProvider
        _cached_providers = [
            OpenAlexProvider(settings=settings),
            SemanticScholarProvider(settings=settings),
            ArxivProvider(settings=settings),
            CrossrefProvider(settings=settings),
        ]
    return _cached_providers


def _reset_cached_instances():
    """Reset cached instances (for testing)."""
    global _cached_providers, _cached_llm_client
    _cached_providers = None
    _cached_llm_client = None


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
    logger.info(f"[{state['task_id']}] Planning queries")
    t0 = time.time()

    llm = _get_llm_client()
    settings = get_settings()

    system_prompt = """You are an expert academic research strategist. Given a research question, create a structured search plan.

Analyze the research question and produce:
1. A refined research topic statement
2. Core concepts (3-6)
3. Synonyms for each concept
4. 3-6 structured search queries (in English, optimized for academic search APIs)
5. Appropriate year range if applicable
6. Relevant academic domains
7. Inclusion and exclusion criteria

Each query should be a concise string suitable for API search (not natural language questions).
Focus on Boolean-like keyword combinations that work well with academic search engines."""

    user_prompt = f"""Research Question: {state['original_question']}

Year constraints: from {state.get('year_from') or 'any'} to {state.get('year_to') or 'any'}

Generate a structured search plan in JSON format."""

    from app.models.search_plan import SearchPlan as SP
    result, usage = await llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=SP,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=settings.LLM_FAST_ENABLE_THINKING,
    )

    if result:
        state["search_plan"] = result
        state["queries"] = result.all_query_strings()
    else:
        logger.warning("Search plan generation failed, using fallback")
        from app.models.search_plan import SearchQuery, InclusionExclusionCriteria
        fallback = SP(
            research_topic=state["original_question"],
            core_concepts=state["original_question"].split(),
            queries=[SearchQuery(query_string=state["original_question"], rationale="Fallback query", keywords=[])],
            criteria=InclusionExclusionCriteria(),
        )
        state["search_plan"] = fallback
        state["queries"] = fallback.all_query_strings()
        state.setdefault("warnings", []).append("Search plan generation failed, using basic fallback query")

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)
    sd = getattr(mt, "stage_durations", {})
    sd["plan_queries"] = time.time() - t0
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Generated {len(state['queries'])} queries")
    return state


async def search_sources_node(state: dict) -> dict:
    """Execute searches across all available providers in parallel."""
    logger.info(f"[{state['task_id']}] Searching sources (round {state['current_round']})")
    t0 = time.time()

    providers = _get_providers()
    available_providers = await _check_availability(providers)
    if not available_providers:
        state.setdefault("errors", []).append("No providers available for search")
        return state

    queries = state.get("queries", [])
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    mt = state.get("metrics", TaskMetrics())

    all_papers: List[Paper] = []

    async def search_provider(provider, query: str):
        try:
            papers = await provider.search(query, year_from=year_from, year_to=year_to)
            return provider.name, papers, True
        except Exception as e:
            logger.error(f"Provider {provider.name} failed for query '{query[:50]}...': {e}")
            state.setdefault("warnings", []).append(f"Provider {provider.name} failed: {str(e)[:200]}")
            return provider.name, [], False

    for query in queries:
        tasks = [search_provider(p, query) for p in available_providers]
        results = await asyncio.gather(*tasks)

        for provider_name, papers, success in results:
            reqs = getattr(mt, "provider_requests", {})
            reqs[provider_name] = reqs.get(provider_name, 0) + 1
            mt.provider_requests = reqs

            results_count = getattr(mt, "provider_results", {})
            results_count[provider_name] = results_count.get(provider_name, 0) + len(papers)
            mt.provider_results = results_count

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
        max_selected=settings.MAX_SELECTED_PAPERS,
        llm_client=llm,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=settings.LLM_FAST_ENABLE_THINKING,
    )

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
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


async def extract_evidence_node(state: dict) -> dict:
    """Extract structured evidence from selected papers."""
    logger.info(f"[{state['task_id']}] Extracting evidence")
    t0 = time.time()

    papers = state.get("selected_papers", [])
    if not papers:
        state.setdefault("warnings", []).append("No papers for evidence extraction")
        return state

    evidence_list: List[ExtractedEvidence] = []
    llm = _get_llm_client()
    settings = get_settings()

    paper_summaries = []
    for paper in papers:
        has_abstract = bool(paper.abstract)
        summary = (
            f"Paper ID: {paper.internal_id}\n"
            f"Title: {paper.title}\n"
            f"Year: {paper.publication_year or 'N/A'}\n"
            f"Venue: {paper.venue or 'N/A'}\n"
            f"Abstract: {(paper.abstract or 'NO ABSTRACT AVAILABLE')[:500]}\n"
        )
        paper_summaries.append(summary)

    paper_list_text = "\n---\n".join(paper_summaries)

    system_prompt = """You are a research assistant extracting structured evidence from academic papers.

For each paper, extract:
- research_question: What research question does this paper address?
- method: What methodology is used?
- dataset_or_participants: What data or participants?
- key_findings: List the key findings
- limitations: List limitations
- relevance_to_user_question: How does this relate to the user's research question?
- evidence_quote: Verbatim quote from the abstract. If no abstract is available, set this to null.

CRITICAL: The evidence_quote MUST be a verbatim excerpt from the provided abstract. Do NOT fabricate quotes.

Respond with a JSON object with an "evidence" array."""

    user_prompt = f"""User's Research Question: {state['original_question']}

Papers to analyze:
{paper_list_text}

Return JSON: {{"evidence": [ExtractedEvidence objects]}}"""

    from pydantic import BaseModel

    class EvidenceOutput(BaseModel):
        evidence: List[ExtractedEvidence]

    result, usage = await llm.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=EvidenceOutput,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=settings.LLM_FAST_ENABLE_THINKING,
    )

    mt = state.get("metrics", TaskMetrics())
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)

    if result and result.evidence:
        evidence_list = result.evidence
    else:
        logger.warning("Evidence extraction failed, using fallback")
        for paper in papers:
            evidence_list.append(ExtractedEvidence(
                paper_id=paper.internal_id,
                relevance_to_user_question="Extracted from title and abstract",
                evidence_quote=paper.abstract[:200] if paper.abstract else None,
                key_findings=[paper.title] if paper.title else [],
            ))
        state.setdefault("warnings", []).append("Evidence extraction LLM call failed, using basic fallback")

    state["evidence"] = evidence_list
    sd = getattr(mt, "stage_durations", {})
    sd["extract_evidence"] = time.time() - t0
    mt.stage_durations = sd
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(f"[{state['task_id']}] Extracted evidence from {len(evidence_list)} papers")
    return state


async def assess_gaps_node(state: dict) -> dict:
    """Assess evidence gaps and decide whether supplementary search is needed."""
    logger.info(f"[{state['task_id']}] Assessing evidence gaps")
    t0 = time.time()

    evidence = state.get("evidence", [])
    settings = get_settings()

    if not evidence:
        state["gap_analysis"] = GapAnalysis(
            needs_supplementary_search=False,
            rationale="No evidence to analyze",
        )
        return state

    evidence_text = "\n".join(
        f"Paper {ev.paper_id}: {ev.relevance_to_user_question or 'N/A'}\n"
        f"  Findings: {'; '.join(ev.key_findings[:3]) if ev.key_findings else 'N/A'}"
        for ev in evidence
    )

    llm = _get_llm_client()

    system_prompt = """You are a research gap analyst. Identify gaps and decide if supplementary search is needed.

Consider:
1. Which aspects of the research question are adequately covered?
2. Which aspects lack evidence?
3. Are there important sub-questions unaddressed?

IMPORTANT: Do NOT recommend supplementary search if max rounds have been reached.

Respond with a JSON GapAnalysis object."""

    user_prompt = f"""Research Question: {state['original_question']}
Supplementary rounds done: {state.get('supplementary_rounds_done', 0)}
Max supplementary rounds: {settings.MAX_SEARCH_ROUNDS - 1}

Current Evidence:
{evidence_text}

Return a GapAnalysis JSON object."""

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
            papers = await provider.search(query, year_from=year_from, year_to=year_to)
            return provider.name, papers
        except Exception as e:
            logger.error(f"Provider {provider.name} failed: {e}")
            return provider.name, []

    for query in state["queries"]:
        tasks = [search_provider(p, query) for p in available_providers]
        results = await asyncio.gather(*tasks)
        for provider_name, papers in results:
            reqs = getattr(mt, "provider_requests", {})
            reqs[provider_name] = reqs.get(provider_name, 0) + 1
            mt.provider_requests = reqs

            res = getattr(mt, "provider_results", {})
            res[provider_name] = res.get(provider_name, 0) + len(papers)
            mt.provider_results = res

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
    logger.info(f"[{state['task_id']}] Synthesizing report")
    t0 = time.time()

    papers = state.get("selected_papers", [])
    evidence = state.get("evidence", [])
    gap_analysis = state.get("gap_analysis")
    settings = get_settings()

    if not papers:
        state["report"] = "# 研究报告\n\n未找到相关文献，无法生成报告。"
        state.setdefault("warnings", []).append("No papers available for report generation")
        return state

    # Safeguard for 8K context: limit papers in prompt and truncate abstracts.
    # At 4096 completion tokens, ~4000 tokens remain for the prompt.
    # Each paper summary ~150-250 tokens; 12 papers ≈ 2400 prompt tokens.
    MAX_PAPERS_IN_PROMPT = 12
    ABSTRACT_CHARS = 200

    papers_for_report = papers[:MAX_PAPERS_IN_PROMPT]
    if len(papers) > MAX_PAPERS_IN_PROMPT:
        state.setdefault("warnings", []).append(
            f"Truncated papers from {len(papers)} to {MAX_PAPERS_IN_PROMPT} for report prompt (8K context budget)"
        )

    paper_summaries = []
    for i, paper in enumerate(papers_for_report):
        ev = next((e for e in evidence if e.paper_id == paper.internal_id), None)
        findings = "; ".join(ev.key_findings[:2]) if ev and ev.key_findings else "N/A"
        abstract_text = (paper.abstract or "N/A")
        if len(abstract_text) > ABSTRACT_CHARS:
            abstract_text = abstract_text[:ABSTRACT_CHARS] + "..."
        summary = (
            f"[P{i + 1}] {paper.title} ({paper.publication_year or 'n.d.'})\n"
            f"    Abstract: {abstract_text}\n"
            f"    Key findings: {findings}\n"
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

    system_prompt = """You are a senior research analyst writing a comprehensive Chinese-language research report.

Requirements:
1. Write in Chinese (Simplified, zh-CN)
2. Structure the report with clear sections
3. Use [P1], [P2], etc. for in-text citations - ONLY cite papers provided to you
4. NEVER fabricate references, findings, or data not present in the provided materials
5. Be honest about evidence strength and gaps

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

Papers:
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
    logger.info(f"[{state['task_id']}] Validating citations")
    t0 = time.time()

    report = state.get("report", "")
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
    logger.info(f"[{state['task_id']}] Finalizing")
    state["status"] = TaskStatus.COMPLETED.value
    mt = state.get("metrics", TaskMetrics())
    mt.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["metrics"] = mt
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

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
    return "synthesize_report"


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
    workflow.add_node("extract_evidence", extract_evidence_node)
    workflow.add_node("assess_gaps", assess_gaps_node)
    workflow.add_node("supplementary_search", supplementary_search_node)
    workflow.add_node("synthesize_report", synthesize_report_node)
    workflow.add_node("validate_citations", validate_citations_node)
    workflow.add_node("finalize", finalize_node)

    # Define the flow
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "plan_queries")
    workflow.add_edge("plan_queries", "search_sources")
    workflow.add_edge("search_sources", "normalize_and_deduplicate")
    workflow.add_edge("normalize_and_deduplicate", "rank_and_select")
    workflow.add_edge("rank_and_select", "extract_evidence")
    workflow.add_edge("extract_evidence", "assess_gaps")

    # Conditional branching
    workflow.add_conditional_edges(
        "assess_gaps",
        should_supplement,
        {
            "supplementary_search": "supplementary_search",
            "synthesize_report": "synthesize_report",
        },
    )

    # After supplementary search, re-rank
    workflow.add_edge("supplementary_search", "rank_and_select")

    # Final steps
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
