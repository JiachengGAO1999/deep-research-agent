"""Quick Research LangGraph subgraph.

Quick mode uses Tavily Search + Extract as the primary content source,
bypassing PDF download, FTS5, and strict EvidenceCard/Claim verification.

Flow:
  classify_question → quick_plan_queries → tavily_search → quick_select_sources
  → tavily_extract → build_research_notes → quick_assess_coverage
  → (loop: quick_supplementary_search → tavily_search → ... → quick_assess_coverage)
  → build_comparison_matrix → synthesize_quick_report → lightweight_citation_check
  → finalize
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.quick_research import (
    AnswerSchema,
    CoverageAssessment,
    ComparisonRow,
    ExtractedWebSource,
    PlannedQuery,
    QuestionType,
    QuickCitationCheckResult,
    ResearchNote,
    SourceType,
    SOURCE_TYPE_PRIORITY,
    WebSearchResult,
)
from app.models.task import TaskMetrics, TaskStatus

logger = logging.getLogger(__name__)


# ---- Progress markers ----

STAGE_PROGRESS_Q = {
    "classify_question": 3,
    "quick_plan_queries": 8,
    "tavily_search": 18,
    "quick_select_sources": 26,
    "tavily_extract": 36,
    "build_research_notes": 50,
    "quick_assess_coverage": 58,
    "quick_supplementary_search": 48,
    "build_comparison_matrix": 68,
    "synthesize_quick_report": 82,
    "lightweight_citation_check": 92,
    "quick_finalize": 100,
}


def _mark_stage(state: dict, stage: str) -> None:
    state["current_stage"] = stage
    state["progress_percent"] = STAGE_PROGRESS_Q.get(stage, 0)


def _record_model_usage(state: dict, usage: dict, purpose: str) -> None:
    if not usage:
        return
    mt = state.get("metrics", TaskMetrics())
    mt.model_calls.append(
        {
            "purpose": purpose,
            "model": usage.get("model", "unknown"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_seconds": usage.get("latency_seconds"),
        }
    )
    settings = get_settings()
    mt.estimated_cost_usd += (
        float(usage.get("prompt_tokens", 0) or 0) * settings.LLM_INPUT_COST_PER_1M
        + float(usage.get("completion_tokens", 0) or 0)
        * settings.LLM_OUTPUT_COST_PER_1M
    ) / 1_000_000
    mt.llm_call_count = getattr(mt, "llm_call_count", 0) + 1
    mt.llm_tokens_used = getattr(mt, "llm_tokens_used", 0) + usage.get("total_tokens", 0)
    state["metrics"] = mt


def _get_llm():
    from app.workflow.graph import _get_llm_client

    return _get_llm_client()


# ---- Academic domain heuristics (lightweight, no hard blocklist) ----

_ACADEMIC_DOMAIN_PATTERNS = [
    re.compile(r"(arxiv\.org|doi\.org|scholar\.google\.)", re.I),
    re.compile(r"(acm\.org|ieee\.org|springer\.com|elsevier\.com)", re.I),
    re.compile(r"(nature\.com|science\.org|cell\.com|lancet\.com)", re.I),
    re.compile(r"(neurips\.cc|icml\.cc|aclweb\.org|aaai\.org)", re.I),
    re.compile(r"(openreview\.net|proceedings\.mlr\.press)", re.I),
    re.compile(r"(semanticscholar\.org|openalex\.org)", re.I),
    re.compile(r"\.edu\b", re.I),
    re.compile(r"(github\.com|gitlab\.com|huggingface\.co)", re.I),
]

_LOW_QUALITY_PATTERNS = [
    re.compile(r"(reddit\.com|quora\.com|medium\.com|substack\.com)", re.I),
    re.compile(r"(twitter\.com|x\.com|facebook\.com|linkedin\.com)", re.I),
    re.compile(r"(youtube\.com|tiktok\.com)", re.I),
    re.compile(r"search\.", re.I),
    re.compile(r"(amazon\.com|ebay\.com|etsy\.com)", re.I),
]


def _classify_source_type(domain: str, url: str, title: str, snippet: str) -> SourceType:
    """Lightweight source type classification based on domain patterns."""
    dl = domain.lower()
    u = url.lower()
    t = (title + " " + snippet).lower()

    if "arxiv.org" in dl:
        return SourceType.PREPRINT
    if any(k in dl for k in ["openreview.net", "mlr.press", "neurips.cc", "icml.cc",
                              "aclweb.org", "aaai.org", "aclrollingreview.org"]):
        return SourceType.PAPER_OFFICIAL
    if any(k in dl for k in ["acm.org", "ieee.org", "springer.com",
                              "elsevier.com", "nature.com", "science.org",
                              "cell.com", "lancet.com", "tandfonline.com",
                              "wiley.com", "sagepub.com"]):
        if "proceedings" in u or "doi" in u or "abstract" in u:
            return SourceType.PAPER_OFFICIAL
        return SourceType.PUBLISHER
    if "github.com" in dl or "gitlab.com" in dl or "huggingface.co" in dl:
        return SourceType.AUTHOR_PROJECT
    if ".edu" in dl and any(k in t for k in ["paper", "research", "lab", "group"]):
        return SourceType.AUTHOR_PROJECT
    if ".edu" in dl:
        return SourceType.INSTITUTIONAL
    if any(k in t for k in ["survey", "review", "systematic review", "meta-analysis"]):
        return SourceType.REVIEW
    if any(k in t for k in ["paper", "research", "study", "method",
                              "arxiv", "preprint", "doi",
                              "experiment", "benchmark"]):
        return SourceType.SECONDARY
    return SourceType.UNKNOWN


def _is_low_quality(domain: str, url: str) -> bool:
    for pat in _LOW_QUALITY_PATTERNS:
        if pat.search(domain) or pat.search(url):
            return True
    return False


def _is_academic_preferred(domain: str) -> int:
    """Return a preference score (higher = more academic)."""
    score = 0
    dl = domain.lower()
    for i, pat in enumerate(_ACADEMIC_DOMAIN_PATTERNS):
        if pat.search(dl):
            score += len(_ACADEMIC_DOMAIN_PATTERNS) - i
    return score


# ---- Node: classify_question ----

_ANSWER_SCHEMA_SYSTEM = """You are a research methodologist. Given a research question,
produce an operationalised AnswerSchema that guides search and extraction.

Analyse the question and determine:
- question_type: descriptive | comparative | causal | trend | methodological | research_landscape
- subject: the primary phenomenon or technology being studied
- comparison_target: what is being compared (for comparative questions)
- outcome: the outcome/dependent variable of interest
- required_dimensions: what dimensions must be covered to answer this question
  (e.g. technique, baseline, task, dataset, metric, reported_result, limitations)
- inclusion_guidance: what types of sources or evidence should be included
- exclusion_guidance: what should be excluded

For COMPARATIVE questions, required_dimensions MUST include:
technique, baseline, task, dataset, metric, reported_result, limitations

If the question type is ambiguous, default to 'descriptive' and include basic dimensions.
Respond with valid JSON matching the AnswerSchema."""


async def classify_question_node(state: dict) -> dict:
    """Classify the research question into an AnswerSchema."""
    _mark_stage(state, "classify_question")
    t0 = time.time()

    llm = _get_llm()
    settings = get_settings()

    result, usage = await llm.generate_structured(
        system_prompt=_ANSWER_SCHEMA_SYSTEM,
        user_prompt=f"Research Question: {state['original_question']}\n\nReturn an AnswerSchema JSON object.",
        output_model=AnswerSchema,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=False,
    )
    _record_model_usage(state, usage, "classify_question")

    if result is None:
        logger.warning("AnswerSchema generation failed, using conservative default")
        result = AnswerSchema.conservative_default(state["original_question"])
        state.setdefault("warnings", []).append(
            "Question classification failed, using conservative defaults"
        )

    state["answer_schema"] = result
    logger.info(
        f"[{state['task_id']}] Question classified: type={result.question_type.value}"
    )
    return state


# ---- Node: quick_plan_queries ----

_PLAN_QUERIES_SYSTEM = """You are a search strategist. Given an AnswerSchema, generate complementary
search queries that together cover all required dimensions.

Rules:
- Maximum 6 queries per round
- Queries should be concise (5-12 words), suitable for web search engines
- Queries MUST use diverse terminology — do NOT repeat the same keywords
- For comparative questions, ensure coverage of:
  - Technique/method categories
  - Empirical comparisons or benchmarks
  - Evaluation metrics and results
  - Limitations and failure cases
  - Review/survey sources
- Each query MUST have a specific purpose
- Include year range terms where appropriate (e.g. \"2023 2024 recent\")
- Write queries in English (academic search is English-dominant)

Return a JSON object with a 'queries' array of {query_id, query, purpose} objects."""


class _QueryList(BaseModel):
    queries: list[PlannedQuery] = Field(default_factory=list)


async def quick_plan_queries_node(state: dict) -> dict:
    """Generate PlannedQuery list from AnswerSchema for the current round."""
    _mark_stage(state, "quick_plan_queries")
    t0 = time.time()

    answer_schema = state.get("answer_schema")
    llm = _get_llm()
    settings = get_settings()
    round_idx = state.get("quick_search_round", 0)

    schema_text = ""
    if answer_schema:
        schema_text = f"""
Question Type: {answer_schema.question_type.value}
Subject: {answer_schema.subject}
Comparison Target: {answer_schema.comparison_target or 'N/A'}
Outcome: {answer_schema.outcome or 'N/A'}
Required Dimensions: {', '.join(answer_schema.required_dimensions)}
Inclusion: {'; '.join(answer_schema.inclusion_guidance) if answer_schema.inclusion_guidance else 'N/A'}"""

    # If supplementary round, include missing dimensions
    prev_coverage = state.get("coverage_assessment")
    coverage_hint = ""
    if prev_coverage and prev_coverage.missing_dimensions:
        coverage_hint = (
            f"\nPrevious round missed these dimensions: {', '.join(prev_coverage.missing_dimensions)}\n"
            f"Underrepresented areas: {', '.join(prev_coverage.underrepresented_areas)}\n"
            f"Generate queries specifically targeting the missing dimensions."
        )

    user_prompt = f"""Research Question: {state['original_question']}
Year range: from {state.get('year_from') or 'any'} to {state.get('year_to') or 'any'}
{schema_text}
{coverage_hint}

Generate up to {get_settings().QUICK_MAX_QUERIES_PER_ROUND} search queries.
Ensure queries use diverse terminology and cover all required dimensions."""

    result, usage = await llm.generate_structured(
        system_prompt=_PLAN_QUERIES_SYSTEM,
        user_prompt=user_prompt,
        output_model=_QueryList,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=False,
    )
    _record_model_usage(state, usage, "quick_plan_queries")

    queries: list[PlannedQuery] = []
    if result and result.queries:
        queries = result.queries[: settings.QUICK_MAX_QUERIES_PER_ROUND]
        for q in queries:
            q.round_index = round_idx
    else:
        # Fallback: use the question itself plus some variants
        q = state["original_question"]
        queries = [
            PlannedQuery(query=q, purpose="primary", round_index=round_idx),
            PlannedQuery(
                query=f"{state.get('answer_schema', AnswerSchema.conservative_default(q)).subject} empirical comparison evaluation",
                purpose="empirical_comparison",
                round_index=round_idx,
            ),
            PlannedQuery(
                query=f"{state.get('answer_schema', AnswerSchema.conservative_default(q)).subject} benchmark survey",
                purpose="review",
                round_index=round_idx,
            ),
        ]
        state.setdefault("warnings", []).append(
            "Query planning failed, using fallback queries"
        )

    # Dedup against previous round queries
    prev_queries = {
        q.query.strip().lower()
        for q in (state.get("prev_round_queries") or [])
    }
    filtered = []
    for q in queries:
        if q.query.strip().lower() not in prev_queries:
            filtered.append(q)
    queries = filtered[: settings.QUICK_MAX_QUERIES_PER_ROUND]

    state["quick_queries"] = queries
    state.setdefault("all_quick_queries", []).extend(queries)
    logger.info(
        f"[{state['task_id']}] Quick planned {len(queries)} queries (round {round_idx})"
    )
    return state


# ---- Node: tavily_search ----

async def tavily_search_node(state: dict) -> dict:
    """Execute Tavily Search for each planned query, dedup results."""
    _mark_stage(state, "tavily_search")
    t0 = time.time()

    queries: list[PlannedQuery] = state.get("quick_queries", [])
    if not queries:
        state.setdefault("warnings", []).append("No queries for Tavily search")
        return state

    settings = get_settings()
    round_idx = state.get("quick_search_round", 0)

    # Check if Tavily is configured
    if not settings.has_tavily_key and not settings.MOCK_MODE:
        state.setdefault("errors", []).append(
            "TAVILY_API_KEY is not configured. Quick Research mode requires Tavily."
        )
        return state

    from app.clients import _normalise_url, _extract_domain

    mt = state.get("metrics", TaskMetrics())

    if settings.MOCK_MODE:
        # Return mock search results
        mock_results = _mock_tavily_search_results(queries, round_idx)
        _merge_search_results(state, mock_results, round_idx)
        mt.provider_requests = getattr(mt, "provider_requests", {})
        mt.provider_requests["tavily"] = (
            mt.provider_requests.get("tavily", 0) + len(queries)
        )
        mt.provider_results = getattr(mt, "provider_results", {})
        mt.provider_results["tavily"] = len(state.get("web_search_results", []))
        state["metrics"] = mt
        logger.info(
            f"[{state['task_id']}] Mock Tavily search: {len(queries)} queries → "
            f"{len(state.get('web_search_results', []))} total results"
        )
        return state

    from app.clients import get_tavily_client
    client = get_tavily_client()

    async def _search_one(q: PlannedQuery) -> list[dict]:
        try:
            data = await client.search(
                query=q.query,
                max_results=settings.TAVILY_MAX_RESULTS_PER_QUERY,
                include_answer=False,
            )
            results = data.get("results", [])
            # Tag each result with query info
            for r in results:
                r["_query"] = q.query
                r["_query_purpose"] = q.purpose
                r["_round_index"] = round_idx
            return results
        except Exception as e:
            logger.error(f"Tavily search failed for '{q.query[:60]}...': {e}")
            state.setdefault("warnings", []).append(
                f"Tavily search failed for '{q.query[:60]}...': {str(e)[:200]}"
            )
            return []

    # Parallel search
    tasks = [_search_one(q) for q in queries]
    all_results = await asyncio.gather(*tasks)
    raw_results = [r for batch in all_results for r in batch]

    # Convert to WebSearchResult with dedup
    seen_urls: dict[str, WebSearchResult] = {}
    for r in raw_results:
        url = _normalise_url(r.get("url", ""))
        if not url:
            continue
        domain = _extract_domain(url)

        if url in seen_urls:
            existing = seen_urls[url]
            qp = r.get("_query_purpose", "")
            if qp and qp not in existing.query_purposes:
                existing.query_purposes.append(qp)
            continue

        ws = WebSearchResult(
            query=r.get("_query", ""),
            query_purpose=r.get("_query_purpose", ""),
            title=r.get("title", ""),
            url=url,
            snippet=r.get("content") or r.get("snippet", ""),
            score=r.get("score"),
            published_date=r.get("published_date"),
            domain=domain,
            round_index=r.get("_round_index", round_idx),
            query_purposes=[r.get("_query_purpose", "")] if r.get("_query_purpose") else [],
        )
        seen_urls[url] = ws

    # Merge with previous round results
    prev_results = state.get("web_search_results", [])
    prev_urls = {r.url for r in prev_results}
    new_results = [ws for url, ws in seen_urls.items() if url not in prev_urls]

    # Merge with previous round results
    prev_results = state.get("web_search_results", [])
    prev_urls = {r.url for r in prev_results}
    new_results = [ws for url, ws in seen_urls.items() if url not in prev_urls]

    all_web_results = prev_results + new_results
    state["web_search_results"] = all_web_results
    state["new_web_results"] = new_results

    mt.provider_requests = getattr(mt, "provider_requests", {})
    mt.provider_requests["tavily"] = (
        mt.provider_requests.get("tavily", 0) + len(queries)
    )
    mt.provider_results = getattr(mt, "provider_results", {})
    mt.provider_results["tavily"] = len(all_web_results)

    logger.info(
        f"[{state['task_id']}] Tavily search: {len(queries)} queries → "
        f"{len(raw_results)} raw → {len(new_results)} new unique URLs "
        f"({len(all_web_results)} total)"
    )
    return state


def _mock_tavily_search_results(
    queries: list[PlannedQuery], round_idx: int
) -> list[dict]:
    """Generate realistic mock Tavily search results."""
    mock_domains = [
        ("arxiv.org", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", 0.95),
        ("arxiv.org", "CRAG: Corrective Retrieval Augmented Generation", 0.92),
        ("arxiv.org", "RAG-Fusion: A Comprehensive Study", 0.88),
        ("aclanthology.org", "The Impact of Retrieval Quality on LLM Factuality", 0.90),
        ("arxiv.org", "Reranking for Retrieval-Augmented Generation", 0.87),
        ("arxiv.org", "REALM: Retrieval-Augmented Language Model Pre-Training", 0.85),
        ("openreview.net", "Atlas: Retrieval Augmented Language Models", 0.83),
        ("arxiv.org", "RAG vs Fine-tuning: A Systematic Comparison", 0.80),
        ("arxiv.org", "Limitations of Current RAG Evaluation Benchmarks", 0.78),
        ("arxiv.org", "Faithful Reasoning Using Large Language Models", 0.76),
        ("aclanthology.org", "Benchmarking Hallucination in LLMs", 0.82),
        ("arxiv.org", "RAG in Non-English Languages: A Preliminary Study", 0.74),
        ("github.com", "langchain-ai/rag-from-scratch", 0.65),
        ("arxiv.org", "Improving RAG with Dynamic Retrieval Scheduling", 0.79),
        ("arxiv.org", "Decomposed Prompting for RAG Systems", 0.75),
        ("semanticscholar.org", "RAG Survey 2024: Methods and Metrics", 0.84),
        ("arxiv.org", "Grounded Generation: A New Framework", 0.72),
        ("arxiv.org", "Multi-hop RAG for Complex Question Answering", 0.70),
    ]

    results = []
    for i, (domain, title, score) in enumerate(mock_domains):
        query_idx = i % len(queries)
        q = queries[query_idx]
        results.append({
            "url": f"https://{domain}/abs/{10000 + i}",
            "title": title,
            "content": f"Abstract and content for: {title}. This paper presents empirical results on RAG techniques and hallucination reduction.",
            "score": score,
            "published_date": f"202{2 + (i % 5)}-0{(i % 12) + 1:02d}-{15 + i:02d}",
            "_query": q.query,
            "_query_purpose": q.purpose,
            "_round_index": round_idx,
        })
    return results


def _merge_search_results(
    state: dict, results: list[dict], round_idx: int
) -> None:
    """Merge raw search results into state as WebSearchResult objects."""
    from app.clients import _normalise_url, _extract_domain

    seen_urls: dict[str, WebSearchResult] = {}
    for r in state.get("web_search_results", []):
        seen_urls[r.url] = r

    for r in results:
        url = _normalise_url(r.get("url", ""))
        if not url:
            continue
        domain = _extract_domain(url)

        if url in seen_urls:
            existing = seen_urls[url]
            qp = r.get("_query_purpose", "")
            if qp and qp not in existing.query_purposes:
                existing.query_purposes.append(qp)
            continue

        ws = WebSearchResult(
            query=r.get("_query", ""),
            query_purpose=r.get("_query_purpose", ""),
            title=r.get("title", ""),
            url=url,
            snippet=r.get("content") or r.get("snippet", ""),
            score=r.get("score"),
            published_date=r.get("published_date"),
            domain=domain,
            round_index=r.get("_round_index", round_idx),
            query_purposes=[r.get("_query_purpose", "")] if r.get("_query_purpose") else [],
        )
        seen_urls[url] = ws

    all_results = list(seen_urls.values())
    prev_count = len(state.get("web_search_results", []))
    state["web_search_results"] = all_results
    state["new_web_results"] = all_results[prev_count:]


def _mock_tavily_extract_results(
    urls: list[str], selected: list[dict]
) -> dict[str, dict]:
    """Generate mock Tavily Extract results for selected sources."""
    mock_content = """## Abstract

This paper investigates {topic} in the context of retrieval-augmented generation (RAG).
We conduct empirical evaluations comparing multiple RAG variants across standard benchmarks
including Natural Questions, TriviaQA, and PubHealth.

## Methods

We evaluate standard RAG, Self-RAG, and reranking-based RAG approaches. Each method is
assessed on factuality, citation accuracy, and hallucination rate. We use both automatic
metrics (FactScore, AlignScore) and human evaluation.

## Results

Our experiments show that methods incorporating retrieval self-reflection consistently
outperform standard RAG by 5-15% on factuality metrics. Reranking improves factual
precision by 10-18%. The combination of self-reflection and reranking achieves the
best performance, reducing hallucination by up to 20% compared to baseline RAG.

## Limitations

The study is limited to English open-domain QA tasks. Additional computational cost
from reflection and reranking modules is significant (20-40% latency increase).
Cross-domain and multi-lingual performance remains unaddressed."""

    results = {}
    for i, url in enumerate(urls):
        s = selected[i] if i < len(selected) else None
        title = s["result"].title if s else f"Source {i}"
        topic = title.lower().replace("self-rag", "").replace("rag", "").strip().strip(":")[:50]
        content = mock_content.replace("{topic}", topic if topic else "retrieval quality")
        results[url] = {
            "url": url,
            "raw_content": content,
        }
    return results


# ---- Node: quick_select_sources ----

async def quick_select_sources_node(state: dict) -> dict:
    """Score, classify, and select web sources for extraction."""
    _mark_stage(state, "quick_select_sources")
    t0 = time.time()

    results: list[WebSearchResult] = state.get("web_search_results", [])
    if not results:
        state.setdefault("warnings", []).append("No web results to select from")
        return state

    answer_schema: Optional[AnswerSchema] = state.get("answer_schema")
    settings = get_settings()

    # Filter low quality
    filtered = []
    low_quality_count = 0
    for r in results:
        if _is_low_quality(r.domain, r.url):
            low_quality_count += 1
            continue
        filtered.append(r)

    # Score each result
    class _ScoredResult(BaseModel):
        result: Any  # WebSearchResult — not validated here
        score: float = 0.0
        source_type: SourceType = SourceType.UNKNOWN

    scored: list[_ScoredResult] = []
    for r in filtered:
        st = _classify_source_type(r.domain, r.url, r.title, r.snippet or "")
        base_score = r.score or 0.0
        # Boost academic preferred domains
        acad_bonus = _is_academic_preferred(r.domain) * 0.02
        # Boost by source type priority
        type_bonus = max(0, (7 - SOURCE_TYPE_PRIORITY.get(st, 7))) * 0.03
        # Boost results that match answer schema subject
        subject_bonus = 0.0
        if answer_schema and answer_schema.subject:
            subj = answer_schema.subject.lower()
            if any(w in r.title.lower() for w in subj.split()):
                subject_bonus += 0.05
            if r.snippet and any(w in r.snippet.lower() for w in subj.split()):
                subject_bonus += 0.03
        # Multiple query purposes = more relevant
        purpose_bonus = min(len(r.query_purposes), 3) * 0.02

        total = base_score + acad_bonus + type_bonus + subject_bonus + purpose_bonus
        scored.append(_ScoredResult(result=r, score=total, source_type=st))

    scored.sort(key=lambda x: x.score, reverse=True)

    # Enforce source diversity: at most 3 from the same domain
    domain_counts: dict[str, int] = {}
    selected: list[tuple[WebSearchResult, SourceType]] = []
    max_sources = min(settings.TAVILY_MAX_SOURCES, len(scored))
    for sr in scored:
        if len(selected) >= max_sources:
            break
        d = sr.result.domain
        if domain_counts.get(d, 0) >= 3:
            continue
        domain_counts[d] = domain_counts.get(d, 0) + 1
        selected.append((sr.result, sr.source_type))

    # Ensure some diversity of source types
    source_types_seen = {st for _, st in selected}
    if len(selected) < max_sources and len(source_types_seen) < 3:
        for sr in scored:
            if len(selected) >= max_sources:
                break
            if sr.source_type not in source_types_seen and sr.result not in [s for s, _ in selected]:
                d = sr.result.domain
                if domain_counts.get(d, 0) >= 3:
                    continue
                domain_counts[d] = domain_counts.get(d, 0) + 1
                selected.append((sr.result, sr.source_type))
                source_types_seen.add(sr.source_type)

    state["selected_web_sources"] = [
        {"result": r, "source_type": st} for r, st in selected
    ]
    state.setdefault("retrieval_diagnostics", {}).update({
        "web_results_total": len(results),
        "web_results_low_quality_filtered": low_quality_count,
        "web_sources_selected": len(selected),
    })

    logger.info(
        f"[{state['task_id']}] Selected {len(selected)} sources from "
        f"{len(results)} results ({low_quality_count} low-quality filtered)"
    )
    return state


# ---- Node: tavily_extract ----

async def tavily_extract_node(state: dict) -> dict:
    """Extract content from selected web sources using Tavily Extract."""
    _mark_stage(state, "tavily_extract")
    t0 = time.time()

    selected: list[dict] = state.get("selected_web_sources", [])
    if not selected:
        state.setdefault("warnings", []).append("No sources selected for extraction")
        return state

    settings = get_settings()
    urls = [s["result"].url for s in selected]

    # Try Tavily Extract
    extracted_map: dict[str, dict] = {}
    if settings.MOCK_MODE:
        extracted_map = _mock_tavily_extract_results(urls, selected)
    elif settings.has_tavily_key:
        try:
            from app.clients import get_tavily_client
            client = get_tavily_client()
            extract_data = await client.extract(urls, extract_depth=settings.TAVILY_EXTRACT_DEPTH)
            # Handle both list and dict response formats
            results_list = extract_data.get("results", extract_data)
            if isinstance(results_list, list):
                for item in results_list:
                    eu = item.get("url", "")
                    extracted_map[eu] = item
            elif isinstance(results_list, dict):
                extracted_map = results_list
        except Exception as e:
            logger.error(f"Tavily Extract failed: {e}")
            state.setdefault("warnings", []).append(
                f"Tavily Extract failed: {str(e)[:200]}"
            )

    # Build ExtractedWebSource objects
    extracted_sources: list[ExtractedWebSource] = []
    import datetime
    now = datetime.datetime.utcnow().isoformat()

    for s in selected:
        r: WebSearchResult = s["result"]
        st: SourceType = s["source_type"]
        extract_info = extracted_map.get(r.url, {})

        content = extract_info.get("raw_content") or extract_info.get("content") or ""
        extraction_status = "success"
        snippet_only = False

        if not content or len(content) < 100:
            # Fall back to snippet
            content = r.snippet or ""
            extraction_status = "snippet_only"
            snippet_only = True
        elif len(content) < 300:
            extraction_status = "success"
            snippet_only = False

        es = ExtractedWebSource(
            title=r.title,
            url=r.url,
            domain=r.domain,
            source_type=st,
            content=content,
            extraction_status=extraction_status,
            extracted_at=now,
            query_purposes=r.query_purposes,
            snippet=r.snippet,
            snippet_only=snippet_only,
            content_length=len(content),
            metadata={
                "score": r.score,
                "published_date": r.published_date,
                "round_index": r.round_index,
            },
        )
        extracted_sources.append(es)

    # Merge with previous round
    prev_extracted = state.get("extracted_sources", [])
    prev_urls = {e.url for e in prev_extracted}
    new_extracted = [e for e in extracted_sources if e.url not in prev_urls]
    all_extracted = prev_extracted + new_extracted

    state["extracted_sources"] = all_extracted
    state["new_extracted_sources"] = new_extracted

    snippet_count = sum(1 for e in all_extracted if e.snippet_only)
    logger.info(
        f"[{state['task_id']}] Extracted {len(all_extracted)} sources "
        f"({snippet_count} snippet-only)"
    )
    return state


# ---- Node: build_research_notes ----

_RESEARCH_NOTE_SYSTEM = """You are a research analyst extracting structured notes from web sources.

Given the extracted content of a web source and the research question's AnswerSchema,
produce a ResearchNote.

CRITICAL RULES:
1. Every field is OPTIONAL — leave it null/empty if the source doesn't contain the info
2. NEVER fabricate data, numbers, DOIs, or author names
3. relevant_quotes MUST be continuous verbatim text from the source content
4. Numbers in reported_results MUST appear in the source content
5. For comparative questions, prioritise: technique + baseline + metric + result
6. Headers, navigation text, author lists, and TOC are NOT reported_results
7. If the source has no useful factual content, mark confidence="low" and note why

Return valid JSON matching the ResearchNote schema."""


async def build_research_notes_node(state: dict) -> dict:
    """Convert each extracted source into a structured ResearchNote via LLM."""
    _mark_stage(state, "build_research_notes")
    t0 = time.time()

    new_extracted: list[ExtractedWebSource] = state.get("new_extracted_sources", [])
    all_extracted: list[ExtractedWebSource] = state.get("extracted_sources", [])

    # If no new sources but we have extracted sources without notes, process all
    sources_to_process = new_extracted if new_extracted else all_extracted

    if not sources_to_process:
        state.setdefault("warnings", []).append("No sources for research note extraction")
        return state

    answer_schema: Optional[AnswerSchema] = state.get("answer_schema")
    llm = _get_llm()
    settings = get_settings()

    # Process in batches of 5 to stay within context limits
    BATCH_SIZE = 5
    all_notes: list[ResearchNote] = state.get("research_notes", [])
    existing_source_ids = {n.source_id for n in all_notes}

    for batch_start in range(0, len(sources_to_process), BATCH_SIZE):
        batch = sources_to_process[batch_start : batch_start + BATCH_SIZE]
        batch = [s for s in batch if s.source_id not in existing_source_ids]
        if not batch:
            continue

        tasks = [_build_single_note(llm, settings, s, answer_schema) for s in batch]
        results = await asyncio.gather(*tasks)

        for note, usage in results:
            if note:
                all_notes.append(note)
                existing_source_ids.add(note.source_id)
            _record_model_usage(state, usage, "build_research_note")

    state["research_notes"] = all_notes
    high_conf = sum(1 for n in all_notes if n.confidence in ("high", "medium"))
    logger.info(
        f"[{state['task_id']}] {len(all_notes)} research notes "
        f"({high_conf} medium+ confidence)"
    )
    return state


async def _build_single_note(
    llm, settings, source: ExtractedWebSource, answer_schema: Optional[AnswerSchema]
) -> tuple[Optional[ResearchNote], dict]:
    """Build a single research note from one extracted source."""
    schema_text = ""
    if answer_schema:
        schema_text = (
            f"Question Type: {answer_schema.question_type.value}\n"
            f"Subject: {answer_schema.subject}\n"
            f"Required Dimensions: {', '.join(answer_schema.required_dimensions)}\n"
        )

    # Truncate content to fit within context
    content = source.content[:6000] if source.content else ""
    source_label = (
        f"Source Type: {source.source_type.value}\n"
        f"Extraction Status: {source.extraction_status}\n"
        f"Content Length: {source.content_length} chars"
    )

    user_prompt = f"""URL: {source.url}
Title: {source.title}
{source_label}

AnswerSchema:
{schema_text}

=== SOURCE CONTENT ===
{content}
=== END CONTENT ===

Extract a structured ResearchNote from this source.
If the content is empty or contains only navigation/snippets, mark confidence="low".
Never invent data not present in the content."""

    result, usage = await llm.generate_structured(
        system_prompt=_RESEARCH_NOTE_SYSTEM,
        user_prompt=user_prompt,
        output_model=ResearchNote,
        model=settings.model_fast,
        max_tokens=settings.LLM_FAST_MAX_TOKENS,
        enable_thinking=False,
    )

    if result is None:
        # Build a minimal note from source fields
        result = ResearchNote(
            source_id=source.source_id,
            title=source.title,
            url=source.url,
            source_type=source.source_type,
            relevance_summary=source.snippet or "",
            confidence="low",
            extraction_failed=True,
        )
    else:
        result.source_id = source.source_id
        result.title = source.title
        result.url = source.url
        result.source_type = source.source_type
        result.extraction_failed = False

    # Validate quotes actually appear in content
    if result.relevant_quotes and content:
        valid_quotes = []
        for quote in result.relevant_quotes:
            if _quote_appears_in(quote, content):
                valid_quotes.append(quote)
            else:
                logger.debug(f"Rejected quote not found in source: {quote[:100]}...")
        result.relevant_quotes = valid_quotes

    # Validate reported_results numbers appear in content
    if result.reported_results and content:
        valid_results = []
        for res in result.reported_results:
            if _result_appears_in(res, content):
                valid_results.append(res)
            else:
                logger.debug(f"Rejected result not found in source: {res[:100]}...")
        result.reported_results = valid_results

    return result, usage


def _quote_appears_in(quote: str, content: str) -> bool:
    """Check if a quote (or close variant) appears in content."""
    if not quote or not content:
        return False
    # Exact match
    if quote in content:
        return True
    # Try fuzzy: 80% of quote words in sequence
    q_words = quote.split()
    if len(q_words) < 4:
        return False
    # Check if most words appear close together
    c_lower = content.lower()
    matched = sum(1 for w in q_words if w.lower() in c_lower)
    return matched >= max(3, len(q_words) * 0.7)


def _result_appears_in(result: str, content: str) -> bool:
    """Check if a reported result (with any numbers) appears supported in content."""
    if not result or not content:
        return False
    # Find numbers in the result
    num_pattern = re.compile(r"\d+(?:\.\d+)?\s*%?")
    nums = num_pattern.findall(result)
    if not nums:
        # No numbers — check if substantial portion of text appears
        words = [w for w in result.split() if len(w) > 3]
        c_lower = content.lower()
        matched = sum(1 for w in words if w.lower() in c_lower)
        return matched >= min(3, len(words) * 0.5)
    # Check if at least one number appears in content
    for n in nums:
        if n in content:
            return True
    return False


# ---- Node: quick_assess_coverage ----

_COVERAGE_SYSTEM = """You are a research coverage analyst. Given research notes and an AnswerSchema,
determine whether coverage is sufficient to answer the research question.

Check:
1. Are all required dimensions covered?
2. Are multiple techniques/methods represented?
3. Are there empirical comparison sources (for comparative questions)?
4. Are metrics and results present?
5. Is there over-concentration on a single domain or technique?
6. Are limitations and failure conditions covered?
7. Is the source count reasonable?
8. Is there enough evidence to answer the user's question?

Stop conditions:
- Coverage is sufficient across all dimensions
- OR maximum search rounds reached
- OR no significant new high-quality sources in the last round

Return a CoverageAssessment JSON.
For supplementary queries, generate specific, targeted queries addressing missing dimensions.
Do NOT repeat queries already used."""


async def quick_assess_coverage_node(state: dict) -> dict:
    """Assess coverage against AnswerSchema and decide on supplementary search."""
    _mark_stage(state, "quick_assess_coverage")
    t0 = time.time()

    notes: list[ResearchNote] = state.get("research_notes", [])
    answer_schema: Optional[AnswerSchema] = state.get("answer_schema")
    round_idx = state.get("quick_search_round", 0)
    settings = get_settings()
    max_rounds = settings.QUICK_MAX_SEARCH_ROUNDS

    # Check hard stop conditions first
    if round_idx >= max_rounds:
        state["coverage_assessment"] = CoverageAssessment(
            sufficient=True,
            reason=f"Maximum search rounds ({max_rounds}) reached",
            source_count=len(notes),
            high_quality_source_count=sum(
                1 for n in notes if n.confidence in ("high", "medium")
            ),
        )
        state["quick_needs_supplementary"] = False
        return state

    # Count new high-quality notes in this round
    new_extracted: list[ExtractedWebSource] = state.get("new_extracted_sources", [])
    new_source_ids = {e.source_id for e in new_extracted}
    new_high_quality = sum(
        1 for n in notes
        if n.source_id in new_source_ids and n.confidence in ("high", "medium")
    )

    if round_idx > 0 and new_high_quality < 2:
        state["coverage_assessment"] = CoverageAssessment(
            sufficient=True,
            reason=f"Only {new_high_quality} new high-quality sources this round",
            source_count=len(notes),
            high_quality_source_count=sum(
                1 for n in notes if n.confidence in ("high", "medium")
            ),
        )
        state["quick_needs_supplementary"] = False
        return state

    # Build a summary for the LLM
    notes_summary = "\n".join(
        f"- [{n.source_type.value}] {n.title}: confidence={n.confidence}, "
        f"technique={n.technique or 'N/A'}, metrics={n.metrics}, "
        f"results={len(n.reported_results)} results"
        for n in notes[:30]
    )

    schema_text = ""
    if answer_schema:
        schema_text = (
            f"Question Type: {answer_schema.question_type.value}\n"
            f"Required Dimensions: {', '.join(answer_schema.required_dimensions)}\n"
        )

    used_queries = [
        q.query
        for q in (state.get("all_quick_queries", []))
    ]

    user_prompt = f"""Research Question: {state['original_question']}
Round: {round_idx + 1}/{max_rounds}
{schema_text}

Used Queries: {'; '.join(used_queries[-15:])}

=== Research Notes ({len(notes)} total) ===
{notes_summary}

=== Task ===
1. Assess whether coverage is sufficient
2. If not, generate new targeted queries for missing dimensions (max {settings.QUICK_MAX_QUERIES_PER_ROUND})
3. Return a CoverageAssessment JSON"""

    llm = _get_llm()
    result, usage = await llm.generate_structured(
        system_prompt=_COVERAGE_SYSTEM,
        user_prompt=user_prompt,
        output_model=CoverageAssessment,
        model=settings.model_strong or settings.model_fast,
        max_tokens=min(settings.LLM_STRONG_MAX_TOKENS, 2048),
        enable_thinking=False,
    )
    _record_model_usage(state, usage, "assess_coverage")

    if result is None:
        result = CoverageAssessment(
            sufficient=True,
            reason="Coverage assessment LLM call failed, proceeding with available sources",
            source_count=len(notes),
            high_quality_source_count=sum(
                1 for n in notes if n.confidence in ("high", "medium")
            ),
        )
        state.setdefault("warnings", []).append(
            "Coverage assessment failed, proceeding with available sources"
        )

    # Enforce max rounds
    if round_idx >= max_rounds:
        result.sufficient = True
        result.reason = (result.reason or "") + " [Max rounds reached]"

    state["coverage_assessment"] = result
    state["quick_needs_supplementary"] = (
        not result.sufficient
        and round_idx < max_rounds
        and len(result.new_queries) > 0
    )

    logger.info(
        f"[{state['task_id']}] Coverage: sufficient={result.sufficient}, "
        f"needs_supplementary={state['quick_needs_supplementary']}"
    )
    return state


# ---- Node: quick_supplementary_search ----

async def quick_supplementary_search_node(state: dict) -> dict:
    """Prepare supplementary search round with new queries."""
    _mark_stage(state, "quick_supplementary_search")

    coverage: Optional[CoverageAssessment] = state.get("coverage_assessment")
    if not coverage or not coverage.new_queries:
        state.setdefault("warnings", []).append(
            "Supplementary search triggered but no new queries generated"
        )
        return state

    state["quick_search_round"] = state.get("quick_search_round", 0) + 1
    state["prev_round_queries"] = [
        q.query for q in (state.get("all_quick_queries", []))
    ]
    state["quick_queries"] = coverage.new_queries

    # Also add fallback queries if very few sources overall
    notes = state.get("research_notes", [])
    if len(notes) < 5 and coverage.new_queries:
        answer_schema = state.get("answer_schema")
        subj = answer_schema.subject if answer_schema else state["original_question"]
        fallback = PlannedQuery(
            query=f"{subj} research paper findings",
            purpose="broad_coverage",
            round_index=state["quick_search_round"],
        )
        coverage.new_queries.append(fallback)
        state["quick_queries"] = coverage.new_queries

    logger.info(
        f"[{state['task_id']}] Supplementary round {state['quick_search_round']}: "
        f"{len(state['quick_queries'])} new queries"
    )
    return state


# ---- Node: build_comparison_matrix ----

_COMPARISON_SYSTEM = """You are a research synthesis specialist. Given ResearchNotes,
build a structured comparison matrix.

CRITICAL RULES:
1. Only use information explicitly present in the ResearchNotes
2. Every row MUST have at least one source_id
3. Numbers must be traceable to a specific note's reported_results or relevant_quotes
4. 'most consistent' conclusions require at least 2 independent sources
5. If evidence is insufficient, say "当前检索结果不足以判断" — never fabricate consensus
6. Results from different tasks/datasets CANNOT be directly compared
7. Mark incomparable results as "domain-specific evidence"
8. support_count is the number of distinct sources supporting this row

Return JSON with a 'matrix' array of ComparisonRow objects."""


class _MatrixOutput(BaseModel):
    matrix: list[ComparisonRow] = Field(default_factory=list)


async def build_comparison_matrix_node(state: dict) -> dict:
    """Build structured comparison matrix from ResearchNotes."""
    _mark_stage(state, "build_comparison_matrix")
    t0 = time.time()

    notes: list[ResearchNote] = state.get("research_notes", [])
    answer_schema: Optional[AnswerSchema] = state.get("answer_schema")

    if not notes:
        state["comparison_matrix"] = []
        return state

    # Only for comparative questions, the matrix is essential.
    # For other question types, build a simpler evidence summary.
    is_comparative = (
        answer_schema is not None
        and answer_schema.question_type == QuestionType.COMPARATIVE
    )

    notes_text = "\n\n".join(
        f"[{n.source_id[:8]}] Title: {n.title}\n"
        f"Technique: {n.technique or 'N/A'}\n"
        f"Baseline: {n.baseline or 'N/A'}\n"
        f"Task: {n.task or 'N/A'}\n"
        f"Datasets: {', '.join(n.datasets) if n.datasets else 'N/A'}\n"
        f"Metrics: {', '.join(n.metrics) if n.metrics else 'N/A'}\n"
        f"Reported Results: {'; '.join(n.reported_results) if n.reported_results else 'N/A'}\n"
        f"Limitations: {'; '.join(n.limitations) if n.limitations else 'N/A'}\n"
        f"Quotes: {'; '.join(n.relevant_quotes[:3]) if n.relevant_quotes else 'N/A'}\n"
        f"Confidence: {n.confidence}"
        for n in notes[:25]
    )

    if not is_comparative:
        # Build a minimal matrix for non-comparative questions
        rows = []
        for n in notes[:15]:
            if n.confidence in ("high", "medium"):
                rows.append(ComparisonRow(
                    technique=n.technique or n.title,
                    task_or_domain=n.task,
                    datasets=n.datasets,
                    metrics=n.metrics,
                    reported_result="; ".join(n.reported_results) if n.reported_results else n.relevance_summary,
                    limitations=n.limitations,
                    source_ids=[n.source_id],
                    support_count=1,
                    confidence=n.confidence,
                ))
        state["comparison_matrix"] = rows
        return state

    user_prompt = f"""Research Question: {state['original_question']}
Question Type: {answer_schema.question_type.value if answer_schema else 'unknown'}
Subject: {answer_schema.subject if answer_schema else 'N/A'}
Comparison Target: {answer_schema.comparison_target if answer_schema else 'N/A'}

=== Research Notes ===
{notes_text}

=== Task ===
Build a comparison matrix comparing different techniques/methods.
Group rows by technique. For each technique, list:
- baselines compared against
- tasks/domains, datasets, metrics, reported results, limitations
- source_ids and support_count
Return JSON with a 'matrix' array."""

    llm = _get_llm()
    settings = get_settings()
    result, usage = await llm.generate_structured(
        system_prompt=_COMPARISON_SYSTEM,
        user_prompt=user_prompt,
        output_model=_MatrixOutput,
        model=settings.model_strong or settings.model_fast,
        max_tokens=min(settings.LLM_STRONG_MAX_TOKENS, 4096),
        enable_thinking=False,
    )
    _record_model_usage(state, usage, "comparison_matrix")

    matrix: list[ComparisonRow] = []
    if result and result.matrix:
        matrix = result.matrix
        # Validate: every row must have at least one source_id
        for row in matrix:
            if not row.source_ids:
                row.source_ids = ["unknown_source"]
                row.note = (row.note or "") + " [Missing source traceability]"
    else:
        # Fallback: build from notes directly
        for n in notes[:12]:
            if n.confidence in ("high", "medium"):
                matrix.append(ComparisonRow(
                    technique=n.technique or n.title,
                    baseline=n.baseline,
                    task_or_domain=n.task,
                    datasets=n.datasets,
                    metrics=n.metrics,
                    reported_result="; ".join(n.reported_results) if n.reported_results else n.relevance_summary,
                    limitations=n.limitations,
                    source_ids=[n.source_id],
                    support_count=1,
                    confidence=n.confidence,
                ))

    state["comparison_matrix"] = matrix
    logger.info(
        f"[{state['task_id']}] Comparison matrix: {len(matrix)} rows"
    )
    return state


# ---- Node: synthesize_quick_report ----

_QUICK_REPORT_SYSTEM = """
<role>
You are an evidence-grounded research report synthesizer.
Write a Simplified Chinese (中文) research report for domain experts.
Your goal is a COMPLETE, READABLE, well-structured report — like a professional
research survey — not a list of isolated findings.
</role>

<source_of_truth>
You may ONLY use the provided ResearchNotes, ComparisonMatrix, source metadata,
and CoverageAssessment. Do NOT use outside knowledge.
Do NOT read raw web page text — only the structured notes and matrix.

Every factual paragraph MUST cite its sources using [S1][S2] markers.
The source mapping is provided below.
</source_of_truth>

<report_structure>
Use Simplified Chinese for all prose. Keep paper titles in their original language.
Generate the following sections:

# 执行摘要
3-6 comprehensive bullet points synthesising the main findings.
Each bullet cites sources.

# 研究问题与范围
Refined research question, search scope, methodology, source coverage.

# 主要方法比较
(for comparative questions) Compare techniques side-by-side.
Use the comparison matrix. Discuss trade-offs, strengths, weaknesses.

# 哪些方法的证据最一致
Identify findings supported by multiple independent sources.
If a finding comes from a single source, mark it as "单一来源证据".
If evidence is insufficient for any conclusion, explicitly state so.

# 适用场景与边界
Under what conditions do the findings apply? What are the boundaries?

# 研究不足与未来方向
Distinguish between:
- Limitations explicitly stated in sources
- Gaps inferred from the current source set ("在本次检索到的来源中...", "当前证据尚未覆盖...")

# 方法与证据限制
Fixed statement about the methodology.

# 参考来源
List all sources with [S#] markers.
</report_structure>

<writing_rules>
1. SYNTHESISE, don't just list quotes. Rewrite for readability.
2. Every factual paragraph must have source markers like [S1][S3].
3. NEVER cite sources you don't have. NEVER invent DOIs, authors, years, or URLs.
4. NEVER claim "全文逐句验证" — this report is based on web sources.
5. For inferred gaps, use hedging: "在本次检索到的来源中...", "当前证据尚未覆盖..."
6. Numbers must come from ResearchNotes or ComparisonMatrix — never invented.
7. If a single source claims something, do NOT present it as consensus.
8. The report should read like a coherent survey, not a bullet-point list.
</writing_rules>

<evidence_limitation_statement>
本报告基于公开网页、论文页面、摘要及可访问正文生成。引用均指向实际检索来源，
但并非所有结论均经过论文PDF全文逐句核验。
</evidence_limitation_statement>"""


async def synthesize_quick_report_node(state: dict) -> dict:
    """Generate the quick research report from ResearchNotes and ComparisonMatrix."""
    _mark_stage(state, "synthesize_quick_report")
    t0 = time.time()

    notes: list[ResearchNote] = state.get("research_notes", [])
    matrix: list[ComparisonRow] = state.get("comparison_matrix", [])
    answer_schema: Optional[AnswerSchema] = state.get("answer_schema")
    coverage: Optional[CoverageAssessment] = state.get("coverage_assessment")

    if not notes:
        state["report"] = (
            "# 研究报告\n\n未找到足够的相关来源，无法生成报告。\n\n"
            "## 方法与证据限制\n\n"
            "本报告基于公开网页、论文页面、摘要及可访问正文生成。"
            "本次检索未能获取足够的高质量来源。"
        )
        return state

    settings = get_settings()

    # Build source index
    source_index: dict[str, int] = {}
    source_entries = []
    for i, n in enumerate(notes):
        idx = i + 1
        source_index[n.source_id] = idx
        st_label = _source_type_label(n.source_type)
        conf_label = {"high": "高", "medium": "中", "low": "低"}.get(n.confidence, "中")
        source_entries.append(
            f"[S{idx}] {n.title}\n"
            f"    URL: {n.url}\n"
            f"    Type: {st_label} | Confidence: {conf_label}\n"
            f"    Summary: {n.relevance_summary[:200]}"
        )

    source_list = "\n\n".join(source_entries)

    # Build notes summary
    notes_text = "\n\n".join(
        f"[S{source_index.get(n.source_id, 0)}] Title: {n.title}\n"
        f"Technique: {n.technique or 'N/A'}\n"
        f"Baseline: {n.baseline or 'N/A'}\n"
        f"Task: {n.task or 'N/A'}\n"
        f"Datasets: {', '.join(n.datasets) if n.datasets else 'N/A'}\n"
        f"Metrics: {', '.join(n.metrics) if n.metrics else 'N/A'}\n"
        f"Reported Results: {'; '.join(n.reported_results) if n.reported_results else 'N/A'}\n"
        f"Limitations: {'; '.join(n.limitations) if n.limitations else 'N/A'}\n"
        f"Quotes: {'; '.join(n.relevant_quotes[:2]) if n.relevant_quotes else 'N/A'}\n"
        f"Relevance: {n.relevance_summary[:300]}\n"
        f"Confidence: {n.confidence}"
        for n in notes[:25]
    )

    # Build matrix text
    if matrix:
        matrix_text = "\n\n".join(
            f"Technique: {r.technique}\n"
            f"Baseline: {r.baseline or 'N/A'}\n"
            f"Task/Domain: {r.task_or_domain or 'N/A'}\n"
            f"Datasets: {', '.join(r.datasets) if r.datasets else 'N/A'}\n"
            f"Metrics: {', '.join(r.metrics) if r.metrics else 'N/A'}\n"
            f"Result: {r.reported_result}\n"
            f"Limitations: {', '.join(r.limitations) if r.limitations else 'N/A'}\n"
            f"Sources: {', '.join(f'S{source_index.get(sid, 0)}' for sid in r.source_ids)}\n"
            f"Support: {r.support_count} sources | Confidence: {r.confidence}"
            for r in matrix[:20]
        )
    else:
        matrix_text = "No comparison matrix available."

    # Coverage info
    coverage_text = ""
    if coverage:
        coverage_text = (
            f"Sufficient: {coverage.sufficient}\n"
            f"Covered Dimensions: {', '.join(coverage.covered_dimensions)}\n"
            f"Missing Dimensions: {', '.join(coverage.missing_dimensions)}\n"
            f"Sources: {coverage.source_count} total, {coverage.high_quality_source_count} high-quality"
        )

    schema_text = ""
    if answer_schema:
        schema_text = (
            f"Question Type: {answer_schema.question_type.value}\n"
            f"Subject: {answer_schema.subject}\n"
            f"Comparison Target: {answer_schema.comparison_target or 'N/A'}\n"
            f"Outcome: {answer_schema.outcome or 'N/A'}\n"
            f"Required Dimensions: {', '.join(answer_schema.required_dimensions)}"
        )

    search_info = (
        f"Search Rounds: {state.get('quick_search_round', 0) + 1}\n"
        f"Web Sources: {len(notes)}\n"
        f"Year Range: {state.get('year_from') or 'any'} – {state.get('year_to') or 'any'}"
    )

    user_prompt = f"""Research Question: {state['original_question']}

=== Search Info ===
{search_info}

=== Answer Schema ===
{schema_text}

=== Coverage Assessment ===
{coverage_text}

=== Research Notes ({len(notes)} total) ===
{notes_text}

=== Comparison Matrix ===
{matrix_text}

=== SOURCE INDEX ===
{source_list}

=== TASK ===
Generate a comprehensive Simplified Chinese research report following the structure specified.
Use [S#] markers for all factual citations.
The report MUST be complete, readable, and synthetic — not a list of raw notes.
Include the evidence limitation statement in the 方法与证据限制 section."""

    llm = _get_llm()
    report_text, usage = await llm.generate_text(
        system_prompt=_QUICK_REPORT_SYSTEM,
        user_prompt=user_prompt,
        model=settings.model_strong or settings.model_fast,
        temperature=0.3,
        max_tokens=min(settings.LLM_STRONG_MAX_TOKENS, 4096),
        enable_thinking=settings.LLM_STRONG_ENABLE_THINKING,
    )
    _record_model_usage(state, usage, "quick_report")

    if not report_text:
        report_text = _build_quick_fallback_report(state)
        state.setdefault("warnings", []).append("Report generation failed, using fallback")

    # Append source references if missing
    if "参考来源" not in report_text and "References" not in report_text:
        report_text += f"\n\n## 参考来源\n\n{source_list}"

    # Ensure evidence limitation statement is present
    if "本报告基于公开网页" not in report_text:
        report_text += (
            "\n\n---\n\n*本报告基于公开网页、论文页面、摘要及可访问正文生成。"
            "引用均指向实际检索来源，但并非所有结论均经过论文PDF全文逐句核验。*"
        )

    state["report"] = report_text
    state["source_index"] = source_index
    logger.info(
        f"[{state['task_id']}] Quick report generated ({len(report_text)} chars)"
    )
    return state


def _source_type_label(st: SourceType) -> str:
    labels = {
        SourceType.PAPER_OFFICIAL: "论文官方页面",
        SourceType.PUBLISHER: "出版社页面",
        SourceType.PREPRINT: "预印本",
        SourceType.AUTHOR_PROJECT: "作者项目页",
        SourceType.INSTITUTIONAL: "研究机构",
        SourceType.REVIEW: "综述",
        SourceType.SECONDARY: "二手来源",
        SourceType.UNKNOWN: "未分类",
    }
    return labels.get(st, "未分类")


def _build_quick_fallback_report(state: dict) -> str:
    """Build a minimal report from ResearchNotes without LLM."""
    notes: list[ResearchNote] = state.get("research_notes", [])
    question = state.get("original_question", "")

    report = f"# 研究报告: {question}\n\n"
    report += "## 研究问题\n\n{question}\n\n"
    report += "## 检索范围\n\n"
    report += f"- 检索轮次: {state.get('quick_search_round', 0) + 1}\n"
    report += f"- 来源数: {len(notes)}\n\n"
    report += "## 核心发现\n\n"

    for i, n in enumerate(notes[:12]):
        report += f"### [{i + 1}] {n.title}\n\n"
        report += f"- 相关度: {n.confidence}\n"
        if n.reported_results:
            report += f"- 结果: {'; '.join(n.reported_results)}\n"
        if n.limitations:
            report += f"- 限制: {'; '.join(n.limitations)}\n"
        report += f"- {n.relevance_summary[:300]}\n"
        report += f"- 来源: {n.url}\n\n"

    report += "\n## 方法与证据限制\n\n"
    report += "本报告基于公开网页、论文页面、摘要及可访问正文生成。"
    report += "引用均指向实际检索来源，但并非所有结论均经过论文PDF全文逐句核验。\n"
    return report


# ---- Node: lightweight_citation_check ----

async def lightweight_citation_check_node(state: dict) -> dict:
    """Lightweight citation check for Quick mode — does NOT use Strict validator."""
    _mark_stage(state, "lightweight_citation_check")
    t0 = time.time()

    report = state.get("report", "")
    notes: list[ResearchNote] = state.get("research_notes", [])
    source_index: dict[str, int] = state.get("source_index", {})

    if not report or not notes:
        state["quick_citation_check"] = QuickCitationCheckResult(
            valid=True, issues=["No report or notes to check"]
        )
        return state

    issues: list[str] = []
    missing_refs: list[str] = []
    unverifiable_numbers: list[str] = []
    orphan_claims: list[str] = []

    # Build a reverse index: S# → source
    sid_to_source: dict[int, ResearchNote] = {}
    for note in notes:
        idx = source_index.get(note.source_id, 0)
        if idx > 0:
            sid_to_source[idx] = note

    # 1. Check all [S#] references exist
    ref_pattern = re.compile(r"\[S(\d+)\]")
    refs_in_report = ref_pattern.findall(report)
    for ref in refs_in_report:
        idx = int(ref)
        if idx not in sid_to_source:
            missing_refs.append(f"[S{idx}]")

    # 2. Check numbers in report are traceable to notes
    num_pattern = re.compile(r"(\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?)")
    all_source_text = " ".join(
        " ".join(n.reported_results + n.relevant_quotes + [n.relevance_summary])
        for n in notes
    )
    report_nums = num_pattern.findall(report)
    for num in set(report_nums):
        if re.match(r"^\d{4}$", num):  # Skip years
            continue
        if num not in all_source_text:
            # Check without %
            num_clean = num.replace("%", "").replace(" ", "")
            if num_clean not in all_source_text.replace("%", "").replace(" ", ""):
                unverifiable_numbers.append(num)

    # 3. Check no citations to un-selected sources
    all_source_ids = {n.source_id for n in notes}
    for note in notes:
        if note.source_id not in all_source_ids:
            issues.append(f"Source {note.source_id[:8]} not in selected sources")

    # 4. Factual paragraphs should have at least one [S#] marker
    paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
    for para in paragraphs:
        # Skip headings and non-factual sections
        if para.startswith("#") or para.startswith("---") or para.startswith("*"):
            continue
        if len(para) > 100 and not ref_pattern.search(para):
            # Only flag substance paragraphs (not purely structural text)
            if any(kw in para.lower() for kw in ["表明", "发现", "结果", "研究", "find", "result", "show"]):
                orphan_claims.append(para[:100] + "...")

    is_valid = (
        len(missing_refs) == 0
        and len(issues) == 0
    )

    # Report warnings but don't block finalize
    if missing_refs:
        issues.append(f"Missing source references: {', '.join(missing_refs[:10])}")
    if unverifiable_numbers:
        issues.append(
            f"Unverifiable numbers: {', '.join(unverifiable_numbers[:8])}"
        )
    if orphan_claims:
        issues.append(
            f"Factual paragraphs without citations: {len(orphan_claims)}"
        )

    result = QuickCitationCheckResult(
        valid=is_valid,
        issues=issues,
        missing_refs=missing_refs,
        unverifiable_numbers=unverifiable_numbers,
        orphan_claims=orphan_claims,
        revision_needed=False,
    )

    if missing_refs and not unverifiable_numbers:
        # Simple fix: try to remove orphan references
        result.revision_needed = False  # Don't auto-rewrite in quick mode

    state["quick_citation_check"] = result
    state["citation_validation"] = result  # Compat with existing state

    for issue in issues:
        state.setdefault("warnings", []).append(f"Citation: {issue}")

    logger.info(
        f"[{state['task_id']}] Citation check: valid={is_valid}, "
        f"{len(missing_refs)} missing refs, "
        f"{len(unverifiable_numbers)} unverifiable numbers"
    )
    return state


# ---- Node: quick_finalize ----

async def quick_finalize_node(state: dict) -> dict:
    """Finalize the Quick Research task."""
    _mark_stage(state, "quick_finalize")
    state["status"] = TaskStatus.COMPLETED.value

    mt = state.get("metrics", TaskMetrics())
    mt.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["metrics"] = mt

    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save run record
    await _save_quick_run_record(state)

    # Add research mode label
    state["_mode_label"] = "research=quick · source=tavily"

    logger.info(f"[{state['task_id']}] Quick research completed")
    return state


async def _save_quick_run_record(state: dict) -> None:
    """Save quick research run record to storage/tasks/{task_id}/run_record.json."""
    import json
    from pathlib import Path

    task_id = state["task_id"]
    run_dir = Path("storage/tasks") / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def _safe(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    notes = state.get("research_notes", [])
    record = {
        "task_id": task_id,
        "original_question": state.get("original_question", ""),
        "status": state.get("status", ""),
        "research_mode": "quick",
        "search_rounds": state.get("quick_search_round", 0) + 1,
        "answer_schema": _safe(state.get("answer_schema")),
        "web_search_results_count": len(state.get("web_search_results", [])),
        "extracted_sources_count": len(state.get("extracted_sources", [])),
        "research_notes_count": len(notes),
        "research_notes": [_safe(n) for n in notes],
        "comparison_matrix": [_safe(r) for r in state.get("comparison_matrix", [])],
        "coverage_assessment": _safe(state.get("coverage_assessment")),
        "citation_check": _safe(state.get("quick_citation_check")),
        "queries": [
            q.model_dump() if hasattr(q, "model_dump") else str(q)
            for q in state.get("all_quick_queries", [])
        ],
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
        "metrics": _safe(state.get("metrics", {})),
        "report_length": len(state.get("report", "")),
    }

    record_path = run_dir / "run_record.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str)
    )

    report_text = state.get("report", "")
    if report_text:
        report_path = run_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")


# ---- Build Quick Research Subgraph ----

def build_quick_research_subgraph() -> list[tuple[str, callable, dict]]:
    """Return nodes and edges for the Quick Research subgraph.

    Returns a list of (op, node_or_edge, metadata) tuples where op is 'node' or 'edge'.
    This is consumed by the main graph builder.
    """
    return [
        ("node", "classify_question", classify_question_node),
        ("node", "quick_plan_queries", quick_plan_queries_node),
        ("node", "tavily_search", tavily_search_node),
        ("node", "quick_select_sources", quick_select_sources_node),
        ("node", "tavily_extract", tavily_extract_node),
        ("node", "build_research_notes", build_research_notes_node),
        ("node", "quick_assess_coverage", quick_assess_coverage_node),
        ("node", "quick_supplementary_search", quick_supplementary_search_node),
        ("node", "build_comparison_matrix", build_comparison_matrix_node),
        ("node", "synthesize_quick_report", synthesize_quick_report_node),
        ("node", "lightweight_citation_check", lightweight_citation_check_node),
        ("node", "quick_finalize", quick_finalize_node),
    ]
