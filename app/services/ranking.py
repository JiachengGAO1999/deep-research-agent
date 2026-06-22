"""Paper ranking and selection service.

Uses a two-stage approach:
1. Deterministic pre-filtering (keywords, year, abstract presence, inclusion criteria)
2. LLM-based structured relevance judgment on the reduced set
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models.paper import Paper
from app.models.search_plan import SearchPlan

logger = logging.getLogger(__name__)

# Default max papers after selection
DEFAULT_MAX_SELECTED = 20
LLM_RANK_BATCH_SIZE = 8


def deterministic_prefilter(
    papers: list[Paper],
    search_plan: SearchPlan,
    max_after_prefilter: int = 40,
) -> list[Paper]:
    """Deterministic pre-filter to reduce candidate set before LLM ranking.

    Filters by:
    - Year range match
    - Abstract presence (bonus, not strict filter)
    - Keyword relevance in title/abstract
    - Inclusion/exclusion criteria (simple keyword match)

    Returns papers sorted by a deterministic score.
    """
    scored: list[tuple[Paper, int]] = []

    include_keywords = _extract_keywords(search_plan.criteria.include)
    exclude_keywords = _extract_keywords(search_plan.criteria.exclude)
    concept_keywords = [c.lower() for c in search_plan.core_concepts]
    # Flatten synonyms
    for syns in search_plan.synonyms.values():
        concept_keywords.extend(s.lower() for s in syns)

    for paper in papers:
        # Exclusion check
        if _matches_exclude(paper, exclude_keywords):
            continue

        score = 0
        title_lower = paper.title.lower()
        abstract_lower = (paper.abstract or "").lower()

        # Title keyword match (strong signal)
        for kw in concept_keywords:
            if kw in title_lower:
                score += 5
            elif kw in abstract_lower:
                score += 2

        # Inclusion criteria match
        for kw in include_keywords:
            if kw in title_lower or kw in abstract_lower:
                score += 3

        # Year match
        if search_plan.year_from and paper.publication_year:
            if paper.publication_year >= search_plan.year_from:
                score += 2
        if search_plan.year_to and paper.publication_year:
            if paper.publication_year <= search_plan.year_to:
                score += 2

        # Has abstract (quality signal)
        if paper.abstract:
            score += 3

        # Citation count as weak signal (capped)
        if paper.citation_count:
            citation_bonus = min(paper.citation_count // 10, 5)
            score += citation_bonus

        scored.append((paper, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    result = [paper for paper, _ in scored[:max_after_prefilter]]
    logger.info(
        f"Pre-filter: {len(papers)} -> {len(result)} candidates "
        f"(max {max_after_prefilter})"
    )
    return result


def _extract_keywords(criteria_list: list[str]) -> list[str]:
    """Extract lowercase keywords from criteria strings."""
    keywords = []
    for criterion in criteria_list:
        words = criterion.lower().split()
        keywords.extend(w for w in words if len(w) > 3)
    return list(set(keywords))


def _matches_exclude(paper: Paper, exclude_keywords: list[str]) -> bool:
    """Check if paper matches exclusion criteria."""
    if not exclude_keywords:
        return False
    text = (paper.title + " " + (paper.abstract or "")).lower()
    # Paper is excluded if it matches ALL exclusion keywords (conservative)
    match_count = sum(1 for kw in exclude_keywords if kw in text)
    # Exclude if more than half of exclusion keywords match
    return match_count > len(exclude_keywords) / 2


async def llm_rank_papers(
    papers: list[Paper],
    research_question: str,
    search_plan: SearchPlan,
    max_selected: int = DEFAULT_MAX_SELECTED,
    llm_client=None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> tuple[list[Paper], dict]:
    """Use LLM to rank and select papers based on relevance.

    Returns (selected_papers, usage_info).
    Returns all papers scored with include=True, sorted by relevance_score.
    """
    if not papers:
        return [], {}

    if len(papers) > LLM_RANK_BATCH_SIZE:
        all_selected: list[Paper] = []
        total_usage: dict[str, int] = {}
        for start in range(0, len(papers), LLM_RANK_BATCH_SIZE):
            batch = papers[start : start + LLM_RANK_BATCH_SIZE]
            batch_selected, usage = await llm_rank_papers(
                batch,
                research_question,
                search_plan,
                max_selected=len(batch),
                llm_client=llm_client,
                model=model,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            all_selected.extend(batch_selected)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[key] = total_usage.get(key, 0) + int(
                    usage.get(key, 0) or 0
                )
            total_usage["call_count"] = total_usage.get("call_count", 0) + int(
                usage.get("call_count", 1)
            )
        all_selected.sort(
            key=lambda paper: paper.relevance_score or 0,
            reverse=True,
        )
        return all_selected[:max_selected], total_usage

    # Build a concise paper list for the LLM
    paper_summaries = []
    for i, paper in enumerate(papers):
        summary = (
            f"[{i}] ID: {paper.internal_id}\n"
            f"    Title: {paper.title}\n"
            f"    Year: {paper.publication_year or 'N/A'}\n"
            f"    Abstract: {(paper.abstract or 'NO ABSTRACT')[:300]}\n"
            f"    Venue: {paper.venue or 'N/A'}\n"
        )
        paper_summaries.append(summary)

    paper_list_text = "\n".join(paper_summaries)

    system_prompt = """You are a research assistant evaluating the relevance of academic papers to a research question.

For each paper, assess its relevance on a scale of 0-100 and decide whether to include it.

Consider:
1. Direct relevance to the research question
2. Quality of the venue/publication
3. Presence and informativeness of the abstract
4. Year (recent is generally preferred but seminal older work is valuable)
5. Alignment with inclusion/exclusion criteria

Select only papers that genuinely contribute to answering the research question. Maximum 15-20 papers should be included.

You MUST respond with a JSON object with this structure:
{
  "rankings": [
    {
      "internal_id": "the paper's ID string",
      "relevance_score": 85,
      "include": true,
      "reason": "Directly addresses the research question with empirical evaluation",
      "matched_aspects": ["multi-turn reasoning", "reliability"]
    }
  ]
}"""

    user_prompt = f"""Research Question: {research_question}

Inclusion Criteria: {', '.join(search_plan.criteria.include) if search_plan.criteria.include else 'None specified'}
Exclusion Criteria: {', '.join(search_plan.criteria.exclude) if search_plan.criteria.exclude else 'None specified'}

Papers to evaluate:
{paper_list_text}

Evaluate each paper and return the JSON with rankings."""

    if llm_client is None:
        from app.llm.client import get_llm_client
        llm_client = get_llm_client()

    from pydantic import BaseModel, Field

    class RankingResult(BaseModel):
        internal_id: str
        relevance_score: int = Field(ge=0, le=100)
        include: bool
        reason: str
        matched_aspects: list[str] = Field(default_factory=list)

    class RankingOutput(BaseModel):
        rankings: list[RankingResult]

    try:
        result, usage = await llm_client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=RankingOutput,
            model=model,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

        if result and result.rankings:
            # Build ranking map
            ranking_map = {r.internal_id: r for r in result.rankings}

            # Apply rankings to papers
            for paper in papers:
                if paper.internal_id in ranking_map:
                    r = ranking_map[paper.internal_id]
                    paper.relevance_score = r.relevance_score
                    paper.include = r.include
                    paper.relevance_reason = r.reason
                    paper.matched_aspects = r.matched_aspects
                else:
                    paper.relevance_score = 0
                    paper.include = False

            # Select included papers, sorted by score, capped
            selected = [p for p in papers if p.include]
            selected.sort(key=lambda p: p.relevance_score or 0, reverse=True)
            selected = selected[:max_selected]

            logger.info(
                f"LLM ranking: {len(papers)} evaluated, {len(selected)} selected"
            )
            usage["call_count"] = usage.get("call_count", 1)
            return selected, usage
        else:
            logger.warning("LLM ranking returned no valid results, using pre-filter")
            usage["call_count"] = usage.get("call_count", 1)
            return _fallback_selection(papers, max_selected), usage

    except Exception as e:
        logger.error(f"LLM ranking failed: {e}, using fallback selection")
        return _fallback_selection(papers, max_selected), {}


def _fallback_selection(papers: list[Paper], max_selected: int) -> list[Paper]:
    """Fallback: select top papers by citation count and abstract presence."""
    def _score(p: Paper) -> int:
        s = 0
        if p.abstract:
            s += 5
        if p.citation_count:
            s += min(p.citation_count, 100)
        if p.publication_year:
            s += max(0, p.publication_year - 2018)
        return s

    scored = [(p, _score(p)) for p in papers]
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in scored[:max_selected]]
    for p in selected:
        p.include = True
        p.relevance_score = 50  # Fallback score
        p.relevance_reason = "Fallback selection (LLM ranking unavailable)"
    return selected
