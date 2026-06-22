"""Semantic Scholar provider — supplementary semantic relevance and citation info."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.models.paper import Paper, PaperSource, AuthorInfo
from app.providers.base import BaseProvider
from app.providers.query_compiler import compile_query

logger = logging.getLogger(__name__)

# Global rate limiter: S2 allows 1 request/second across all endpoints.
# Shared across all provider instances and tasks.
_s2_semaphore = asyncio.Semaphore(1)
_s2_last_request_time = 0.0


class SemanticScholarProvider(BaseProvider):
    """Search Semantic Scholar for papers with semantic relevance."""

    name = "semantic_scholar"

    # Fields to request from the API
    SEARCH_FIELDS = (
        "paperId,title,abstract,year,authors,venue,"
        "externalIds,url,openAccessPdf,citationCount,"
        "publicationTypes,publicationDate,journal"
    )

    def __init__(self, settings=None):
        super().__init__(settings)
        self._base_url = self._settings.SEMANTIC_SCHOLAR_BASE_URL
        self._api_key = self._settings.SEMANTIC_SCHOLAR_API_KEY
        self._max_results = self._settings.MAX_RESULTS_PER_QUERY_S2

    async def _rate_limit(self) -> None:
        """Enforce S2's 1 req/s global limit across all instances."""
        global _s2_semaphore, _s2_last_request_time
        async with _s2_semaphore:
            elapsed = time.monotonic() - _s2_last_request_time
            if elapsed < 1.1:
                await asyncio.sleep(1.1 - elapsed)
            _s2_last_request_time = time.monotonic()

    async def is_available(self) -> bool:
        """Check if Semantic Scholar API is reachable."""
        try:
            headers = self._build_headers()
            await self._request_with_retry(
                f"{self._base_url}/paper/search",
                params={"query": "test", "limit": 1, "fields": "paperId"},
                headers=headers,
            )
            return True
        except Exception:
            return False

    def _build_headers(self) -> dict:
        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    async def search(
        self, query: str, year_from: Optional[int] = None, year_to: Optional[int] = None,
        search_intent=None,
    ) -> list[Paper]:
        """Search Semantic Scholar papers."""
        query = compile_query(self.name, query, search_intent=search_intent)
        params: dict = {
            "query": query,
            "limit": min(self._max_results, 50),
            "fields": self.SEARCH_FIELDS,
        }

        # Semantic Scholar supports year filter as query modifiers
        if year_from and year_to:
            params["year"] = f"{year_from}-{year_to}"
        elif year_from:
            params["year"] = f"{year_from}-"

        headers = self._build_headers()

        await self._rate_limit()

        try:
            response = await self._request_with_retry(
                f"{self._base_url}/paper/search",
                params=params,
                headers=headers,
            )
            data = response.json()
            papers = []
            for paper_data in data.get("data", []):
                try:
                    paper = self._normalize(paper_data)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"{self.name}: failed to normalize paper: {e}")
            # S2 search endpoint does not return total count
            self.last_total_hits = 0
            logger.info(
                f"{self.name}: returned {len(papers)} papers "
                f"(total available: N/A) for query '{query[:60]}...'"
            )
            return papers
        except Exception as e:
            logger.error(f"{self.name}: search failed: {e}")
            return []

    def _normalize(self, paper_data: dict) -> Paper:
        """Convert Semantic Scholar paper to normalized Paper."""
        # Authors
        authors = []
        for author in paper_data.get("authors", []):
            authors.append(AuthorInfo(name=author.get("name", "Unknown")))

        # External IDs
        ext_ids = paper_data.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI")

        # Venue
        venue = None
        if paper_data.get("venue"):
            venue = paper_data["venue"]
        elif paper_data.get("journal"):
            journal = paper_data["journal"] or {}
            venue = journal.get("name")

        # Open access
        oa_pdf = paper_data.get("openAccessPdf") or {}
        full_text_url = oa_pdf.get("url")
        open_access = bool(full_text_url)

        # URL
        url = paper_data.get("url") or f"https://api.semanticscholar.org/CorpusID:{paper_data.get('paperId')}"

        source_id = PaperSource(
            provider=self.name,
            provider_id=paper_data.get("paperId", ""),
        )

        return Paper(
            title=paper_data.get("title", "Untitled") or "Untitled",
            abstract=paper_data.get("abstract"),
            authors=authors,
            publication_year=paper_data.get("year"),
            venue=venue,
            doi=doi,
            url=url,
            full_text_url=full_text_url,
            citation_count=paper_data.get("citationCount"),
            source_names=[self.name],
            source_ids=[source_id],
            open_access=open_access,
        )
