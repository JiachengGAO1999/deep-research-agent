"""OpenAlex provider — primary search source."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlencode

import httpx

from app.core.config import get_settings
from app.models.paper import Paper, PaperSource, AuthorInfo
from app.providers.base import BaseProvider
from app.providers.query_compiler import compile_query

logger = logging.getLogger(__name__)


class OpenAlexProvider(BaseProvider):
    """Search OpenAlex for academic works."""

    name = "openalex"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._base_url = self._settings.OPENALEX_BASE_URL
        self._api_key = self._settings.OPENALEX_API_KEY
        self._mailto = self._settings.OPENALEX_MAILTO
        self._max_results = self._settings.MAX_PAPERS_PER_SOURCE

    async def is_available(self) -> bool:
        """OpenAlex is always available (no auth required)."""
        try:
            await self._request_with_retry(f"{self._base_url}/works?per_page=1")
            return True
        except Exception:
            return False

    async def search(
        self, query: str, year_from: Optional[int] = None, year_to: Optional[int] = None
    ) -> list[Paper]:
        """Search OpenAlex works."""
        query = compile_query(self.name, query)
        params: dict = {
            "search": query,
            "per_page": min(self._max_results, 50),
            "sort": "relevance_score:desc",
        }

        # Build filter string
        filters = []
        if year_from:
            filters.append(f"publication_year:>{year_from - 1}")
        if year_to:
            filters.append(f"publication_year:<{year_to + 1}")
        if filters:
            params["filter"] = ",".join(filters)

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._mailto:
            params["mailto"] = self._mailto

        try:
            response = await self._request_with_retry(
                f"{self._base_url}/works", params=params, headers=headers
            )
            data = response.json()
            meta = data.get("meta", {})
            total_hits = meta.get("count", 0)
            self.last_total_hits = total_hits
            papers = []
            for work in data.get("results", []):
                try:
                    paper = self._normalize(work)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"{self.name}: failed to normalize work: {e}")
            logger.info(
                f"{self.name}: returned {len(papers)} papers "
                f"(total available: {total_hits}) for query '{query[:60]}...'"
            )
            return papers
        except Exception as e:
            logger.error(f"{self.name}: search failed: {e}")
            return []  # Degrade gracefully — don't fail the whole task

    def _normalize(self, work: dict) -> Paper:
        """Convert OpenAlex work to normalized Paper."""
        # Extract DOI
        doi = None
        if work.get("doi"):
            doi = work["doi"].replace("https://doi.org/", "")

        # Extract authors
        authors = []
        for auth in work.get("authorships", []):
            author_info = auth.get("author", {})
            name = author_info.get("display_name", "Unknown")
            affiliation = None
            insts = auth.get("institutions", [])
            if insts:
                affiliation = insts[0].get("display_name")
            orcid = author_info.get("orcid")
            authors.append(AuthorInfo(name=name, affiliation=affiliation, orcid=orcid))

        # Extract venue
        venue = None
        primary_location = work.get("primary_location", {}) or {}
        source = primary_location.get("source", {}) or {}
        if source.get("display_name"):
            venue = source["display_name"]

        # Extract URL
        url = work.get("open_access", {}).get("oa_url") or work.get("doi") or ""

        # Open access
        oa = work.get("open_access", {}) or {}
        open_access = oa.get("is_oa", False)

        # Full text URL
        full_text_url = oa.get("oa_url") or None

        # Citation count
        cited_by = work.get("cited_by_count")

        # Referenced works
        referenced = []
        for ref in work.get("referenced_works", [])[:20]:
            referenced.append(ref)

        source_id = PaperSource(
            provider=self.name,
            provider_id=work.get("id", "").split("/")[-1] or work.get("id", ""),
        )

        return Paper(
            title=work.get("title", "Untitled") or "Untitled",
            abstract=self._clean_abstract(
                work.get("abstract_inverted_index", {})
            ),
            authors=authors,
            publication_year=work.get("publication_year"),
            venue=venue,
            doi=doi,
            url=url,
            full_text_url=full_text_url,
            citation_count=cited_by,
            referenced_works=referenced,
            source_names=[self.name],
            source_ids=[source_id],
            open_access=open_access,
        )

    @staticmethod
    def _clean_abstract(inverted_index: dict) -> Optional[str]:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return None
        try:
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort(key=lambda x: x[0])
            return " ".join(w for _, w in word_positions)
        except Exception:
            return None
