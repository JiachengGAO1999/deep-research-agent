"""Crossref provider — DOI verification and publication metadata."""

from __future__ import annotations

import logging
from typing import Optional

from app.models.paper import Paper, PaperSource, AuthorInfo
from app.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class CrossrefProvider(BaseProvider):
    """Search Crossref for publication metadata."""

    name = "crossref"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._base_url = self._settings.CROSSREF_BASE_URL
        self._max_results = min(self._settings.MAX_PAPERS_PER_SOURCE, 20)

    async def is_available(self) -> bool:
        """Check if Crossref API is reachable."""
        try:
            await self._request_with_retry(
                f"{self._base_url}/works",
                params={"query": "test", "rows": 1},
            )
            return True
        except Exception:
            return False

    async def search(
        self, query: str, year_from: Optional[int] = None, year_to: Optional[int] = None
    ) -> list[Paper]:
        """Search Crossref works."""
        params: dict = {
            "query": query,
            "rows": self._max_results,
            "sort": "relevance",
            "select": "DOI,title,abstract,author,issued,container-title,publisher,URL,is-referenced-by-count,reference",
        }

        # Year filter
        filter_parts = []
        if year_from:
            filter_parts.append(f"from-created-date:{year_from}-01-01")
        if year_to:
            filter_parts.append(f"until-created-date:{year_to}-12-31")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        try:
            response = await self._request_with_retry(
                f"{self._base_url}/works", params=params
            )
            data = response.json()
            items = data.get("message", {}).get("items", [])
            papers = []
            for item in items:
                try:
                    paper = self._normalize(item)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"{self.name}: failed to normalize item: {e}")
            logger.info(f"{self.name}: found {len(papers)} results for query '{query[:60]}...'")
            return papers
        except Exception as e:
            logger.error(f"{self.name}: search failed: {e}")
            return []

    def _normalize(self, item: dict) -> Paper:
        """Convert Crossref work to normalized Paper."""
        # DOI
        doi = item.get("DOI")

        # Title
        title_list = item.get("title", ["Untitled"])
        title = title_list[0] if title_list else "Untitled"

        # Abstract
        abstract = item.get("abstract")
        # Crossref abstracts sometimes have HTML tags
        if abstract and "<" in abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)

        # Authors
        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            name = f"{given} {family}".strip()
            if not name:
                name = author.get("name", "Unknown")
            authors.append(AuthorInfo(
                name=name,
                affiliation=author.get("affiliation", [{}])[0].get("name") if author.get("affiliation") else None,
                orcid=author.get("ORCID"),
            ))

        # Year from issued date
        issued = item.get("issued", {})
        date_parts = issued.get("date-parts", [[None]])[0]
        pub_year = date_parts[0] if date_parts and date_parts[0] else None

        # Venue
        container = item.get("container-title", [])
        venue = container[0] if container else None

        # URL
        url = item.get("URL") or f"https://doi.org/{doi}" if doi else None

        # Citation count
        cited_by = item.get("is-referenced-by-count")

        # References
        references = []
        for ref in item.get("reference", []):
            ref_doi = ref.get("DOI")
            if ref_doi:
                references.append(ref_doi)

        source_id = PaperSource(
            provider=self.name,
            provider_id=doi or f"crossref:{item.get('indexed', {}).get('id', '')}",
        )

        return Paper(
            title=title,
            abstract=abstract,
            authors=authors,
            publication_year=pub_year,
            venue=venue,
            doi=doi,
            url=url,
            citation_count=cited_by,
            referenced_works=references,
            source_names=[self.name],
            source_ids=[source_id],
        )
