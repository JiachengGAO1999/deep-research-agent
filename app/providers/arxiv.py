"""arXiv provider — supplementary for CS/NLP preprints."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urlencode

from app.models.paper import Paper, PaperSource, AuthorInfo
from app.providers.base import BaseProvider
from app.providers.query_compiler import compile_query

logger = logging.getLogger(__name__)

# arXiv API namespace
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivProvider(BaseProvider):
    """Search arXiv for preprints."""

    name = "arxiv"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._base_url = self._settings.ARXIV_BASE_URL
        self._max_results = min(
            self._settings.MAX_RESULTS_PER_QUERY_ARXIV,
            self._settings.MAX_PAPERS_PER_SOURCE,
            30,
        )

    async def is_available(self) -> bool:
        """Check if arXiv API is reachable."""
        try:
            await self._request_with_retry(
                f"{self._base_url}/query",
                params={"search_query": "all:test", "max_results": 1},
            )
            return True
        except Exception:
            return False

    async def search(
        self, query: str, year_from: Optional[int] = None, year_to: Optional[int] = None,
        search_intent=None,
    ) -> list[Paper]:
        """Search arXiv for papers."""
        search_query = compile_query(self.name, query, search_intent=search_intent)
        params = {
            "search_query": search_query,
            "max_results": str(self._max_results),
            "sortBy": "relevance",
        }

        try:
            response = await self._request_with_retry(
                f"{self._base_url}/query", params=params
            )
            papers, total_hits = self._parse_atom(response.text)
            self.last_total_hits = total_hits
            logger.info(
                f"{self.name}: returned {len(papers)} papers "
                f"(total available: {total_hits}) for query '{query[:60]}...'"
            )
            return papers
        except Exception as e:
            logger.error(f"{self.name}: search failed: {e}")
            return []

    def _parse_atom(self, xml_text: str) -> tuple:
        """Parse arXiv Atom XML response. Returns (papers, total_hits)."""
        papers = []
        total_hits = 0
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"{self.name}: XML parse error: {e}")
            return [], 0

        # Extract total results from opensearch namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        }
        total_el = root.find("opensearch:totalResults", ns)
        if total_el is not None and total_el.text:
            try:
                total_hits = int(total_el.text)
            except ValueError:
                pass

        for entry in root.findall("atom:entry", ns):
            try:
                paper = self._normalize_entry(entry)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"{self.name}: failed to normalize entry: {e}")

        return papers, total_hits

    def _normalize_entry(self, entry: ET.Element) -> Paper:
        """Convert arXiv Atom entry to normalized Paper."""
        def _text(tag: str) -> Optional[str]:
            el = entry.find(f"atom:{tag}", ARXIV_NS)
            return el.text.strip() if el is not None and el.text else None

        title = _text("title") or "Untitled"
        # Remove newlines from title
        title = " ".join(title.split())

        abstract = _text("summary")
        if abstract:
            abstract = " ".join(abstract.split())

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name_el = author_el.find("atom:name", ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(AuthorInfo(name=name_el.text.strip()))

        # Extract arXiv ID from the id URL
        arxiv_id = ""
        id_url = _text("id") or ""
        if "abs/" in id_url:
            arxiv_id = id_url.split("abs/")[-1]

        # Extract year from published date
        published = _text("published") or ""
        pub_year = None
        if published:
            try:
                pub_year = int(published[:4])
            except (ValueError, IndexError):
                pass

        # Categories
        categories = []
        for cat_el in entry.findall("atom:category", ARXIV_NS):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)

        # DOI (sometimes available in arxiv entries)
        doi = None
        for link_el in entry.findall("atom:link", ARXIV_NS):
            href = link_el.get("href", "")
            if "doi.org" in href:
                doi = href.split("doi.org/")[-1]

        source_id = PaperSource(
            provider=self.name,
            provider_id=arxiv_id,
        )

        return Paper(
            title=title,
            abstract=abstract,
            authors=authors,
            publication_year=pub_year,
            venue="arXiv preprint",
            doi=doi,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            full_text_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
            source_names=[self.name],
            source_ids=[source_id],
            open_access=True,  # arXiv is always open access
        )
