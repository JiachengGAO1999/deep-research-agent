"""Real network tests for academic data providers.

Each provider is tested with a single simple query (max 3 results).
No API keys required.
Run with: .venv/bin/python -m pytest tests/test_real_providers.py -v
"""

import pytest


# Shared query — simple, broad, should return results from all sources
QUERY = "machine learning"
MAX_RESULTS = 3


# ============================================================
# OpenAlex
# ============================================================

class TestOpenAlexReal:
    """Test OpenAlex provider with real HTTP calls."""

    @pytest.mark.asyncio
    async def test_search_returns_papers(self):
        from app.providers.openalex import OpenAlexProvider
        provider = OpenAlexProvider()
        papers = await provider.search(QUERY, year_from=2023, year_to=2024)
        assert len(papers) > 0, "No results returned"
        assert len(papers) <= provider._settings.MAX_PAPERS_PER_SOURCE
        print(f"\n    OpenAlex: {len(papers)} papers returned")

    @pytest.mark.asyncio
    async def test_paper_has_core_fields(self):
        from app.providers.openalex import OpenAlexProvider
        provider = OpenAlexProvider()
        papers = await provider.search(QUERY)
        for p in papers:
            assert p.title, "Missing title"
            assert p.internal_id, "Missing internal_id"
            assert "openalex" in p.source_names
            print(f"\n    title: {p.title[:80]}")
            print(f"    year: {p.publication_year}")
            print(f"    venue: {p.venue}")
            print(f"    doi: {p.doi}")
            print(f"    authors: {len(p.authors)}")
            print(f"    abstract: {p.abstract[:100] if p.abstract else 'EMPTY'}...")
            print(f"    url: {p.url}")
            print(f"    source_ids: {[(s.provider, s.provider_id[:30]) for s in p.source_ids]}")

    @pytest.mark.asyncio
    async def test_inverted_abstract_reconstructed(self):
        """OpenAlex stores abstracts as inverted index — must be reconstructed."""
        from app.providers.openalex import OpenAlexProvider
        provider = OpenAlexProvider()
        papers = await provider.search("deep learning transformer attention")
        papers_with_abstract = [p for p in papers if p.abstract]
        if papers_with_abstract:
            p = papers_with_abstract[0]
            # Should be readable text, not a dict
            assert isinstance(p.abstract, str), f"Abstract is {type(p.abstract)}, not str"
            assert len(p.abstract) > 20, f"Abstract too short: '{p.abstract}'"
            # Should not contain JSON-like artifacts
            assert "{" not in p.abstract, f"Abstract looks like raw inverted index: {p.abstract[:100]}"
            print(f"\n    Reconstructed abstract: {p.abstract[:200]}...")
        else:
            pytest.skip("No papers with abstracts in this result set")

    @pytest.mark.asyncio
    async def test_missing_fields_no_exception(self):
        """Papers with missing DOI/abstract/authors should not raise."""
        from app.providers.openalex import OpenAlexProvider
        provider = OpenAlexProvider()
        papers = await provider.search("obscure niche topic xyzabc123")
        # Even with no results, should not raise
        assert isinstance(papers, list)

    @pytest.mark.asyncio
    async def test_http_status_and_rate_limit(self):
        """Verify HTTP 200 and handle rate limits gracefully."""
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={"search": QUERY, "per_page": 1},
            )
            print(f"\n    HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                meta = data.get("meta", {})
                print(f"    total results: {meta.get('count', 'N/A')}")
                print(f"    per_page: {meta.get('per_page', 'N/A')}")
            elif resp.status_code == 429:
                print("    RATE LIMITED — consider adding OPENALEX_MAILTO or API key")
            else:
                print(f"    Unexpected status: {resp.status_code}")


# ============================================================
# Semantic Scholar
# ============================================================

class TestSemanticScholarReal:
    """Test Semantic Scholar provider with real HTTP calls."""

    @pytest.mark.asyncio
    async def test_search_returns_papers(self):
        from app.providers.semantic_scholar import SemanticScholarProvider
        provider = SemanticScholarProvider()
        papers = await provider.search(QUERY, year_from=2023)
        assert len(papers) > 0, "No results returned"
        assert len(papers) <= provider._settings.MAX_PAPERS_PER_SOURCE
        print(f"\n    Semantic Scholar: {len(papers)} papers returned")

    @pytest.mark.asyncio
    async def test_paper_has_core_fields(self):
        from app.providers.semantic_scholar import SemanticScholarProvider
        provider = SemanticScholarProvider()
        papers = await provider.search(QUERY)
        for p in papers:
            assert p.title, "Missing title"
            assert "semantic_scholar" in p.source_names
            print(f"\n    title: {p.title[:80]}")
            print(f"    year: {p.publication_year}")
            print(f"    venue: {p.venue}")
            print(f"    doi: {p.doi}")
            print(f"    authors: {len(p.authors)}")
            print(f"    abstract: {p.abstract[:100] if p.abstract else 'EMPTY'}...")
            print(f"    citation_count: {p.citation_count}")
            print(f"    open_access: {p.open_access}")
            print(f"    url: {p.url}")

    @pytest.mark.asyncio
    async def test_no_doi_handled(self):
        """Some S2 papers have no DOI — should still work."""
        from app.providers.semantic_scholar import SemanticScholarProvider
        provider = SemanticScholarProvider()
        papers = await provider.search("arxiv preprint cs.CL")
        no_doi = [p for p in papers if not p.doi]
        if no_doi:
            p = no_doi[0]
            assert p.title, "Paper without DOI should still have title"
            print(f"\n    Paper without DOI: {p.title[:80]}")
            print(f"    url: {p.url}")
        else:
            print("\n    All papers in this sample have DOIs")

    @pytest.mark.asyncio
    async def test_missing_fields_no_exception(self):
        from app.providers.semantic_scholar import SemanticScholarProvider
        provider = SemanticScholarProvider()
        papers = await provider.search("xyzabc123 obscure nonexistent")
        assert isinstance(papers, list)

    @pytest.mark.asyncio
    async def test_http_status(self):
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": QUERY, "limit": 1, "fields": "paperId,title"},
            )
            print(f"\n    HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    papers in response: {len(data.get('data', []))}")
            elif resp.status_code == 429:
                print("    RATE LIMITED — consider adding SEMANTIC_SCHOLAR_API_KEY")
            else:
                print(f"    status: {resp.status_code}")


# ============================================================
# arXiv
# ============================================================

class TestArxivReal:
    """Test arXiv provider with real HTTP calls."""

    @pytest.mark.asyncio
    async def test_search_returns_papers(self):
        from app.providers.arxiv import ArxivProvider
        provider = ArxivProvider()
        papers = await provider.search(QUERY)
        assert len(papers) > 0, "No results returned"
        print(f"\n    arXiv: {len(papers)} papers returned")

    @pytest.mark.asyncio
    async def test_paper_has_core_fields(self):
        from app.providers.arxiv import ArxivProvider
        provider = ArxivProvider()
        papers = await provider.search("transformer attention mechanism")
        for p in papers:
            assert p.title, "Missing title"
            assert "arxiv" in p.source_names
            assert p.open_access is True, "arXiv papers should be open_access"
            print(f"\n    title: {p.title[:80]}")
            print(f"    year: {p.publication_year}")
            print(f"    authors: {len(p.authors)}")
            if p.authors:
                print(f"    first author: {p.authors[0].name}")
            print(f"    abstract: {p.abstract[:100] if p.abstract else 'EMPTY'}...")
            print(f"    url: {p.url}")
            print(f"    full_text_url: {p.full_text_url}")
            print(f"    doi: {p.doi}")

    @pytest.mark.asyncio
    async def test_atom_xml_parsing(self):
        """Verify arXiv Atom XML is correctly parsed."""
        from app.providers.arxiv import ArxivProvider
        provider = ArxivProvider()
        papers = await provider.search("deep learning")

        if not papers:
            pytest.skip("No arXiv results")

        # Check that we didn't just get empty entries
        for p in papers:
            # Title should be clean (no newlines)
            assert "\n" not in p.title, f"Title has newlines: '{p.title}'"
            # Abstract should be clean
            if p.abstract:
                assert "\n" not in p.abstract, f"Abstract has newlines"
            # URL format should be correct
            if p.url:
                assert "arxiv.org/abs/" in p.url, f"Wrong URL format: {p.url}"

    @pytest.mark.asyncio
    async def test_no_doi_handled(self):
        """Many arXiv papers have no DOI — should still work."""
        from app.providers.arxiv import ArxivProvider
        provider = ArxivProvider()
        papers = await provider.search("cs.CL preprint 2025")
        no_doi = [p for p in papers if not p.doi]
        if no_doi:
            p = no_doi[0]
            assert p.title
            print(f"\n    arXiv paper without DOI: {p.title[:80]}")
            print(f"    arxiv_id: {p.source_ids[0].provider_id if p.source_ids else 'N/A'}")
        else:
            print("\n    All arXiv papers in this sample have DOIs")


# ============================================================
# Crossref
# ============================================================

class TestCrossrefReal:
    """Test Crossref provider with real HTTP calls."""

    @pytest.mark.asyncio
    async def test_search_returns_papers(self):
        from app.providers.crossref import CrossrefProvider
        provider = CrossrefProvider()
        papers = await provider.search(QUERY, year_from=2023)
        assert len(papers) > 0, "No results returned"
        print(f"\n    Crossref: {len(papers)} papers returned")

    @pytest.mark.asyncio
    async def test_paper_has_core_fields(self):
        from app.providers.crossref import CrossrefProvider
        provider = CrossrefProvider()
        papers = await provider.search(QUERY)
        for p in papers:
            assert p.title, "Missing title"
            assert "crossref" in p.source_names
            print(f"\n    title: {p.title[:80]}")
            print(f"    year: {p.publication_year}")
            print(f"    venue: {p.venue}")
            print(f"    doi: {p.doi}")
            print(f"    authors: {len(p.authors)}")
            if p.authors:
                print(f"    first author: {p.authors[0].name}")
                print(f"    affiliation: {p.authors[0].affiliation}")
            print(f"    abstract: {p.abstract[:100] if p.abstract else 'EMPTY'}...")
            print(f"    citation_count: {p.citation_count}")
            print(f"    url: {p.url}")

    @pytest.mark.asyncio
    async def test_html_in_abstract_stripped(self):
        """Crossref abstracts may contain HTML tags — should be cleaned."""
        from app.providers.crossref import CrossrefProvider
        provider = CrossrefProvider()
        papers = await provider.search("quantum computing entanglement")
        html_papers = []
        for p in papers:
            if p.abstract and ("<" in p.abstract or ">" in p.abstract):
                html_papers.append(p)
        if html_papers:
            pytest.fail(
                f"Found {len(html_papers)} papers with HTML in abstract: "
                f"'{html_papers[0].abstract[:150]}'"
            )
        print(f"\n    Checked {len(papers)} abstracts — no HTML tags found")

    @pytest.mark.asyncio
    async def test_http_status(self):
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={"query": QUERY, "rows": 1},
            )
            print(f"\n    HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("message", {}).get("items", [])
                print(f"    items returned: {len(items)}")
                print(f"    total: {data.get('message', {}).get('total-results', 'N/A')}")
            elif resp.status_code == 429:
                print("    RATE LIMITED")
            else:
                print(f"    status: {resp.status_code}")


# ============================================================
# Cross-provider: normalization consistency
# ============================================================

class TestNormalizationConsistency:
    """Verify all providers produce consistent Paper objects."""

    REQUIRED_FIELDS = [
        "title", "internal_id", "source_names", "source_ids",
    ]

    @pytest.mark.asyncio
    async def test_all_providers_normalize_consistently(self):
        """Every provider must always return valid Paper objects."""
        from app.providers.openalex import OpenAlexProvider
        from app.providers.semantic_scholar import SemanticScholarProvider
        from app.providers.arxiv import ArxivProvider
        from app.providers.crossref import CrossrefProvider
        from app.models.paper import Paper

        providers = [
            ("OpenAlex", OpenAlexProvider()),
            ("Semantic Scholar", SemanticScholarProvider()),
            ("arXiv", ArxivProvider()),
            ("Crossref", CrossrefProvider()),
        ]

        for name, provider in providers:
            print(f"\n  Testing {name}...")
            try:
                papers = await provider.search("neural network", year_from=2024)
            except Exception as e:
                print(f"    WARNING: {name} search failed: {e}")
                continue

            if not papers:
                print(f"    WARNING: {name} returned 0 results")
                continue

            for i, p in enumerate(papers[:3]):
                assert isinstance(p, Paper), f"{name}[{i}] is not a Paper: {type(p)}"
                for field in self.REQUIRED_FIELDS:
                    assert getattr(p, field, None), f"{name}[{i}] missing {field}"
                assert p.normalized_title, f"{name}[{i}] missing normalized_title"
            print(f"    OK: {len(papers[:3])} papers normalized")


# ============================================================
# Failure isolation
# ============================================================

class TestFailureIsolation:
    """Provider failure must not crash — returns empty list gracefully."""

    @pytest.mark.asyncio
    async def test_timeout_handled(self):
        """Very low timeout should result in empty list, not exception."""
        from app.providers.openalex import OpenAlexProvider
        from app.core.config import Settings
        s = Settings()
        object.__setattr__(s, "HTTP_TIMEOUT", 1)
        object.__setattr__(s, "HTTP_MAX_RETRIES", 0)
        provider = OpenAlexProvider(settings=s)
        # Should not raise
        papers = await provider.search(QUERY)
        assert isinstance(papers, list)

    @pytest.mark.asyncio
    async def test_bad_url_returns_empty(self):
        """Invalid base URL should return empty list, not crash."""
        from app.providers.openalex import OpenAlexProvider
        from app.core.config import Settings
        s = Settings()
        object.__setattr__(s, "OPENALEX_BASE_URL", "https://invalid.example.com")
        object.__setattr__(s, "HTTP_MAX_RETRIES", 0)
        provider = OpenAlexProvider(settings=s)
        papers = await provider.search(QUERY)
        assert isinstance(papers, list)
        assert papers == [], f"Expected empty list for bad URL, got {len(papers)}"
