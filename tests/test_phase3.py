"""Tests for Phase 3: PDF download, parsing, chunking, FTS5, lifecycle."""

import os
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def _init_test_db():
    """Initialize DB tables + FTS5 once per module."""
    from app.db.database import init_db
    await init_db()


# ============================================================
# Storage
# ============================================================

class TestLocalDocumentStorage:
    def test_sha256_computation(self):
        from app.core.storage import LocalDocumentStorage
        storage = LocalDocumentStorage()
        sha = storage.compute_sha256(b"hello")
        assert len(sha) == 64
        assert sha == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    @pytest.mark.asyncio
    async def test_pdf_signature_check(self):
        """Non-PDF content should be rejected."""
        from app.core.storage import LocalDocumentStorage
        import pytest as pt
        storage = LocalDocumentStorage()

        with pt.raises(ValueError, match="PDF signature"):
            await storage.save_pdf(b"not a pdf", "http://example.com/test.pdf")

    @pytest.mark.asyncio
    async def test_exists_and_get_path(self):
        from app.core.storage import LocalDocumentStorage
        storage = LocalDocumentStorage()
        sha256, path = await storage.save_pdf(
            b"%PDF-1.4 fake pdf content", "http://example.com/test.pdf"
        )
        assert storage.exists(sha256)
        assert storage.get_pdf_path(sha256) == path
        # Cleanup
        import os
        os.remove(path)

    @pytest.mark.asyncio
    async def test_duplicate_save_does_not_overwrite(self):
        from app.core.storage import LocalDocumentStorage
        storage = LocalDocumentStorage()
        sha1, p1 = await storage.save_pdf(
            b"%PDF-1.4 content", "http://a.com/a.pdf"
        )
        sha2, p2 = await storage.save_pdf(
            b"%PDF-1.4 content", "http://b.com/b.pdf"
        )
        assert sha1 == sha2, "Same content should have same SHA-256"
        assert p1 == p2, "Same content should use same file"
        path = p1
        import os
        os.remove(path)


# ============================================================
# PDF Downloader (mock behavior)
# ============================================================

class TestPDFDownloader:
    @pytest.mark.asyncio
    async def test_invalid_url_returns_error(self):
        from app.services.pdf_downloader import PDFDownloader
        downloader = PDFDownloader()
        sha256, path, error = await downloader.download("")
        assert sha256 is None
        assert error is not None

    @pytest.mark.asyncio
    async def test_non_pdf_url_warns(self):
        from app.services.pdf_downloader import PDFDownloader
        downloader = PDFDownloader()
        sha256, path, error = await downloader.download("http://example.com/not-a-pdf.html")
        assert sha256 is None or error is not None


# ============================================================
# PDF Parser (PyMuPDF)
# ============================================================

class TestPDFParser:
    def test_pymupdf_available(self):
        import fitz
        assert fitz.__version__

    def test_parser_initializes(self):
        from app.services.pdf_parser import PDFParser
        parser = PDFParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self):
        from app.services.pdf_parser import PDFParser
        from app.models.paper import Paper
        parser = PDFParser()
        result = await parser.parse("/nonexistent/path.pdf", Paper(title="Test", internal_id="t1"), "task1")
        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_parse_minimal_pdf(self):
        """Create a minimal valid PDF and parse it."""
        from app.services.pdf_parser import PDFParser
        from app.models.paper import Paper

        import tempfile
        import fitz
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Reasoning accuracy degrades across multiple conversation turns.",
        )
        doc.save(pdf_path)
        doc.close()

        try:
            parser = PDFParser()
            paper = Paper(
                internal_id="test_p1",
                title="Test PDF",
                full_text_url="http://example.com/test.pdf",
            )
            result = await parser.parse(pdf_path, paper, "task_test")

            assert result.status.value == "completed", result.error_message
            assert result.num_pages == 1
            assert result.parent_chunks
            assert "reasoning accuracy" in result.parent_chunks[0].text.lower()
        finally:
            os.unlink(pdf_path)


# ============================================================
# FTS5
# ============================================================

class TestFTS5:
    @pytest.mark.asyncio
    async def test_fts5_init(self):
        from app.services.fts_search import init_fts5
        await init_fts5()  # Should not raise

    @pytest.mark.asyncio
    async def test_save_and_search_chunks(self):
        from app.services.fts_search import init_fts5, save_chunks, search_chunks
        from app.models.document import DocumentChunk

        await init_fts5()

        # Create a test chunk
        chunk = DocumentChunk(
            chunk_id="test_fts_chunk_1",
            paper_id="test_p1",
            task_id="test_task_fts",
            chunk_index=0,
            section_title="Introduction",
            text="Large language models exhibit reasoning degradation in multi-turn dialogue settings.",
            source_url="http://example.com/test.pdf",
            pdf_sha256="a" * 64,
            parser_name="pymupdf",
            parser_version="1.0",
        )

        await save_chunks([chunk])

        # Search — use simple keywords without hyphens (FTS5 treats hyphens as column refs)
        results = await search_chunks(
            task_id="test_task_fts",
            query="reasoning degradation dialogue",
            limit=5,
        )
        assert results
        assert results[0].paper_id == "test_p1"

    @pytest.mark.asyncio
    async def test_search_by_keywords(self):
        from app.services.fts_search import search_by_keywords, save_chunks, init_fts5
        from app.models.document import DocumentChunk

        await init_fts5()

        chunk = DocumentChunk(
            chunk_id="test_fts_kw_1",
            paper_id="test_kw_p1",
            task_id="test_task_kw",
            chunk_index=0,
            section_title="Results",
            text="Our experiments show that context accumulation negatively affects reasoning accuracy across multiple conversation turns.",
            parser_name="pymupdf",
            parser_version="1.0",
        )
        await save_chunks([chunk])

        results = await search_by_keywords(
            task_id="test_task_kw",
            keywords=["context accumulation", "reasoning accuracy", "conversation"],
            limit=5,
        )
        assert results
        assert results[0].paper_id == "test_kw_p1"


# ============================================================
# PDF Lifecycle
# ============================================================

class TestPDFLifecycle:
    @pytest.mark.asyncio
    async def test_cleanup_noop_when_empty(self):
        from app.services.pdf_lifecycle import PDFLifecycleManager
        manager = PDFLifecycleManager()
        stats = await manager.cleanup()
        assert stats["deleted_count"] == 0
        assert isinstance(stats["cache_stats"], dict)


# ============================================================
# Document models
# ============================================================

class TestDocumentModels:
    def test_chunk_has_required_fields(self):
        from app.models.document import DocumentChunk
        chunk = DocumentChunk(
            paper_id="p1",
            task_id="t1",
            text="Sample text",
        )
        assert chunk.chunk_id
        assert chunk.paper_id == "p1"
        assert chunk.task_id == "t1"

    def test_pdf_cache_entry_expiry(self):
        from app.models.document import PDFCacheEntry
        from datetime import datetime, timedelta

        # Old entry with no active tasks should be expired
        old = PDFCacheEntry(
            sha256="a" * 64,
            source_url="http://example.com/old.pdf",
            file_path="/tmp/old.pdf",
            acquired_at=(datetime.utcnow() - timedelta(days=10)).isoformat(),
            ttl_days=7,
        )
        assert old.is_expired

        # Recent entry should not be expired
        recent = PDFCacheEntry(
            sha256="b" * 64,
            source_url="http://example.com/new.pdf",
            file_path="/tmp/new.pdf",
            acquired_at=datetime.utcnow().isoformat(),
            ttl_days=7,
        )
        assert not recent.is_expired

        # Old entry with active tasks should NOT be expired
        old_active = PDFCacheEntry(
            sha256="c" * 64,
            source_url="http://example.com/active.pdf",
            file_path="/tmp/active.pdf",
            acquired_at=(datetime.utcnow() - timedelta(days=10)).isoformat(),
            ttl_days=7,
            active_task_ids=["task_123"],
        )
        assert not old_active.is_expired
