"""PDF parsing service — Docling primary, PyMuPDF fallback."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import re
from typing import List, Optional

from app.models.document import DocumentChunk, ParseResult, ParseStatus
from app.models.paper import Paper

logger = logging.getLogger(__name__)

_DOCLING_AVAILABLE = importlib.util.find_spec("docling") is not None
_PYMUPDF_AVAILABLE = importlib.util.find_spec("fitz") is not None
_DOCLING_VERSION = "unknown"
_PYMUPDF_VERSION = "unknown"


class PDFParser:
    """Parse PDFs into structured DocumentChunks.

    Primary: Docling (structure-aware, headings, sections, tables, figures)
    Fallback: PyMuPDF (page-by-page text extraction)
    """

    def __init__(self, backend: Optional[str] = None):
        from app.core.config import get_settings

        self._backend = (backend or get_settings().PDF_PARSER_BACKEND).lower()
        self._docling_converter = None

    def _get_docling_converter(self):
        """Initialize Docling only when explicitly selected and first used."""
        global _DOCLING_VERSION
        if self._docling_converter is not None:
            return self._docling_converter
        if not _DOCLING_AVAILABLE:
            return None
        try:
            docling = importlib.import_module("docling")
            converter_module = importlib.import_module("docling.document_converter")
            converter_cls = getattr(converter_module, "DocumentConverter")
            _DOCLING_VERSION = getattr(docling, "__version__", "installed")
            self._docling_converter = converter_cls()
            logger.info("Docling parser ready (v%s)", _DOCLING_VERSION)
        except Exception as e:
            logger.warning("Failed to initialize Docling: %s", e)
        return self._docling_converter

    @staticmethod
    def _get_fitz():
        global _PYMUPDF_VERSION
        if not _PYMUPDF_AVAILABLE:
            return None
        fitz_module = importlib.import_module("fitz")
        _PYMUPDF_VERSION = getattr(fitz_module, "__version__", "installed")
        return fitz_module

    async def parse(
        self,
        pdf_path: str,
        paper: Paper,
        task_id: str,
    ) -> ParseResult:
        """Parse a PDF file into structured chunks."""
        # Compute SHA-256
        sha256 = ""
        try:
            with open(pdf_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

        # Docling is opt-in because initialization may download model assets.
        if self._backend == "docling" and self._get_docling_converter() is not None:
            try:
                return await self._parse_with_docling(pdf_path, paper, task_id, sha256)
            except Exception as e:
                logger.warning(
                    f"Docling failed for {paper.internal_id}: {e}. Falling back to PyMuPDF."
                )

        # Fallback to PyMuPDF
        if _PYMUPDF_AVAILABLE:
            try:
                return await self._parse_with_pymupdf(pdf_path, paper, task_id, sha256)
            except Exception as e:
                logger.error(f"PyMuPDF also failed for {paper.internal_id}: {e}")
                return ParseResult(
                    paper_id=paper.internal_id,
                    pdf_sha256=sha256,
                    parser_name="none",
                    parser_version="0",
                    status=ParseStatus.FAILED,
                    error_message=f"Both parsers failed: {e}",
                )

        return ParseResult(
            paper_id=paper.internal_id,
            pdf_sha256=sha256,
            parser_name="none",
            parser_version="0",
            status=ParseStatus.FAILED,
            error_message="No PDF parser available",
        )

    async def _parse_with_docling(
        self, pdf_path: str, paper: Paper, task_id: str, sha256: str
    ) -> ParseResult:
        """Docling: structure-aware parsing with heading/section detection."""
        import asyncio

        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(None, self._docling_convert, pdf_path)

        # num_pages is a method on some versions, property on others
        num_pages = 0
        try:
            np = doc.num_pages
            if callable(np):
                num_pages = np()
            else:
                num_pages = int(np)
        except Exception:
            pass

        parent_chunks: List[DocumentChunk] = []
        child_chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # Strategy: use Docling's item tree. Group items under headings.
        sections = self._extract_sections_from_docling(doc)
        if not sections:
            # Fallback: export full markdown, split by ## headings
            md_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
            sections = self._extract_sections_from_markdown(md_text)

        for sec_idx, section in enumerate(sections):
            section_title = section["title"]
            section_text = section["text"]
            page_start = section.get("page_start")
            page_end = section.get("page_end")

            if not section_text or len(section_text.strip()) < 20:
                continue

            # Parent chunk = entire section
            parent = DocumentChunk(
                chunk_id=f"{paper.internal_id}_s{chunk_idx}",
                paper_id=paper.internal_id,
                task_id=task_id,
                chunk_index=chunk_idx,
                section_title=section_title,
                page_start=page_start,
                page_end=page_end,
                text=section_text,
                source_url=paper.full_text_url or paper.url,
                pdf_sha256=sha256,
                parser_name="docling",
                parser_version=_DOCLING_VERSION,
            )
            parent_chunks.append(parent)
            chunk_idx += 1

            # Child chunks = paragraphs within section
            paragraphs = self._split_into_paragraphs(section_text)
            for para_text in paragraphs:
                if len(para_text.strip()) < 30:
                    continue
                child = DocumentChunk(
                    chunk_id=f"{paper.internal_id}_c{chunk_idx}",
                    paper_id=paper.internal_id,
                    task_id=task_id,
                    chunk_index=chunk_idx,
                    section_title=section_title,
                    page_start=page_start,
                    page_end=page_end,
                    text=para_text,
                    parent_chunk_id=parent.chunk_id,
                    source_url=paper.full_text_url or paper.url,
                    pdf_sha256=sha256,
                    parser_name="docling",
                    parser_version=_DOCLING_VERSION,
                )
                parent.child_chunk_ids.append(child.chunk_id)
                child_chunks.append(child)
                chunk_idx += 1

        logger.info(
            f"Docling: {paper.internal_id} → {len(parent_chunks)} sections, "
            f"{len(child_chunks)} paragraphs (pages: {num_pages})"
        )

        return ParseResult(
            paper_id=paper.internal_id,
            pdf_sha256=sha256,
            parser_name="docling",
            parser_version=_DOCLING_VERSION,
            num_pages=num_pages,
            num_sections=len(parent_chunks),
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )

    async def _parse_with_pymupdf(
        self, pdf_path: str, paper: Paper, task_id: str, sha256: str
    ) -> ParseResult:
        """Fallback: page-by-page text extraction using PyMuPDF."""
        import asyncio

        loop = asyncio.get_running_loop()

        def _extract():
            fitz = self._get_fitz()
            if fitz is None:
                raise RuntimeError("PyMuPDF is not installed")
            doc = fitz.open(pdf_path)
            pages_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text and text.strip():
                    pages_text.append((page_num + 1, text.strip()))
            doc.close()
            return pages_text

        pages_text = await loop.run_in_executor(None, _extract)

        parent_chunks: List[DocumentChunk] = []
        child_chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page_num, page_text in pages_text:
            parent = DocumentChunk(
                chunk_id=f"{paper.internal_id}_p{chunk_idx}",
                paper_id=paper.internal_id,
                task_id=task_id,
                chunk_index=chunk_idx,
                section_title=f"Page {page_num}",
                page_start=page_num,
                page_end=page_num,
                text=page_text,
                source_url=paper.full_text_url or paper.url,
                pdf_sha256=sha256,
                parser_name="pymupdf",
                parser_version=_PYMUPDF_VERSION,
            )
            parent_chunks.append(parent)
            chunk_idx += 1

            paragraphs = self._split_into_paragraphs(page_text)
            for para_text in paragraphs:
                if len(para_text.strip()) < 30:
                    continue
                child = DocumentChunk(
                    chunk_id=f"{paper.internal_id}_c{chunk_idx}",
                    paper_id=paper.internal_id,
                    task_id=task_id,
                    chunk_index=chunk_idx,
                    section_title=f"Page {page_num}",
                    page_start=page_num,
                    page_end=page_num,
                    text=para_text,
                    parent_chunk_id=parent.chunk_id,
                    source_url=paper.full_text_url or paper.url,
                    pdf_sha256=sha256,
                    parser_name="pymupdf",
                    parser_version=_PYMUPDF_VERSION,
                )
                parent.child_chunk_ids.append(child.chunk_id)
                child_chunks.append(child)
                chunk_idx += 1

        logger.info(
            f"PyMuPDF: {paper.internal_id} → {len(pages_text)} pages, "
            f"{len(child_chunks)} paragraphs"
        )

        return ParseResult(
            paper_id=paper.internal_id,
            pdf_sha256=sha256,
            parser_name="pymupdf",
            parser_version=_PYMUPDF_VERSION,
            num_pages=len(pages_text),
            num_sections=len(parent_chunks),
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )

    # ---- Docling item tree → sections ----

    def _extract_sections_from_docling(self, doc) -> List[dict]:
        """Walk Docling's item tree and group items under headings."""
        sections: List[dict] = []
        current = {"title": "Abstract / Introduction", "text": "", "page_start": None, "page_end": None}

        # Try iterate_items() first
        items = []
        if hasattr(doc, "iterate_items"):
            try:
                items = list(doc.iterate_items())
            except Exception:
                pass

        # If no items, try doc.texts
        if not items and hasattr(doc, "texts"):
            items = doc.texts or []

        for item in items:
            label = getattr(item, "label", "text")
            text = ""
            if hasattr(item, "text"):
                text = item.text or ""
            elif isinstance(item, str):
                text = item

            if not text:
                # Tables, figures: record as note
                if label in ("table", "figure", "picture"):
                    caption = ""
                    for c in (getattr(item, "children", None) or []):
                        if hasattr(c, "text") and c.text:
                            caption += c.text + " "
                    if caption:
                        current["text"] += f"\n[{label}: {caption.strip()}]\n"
                continue

            page = None
            prov = getattr(item, "prov", None)
            if prov and hasattr(prov, "page_no"):
                page = prov.page_no

            # Is this a heading?
            is_heading = label in (
                "section_header", "section-header", "heading", "title",
                "chapter_title", "subtitle",
            ) or (label == "text" and text and text.strip().startswith("#"))

            if is_heading and len(current["text"].strip()) > 50:
                # Save current section, start new
                sections.append(current)
                current = {"title": text.strip()[:120], "text": "", "page_start": page, "page_end": page}
            else:
                if is_heading:
                    # Small heading — update title but keep accumulating
                    current["title"] = text.strip()[:120]
                current["text"] += text + "\n"
                if page and current["page_start"] is None:
                    current["page_start"] = page
                if page:
                    current["page_end"] = page

        if current["text"].strip():
            sections.append(current)

        return sections

    def _extract_sections_from_markdown(self, md_text: str) -> List[dict]:
        """Fallback: split markdown by ## headings."""
        if not md_text:
            return []
        sections = []
        parts = re.split(r"\n(?=#{1,3}\s)", md_text)
        for part in parts:
            lines = part.strip().split("\n", 1)
            title = lines[0].lstrip("#").strip() if lines else "Untitled"
            body = lines[1] if len(lines) > 1 else ""
            if len(body.strip()) > 20:
                sections.append({"title": title[:120], "text": body, "page_start": None, "page_end": None})
        return sections

    # ---- Helpers ----

    def _docling_convert(self, pdf_path: str):
        """Synchronous Docling conversion (runs in thread pool)."""
        result = self._docling_converter.convert(pdf_path)
        return result.document

    @staticmethod
    def _split_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs by blank lines."""
        parts = text.split("\n\n")
        return [p.strip() for p in parts if len(p.strip()) > 20]
