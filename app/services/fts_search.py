"""Full-text search using SQLite FTS5 over DocumentChunks."""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import engine, async_session_factory
from app.db.models import DocumentChunkRecord

logger = logging.getLogger(__name__)

# FTS5 virtual table DDL
FTS5_CREATE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    paper_id,
    task_id,
    section_title,
    text,
    source_url,
    content='document_chunks',
    content_rowid='id'
)
"""

# Triggers to keep FTS5 in sync with document_chunks
FTS5_TRIGGER_SQLS = (
"""CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, paper_id, task_id, section_title, text, source_url)
    VALUES (NEW.id, NEW.chunk_id, NEW.paper_id, NEW.task_id, NEW.section_title, NEW.text, NEW.source_url);
END""",
"""CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, paper_id, task_id, section_title, text, source_url)
    VALUES ('delete', OLD.id, OLD.chunk_id, OLD.paper_id, OLD.task_id, OLD.section_title, OLD.text, OLD.source_url);
END""",
"""CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, paper_id, task_id, section_title, text, source_url)
    VALUES ('delete', OLD.id, OLD.chunk_id, OLD.paper_id, OLD.task_id, OLD.section_title, OLD.text, OLD.source_url);
    INSERT INTO chunks_fts(rowid, chunk_id, paper_id, task_id, section_title, text, source_url)
    VALUES (NEW.id, NEW.chunk_id, NEW.paper_id, NEW.task_id, NEW.section_title, NEW.text, NEW.source_url);
END""",
)


async def init_fts5() -> None:
    """Create FTS5 virtual table and triggers if not exist."""
    async with engine.begin() as conn:
        await conn.execute(text(FTS5_CREATE_SQL))
        for trigger in FTS5_TRIGGER_SQLS:
            await conn.execute(text(trigger))
    logger.info("FTS5 index initialized")


class FTSSearchResult:
    """One search result from FTS5."""

    def __init__(self, row):
        self.chunk_id: str = row.chunk_id
        self.paper_id: str = row.paper_id
        self.task_id: str = row.task_id
        self.section_title: Optional[str] = row.section_title
        self.text: str = row.text
        self.source_url: Optional[str] = row.source_url
        self.page_start: Optional[int] = row.page_start
        self.page_end: Optional[int] = row.page_end
        self.parser_name: Optional[str] = row.parser_name
        self.document_hash: Optional[str] = row.pdf_sha256
        self.fts_score: float = getattr(row, "score", 0.0)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "task_id": self.task_id,
            "section_title": self.section_title,
            "text": self.text[:500],
            "source_url": self.source_url,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "fts_score": self.fts_score,
        }


def _sanitize_fts5_query(query: str) -> str:
    """Escape special FTS5 characters to avoid syntax errors.

    FTS5 treats hyphens and some punctuation as column references.
    Wrap phrases in double quotes and remove problematic chars.
    """
    # Remove characters that FTS5 misinterprets
    import re
    # Keep alphanumeric, spaces, and double quotes (for phrase search)
    cleaned = re.sub(r'[^\w\s"]', " ", query)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def search_chunks(
    task_id: str,
    query: str,
    paper_ids: Optional[List[str]] = None,
    limit: int = 20,
) -> List[FTSSearchResult]:
    """Search document chunks using FTS5 BM25.

    Args:
        task_id: Limit search to chunks from this task.
        query: FTS5 query string (supports keywords, simple phrases).
        paper_ids: Optional list of paper IDs to restrict search to.
        limit: Max results to return.

    Returns:
        List of FTSSearchResult sorted by BM25 score descending.
    """
    query = _sanitize_fts5_query(query)
    if not query:
        return []

    # Build WHERE clause
    conditions = ["document_chunks.task_id = :task_id"]
    params = {"task_id": task_id, "limit": limit}

    if paper_ids:
        placeholders = ",".join(f":pid_{i}" for i in range(len(paper_ids)))
        conditions.append(f"document_chunks.paper_id IN ({placeholders})")
        for i, pid in enumerate(paper_ids):
            params[f"pid_{i}"] = pid

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            document_chunks.chunk_id,
            document_chunks.paper_id,
            document_chunks.task_id,
            document_chunks.section_title,
            document_chunks.text,
            document_chunks.source_url,
            document_chunks.page_start,
            document_chunks.page_end,
            document_chunks.parser_name,
            document_chunks.pdf_sha256,
            rank AS score
        FROM chunks_fts
        JOIN document_chunks ON chunks_fts.rowid = document_chunks.id
        WHERE chunks_fts MATCH :query
          AND {where_clause}
        ORDER BY rank
        LIMIT :limit
    """

    params["query"] = query

    async with async_session_factory() as session:
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    results = []
    for row in rows:
        results.append(FTSSearchResult(row))

    logger.info(
        f"FTS5 search: '{query[:80]}' → {len(results)} results "
        f"(task={task_id[:8]}...)"
    )
    return results


async def search_chunks_direct(
    task_id: str,
    keywords: List[str],
    paper_ids: Optional[List[str]] = None,
    limit: int = 20,
) -> List[FTSSearchResult]:
    """Fallback: search document_chunks with SQL LIKE when FTS5 is unavailable."""
    conditions = ["task_id = :task_id"]
    params = {"task_id": task_id, "limit": limit}

    if paper_ids:
        placeholders = ",".join(f":pid_{i}" for i in range(len(paper_ids)))
        conditions.append(f"paper_id IN ({placeholders})")
        for i, pid in enumerate(paper_ids):
            params[f"pid_{i}"] = pid

    # Build LIKE clauses for keyword matching
    like_clauses = []
    for i, kw in enumerate(keywords[:10]):
        kw_clean = kw.strip().lower()
        if len(kw_clean) > 2:
            like_clauses.append(f"(LOWER(text) LIKE :kw_{i} OR LOWER(section_title) LIKE :kw_{i})")
            params[f"kw_{i}"] = f"%{kw_clean}%"

    if not like_clauses:
        sql = f"""
            SELECT chunk_id, paper_id, task_id, section_title, text, source_url,
                   page_start, page_end, parser_name, pdf_sha256, 0.5 AS score
            FROM document_chunks
            WHERE {' AND '.join(conditions)}
            LIMIT :limit
        """
    else:
        where = " AND ".join(conditions + [f"({' OR '.join(like_clauses)})"])
        sql = f"""
            SELECT chunk_id, paper_id, task_id, section_title, text, source_url,
                   page_start, page_end, parser_name, pdf_sha256,
                   1.0 AS score
            FROM document_chunks
            WHERE {where}
            LIMIT :limit
        """

    async with async_session_factory() as session:
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    results = [FTSSearchResult(row) for row in rows]
    logger.info(f"Direct search: {len(keywords)} keywords → {len(results)} chunks")
    return results


async def search_by_keywords(
    task_id: str,
    keywords: List[str],
    paper_ids: Optional[List[str]] = None,
    limit: int = 20,
) -> List[FTSSearchResult]:
    """Search using keyword OR query (simpler than FTS5 syntax)."""
    # Build FTS5 query: keyword1 OR keyword2 OR ...
    fts_query = " OR ".join(
        k.replace('"', '').replace("'", "") for k in keywords if len(k) > 2
    )
    if not fts_query:
        return []
    return await search_chunks(task_id, fts_query, paper_ids, limit)


async def save_chunks(chunks: list) -> None:
    """Persist a list of DocumentChunk Pydantic models to DB."""
    async with async_session_factory() as session:
        for chunk in chunks:
            from sqlalchemy import select

            existing = await session.execute(
                select(DocumentChunkRecord.id).where(
                    DocumentChunkRecord.chunk_id == chunk.chunk_id
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue
            rec = DocumentChunkRecord(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                task_id=chunk.task_id,
                chunk_index=chunk.chunk_index,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                parent_chunk_id=chunk.parent_chunk_id,
                child_chunk_ids_json=json.dumps(chunk.child_chunk_ids),
                source_url=chunk.source_url,
                pdf_sha256=chunk.pdf_sha256,
                parser_name=chunk.parser_name,
                parser_version=chunk.parser_version,
                created_at=chunk.created_at,
            )
            session.add(rec)
        await session.commit()
    logger.info(f"Saved {len(chunks)} chunks to DB")
