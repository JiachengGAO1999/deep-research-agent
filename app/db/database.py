"""Database engine and session management."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

_settings = get_settings()

# Convert sqlite+aiosqlite:///./storage/app.db to proper path if needed
database_url = _settings.DATABASE_URL
if database_url.startswith("sqlite+aiosqlite:///./"):
    import os

    db_path = database_url.replace("sqlite+aiosqlite:///./", "")
    abs_path = os.path.join(os.getcwd(), db_path)
    database_url = f"sqlite+aiosqlite:///{abs_path}"

engine = create_async_engine(database_url, echo=False, future=True)

async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncSession:
    """Get a new async database session."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables and FTS5 index."""
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if database_url.startswith("sqlite"):
            await _ensure_sqlite_columns(conn)

    # Initialize FTS5 virtual table
    try:
        from app.services.fts_search import init_fts5
        await init_fts5()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"FTS5 init failed (non-fatal): {e}")


async def _ensure_sqlite_columns(conn) -> None:
    """Small additive migration shim for development SQLite databases."""
    required = {
        "tasks": {
            "max_papers": "INTEGER DEFAULT 12",
            "research_depth": "VARCHAR(20) DEFAULT 'standard'",
            "evidence_backend": "VARCHAR(20) DEFAULT 'abstract'",
            "enable_full_text": "BOOLEAN DEFAULT 0",
            "report_language": "VARCHAR(20) DEFAULT 'zh-CN'",
            "retrieved_passages_json": "TEXT",
            "claims_json": "TEXT",
            "evidence_quality_json": "TEXT",
            "report_paper_ids_json": "TEXT",
        },
        "evidence": {
            "evidence_id": "VARCHAR(20)",
            "passage_id": "VARCHAR(100)",
            "sub_question_id": "VARCHAR(30)",
            "chunk_id": "VARCHAR(20)",
            "section_title": "TEXT",
            "page_start": "INTEGER",
            "page_end": "INTEGER",
            "source_url": "TEXT",
            "evidence_level": "VARCHAR(20)",
            "stance": "VARCHAR(20)",
            "evidence_type": "VARCHAR(30)",
            "verification_status": "VARCHAR(20)",
            "verification_reason": "TEXT",
            "confidence": "FLOAT",
        },
    }
    for table, columns in required.items():
        rows = await conn.execute(text(f"PRAGMA table_info({table})"))
        existing = {row[1] for row in rows.fetchall()}
        for column, ddl_type in columns.items():
            if column not in existing:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                )


async def close_db() -> None:
    """Dispose the engine."""
    await engine.dispose()
