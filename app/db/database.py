"""Database engine and session management."""

from __future__ import annotations

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
    """Create all tables."""
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the engine."""
    await engine.dispose()
