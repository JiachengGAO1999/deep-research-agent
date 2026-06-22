"""Document storage abstraction — local filesystem implementation."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class DocumentStorage(ABC):
    """Abstract interface for PDF document storage."""

    @abstractmethod
    async def save_pdf(self, content: bytes, source_url: str) -> tuple[str, str]:
        """Save PDF bytes, return (sha256, file_path)."""
        ...

    @abstractmethod
    def get_pdf_path(self, sha256: str) -> Optional[str]:
        """Get the filesystem path for a cached PDF by SHA-256."""
        ...

    @abstractmethod
    def exists(self, sha256: str) -> bool:
        """Check if a PDF with this SHA-256 is already cached."""
        ...

    @abstractmethod
    async def delete(self, sha256: str) -> bool:
        """Delete a cached PDF by SHA-256. Returns True if deleted."""
        ...

    @abstractmethod
    async def cleanup_expired(self, active_task_ids: set[str]) -> list[str]:
        """Delete PDFs past TTL that have no active tasks. Returns list of deleted SHAs."""
        ...

    @abstractmethod
    def get_cache_stats(self) -> dict:
        """Return cache statistics: count, total_bytes."""
        ...


class LocalDocumentStorage(DocumentStorage):
    """Local filesystem PDF storage with SHA-256 content addressing."""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        base = Path(self._settings.PDF_CACHE_DIR)
        if not base.is_absolute():
            base = Path.cwd() / base
        self._cache_dir = base
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _sha256_path(self, sha256: str) -> Path:
        """Path to the cached PDF file."""
        return self._cache_dir / f"{sha256}.pdf"

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def exists(self, sha256: str) -> bool:
        return self._sha256_path(sha256).is_file()

    def get_pdf_path(self, sha256: str) -> Optional[str]:
        p = self._sha256_path(sha256)
        return str(p) if p.is_file() else None

    async def save_pdf(self, content: bytes, source_url: str) -> tuple[str, str]:
        """Save PDF bytes. Returns (sha256, file_path). Does NOT overwrite if exists."""
        sha256 = self.compute_sha256(content)
        file_path = self._sha256_path(sha256)

        if not file_path.is_file():
            # Verify PDF signature before saving
            if not content.startswith(b"%PDF"):
                raise ValueError(f"File does not start with PDF signature: {source_url}")
            file_path.write_bytes(content)
            logger.info(f"PDF cached: {sha256[:12]}... ({len(content)} bytes) from {source_url[:80]}")
        else:
            logger.debug(f"PDF already cached: {sha256[:12]}...")

        return sha256, str(file_path)

    async def delete(self, sha256: str) -> bool:
        p = self._sha256_path(sha256)
        if p.is_file():
            p.unlink()
            logger.info(f"PDF deleted from cache: {sha256[:12]}...")
            return True
        return False

    async def cleanup_expired(self, active_task_ids: set[str]) -> list[str]:
        """Delete PDFs past TTL that have no active tasks. Returns list of deleted SHAs."""
        from app.db.database import async_session_factory
        from app.db.models import PDFCacheRecord
        from sqlalchemy import select

        ttl_days = self._settings.PDF_CACHE_TTL_DAYS
        deadline = time.time() - (ttl_days * 86400)
        deleted = []

        async with async_session_factory() as session:
            result = await session.execute(
                select(PDFCacheRecord)
            )
            records = result.scalars().all()

            for rec in records:
                # Check TTL
                try:
                    from datetime import datetime
                    acquired = datetime.fromisoformat(rec.acquired_at)
                    age_days = (datetime.utcnow() - acquired).days
                except Exception:
                    continue

                if age_days <= ttl_days:
                    continue

                # Check active task references
                import json
                active_tasks = json.loads(rec.active_task_ids_json or "[]")
                if any(t in active_task_ids for t in active_tasks):
                    continue

                # Safe to delete
                p = self._sha256_path(rec.sha256)
                if p.is_file():
                    p.unlink()
                    await session.delete(rec)
                    deleted.append(rec.sha256)
                    logger.info(f"Cleanup: deleted expired PDF {rec.sha256[:12]}... (age: {age_days}d)")

            if deleted:
                await session.commit()

        return deleted

    def get_cache_stats(self) -> dict:
        total_bytes = 0
        count = 0
        for f in self._cache_dir.glob("*.pdf"):
            total_bytes += f.stat().st_size
            count += 1
        return {"count": count, "total_bytes": total_bytes}
