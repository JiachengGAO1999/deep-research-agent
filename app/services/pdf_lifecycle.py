"""PDF cache lifecycle management — TTL cleanup, task reference tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional, Set

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.storage import LocalDocumentStorage
from app.db.database import async_session_factory
from app.db.models import PDFCacheRecord

logger = logging.getLogger(__name__)


class PDFLifecycleManager:
    """Manage PDF cache lifecycle: track references, cleanup expired PDFs."""

    def __init__(self, settings=None, storage=None):
        self._settings = settings or get_settings()
        self._storage = storage or LocalDocumentStorage(settings=self._settings)

    async def register_download(
        self,
        session: AsyncSession,
        sha256: str,
        source_url: str,
        file_path: str,
        file_size_bytes: int,
        task_id: str,
        content_type: Optional[str] = None,
        open_access_status: Optional[str] = None,
    ) -> PDFCacheRecord:
        """Register a newly downloaded PDF in the DB."""
        existing = await session.execute(
            select(PDFCacheRecord).where(PDFCacheRecord.sha256 == sha256)
        )
        current = existing.scalar_one_or_none()
        if current is not None:
            tasks = json.loads(current.active_task_ids_json or "[]")
            if task_id not in tasks:
                tasks.append(task_id)
            current.active_task_ids_json = json.dumps(tasks)
            current.last_accessed_at = datetime.utcnow().isoformat()
            await session.commit()
            return current
        rec = PDFCacheRecord(
            sha256=sha256,
            source_url=source_url,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            content_type=content_type,
            open_access_status=open_access_status,
            active_task_ids_json=json.dumps([task_id]),
            acquired_at=datetime.utcnow().isoformat(),
            last_accessed_at=datetime.utcnow().isoformat(),
        )
        session.add(rec)
        await session.commit()
        return rec

    async def add_task_reference(self, sha256: str, task_id: str) -> None:
        """Mark a PDF as referenced by an active task."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(PDFCacheRecord).where(PDFCacheRecord.sha256 == sha256)
            )
            rec = result.scalar_one_or_none()
            if rec:
                tasks = json.loads(rec.active_task_ids_json or "[]")
                if task_id not in tasks:
                    tasks.append(task_id)
                    rec.active_task_ids_json = json.dumps(tasks)
                    rec.last_accessed_at = datetime.utcnow().isoformat()
                    await session.commit()

    async def remove_task_reference(self, sha256: str, task_id: str) -> None:
        """Remove a task reference — called when task completes."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(PDFCacheRecord).where(PDFCacheRecord.sha256 == sha256)
            )
            rec = result.scalar_one_or_none()
            if rec:
                tasks = json.loads(rec.active_task_ids_json or "[]")
                if task_id in tasks:
                    tasks.remove(task_id)
                    rec.active_task_ids_json = json.dumps(tasks)
                    await session.commit()

    async def cleanup(self) -> dict:
        """Run one lightweight cleanup pass. Returns stats dict."""
        settings = self._settings
        ttl_days = settings.PDF_CACHE_TTL_DAYS
        deadline = datetime.utcnow()
        deleted_count = 0
        deleted_bytes = 0
        errors = 0

        async with async_session_factory() as session:
            result = await session.execute(select(PDFCacheRecord))
            records = result.scalars().all()

            for rec in records:
                try:
                    acquired = datetime.fromisoformat(rec.acquired_at)
                    age_days = (deadline - acquired).days
                except Exception:
                    continue

                if age_days <= ttl_days:
                    continue

                # Check active tasks
                tasks = json.loads(rec.active_task_ids_json or "[]")
                if tasks:
                    logger.debug(f"PDF {rec.sha256[:12]}... has active tasks: {tasks}")
                    continue

                # Delete file
                try:
                    deleted = await self._storage.delete(rec.sha256)
                    if deleted:
                        deleted_count += 1
                        deleted_bytes += rec.file_size_bytes
                except Exception as e:
                    logger.warning(f"Failed to delete PDF {rec.sha256[:12]}...: {e}")
                    errors += 1
                    continue

                # Remove DB record
                await session.delete(rec)

            await session.commit()

        stats = {
            "deleted_count": deleted_count,
            "deleted_bytes": deleted_bytes,
            "errors": errors,
            "cache_stats": self._storage.get_cache_stats(),
        }
        if deleted_count > 0:
            logger.info(f"PDF cleanup: removed {deleted_count} files ({deleted_bytes} bytes)")
        return stats

    async def get_active_pdf_tasks(self) -> Set[str]:
        """Get all task IDs that currently reference cached PDFs."""
        active: Set[str] = set()
        async with async_session_factory() as session:
            result = await session.execute(select(PDFCacheRecord))
            for rec in result.scalars().all():
                tasks = json.loads(rec.active_task_ids_json or "[]")
                active.update(tasks)
        return active
