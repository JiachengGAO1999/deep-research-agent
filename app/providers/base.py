"""Base provider interface and utilities."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.core.config import get_settings
from app.models.paper import Paper

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base for academic data source providers."""

    name: str = "base"
    # Set by each search() call: total hits in the API for this query
    last_total_hits: int = 0

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.HTTP_TIMEOUT),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> httpx.Response:
        """Make an HTTP GET request with exponential backoff."""
        client = await self._get_client()
        last_exception = None

        for attempt in range(self._settings.HTTP_MAX_RETRIES + 1):
            try:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 429:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        f"{self.name}: rate limited (429), waiting {wait}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                return response
            except httpx.TimeoutException as e:
                last_exception = e
                wait = 2 ** attempt
                logger.warning(f"{self.name}: timeout, retrying in {wait}s")
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_exception = e
                    wait = 2 ** attempt
                    logger.warning(
                        f"{self.name}: server error {e.response.status_code}, retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if attempt < self._settings.HTTP_MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(f"{self.name}: error, retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                    continue
                raise

        raise last_exception or RuntimeError(f"{self.name}: max retries exceeded")

    @abstractmethod
    async def search(
        self, query: str, year_from: Optional[int] = None, year_to: Optional[int] = None,
        search_intent=None,
    ) -> list[Paper]:
        """Execute a search and return normalized Paper objects."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is available."""
        ...
