"""Tavily Search API client for Quick Research mode.

Encapsulates Tavily Search and Extract endpoints with:
- Timeout, limited retries, and 429 handling
- No logging of full API key
- Search and Extract separated
- Tavily's own 'answer' field is never used as evidence
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyClientError(Exception):
    """Raised when Tavily API returns an error or is unavailable."""


class TavilyNotConfiguredError(TavilyClientError):
    """Raised when TAVILY_API_KEY is missing."""


def _normalise_url(url: str) -> str:
    """Canonicalise a URL for dedup: lowercase host, strip www, strip trailing slash."""
    try:
        parsed = urlparse(url.strip().lower())
        netloc = parsed.netloc.removeprefix("www.")
        path = parsed.path.rstrip("/") or "/"
        query = parsed.query
        return f"{parsed.scheme}://{netloc}{path}{'?' + query if query else ''}"
    except Exception:
        return url.strip().lower()


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


class TavilyClient:
    """Async client for Tavily Search and Extract APIs."""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    def _check_key(self) -> None:
        if not self._settings.has_tavily_key:
            raise TavilyNotConfiguredError(
                "TAVILY_API_KEY is not set or is empty. "
                "Quick Research mode requires a Tavily API key. "
                "Set TAVILY_API_KEY in your .env file or environment."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._check_key()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.TAVILY_TIMEOUT),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post(self, url: str, payload: dict) -> dict:
        """POST to Tavily with retries and 429 handling."""
        client = await self._get_client()

        for attempt in range(3):
            try:
                response = await client.post(url, json=payload)
                if response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"Tavily rate limited (429), retrying in {wait}s "
                        f"(attempt {attempt + 1}/3)"
                    )
                    await _asleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"Tavily rate limited (429), retrying in {wait}s "
                        f"(attempt {attempt + 1}/3)"
                    )
                    await _asleep(wait)
                    continue
                logger.error(
                    f"Tavily HTTP {e.response.status_code}: "
                    f"{e.response.text[:500]}"
                )
                raise TavilyClientError(
                    f"Tavily API error: {e.response.status_code}"
                ) from e
            except httpx.TimeoutException:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Tavily timeout, retrying in {wait}s")
                    await _asleep(wait)
                    continue
                raise TavilyClientError("Tavily API timeout after 3 attempts")

        raise TavilyClientError("Tavily API request failed after all retries")

    async def search(
        self,
        query: str,
        search_depth: Optional[str] = None,
        max_results: Optional[int] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        include_answer: bool = False,
        days: Optional[int] = None,
    ) -> dict:
        """Execute a Tavily search query.

        Args:
            query: The search query string.
            search_depth: "basic" or "advanced". Defaults to TAVILY_SEARCH_DEPTH.
            max_results: Max results to return. Defaults to TAVILY_MAX_RESULTS_PER_QUERY.
            include_domains: Optional list of domains to include.
            exclude_domains: Optional list of domains to exclude.
            include_answer: Whether to include Tavily's generated answer.
                            ALWAYS False for evidence-based research.
            days: Optional number of days back to search.

        Returns:
            Raw Tavily search response dict (without the 'answer' field).
        """
        payload = {
            "api_key": self._settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": search_depth or self._settings.TAVILY_SEARCH_DEPTH,
            "max_results": max_results or self._settings.TAVILY_MAX_RESULTS_PER_QUERY,
            "include_answer": include_answer,
            "include_raw_content": False,
            "include_images": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if days is not None:
            payload["days"] = days

        logger.info(
            f"Tavily search: query='{query[:80]}...' depth={payload['search_depth']}"
        )
        data = await self._post(TAVILY_SEARCH_URL, payload)
        # Strip answer field — we never use it as evidence
        data.pop("answer", None)
        return data

    async def extract(
        self,
        urls: list[str],
        extract_depth: Optional[str] = None,
        include_images: bool = False,
    ) -> dict:
        """Extract cleaned content from URLs using Tavily Extract.

        Args:
            urls: List of URLs to extract content from.
            extract_depth: "basic" or "advanced". Defaults to TAVILY_EXTRACT_DEPTH.
            include_images: Whether to include images in extracted content.

        Returns:
            Raw Tavily extract response dict.
        """
        # Cap URLs to avoid oversized requests (Tavily limit is ~20 per call)
        urls = urls[:20]

        payload = {
            "api_key": self._settings.TAVILY_API_KEY,
            "urls": urls,
            "extract_depth": extract_depth or self._settings.TAVILY_EXTRACT_DEPTH,
            "include_images": include_images,
        }

        logger.info(f"Tavily extract: {len(urls)} URLs")
        data = await self._post(TAVILY_EXTRACT_URL, payload)
        return data


async def _asleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)


# Module-level cached client
_tavily_client: Optional[TavilyClient] = None


def get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


def reset_tavily_client() -> None:
    global _tavily_client
    _tavily_client = None
