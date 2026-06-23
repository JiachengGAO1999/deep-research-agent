"""PDF download service — OA resolution chain + streaming download with SHA-256 cache.

Resolution order (first success wins):
    1. arXiv ID → arxiv.org/pdf/{id}
    2. Paper's full_text_url (from OpenAlex oa_url / S2 openAccessPdf)
    3. Unpaywall API → best_oa_location (requires UNPAYWALL_EMAIL)
    4. DOI → follow redirects, check Content-Type
    5. Paper's url as last resort

Every resolved URL is validated: Content-Type must include "pdf" and body must
start with %%PDF.  No host allowlist — the metadata-driven chain replaces it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.core.storage import LocalDocumentStorage

logger = logging.getLogger(__name__)

# ── arXiv ID extraction ──────────────────────────────────────────────

_ARXIV_ID_RE = re.compile(r"arxiv(?:\.org)?[/:](\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)


def _extract_arxiv_id(paper) -> Optional[str]:
    """Extract a clean arXiv ID from paper metadata."""
    # Try DOI-style arXiv ID first (e.g. 10.48550/arxiv.2505.06120)
    doi = paper.doi or ""
    m = re.search(r"10\.\d{4,9}/arxiv\.(\d{4}\.\d{4,5})", doi, re.I)
    if m:
        return m.group(1)
    # Try URL fields
    for field in (paper.url, paper.full_text_url):
        if field:
            m = _ARXIV_ID_RE.search(field) or _ARXIV_ABS_RE.search(field)
            if m:
                return m.group(1).rstrip("v1234567890")  # strip version
    return None


# ── PDF downloader ───────────────────────────────────────────────────


class PDFDownloader:
    """OA-aware PDF downloader with metadata-driven URL resolution."""

    def __init__(self, settings=None, storage: Optional[LocalDocumentStorage] = None):
        self._settings = settings or get_settings()
        self._storage = storage or LocalDocumentStorage(settings=self._settings)
        self._max_size = self._settings.PDF_MAX_SIZE_MB * 1024 * 1024
        self._download_timeout = self._settings.PDF_DOWNLOAD_TIMEOUT
        self._unpaywall_email: Optional[str] = getattr(
            self._settings, "UNPAYWALL_EMAIL", None
        ) or None

    # ── public API ────────────────────────────────────────────────────

    async def resolve_and_download(self, paper) -> tuple[Optional[str], Optional[str], Optional[str], str]:
        """Try every OA resolution strategy. Returns (sha256, file_path, error, source).

        ``source`` describes which strategy succeeded (for logging / lifecycle).
        """
        arxiv_id = _extract_arxiv_id(paper)

        strategies = [
            ("arxiv", lambda: self._arxiv_url(arxiv_id)) if arxiv_id else None,
            ("full_text_url", lambda: paper.full_text_url) if paper.full_text_url else None,
            ("unpaywall", lambda: self._unpaywall_resolve(paper.doi))
            if paper.doi and self._unpaywall_email else None,
            ("doi_redirect", lambda: self._doi_resolve(paper.doi))
            if paper.doi else None,
            ("direct_url", lambda: paper.url) if paper.url else None,
        ]

        for label, resolver in strategies:
            if resolver is None:
                continue
            try:
                result = resolver()
                if asyncio.iscoroutine(result):
                    url = await result
                else:
                    url = result
            except Exception as exc:
                logger.debug("OA resolver %s raised: %s", label, exc)
                continue
            if not url:
                continue
            sha256, file_path, error = await self._try_download(url)
            if sha256 and file_path:
                return sha256, file_path, None, label

        return None, None, "All OA resolution strategies exhausted", "none"

    async def download(self, url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Legacy single-URL download (used by tests / baselines)."""
        sha256, file_path, error = await self._try_download(url)
        return sha256, file_path, error

    # ── strategy helpers ──────────────────────────────────────────────

    @staticmethod
    def _arxiv_url(arxiv_id: str) -> str:
        return f"https://arxiv.org/pdf/{arxiv_id}"

    async def _unpaywall_resolve(self, doi: str) -> Optional[str]:
        """Query Unpaywall for the best OA PDF location."""
        url = f"https://api.unpaywall.org/v2/{doi}?email={self._unpaywall_email}"
        client = httpx.AsyncClient(timeout=httpx.Timeout(15), follow_redirects=True)
        try:
            async with client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.debug("Unpaywall HTTP %s for %s", resp.status_code, doi)
                    return None
                data = resp.json()
        except Exception as exc:
            logger.debug("Unpaywall error for %s: %s", doi, exc)
            return None
        finally:
            await client.aclose()

        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url_for_landing_page")
        if not pdf_url:
            # Fallback: try all oa_locations
            for loc in data.get("oa_locations") or []:
                pdf_url = loc.get("url_for_pdf")
                if pdf_url:
                    break
        return pdf_url

    async def _doi_resolve(self, doi: str) -> Optional[str]:
        """Follow DOI redirect chain, return final URL if it's a PDF."""
        doi_url = f"https://doi.org/{doi}"
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(15),
            follow_redirects=True,
        )
        try:
            async with client:
                # HEAD first to check final Content-Type
                head = await client.head(
                    doi_url,
                    headers={"Accept": "application/pdf, text/html"},
                )
                final_url = str(head.url) if head.url else doi_url
                ct = head.headers.get("content-type", "")
                if "pdf" in ct.lower():
                    return final_url
                # If not PDF, return the resolved URL anyway — caller will validate
                return final_url if final_url != doi_url else None
        except Exception as exc:
            logger.debug("DOI resolve error for %s: %s", doi, exc)
            return None
        finally:
            await client.aclose()

    # ── core download ─────────────────────────────────────────────────

    async def _try_download(self, url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Download and validate a single URL. Returns (sha256, file_path, error)."""
        if not url or not url.startswith("http"):
            return None, None, f"Invalid URL: {url}"

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._download_timeout),
            follow_redirects=True,
        )

        try:
            async with client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        return None, None, f"HTTP {response.status_code}"

                    # Check Content-Type
                    content_type = response.headers.get("content-type", "")
                    if "pdf" not in content_type.lower():
                        host = urlparse(url).hostname or ""
                        # arxiv sometimes serves PDFs without proper Content-Type
                        if host.endswith("arxiv.org"):
                            pass  # continue anyway
                        else:
                            return None, None, f"Not a PDF (Content-Type: {content_type})"

                    # Check Content-Length
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self._max_size:
                        return None, None, (
                            f"PDF too large: {int(content_length)} bytes (max: {self._max_size})"
                        )

                    # Stream to memory, enforcing size limit
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > self._max_size:
                            return None, None, f"PDF exceeds max size ({self._max_size} bytes)"
                        chunks.append(chunk)

                    content = b"".join(chunks)

                    # Verify PDF signature
                    if not content.startswith(b"%PDF"):
                        return None, None, "Not a valid PDF (missing %%PDF signature)"

                    sha256, file_path = await self._storage.save_pdf(content, url)
                    logger.info(
                        "PDF downloaded: %s... (%d bytes) from %s",
                        sha256[:12],
                        len(content),
                        url[:100],
                    )
                    return sha256, file_path, None

        except httpx.TimeoutException:
            return None, None, "Timeout downloading PDF"
        except Exception as exc:
            logger.error("PDF download error: %s", exc)
            return None, None, str(exc)
        finally:
            await client.aclose()

    # ── cache helpers ─────────────────────────────────────────────────

    def get_pdf_path(self, sha256: str) -> Optional[str]:
        return self._storage.get_pdf_path(sha256)

    def is_cached(self, sha256: str) -> bool:
        return self._storage.exists(sha256)
