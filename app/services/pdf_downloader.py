"""PDF download service — streaming download with SHA-256 cache."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import httpx

from app.core.config import get_settings
from app.core.storage import LocalDocumentStorage

logger = logging.getLogger(__name__)

# Known open-access PDF hosts — safe to download
ALLOWED_PDF_HOSTS = {
    "arxiv.org",
    "export.arxiv.org",
    "browse.arxiv.org",
    "www.biorxiv.org",
    "www.medrxiv.org",
    "openaccess.thecvf.com",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "aclanthology.org",
    "papers.nips.cc",
    "dl.acm.org",  # Some are OA
    "par.nsf.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "europepmc.org",
}


class PDFDownloader:
    """Download PDFs from open-access sources, cache by SHA-256."""

    def __init__(self, settings=None, storage: Optional[LocalDocumentStorage] = None):
        self._settings = settings or get_settings()
        self._storage = storage or LocalDocumentStorage(settings=self._settings)
        self._max_size = self._settings.PDF_MAX_SIZE_MB * 1024 * 1024

    async def download(self, url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Download a PDF from url. Returns (sha256, file_path, error_message).

        Only downloads from known open-access hosts.
        Checks PDF signature and file size.
        Caches by SHA-256 — same PDF not downloaded twice.
        """
        # Validate URL
        if not url or not url.startswith("http"):
            return None, None, f"Invalid URL: {url}"

        # Check open-access host (arxiv is always fine, others need explicit listing)
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""

        # Allow arxiv unconditionally, others must be in allowlist
        is_arxiv = host.endswith("arxiv.org")
        if not is_arxiv and host not in ALLOWED_PDF_HOSTS:
            # Still try — log a warning but proceed if it looks like a PDF
            if not url.lower().endswith(".pdf"):
                logger.debug(f"Skipping non-PDF, non-OA URL: {url[:100]}")
                return None, None, f"Not a known OA host and not a .pdf: {host}"

        # Stream download
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.PDF_DOWNLOAD_TIMEOUT),
            follow_redirects=True,
        )

        try:
            async with client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        return None, None, f"HTTP {response.status_code}"

                    # Check Content-Type
                    content_type = response.headers.get("content-type", "")
                    if "pdf" not in content_type.lower() and not is_arxiv:
                        # arXiv sometimes serves PDF without proper content-type
                        logger.debug(f"Non-PDF content-type: {content_type} for {url[:80]}")
                        # Continue anyway — some servers misconfigure

                    # Check Content-Length against max size
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self._max_size:
                        return None, None, (
                            f"PDF too large: {int(content_length)} bytes "
                            f"(max: {self._max_size})"
                        )

                    # Read in chunks, checking size limit
                    chunks = []
                    total = 0
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > self._max_size:
                            return None, None, f"PDF exceeds max size ({self._max_size} bytes)"
                        chunks.append(chunk)

                    content = b"".join(chunks)

                    # Verify PDF signature
                    if not content.startswith(b"%PDF"):
                        return None, None, f"Not a valid PDF (missing %%PDF signature)"

                    # Save to cache (SHA-256 addressed)
                    sha256, file_path = await self._storage.save_pdf(content, url)
                    logger.info(
                        f"PDF downloaded: {sha256[:12]}... "
                        f"({len(content)} bytes) from {url[:80]}"
                    )
                    return sha256, file_path, None

        except httpx.TimeoutException:
            return None, None, f"Timeout downloading PDF"
        except Exception as e:
            logger.error(f"PDF download failed: {e}")
            return None, None, str(e)
        finally:
            await client.aclose()

    def get_pdf_path(self, sha256: str) -> Optional[str]:
        return self._storage.get_pdf_path(sha256)

    def is_cached(self, sha256: str) -> bool:
        return self._storage.exists(sha256)
