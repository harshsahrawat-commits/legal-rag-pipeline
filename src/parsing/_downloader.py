"""PDF downloader with caching for India Code documents.

Downloads PDFs from ``download_url`` in Phase 1 metadata and caches
them locally so the Docling PDF parser can operate on local files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp

from src.parsing._exceptions import PDFDownloadError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.parsing._models import ParsingSettings

_log = get_logger(__name__)

_BYTES_PER_MB = 1024 * 1024


class PdfDownloader:
    """Download and cache PDFs for Phase 2 parsing.

    Features:
    - Cache-based idempotency: skips download if cached file exists.
    - Atomic writes: writes to a temp file then renames.
    - Size limit enforcement from ``ParsingSettings.max_pdf_size_mb``.
    - Configurable timeout from ``ParsingSettings.download_timeout_seconds``.
    """

    def __init__(self, settings: ParsingSettings) -> None:
        self._settings = settings

    async def download(self, url: str, doc_id: str) -> Path:
        """Download a PDF and return the local cache path.

        Args:
            url: Remote URL pointing to the PDF file.
            doc_id: Document identifier used for the cache filename.

        Returns:
            Path to the cached PDF file.

        Raises:
            PDFDownloadError: On HTTP errors, timeout, or size limit exceeded.
        """
        cache_dir = Path(self._settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{doc_id}.pdf"

        # Cache hit — skip download
        if cache_path.exists() and cache_path.stat().st_size > 0:
            _log.info("pdf_cache_hit", doc_id=doc_id, path=str(cache_path))
            return cache_path

        _log.info("pdf_downloading", doc_id=doc_id, url=url)

        timeout = aiohttp.ClientTimeout(
            total=self._settings.download_timeout_seconds,
        )
        max_bytes = self._settings.max_pdf_size_mb * _BYTES_PER_MB

        try:
            async with (
                aiohttp.ClientSession(
                    timeout=timeout,
                    headers={"User-Agent": self._settings.user_agent},
                ) as session,
                session.get(url) as resp,
            ):
                if resp.status >= 400:
                    raise PDFDownloadError(
                        f"HTTP {resp.status} downloading PDF for {doc_id}: {url}"
                    )

                # Check Content-Length header if available
                content_length = resp.content_length
                if content_length is not None and content_length > max_bytes:
                    raise PDFDownloadError(
                        f"PDF too large for {doc_id}: "
                        f"{content_length / _BYTES_PER_MB:.1f} MB "
                        f"(limit: {self._settings.max_pdf_size_mb} MB)"
                    )

                # Stream to temp file, enforcing size limit
                downloaded = 0
                tmp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=cache_dir,
                        suffix=".tmp",
                        delete=False,
                    ) as fd:
                        tmp_path = Path(fd.name)
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise PDFDownloadError(
                                    f"PDF too large for {doc_id}: "
                                    f">{self._settings.max_pdf_size_mb} MB "
                                    f"(limit exceeded during download)"
                                )
                            fd.write(chunk)
                    # fd is closed here; atomic rename
                    tmp_path.replace(cache_path)
                    tmp_path = None
                except BaseException:
                    # fd is closed (with-block exited); safe to unlink on Windows
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)
                    raise

        except PDFDownloadError:
            raise
        except TimeoutError as exc:
            raise PDFDownloadError(f"Timeout downloading PDF for {doc_id}: {url}") from exc
        except aiohttp.ClientError as exc:
            raise PDFDownloadError(f"Network error downloading PDF for {doc_id}: {exc}") from exc

        size_mb = cache_path.stat().st_size / _BYTES_PER_MB
        _log.info(
            "pdf_downloaded",
            doc_id=doc_id,
            size_mb=round(size_mb, 2),
            path=str(cache_path),
        )
        return cache_path
