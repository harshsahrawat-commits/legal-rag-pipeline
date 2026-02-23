from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.parsing._downloader import PdfDownloader
from src.parsing._exceptions import PDFDownloadError
from src.parsing._models import ParsingSettings

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PDF_URL = "https://www.indiacode.nic.in/bitstream/123456789/2110/1/a2012-13.pdf"
DOC_ID = "ic_2110"
FAKE_PDF = b"%PDF-1.4 fake content for testing purposes " + b"x" * 200


def _settings(tmp_path: Path, **overrides) -> ParsingSettings:
    defaults = {
        "input_dir": tmp_path / "raw",
        "output_dir": tmp_path / "parsed",
        "pdf_cache_dir": tmp_path / "cache" / "pdf",
        "download_timeout_seconds": 10,
        "max_pdf_size_mb": 50,
    }
    defaults.update(overrides)
    return ParsingSettings(**defaults)


def _mock_response(
    *,
    status: int = 200,
    content: bytes = FAKE_PDF,
    content_length: int | None = None,
) -> MagicMock:
    """Build a mock aiohttp response with async iteration."""
    resp = MagicMock()
    resp.status = status
    resp.content_length = content_length if content_length is not None else len(content)

    async def _iter_chunked(chunk_size: int):
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    resp.content = MagicMock()
    resp.content.iter_chunked = _iter_chunked

    # Support async context manager
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(resp: MagicMock) -> MagicMock:
    """Build a mock aiohttp.ClientSession."""
    session = MagicMock()
    session.get = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# TestCacheHit
# ---------------------------------------------------------------------------


class TestCacheHit:
    @pytest.mark.asyncio()
    async def test_returns_cached_file_without_downloading(self, tmp_path: Path):
        settings = _settings(tmp_path)
        cache_dir = settings.pdf_cache_dir
        cache_dir.mkdir(parents=True)
        cached = cache_dir / f"{DOC_ID}.pdf"
        cached.write_bytes(FAKE_PDF)

        downloader = PdfDownloader(settings)

        with patch("src.parsing._downloader.aiohttp.ClientSession") as mock_cls:
            result = await downloader.download(PDF_URL, DOC_ID)
            mock_cls.assert_not_called()

        assert result == cached
        assert result.read_bytes() == FAKE_PDF

    @pytest.mark.asyncio()
    async def test_skips_empty_cached_file(self, tmp_path: Path):
        """An empty cached file should trigger a fresh download."""
        settings = _settings(tmp_path)
        cache_dir = settings.pdf_cache_dir
        cache_dir.mkdir(parents=True)
        cached = cache_dir / f"{DOC_ID}.pdf"
        cached.write_bytes(b"")  # empty

        resp = _mock_response()
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with patch("src.parsing._downloader.aiohttp.ClientSession", return_value=session):
            result = await downloader.download(PDF_URL, DOC_ID)

        assert result == cached
        assert result.read_bytes() == FAKE_PDF


# ---------------------------------------------------------------------------
# TestSuccessfulDownload
# ---------------------------------------------------------------------------


class TestSuccessfulDownload:
    @pytest.mark.asyncio()
    async def test_downloads_and_caches_pdf(self, tmp_path: Path):
        settings = _settings(tmp_path)
        resp = _mock_response()
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with patch("src.parsing._downloader.aiohttp.ClientSession", return_value=session):
            result = await downloader.download(PDF_URL, DOC_ID)

        expected = settings.pdf_cache_dir / f"{DOC_ID}.pdf"
        assert result == expected
        assert result.exists()
        assert result.read_bytes() == FAKE_PDF

    @pytest.mark.asyncio()
    async def test_creates_cache_directory(self, tmp_path: Path):
        settings = _settings(tmp_path)
        assert not settings.pdf_cache_dir.exists()

        resp = _mock_response()
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with patch("src.parsing._downloader.aiohttp.ClientSession", return_value=session):
            result = await downloader.download(PDF_URL, DOC_ID)

        assert result.parent.exists()
        assert result.read_bytes() == FAKE_PDF


# ---------------------------------------------------------------------------
# TestHTTPErrors
# ---------------------------------------------------------------------------


class TestHTTPErrors:
    @pytest.mark.asyncio()
    async def test_raises_on_http_404(self, tmp_path: Path):
        settings = _settings(tmp_path)
        resp = _mock_response(status=404)
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError, match="HTTP 404"),
        ):
            await downloader.download(PDF_URL, DOC_ID)

    @pytest.mark.asyncio()
    async def test_raises_on_http_500(self, tmp_path: Path):
        settings = _settings(tmp_path)
        resp = _mock_response(status=500)
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError, match="HTTP 500"),
        ):
            await downloader.download(PDF_URL, DOC_ID)

    @pytest.mark.asyncio()
    async def test_raises_on_network_error(self, tmp_path: Path):
        settings = _settings(tmp_path)
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))

        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError, match="Network error"),
        ):
            await downloader.download(PDF_URL, DOC_ID)

    @pytest.mark.asyncio()
    async def test_raises_on_timeout(self, tmp_path: Path):
        settings = _settings(tmp_path)
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.get = MagicMock(side_effect=TimeoutError("Request timed out"))

        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError, match="Timeout"),
        ):
            await downloader.download(PDF_URL, DOC_ID)


# ---------------------------------------------------------------------------
# TestSizeLimits
# ---------------------------------------------------------------------------


class TestSizeLimits:
    @pytest.mark.asyncio()
    async def test_rejects_large_content_length(self, tmp_path: Path):
        settings = _settings(tmp_path, max_pdf_size_mb=1)
        # 2 MB content-length header
        resp = _mock_response(content_length=2 * 1024 * 1024)
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError, match="too large"),
        ):
            await downloader.download(PDF_URL, DOC_ID)

    @pytest.mark.asyncio()
    async def test_rejects_large_body_during_streaming(self, tmp_path: Path):
        settings = _settings(tmp_path, max_pdf_size_mb=1)
        # Content larger than 1MB but Content-Length header is None
        big_content = b"x" * (2 * 1024 * 1024)
        resp = _mock_response(content=big_content, content_length=None)
        session = _mock_session(resp)

        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError, match="too large"),
        ):
            await downloader.download(PDF_URL, DOC_ID)

        # Ensure no temp files left behind
        cache_dir = settings.pdf_cache_dir
        if cache_dir.exists():
            tmp_files = list(cache_dir.glob("*.tmp"))
            assert tmp_files == [], f"Temp files not cleaned up: {tmp_files}"


# ---------------------------------------------------------------------------
# TestAtomicWrite
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    @pytest.mark.asyncio()
    async def test_no_partial_file_on_error(self, tmp_path: Path):
        """If download fails mid-stream, no .pdf file should exist."""
        settings = _settings(tmp_path)

        async def _failing_iter(chunk_size: int):
            yield b"partial data"
            raise aiohttp.ClientError("Connection lost")

        resp = MagicMock()
        resp.status = 200
        resp.content_length = None
        resp.content = MagicMock()
        resp.content.iter_chunked = _failing_iter
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)

        session = _mock_session(resp)
        downloader = PdfDownloader(settings)

        with (
            patch(
                "src.parsing._downloader.aiohttp.ClientSession",
                return_value=session,
            ),
            pytest.raises(PDFDownloadError),
        ):
            await downloader.download(PDF_URL, DOC_ID)

        cache_path = settings.pdf_cache_dir / f"{DOC_ID}.pdf"
        assert not cache_path.exists(), "Partial file should not exist"
