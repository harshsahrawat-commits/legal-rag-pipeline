from __future__ import annotations

import pytest
from aioresponses import aioresponses

from src.acquisition._exceptions import RateLimitError, ScrapingError
from src.acquisition._http_client import HttpClient
from src.acquisition._models import ContentFormat
from src.acquisition._rate_limiter import AsyncRateLimiter


@pytest.fixture()
def rate_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(requests_per_second=100.0)  # Fast for tests


class TestHttpClient:
    async def test_fetch_success(self, rate_limiter: AsyncRateLimiter):
        with aioresponses() as m:
            m.get("https://example.com/doc/1", body="<html>test</html>", status=200)

            async with HttpClient(rate_limiter=rate_limiter) as client:
                result = await client.fetch("https://example.com/doc/1")

            assert result.status_code == 200
            assert result.content == "<html>test</html>"
            assert result.content_format == ContentFormat.HTML
            assert len(result.content_hash) == 64

    async def test_fetch_rate_limit_429(self, rate_limiter: AsyncRateLimiter):
        with aioresponses() as m:
            m.get("https://example.com/doc/2", status=429)

            async with HttpClient(rate_limiter=rate_limiter, max_retries=1) as client:
                with pytest.raises(RateLimitError):
                    await client.fetch("https://example.com/doc/2")

    async def test_fetch_server_error_raises(self, rate_limiter: AsyncRateLimiter):
        with aioresponses() as m:
            m.get("https://example.com/doc/3", status=500)

            async with HttpClient(rate_limiter=rate_limiter, max_retries=1) as client:
                with pytest.raises(ScrapingError):
                    await client.fetch("https://example.com/doc/3")

    async def test_fetch_content_format_passed(self, rate_limiter: AsyncRateLimiter):
        with aioresponses() as m:
            m.get("https://example.com/doc.pdf", body="%PDF-content", status=200)

            async with HttpClient(rate_limiter=rate_limiter) as client:
                result = await client.fetch(
                    "https://example.com/doc.pdf",
                    content_format=ContentFormat.PDF,
                )

            assert result.content_format == ContentFormat.PDF

    async def test_fetch_without_context_manager_raises(self, rate_limiter: AsyncRateLimiter):
        client = HttpClient(rate_limiter=rate_limiter, max_retries=1)
        with pytest.raises(ScrapingError, match="context manager"):
            await client.fetch("https://example.com/fail")

    async def test_custom_user_agent(self, rate_limiter: AsyncRateLimiter):
        with aioresponses() as m:
            m.get("https://example.com/doc/ua", body="ok", status=200)

            async with HttpClient(rate_limiter=rate_limiter, user_agent="TestBot/1.0") as client:
                result = await client.fetch("https://example.com/doc/ua")
                assert result.status_code == 200
