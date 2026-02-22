from __future__ import annotations

from typing import TYPE_CHECKING

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.acquisition._exceptions import RateLimitError, ScrapingError
from src.acquisition._models import ContentFormat, ScrapedContent
from src.utils._hashing import content_hash
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.acquisition._rate_limiter import AsyncRateLimiter

_log = get_logger(__name__)


class HttpClient:
    """Async HTTP client with rate limiting, retries, and timeouts."""

    def __init__(
        self,
        *,
        rate_limiter: AsyncRateLimiter,
        user_agent: str = "LegalRAGBot/0.1",
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._user_agent = user_agent
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> HttpClient:
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={"User-Agent": self._user_agent},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _make_fetch(self) -> object:
        """Build a retrying fetch function bound to current retry config."""

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
            reraise=True,
        )
        async def _fetch(url: str) -> tuple[str, int, dict[str, str]]:
            if not self._session:
                msg = "HttpClient must be used as async context manager"
                raise ScrapingError(msg)

            await self._rate_limiter.acquire()
            _log.debug("http_request", url=url)

            async with self._session.get(url) as resp:
                if resp.status == 429:
                    raise RateLimitError(f"Rate limited by server: {url}")
                if resp.status >= 400:
                    raise ScrapingError(f"HTTP {resp.status} for {url}")

                text = await resp.text()
                headers = {k: v for k, v in resp.headers.items()}
                return text, resp.status, headers

        return _fetch

    async def fetch(
        self, url: str, *, content_format: ContentFormat = ContentFormat.HTML
    ) -> ScrapedContent:
        """Fetch a URL with rate limiting and retries.

        Args:
            url: The URL to fetch.
            content_format: Expected content format.

        Returns:
            ScrapedContent with the response data.

        Raises:
            ScrapingError: On HTTP errors after retries exhausted.
            RateLimitError: If server returns 429.
        """
        fetch_fn = self._make_fetch()
        try:
            text, status, headers = await fetch_fn(url)
        except RateLimitError:
            raise
        except Exception as exc:
            raise ScrapingError(f"Failed to fetch {url}: {exc}") from exc

        return ScrapedContent(
            url=url,
            content=text,
            content_format=content_format,
            content_hash=content_hash(text),
            status_code=status,
            headers=headers,
        )
