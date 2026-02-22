from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket rate limiter for async HTTP requests.

    Enforces a maximum request rate per domain/source. Thread-safe via asyncio.Lock.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            msg = "requests_per_second must be positive"
            raise ValueError(msg)
        self._interval = 1.0 / requests_per_second
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def interval(self) -> float:
        """Minimum seconds between requests."""
        return self._interval

    async def acquire(self) -> None:
        """Wait until a request is allowed, then consume a token."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                wait_time = self._interval - elapsed
                await asyncio.sleep(wait_time)
            self._last_request = time.monotonic()
