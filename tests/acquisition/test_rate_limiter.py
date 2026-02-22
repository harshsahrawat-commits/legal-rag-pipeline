from __future__ import annotations

import time

import pytest

from src.acquisition._rate_limiter import AsyncRateLimiter


class TestAsyncRateLimiter:
    async def test_first_request_immediate(self):
        limiter = AsyncRateLimiter(requests_per_second=10.0)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

    async def test_enforces_interval(self):
        limiter = AsyncRateLimiter(requests_per_second=5.0)  # 0.2s interval
        await limiter.acquire()
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # Allow some tolerance

    async def test_multiple_requests_spaced(self):
        limiter = AsyncRateLimiter(requests_per_second=10.0)  # 0.1s interval
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        # 3 requests at 10/s: first immediate, then 2 * 0.1s waits ≈ 0.2s
        assert elapsed >= 0.15

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            AsyncRateLimiter(requests_per_second=0)

        with pytest.raises(ValueError, match="positive"):
            AsyncRateLimiter(requests_per_second=-1.0)

    def test_interval_property(self):
        limiter = AsyncRateLimiter(requests_per_second=0.5)
        assert limiter.interval == pytest.approx(2.0)
