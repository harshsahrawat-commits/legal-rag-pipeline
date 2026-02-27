"""Shared fixtures for query intelligence tests."""

from __future__ import annotations

import pytest

from src.query._models import QueryConfig, QuerySettings


@pytest.fixture
def default_settings() -> QuerySettings:
    """Default query settings for tests."""
    return QuerySettings()


@pytest.fixture
def default_config() -> QueryConfig:
    """Default query config for tests."""
    return QueryConfig()


@pytest.fixture
def custom_settings() -> QuerySettings:
    """Custom settings with non-default values."""
    return QuerySettings(
        cache_enabled=False,
        cache_similarity_threshold=0.85,
        cache_ttl_seconds=3600,
        hyde_enabled=False,
        hyde_routes=["analytical"],
        device="cuda",
    )
