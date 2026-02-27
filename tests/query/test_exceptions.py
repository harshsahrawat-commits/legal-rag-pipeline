"""Tests for query intelligence exception hierarchy."""

from __future__ import annotations

import pytest

from src.query._exceptions import (
    CacheError,
    EmbeddingError,
    HyDEError,
    QueryIntelligenceError,
    RouterError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    """Verify the exception class hierarchy."""

    def test_query_intelligence_error_is_legal_rag_error(self) -> None:
        assert issubclass(QueryIntelligenceError, LegalRAGError)

    def test_cache_error_is_query_intelligence_error(self) -> None:
        assert issubclass(CacheError, QueryIntelligenceError)

    def test_router_error_is_query_intelligence_error(self) -> None:
        assert issubclass(RouterError, QueryIntelligenceError)

    def test_hyde_error_is_query_intelligence_error(self) -> None:
        assert issubclass(HyDEError, QueryIntelligenceError)

    def test_embedding_error_is_query_intelligence_error(self) -> None:
        assert issubclass(EmbeddingError, QueryIntelligenceError)

    def test_cache_error_caught_by_legal_rag_error(self) -> None:
        with pytest.raises(LegalRAGError):
            raise CacheError("cache broken")

    def test_router_error_caught_by_query_intelligence_error(self) -> None:
        with pytest.raises(QueryIntelligenceError):
            raise RouterError("classification failed")

    def test_hyde_error_message(self) -> None:
        exc = HyDEError("LLM timeout")
        assert str(exc) == "LLM timeout"

    def test_embedding_error_message(self) -> None:
        exc = EmbeddingError("model not loaded")
        assert str(exc) == "model not loaded"
