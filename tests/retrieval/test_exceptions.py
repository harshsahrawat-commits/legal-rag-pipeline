"""Tests for retrieval exception hierarchy."""

from __future__ import annotations

import pytest

from src.retrieval._exceptions import (
    ContextExpansionError,
    FLAREError,
    RerankerError,
    RerankerNotAvailableError,
    RetrievalError,
    SearchError,
    SearchNotAvailableError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    """All retrieval exceptions inherit from RetrievalError and LegalRAGError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            RetrievalError,
            SearchError,
            RerankerError,
            RerankerNotAvailableError,
            ContextExpansionError,
            FLAREError,
            SearchNotAvailableError,
        ],
    )
    def test_inherits_from_legal_rag_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, LegalRAGError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SearchError,
            RerankerError,
            RerankerNotAvailableError,
            ContextExpansionError,
            FLAREError,
            SearchNotAvailableError,
        ],
    )
    def test_inherits_from_retrieval_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, RetrievalError)

    def test_retrieval_error_is_base(self) -> None:
        assert issubclass(RetrievalError, LegalRAGError)
        assert not issubclass(LegalRAGError, RetrievalError)

    def test_exception_message(self) -> None:
        exc = SearchError("Dense search failed")
        assert str(exc) == "Dense search failed"

    def test_exception_can_be_raised_and_caught(self) -> None:
        with pytest.raises(RetrievalError):
            raise SearchError("test")

    def test_exception_chain(self) -> None:
        try:
            try:
                raise ConnectionError("connection refused")
            except ConnectionError as inner:
                raise SearchError("Qdrant unavailable") from inner
        except SearchError as outer:
            assert outer.__cause__ is not None
            assert isinstance(outer.__cause__, ConnectionError)

    def test_all_exceptions_have_unique_names(self) -> None:
        names = {
            RetrievalError.__name__,
            SearchError.__name__,
            RerankerError.__name__,
            RerankerNotAvailableError.__name__,
            ContextExpansionError.__name__,
            FLAREError.__name__,
            SearchNotAvailableError.__name__,
        }
        assert len(names) == 7
