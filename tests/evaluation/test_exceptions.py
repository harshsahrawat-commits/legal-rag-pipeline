"""Tests for the evaluation exception hierarchy."""

from __future__ import annotations

import pytest

from src.evaluation._exceptions import (
    EvaluationConfigError,
    EvaluationError,
    HumanEvalError,
    LegalMetricError,
    RagasEvaluationError,
    RagasNotAvailableError,
    ReportError,
    TestDatasetError,
)
from src.utils._exceptions import LegalRAGError

ALL_EXCEPTIONS = [
    EvaluationError,
    EvaluationConfigError,
    TestDatasetError,
    RagasNotAvailableError,
    RagasEvaluationError,
    LegalMetricError,
    HumanEvalError,
    ReportError,
]


class TestExceptionHierarchy:
    """Verify all exceptions inherit from LegalRAGError."""

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_inherits_from_legal_rag_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, LegalRAGError)

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_inherits_from_evaluation_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, EvaluationError)

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_instantiable_with_message(self, exc_class: type) -> None:
        exc = exc_class("test message")
        assert str(exc) == "test message"

    def test_exception_names_end_with_error(self) -> None:
        for exc_class in ALL_EXCEPTIONS:
            assert exc_class.__name__.endswith("Error"), (
                f"{exc_class.__name__} does not end with 'Error'"
            )

    def test_total_exception_count(self) -> None:
        assert len(ALL_EXCEPTIONS) == 8
