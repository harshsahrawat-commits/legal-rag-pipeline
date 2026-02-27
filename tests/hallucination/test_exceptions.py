"""Tests for hallucination exception hierarchy."""

from __future__ import annotations

import pytest

from src.hallucination._exceptions import (
    CitationExtractionError,
    CitationVerificationError,
    ConfidenceScoringError,
    GenGroundError,
    GenGroundNotAvailableError,
    HallucinationConfigError,
    HallucinationError,
    TemporalCheckError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    """Verify all exceptions are rooted at LegalRAGError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            HallucinationError,
            CitationExtractionError,
            CitationVerificationError,
            TemporalCheckError,
            ConfidenceScoringError,
            GenGroundError,
            GenGroundNotAvailableError,
            HallucinationConfigError,
        ],
    )
    def test_inherits_from_legal_rag_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, LegalRAGError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            CitationExtractionError,
            CitationVerificationError,
            TemporalCheckError,
            ConfidenceScoringError,
            GenGroundError,
            GenGroundNotAvailableError,
            HallucinationConfigError,
        ],
    )
    def test_inherits_from_hallucination_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, HallucinationError)

    def test_exception_message(self) -> None:
        exc = HallucinationError("test message")
        assert str(exc) == "test message"

    def test_raise_and_catch_hierarchy(self) -> None:
        with pytest.raises(HallucinationError):
            raise CitationVerificationError("node not found")

    def test_raise_and_catch_base(self) -> None:
        with pytest.raises(LegalRAGError):
            raise GenGroundError("LLM call failed")

    def test_genground_not_available_is_hallucination_error(self) -> None:
        with pytest.raises(HallucinationError):
            raise GenGroundNotAvailableError("anthropic not installed")

    def test_config_error_message(self) -> None:
        exc = HallucinationConfigError("bad yaml")
        assert "bad yaml" in str(exc)
