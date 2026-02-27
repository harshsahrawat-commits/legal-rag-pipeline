"""Exception hierarchy for the evaluation module."""

from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class EvaluationError(LegalRAGError):
    """Base exception for the evaluation module."""


class EvaluationConfigError(EvaluationError):
    """Invalid or missing evaluation config."""


class TestDatasetError(EvaluationError):
    """Failed to load or validate test query dataset."""


class RagasNotAvailableError(EvaluationError):
    """ragas or langchain-anthropic not installed."""


class RagasEvaluationError(EvaluationError):
    """RAGAS evaluation failed during execution."""


class LegalMetricError(EvaluationError):
    """Custom legal metric computation failed."""


class HumanEvalError(EvaluationError):
    """Human evaluation harness error."""


class ReportError(EvaluationError):
    """Failed to generate evaluation report."""
