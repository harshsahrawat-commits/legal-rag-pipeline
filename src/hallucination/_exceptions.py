"""Exception hierarchy for the hallucination mitigation module."""

from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class HallucinationError(LegalRAGError):
    """Base exception for the hallucination mitigation module."""


class CitationExtractionError(HallucinationError):
    """Failed to extract citations from response text."""


class CitationVerificationError(HallucinationError):
    """Failed to verify a citation against the knowledge graph."""


class TemporalCheckError(HallucinationError):
    """Failed to perform temporal consistency check."""


class ConfidenceScoringError(HallucinationError):
    """Failed to compute confidence score."""


class GenGroundError(HallucinationError):
    """GenGround verification failed (LLM call or claim processing)."""


class GenGroundNotAvailableError(HallucinationError):
    """Anthropic SDK not installed for GenGround verification."""


class HallucinationConfigError(HallucinationError):
    """Invalid or missing hallucination config."""
