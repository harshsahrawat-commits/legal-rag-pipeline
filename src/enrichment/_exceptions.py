from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class EnrichmentError(LegalRAGError):
    """Base exception for the enrichment module."""


class ContextualRetrievalError(EnrichmentError):
    """Failed to generate contextual retrieval text for a chunk."""


class QuIMGenerationError(EnrichmentError):
    """Failed to generate QuIM-RAG questions for a chunk."""


class EnricherNotAvailableError(EnrichmentError):
    """A required enricher dependency is not installed."""


class LLMRateLimitError(EnrichmentError):
    """LLM API rate limit exceeded."""


class DocumentTextTooLargeError(EnrichmentError):
    """Document text exceeds the maximum supported size."""
