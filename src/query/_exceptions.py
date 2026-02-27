"""Exception hierarchy for the Query Intelligence Layer (Phase 0)."""

from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class QueryIntelligenceError(LegalRAGError):
    """Base exception for query intelligence operations."""


class CacheError(QueryIntelligenceError):
    """Cache read/write/invalidation failure."""


class RouterError(QueryIntelligenceError):
    """Query classification failure."""


class HyDEError(QueryIntelligenceError):
    """Hypothetical document generation failure."""


class EmbeddingError(QueryIntelligenceError):
    """Query embedding failure."""
