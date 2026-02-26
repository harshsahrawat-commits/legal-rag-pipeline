from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class RetrievalError(LegalRAGError):
    """Base exception for the retrieval module."""


class SearchError(RetrievalError):
    """A search channel (dense, BM25, QuIM, graph) failed."""


class RerankerError(RetrievalError):
    """Cross-encoder reranking failed."""


class RerankerNotAvailableError(RetrievalError):
    """transformers package not installed for cross-encoder reranker."""


class ContextExpansionError(RetrievalError):
    """Failed to expand context from Redis parent store."""


class FLAREError(RetrievalError):
    """FLARE active retrieval failed."""


class SearchNotAvailableError(RetrievalError):
    """A required retrieval dependency (qdrant-client, redis, neo4j) is not installed."""
