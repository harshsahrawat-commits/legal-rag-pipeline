from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class EmbeddingError(LegalRAGError):
    """Base exception for the embedding module."""


class ModelLoadError(EmbeddingError):
    """Failed to load the embedding model."""


class EmbeddingInferenceError(EmbeddingError):
    """Forward pass or pooling failed during embedding."""


class IndexingError(EmbeddingError):
    """Failed to upsert vectors or payloads to the vector database."""


class RedisStoreError(EmbeddingError):
    """Failed to store or retrieve data from Redis."""


class EmbedderNotAvailableError(EmbeddingError):
    """A required embedding dependency (torch, transformers) is not installed."""


class CollectionCreationError(EmbeddingError):
    """Failed to create or verify a Qdrant collection."""
