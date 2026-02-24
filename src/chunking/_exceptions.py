from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class ChunkingError(LegalRAGError):
    """Base exception for the chunking module."""


class TokenLimitExceededError(ChunkingError):
    """Chunk exceeds the configured maximum token limit."""


class ChunkerNotAvailableError(ChunkingError):
    """A required chunker dependency is not installed."""


class DocumentStructureError(ChunkingError):
    """Document lacks the expected structure for the selected chunker."""


class MetadataBuildError(ChunkingError):
    """Failed to construct metadata for a chunk."""
