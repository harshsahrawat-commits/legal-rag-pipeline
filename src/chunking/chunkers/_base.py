from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings, ChunkStrategy, LegalChunk
    from src.chunking._token_counter import TokenCounter
    from src.parsing._models import ParsedDocument


class BaseChunker(abc.ABC):
    """Abstract base for all chunking strategies.

    Chunkers are pure transforms: they accept a ``ParsedDocument`` in memory
    and return a list of ``LegalChunk`` objects. I/O is handled by the pipeline.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        self._settings = settings
        self._tc = token_counter

    @abc.abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        """Produce chunks from *doc*."""

    @abc.abstractmethod
    def can_chunk(self, doc: ParsedDocument) -> bool:
        """Return True if this chunker can handle *doc*."""

    @property
    @abc.abstractmethod
    def strategy(self) -> ChunkStrategy:
        """The strategy enum value this chunker implements."""
