from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.chunking._models import ChunkingSettings, ChunkStrategy, LegalChunk
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._base import BaseChunker

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument


class ConcreteChunker(BaseChunker):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.PAGE_LEVEL

    def can_chunk(self, doc: ParsedDocument) -> bool:
        return True

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        return []


class TestBaseChunker:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseChunker(ChunkingSettings(), TokenCounter())  # type: ignore[abstract]

    def test_concrete_instantiation(self):
        chunker = ConcreteChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.PAGE_LEVEL

    def test_settings_accessible(self):
        settings = ChunkingSettings(max_tokens=1000)
        chunker = ConcreteChunker(settings, TokenCounter())
        assert chunker._settings.max_tokens == 1000

    def test_token_counter_accessible(self):
        tc = TokenCounter()
        chunker = ConcreteChunker(ChunkingSettings(), tc)
        assert chunker._tc is tc

    def test_can_chunk_returns_bool(self, sample_statute_doc: ParsedDocument):
        chunker = ConcreteChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_statute_doc) is True

    def test_chunk_returns_list(self, sample_statute_doc: ParsedDocument):
        chunker = ConcreteChunker(ChunkingSettings(), TokenCounter())
        result = chunker.chunk(sample_statute_doc)
        assert isinstance(result, list)
