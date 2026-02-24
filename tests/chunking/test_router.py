from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.chunking._models import ChunkingSettings, ChunkStrategy
from src.chunking._router import ChunkerRouter
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._judgment_structural import JudgmentStructuralChunker
from src.chunking.chunkers._page_level import PageLevelChunker
from src.chunking.chunkers._statute_boundary import StatuteBoundaryChunker

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument


def _make_router() -> ChunkerRouter:
    settings = ChunkingSettings()
    tc = TokenCounter()
    router = ChunkerRouter(settings, tc)
    router.register(StatuteBoundaryChunker(settings, tc))
    router.register(JudgmentStructuralChunker(settings, tc))
    router.register(PageLevelChunker(settings, tc))
    return router


class TestChunkerRouter:
    def test_statute_routes_to_statute_chunker(self, sample_statute_doc: ParsedDocument):
        router = _make_router()
        selected = router.select(sample_statute_doc)
        assert selected.strategy == ChunkStrategy.STRUCTURE_BOUNDARY

    def test_judgment_routes_to_judgment_chunker(self, sample_judgment_doc: ParsedDocument):
        router = _make_router()
        selected = router.select(sample_judgment_doc)
        assert selected.strategy == ChunkStrategy.JUDGMENT_STRUCTURAL

    def test_degraded_scan_routes_to_page_level(self, sample_degraded_scan_doc: ParsedDocument):
        router = _make_router()
        selected = router.select(sample_degraded_scan_doc)
        assert selected.strategy == ChunkStrategy.PAGE_LEVEL

    def test_unstructured_routes_to_page_level(self, sample_unstructured_doc: ParsedDocument):
        router = _make_router()
        selected = router.select(sample_unstructured_doc)
        assert selected.strategy == ChunkStrategy.PAGE_LEVEL

    def test_no_chunkers_raises(self, sample_statute_doc: ParsedDocument):
        settings = ChunkingSettings()
        tc = TokenCounter()
        router = ChunkerRouter(settings, tc)
        with pytest.raises(ValueError, match="No chunker available"):
            router.select(sample_statute_doc)

    def test_chunkers_property(self):
        router = _make_router()
        assert len(router.chunkers) == 3

    def test_register_order_preserved(self):
        settings = ChunkingSettings()
        tc = TokenCounter()
        router = ChunkerRouter(settings, tc)
        page = PageLevelChunker(settings, tc)
        statute = StatuteBoundaryChunker(settings, tc)
        router.register(page)
        router.register(statute)
        assert router.chunkers[0].strategy == ChunkStrategy.PAGE_LEVEL
        assert router.chunkers[1].strategy == ChunkStrategy.STRUCTURE_BOUNDARY
