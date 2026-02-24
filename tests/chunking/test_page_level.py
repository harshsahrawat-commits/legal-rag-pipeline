from __future__ import annotations

from uuid import uuid4

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._models import ChunkingSettings, ChunkStrategy
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._page_level import PageLevelChunker
from src.parsing._models import ParsedDocument, ParserType, QualityReport


def _make_doc(raw_text: str, **overrides) -> ParsedDocument:
    defaults = {
        "document_id": uuid4(),
        "source_type": SourceType.INDIAN_KANOON,
        "document_type": DocumentType.STATUTE,
        "content_format": ContentFormat.HTML,
        "raw_text": raw_text,
        "parser_used": ParserType.HTML_INDIAN_KANOON,
        "quality": QualityReport(overall_score=0.9, passed=True),
        "raw_content_path": "data/raw/test.html",
    }
    defaults.update(overrides)
    return ParsedDocument(**defaults)


class TestPageLevelChunker:
    def test_strategy(self):
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.PAGE_LEVEL

    def test_can_chunk_any_doc(self, sample_statute_doc: ParsedDocument):
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_statute_doc) is True

    def test_single_page(self):
        doc = _make_doc("This is a single page document with no form feeds.")
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "single page" in chunks[0].text

    def test_multiple_pages(self):
        doc = _make_doc("Page one content.\fPage two content.\fPage three content.")
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3
        assert "Page one" in chunks[0].text
        assert "Page three" in chunks[2].text

    def test_empty_pages_skipped(self):
        doc = _make_doc("Page one.\f\f\fPage four.")
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2

    def test_degraded_scan_flagged(self):
        doc = _make_doc(
            "Degraded page text.",
            ocr_applied=True,
            ocr_confidence=0.65,
        )
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].ingestion.requires_manual_review is True

    def test_good_ocr_not_flagged(self):
        doc = _make_doc(
            "Good quality text.",
            ocr_applied=True,
            ocr_confidence=0.95,
        )
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert chunks[0].ingestion.requires_manual_review is False

    def test_token_count_populated(self):
        doc = _make_doc("This is page one text.\fThis is page two text.")
        tc = TokenCounter()
        chunker = PageLevelChunker(ChunkingSettings(), tc)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.token_count > 0
            assert c.token_count == tc.count(c.text)

    def test_document_id_propagated(self, sample_statute_doc: ParsedDocument):
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        for c in chunks:
            assert c.document_id == sample_statute_doc.document_id

    def test_statute_metadata_set(self, sample_statute_doc: ParsedDocument):
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        assert chunks[0].statute is not None
        assert chunks[0].judgment is None

    def test_judgment_metadata_set(self, sample_judgment_doc: ParsedDocument):
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        assert chunks[0].judgment is not None
        assert chunks[0].statute is None

    def test_oversized_page_split(self):
        # Create a page that exceeds max_tokens
        long_text = " ".join(f"word{i}" for i in range(2000))
        doc = _make_doc(long_text)
        settings = ChunkingSettings(max_tokens=100)
        chunker = PageLevelChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 100

    def test_chunk_ids_are_unique(self):
        doc = _make_doc("Page 1.\fPage 2.\fPage 3.")
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_ingestion_strategy(self):
        doc = _make_doc("Some text.")
        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert chunks[0].ingestion.chunk_strategy == ChunkStrategy.PAGE_LEVEL
