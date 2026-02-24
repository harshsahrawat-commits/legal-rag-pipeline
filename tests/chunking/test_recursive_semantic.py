from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._exceptions import ChunkerNotAvailableError
from src.chunking._models import ChunkingSettings, ChunkStrategy
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._recursive_semantic import (
    RecursiveSemanticChunker,
    _recursive_split,
)
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)


def _make_doc(raw_text: str, sections: list[ParsedSection] | None = None, **kw) -> ParsedDocument:
    defaults = {
        "document_id": uuid4(),
        "source_type": SourceType.INDIAN_KANOON,
        "document_type": DocumentType.STATUTE,
        "content_format": ContentFormat.HTML,
        "raw_text": raw_text,
        "sections": sections or [],
        "parser_used": ParserType.HTML_INDIAN_KANOON,
        "quality": QualityReport(overall_score=0.9, passed=True),
        "raw_content_path": "data/raw/test.html",
    }
    defaults.update(kw)
    return ParsedDocument(**defaults)


def _mock_model():
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()
    call_count = [0]

    def fake_encode(text, **kwargs):
        call_count[0] += 1
        np.random.seed(hash(text) % 2**31)
        return np.random.randn(768).astype(np.float32)

    model.encode = fake_encode
    return model


class TestStrategy:
    def test_strategy_value(self):
        chunker = RecursiveSemanticChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.RECURSIVE_SEMANTIC


class TestCanChunk:
    def test_statute_with_partial_structure(self):
        sections = [
            ParsedSection(id="s1", level=SectionLevel.SECTION, number="1", text="Sec 1."),
        ]
        doc = _make_doc("text", sections=sections)
        chunker = RecursiveSemanticChunker(
            ChunkingSettings(min_section_count_statute=3), TokenCounter()
        )
        assert chunker.can_chunk(doc) is True

    def test_statute_with_full_structure(self):
        sections = [
            ParsedSection(id=f"s{i}", level=SectionLevel.SECTION, number=str(i), text=f"Sec {i}.")
            for i in range(5)
        ]
        doc = _make_doc("text", sections=sections)
        chunker = RecursiveSemanticChunker(
            ChunkingSettings(min_section_count_statute=3), TokenCounter()
        )
        assert chunker.can_chunk(doc) is False

    def test_no_sections(self):
        doc = _make_doc("text")
        chunker = RecursiveSemanticChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(doc) is False


class TestChunkWithMockedModel:
    def test_basic_chunking(self):
        text = "First paragraph about contracts.\n\nSecond paragraph about torts.\n\nThird about evidence."
        sections = [ParsedSection(id="s1", level=SectionLevel.SECTION, number="1", text="X")]
        doc = _make_doc(text, sections=sections)

        settings = ChunkingSettings(max_tokens=1500)
        chunker = RecursiveSemanticChunker(settings, TokenCounter())
        chunker._model = _mock_model()  # inject mock

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        # All text should be covered
        all_text = " ".join(c.text for c in chunks)
        assert "contracts" in all_text
        assert "evidence" in all_text

    def test_empty_text(self):
        doc = _make_doc("", sections=[ParsedSection(id="s1", level=SectionLevel.SECTION, text="X")])
        chunker = RecursiveSemanticChunker(ChunkingSettings(), TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_chunk_metadata(self):
        text = "Paragraph one.\n\nParagraph two."
        sections = [ParsedSection(id="s1", level=SectionLevel.SECTION, number="1", text="X")]
        doc = _make_doc(text, sections=sections, act_name="Test Act")
        chunker = RecursiveSemanticChunker(ChunkingSettings(), TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.ingestion.chunk_strategy == ChunkStrategy.RECURSIVE_SEMANTIC
            assert c.token_count > 0
            assert c.document_id == doc.document_id

    def test_oversized_text_split(self):
        text = " ".join(f"word{i}" for i in range(3000))
        sections = [ParsedSection(id="s1", level=SectionLevel.SECTION, text="X")]
        doc = _make_doc(text, sections=sections)
        settings = ChunkingSettings(max_tokens=200)
        chunker = RecursiveSemanticChunker(settings, TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 200


class TestMissingDependency:
    def test_raises_when_no_sentence_transformers(self):
        sections = [ParsedSection(id="s1", level=SectionLevel.SECTION, text="X")]
        doc = _make_doc("Some text.", sections=sections)
        chunker = RecursiveSemanticChunker(ChunkingSettings(), TokenCounter())
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ChunkerNotAvailableError, match="sentence-transformers"),
        ):
            chunker._model = None
            chunker.chunk(doc)


class TestRecursiveSplit:
    def test_short_text(self):
        tc = TokenCounter()
        result = _recursive_split("hello world", ["\n\n", "\n", ". "], 100, tc)
        assert result == ["hello world"]

    def test_splits_on_double_newline(self):
        tc = TokenCounter()
        text = "First part.\n\nSecond part."
        result = _recursive_split(text, ["\n\n", "\n", ". "], 4, tc)
        assert len(result) == 2

    def test_force_split_when_no_separator(self):
        tc = TokenCounter()
        text = " ".join(f"word{i}" for i in range(200))
        result = _recursive_split(text, ["\n\n"], 50, tc)
        assert len(result) > 1
        for chunk in result:
            assert tc.count(chunk) <= 50
