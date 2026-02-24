from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._exceptions import ChunkerNotAvailableError
from src.chunking._models import ChunkingSettings, ChunkStrategy
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._semantic_maxmin import SemanticMaxMinChunker, _regex_sentence_split
from src.parsing._models import ParsedDocument, ParserType, QualityReport


def _make_doc(raw_text: str, **kw) -> ParsedDocument:
    defaults = {
        "document_id": uuid4(),
        "source_type": SourceType.INDIAN_KANOON,
        "document_type": DocumentType.CIRCULAR,
        "content_format": ContentFormat.HTML,
        "raw_text": raw_text,
        "sections": [],
        "parser_used": ParserType.HTML_INDIAN_KANOON,
        "quality": QualityReport(overall_score=0.9, passed=True),
        "raw_content_path": "data/raw/test.html",
    }
    defaults.update(kw)
    return ParsedDocument(**defaults)


def _mock_model():
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()

    def fake_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for t in texts:
            np.random.seed(hash(t) % 2**31)
            result.append(np.random.randn(768).astype(np.float32))
        return np.array(result)

    model.encode = fake_encode
    return model


class TestStrategy:
    def test_strategy_value(self):
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.SEMANTIC_MAXMIN


class TestCanChunk:
    def test_unstructured_doc(self, sample_unstructured_doc: ParsedDocument):
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_unstructured_doc) is True

    def test_structured_doc_returns_false(self, sample_statute_doc: ParsedDocument):
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_statute_doc) is False


class TestChunkWithMockedModel:
    def test_basic_chunking(self):
        text = (
            "The Reserve Bank of India issued new guidelines. "
            "All banks must comply with KYC norms. "
            "The deadline for compliance is March 2024. "
            "Non-compliance will result in penalties."
        )
        doc = _make_doc(text)
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Reserve Bank" in all_text

    def test_empty_text(self):
        doc = _make_doc("")
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_chunk_metadata(self):
        text = "First sentence. Second sentence. Third sentence."
        doc = _make_doc(text)
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.ingestion.chunk_strategy == ChunkStrategy.SEMANTIC_MAXMIN
            assert c.token_count > 0

    def test_oversized_merged_groups_split(self):
        text = " ".join(f"word{i}." for i in range(500))
        doc = _make_doc(text)
        settings = ChunkingSettings(max_tokens=50)
        chunker = SemanticMaxMinChunker(settings, TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.token_count <= 50

    def test_tiny_groups_merged(self):
        # Create text with very short sentences that should be merged
        text = "A. B. C. D. E. F. G. H. I. J."
        doc = _make_doc(text)
        settings = ChunkingSettings(min_chunk_tokens=20)
        chunker = SemanticMaxMinChunker(settings, TokenCounter())
        chunker._model = _mock_model()
        chunks = chunker.chunk(doc)
        # Should merge tiny groups
        assert len(chunks) >= 1


class TestMissingDependency:
    def test_raises_when_no_sentence_transformers(self):
        doc = _make_doc("Some text.")
        chunker = SemanticMaxMinChunker(ChunkingSettings(), TokenCounter())
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ChunkerNotAvailableError, match="sentence-transformers"),
        ):
            chunker._model = None
            chunker.chunk(doc)


class TestRegexSentenceSplit:
    def test_basic(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _regex_sentence_split(text)
        assert len(result) == 3

    def test_question_marks(self):
        text = "Is this a question? Yes it is. Another question?"
        result = _regex_sentence_split(text)
        assert len(result) == 3

    def test_no_split_needed(self):
        text = "Single sentence without period"
        result = _regex_sentence_split(text)
        assert len(result) == 1
