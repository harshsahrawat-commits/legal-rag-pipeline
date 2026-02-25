"""Tests for LateChunkingEmbedder with mocked transformer model."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from src.acquisition._models import DocumentType
from src.chunking._models import (
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    IngestionMetadata,
    LegalChunk,
    SourceInfo,
)
from src.embedding._embedder import LateChunkingEmbedder
from src.embedding._exceptions import EmbedderNotAvailableError, ModelLoadError
from src.embedding._models import EmbeddingSettings

# --- Helpers ---


def _make_chunk(text: str, chunk_index: int = 0, doc_id=None) -> LegalChunk:
    now = datetime.now(UTC)
    return LegalChunk(
        document_id=doc_id or uuid4(),
        text=text,
        document_type=DocumentType.STATUTE,
        chunk_type=ChunkType.STATUTORY_TEXT,
        chunk_index=chunk_index,
        token_count=len(text.split()),
        source=SourceInfo(
            url="https://example.com",
            source_name="test",
            scraped_at=now,
            last_verified=now,
        ),
        content=ContentMetadata(),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="test",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
    )


def _make_mock_model(embedding_dim: int = 768, seq_len: int = 50):
    """Create mock torch model that returns deterministic last_hidden_state."""
    import torch

    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def mock_forward(input_ids=None, attention_mask=None):
        batch_size = input_ids.shape[0]
        actual_seq_len = input_ids.shape[1]
        np.random.seed(42)
        hidden = torch.tensor(
            np.random.randn(batch_size, actual_seq_len, embedding_dim).astype(np.float32)
        )
        result = MagicMock()
        result.last_hidden_state = hidden
        return result

    model.__call__ = mock_forward
    model.side_effect = mock_forward
    return model


def _make_mock_tokenizer(full_text: str, max_length: int = 8192):
    """Create mock tokenizer that produces char-aligned offset_mapping."""
    import torch

    tokenizer = MagicMock()
    tokenizer.model_max_length = max_length

    # Simulate word-level tokens with offset mapping
    words = full_text.split()
    offsets = []
    input_ids_list = []
    pos = 0
    for i, word in enumerate(words):
        start = full_text.find(word, pos)
        end = start + len(word)
        offsets.append([start, end])
        input_ids_list.append(i + 1)  # token id
        pos = end

    offset_tensor = torch.tensor([offsets])
    ids_tensor = torch.tensor([input_ids_list])
    mask_tensor = torch.ones_like(ids_tensor)

    def mock_call(text_or_list, **kwargs):
        if isinstance(text_or_list, list):
            # Batch mode (embed_texts)
            batch_size = len(text_or_list)
            max_len = max(len(t.split()) for t in text_or_list)
            batch_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            batch_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
            for j, t in enumerate(text_or_list):
                n = len(t.split())
                batch_ids[j, :n] = torch.arange(1, n + 1)
                batch_mask[j, :n] = 1
            result = {"input_ids": batch_ids, "attention_mask": batch_mask}
            return result
        # Single text mode (late chunking)
        result = {
            "input_ids": ids_tensor,
            "attention_mask": mask_tensor,
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offset_tensor
        return result

    tokenizer.__call__ = mock_call
    tokenizer.side_effect = mock_call
    return tokenizer


class TestInit:
    def test_creates_with_settings(self) -> None:
        settings = EmbeddingSettings()
        embedder = LateChunkingEmbedder(settings)
        assert embedder._settings is settings
        assert embedder._model is None
        assert embedder._tokenizer is None


class TestEnsureModel:
    def test_raises_when_model_not_loaded(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings())
        with pytest.raises(ModelLoadError, match="not loaded"):
            embedder._ensure_model()


class TestMissingDependency:
    def test_load_model_without_torch(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings())
        with (
            patch.dict("sys.modules", {"torch": None, "transformers": None}),
            pytest.raises(EmbedderNotAvailableError, match="torch"),
        ):
            embedder.load_model()


class TestMatryoshkaSlice:
    def test_slices_to_matryoshka_dim(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings(matryoshka_dim=64))
        full = np.random.randn(768).astype(np.float32)
        sliced = embedder.matryoshka_slice(full)
        assert sliced.shape == (64,)

    def test_normalized(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings(matryoshka_dim=64))
        full = np.random.randn(768).astype(np.float32)
        sliced = embedder.matryoshka_slice(full)
        assert abs(np.linalg.norm(sliced) - 1.0) < 1e-5

    def test_zero_vector_not_normalized(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings(matryoshka_dim=64))
        full = np.zeros(768, dtype=np.float32)
        sliced = embedder.matryoshka_slice(full)
        assert np.all(sliced == 0)

    def test_does_not_modify_input(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings(matryoshka_dim=64))
        full = np.ones(768, dtype=np.float32)
        original = full.copy()
        embedder.matryoshka_slice(full)
        np.testing.assert_array_equal(full, original)

    def test_custom_dim(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings(matryoshka_dim=128))
        full = np.random.randn(768).astype(np.float32)
        sliced = embedder.matryoshka_slice(full)
        assert sliced.shape == (128,)


class TestMeanPool:
    def test_mean_of_two_vectors(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        tokens = np.stack([v1, v2])
        result = LateChunkingEmbedder._mean_pool(tokens)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.0])

    def test_single_token(self) -> None:
        tokens = np.array([[3.0, 4.0]])
        result = LateChunkingEmbedder._mean_pool(tokens)
        np.testing.assert_allclose(result, [3.0, 4.0])

    def test_returns_float32(self) -> None:
        tokens = np.array([[1.0, 2.0]], dtype=np.float64)
        result = LateChunkingEmbedder._mean_pool(tokens)
        assert result.dtype == np.float32


class TestFindChunkTokenSpans:
    def test_finds_two_chunks(self) -> None:
        full_text = "Hello world foo bar baz"
        offsets = np.array(
            [
                [0, 5],  # Hello
                [6, 11],  # world
                [12, 15],  # foo
                [16, 19],  # bar
                [20, 23],  # baz
            ]
        )
        embedder = LateChunkingEmbedder(EmbeddingSettings())
        spans = embedder._find_chunk_token_spans(
            ["Hello world", "foo bar baz"],
            full_text,
            offsets,
        )
        assert len(spans) == 2
        assert spans[0] == (0, 2)  # tokens 0,1
        assert spans[1] == (2, 5)  # tokens 2,3,4

    def test_missing_chunk_returns_zero_span(self) -> None:
        full_text = "Hello world"
        offsets = np.array([[0, 5], [6, 11]])
        embedder = LateChunkingEmbedder(EmbeddingSettings())
        spans = embedder._find_chunk_token_spans(
            ["nonexistent text"],
            full_text,
            offsets,
        )
        assert spans == [(0, 0)]

    def test_sequential_search(self) -> None:
        """Chunks appear in order; second search starts after first match."""
        full_text = "aaa bbb aaa ccc"
        offsets = np.array(
            [
                [0, 3],  # aaa
                [4, 7],  # bbb
                [8, 11],  # aaa
                [12, 15],  # ccc
            ]
        )
        embedder = LateChunkingEmbedder(EmbeddingSettings())
        spans = embedder._find_chunk_token_spans(
            ["aaa bbb", "aaa ccc"],
            full_text,
            offsets,
        )
        assert len(spans) == 2
        # First "aaa" matches at char 0
        assert spans[0][0] == 0
        # Second "aaa" should match at char 8 (sequential search)
        assert spans[1][0] == 2


class TestCharToToken:
    def test_start_position(self) -> None:
        offsets = np.array([[0, 3], [4, 7], [8, 11]])
        # Find token starting at char 4
        idx = LateChunkingEmbedder._char_to_token(offsets, 4, start=True)
        assert idx == 1

    def test_end_position(self) -> None:
        offsets = np.array([[0, 3], [4, 7], [8, 11]])
        # Find token ending at char 7
        idx = LateChunkingEmbedder._char_to_token(offsets, 7, start=False)
        assert idx == 2

    def test_past_end_returns_len(self) -> None:
        offsets = np.array([[0, 3], [4, 7]])
        idx = LateChunkingEmbedder._char_to_token(offsets, 100, start=True)
        assert idx == 2


class TestEmbedDocumentLateChunking:
    def test_empty_chunks_returns_empty(self) -> None:
        settings = EmbeddingSettings()
        embedder = LateChunkingEmbedder(settings)
        # Inject mock model to pass _ensure_model
        embedder._model = MagicMock()
        embedder._tokenizer = MagicMock()
        result = embedder.embed_document_late_chunking("some text", [])
        assert result == []

    def test_produces_correct_count(self) -> None:
        full_text = "Section 10 All agreements are contracts. Section 11 Every person is competent."
        settings = EmbeddingSettings(embedding_dim=768)
        embedder = LateChunkingEmbedder(settings)

        embedder._model = _make_mock_model(768)
        embedder._tokenizer = _make_mock_tokenizer(full_text)

        doc_id = uuid4()
        chunks = [
            _make_chunk("Section 10 All agreements are contracts.", 0, doc_id),
            _make_chunk("Section 11 Every person is competent.", 1, doc_id),
        ]
        result = embedder.embed_document_late_chunking(full_text, chunks)
        assert len(result) == 2
        assert result[0].shape == (768,)
        assert result[1].shape == (768,)

    def test_different_chunks_different_embeddings(self) -> None:
        full_text = "alpha beta gamma delta epsilon zeta eta theta"
        settings = EmbeddingSettings(embedding_dim=768)
        embedder = LateChunkingEmbedder(settings)
        embedder._model = _make_mock_model(768)
        embedder._tokenizer = _make_mock_tokenizer(full_text)

        doc_id = uuid4()
        chunks = [
            _make_chunk("alpha beta gamma delta", 0, doc_id),
            _make_chunk("epsilon zeta eta theta", 1, doc_id),
        ]
        result = embedder.embed_document_late_chunking(full_text, chunks)
        # Different token spans → different embeddings
        assert not np.allclose(result[0], result[1])


class TestEmbedTexts:
    def test_embed_single_text(self) -> None:
        settings = EmbeddingSettings(embedding_dim=768)
        embedder = LateChunkingEmbedder(settings)
        embedder._model = _make_mock_model(768)
        embedder._tokenizer = _make_mock_tokenizer("placeholder")

        result = embedder.embed_texts(["Hello world test"])
        assert len(result) == 1
        assert result[0].shape == (768,)

    def test_embed_empty_list(self) -> None:
        settings = EmbeddingSettings()
        embedder = LateChunkingEmbedder(settings)
        embedder._model = MagicMock()
        embedder._tokenizer = MagicMock()
        result = embedder.embed_texts([])
        assert result == []

    def test_embed_multiple_texts(self) -> None:
        settings = EmbeddingSettings(embedding_dim=768, batch_size=2)
        embedder = LateChunkingEmbedder(settings)
        embedder._model = _make_mock_model(768)
        embedder._tokenizer = _make_mock_tokenizer("placeholder")

        result = embedder.embed_texts(["text one", "text two", "text three"])
        assert len(result) == 3
        for emb in result:
            assert emb.shape == (768,)
            assert emb.dtype == np.float32

    def test_raises_without_model(self) -> None:
        embedder = LateChunkingEmbedder(EmbeddingSettings())
        with pytest.raises(ModelLoadError):
            embedder.embed_texts(["test"])


class TestModelLoading:
    def test_load_model_sets_attributes(self) -> None:
        """Test that load_model populates model and tokenizer."""
        settings = EmbeddingSettings(model_name_or_path="test/model")
        embedder = LateChunkingEmbedder(settings)

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": MagicMock()}):
            embedder.load_model()

        assert embedder._model is mock_model
        assert embedder._tokenizer is mock_tokenizer
        mock_model.eval.assert_called_once()
