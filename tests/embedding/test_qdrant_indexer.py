"""Tests for QdrantIndexer with mocked QdrantClient."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from types import ModuleType
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
from src.embedding._exceptions import (
    CollectionCreationError,
    EmbedderNotAvailableError,
    IndexingError,
)
from src.embedding._models import EmbeddingSettings, SparseVector
from src.embedding._qdrant_indexer import QdrantIndexer
from src.enrichment._models import QuIMDocument, QuIMEntry


def _make_chunk(text: str = "test text", doc_id=None) -> LegalChunk:
    now = datetime.now(UTC)
    return LegalChunk(
        document_id=doc_id or uuid4(),
        text=text,
        document_type=DocumentType.STATUTE,
        chunk_type=ChunkType.STATUTORY_TEXT,
        chunk_index=0,
        token_count=5,
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


def _build_mock_qdrant_module():
    """Build a fake qdrant_client module with all necessary model classes."""
    mod = ModuleType("qdrant_client")
    models_mod = ModuleType("qdrant_client.models")

    # Mock model classes
    models_mod.PointStruct = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.VectorParams = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.SparseVectorParams = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.SparseVector = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.Distance = MagicMock()
    models_mod.Distance.COSINE = "Cosine"

    mod.models = models_mod
    mod.QdrantClient = MagicMock
    return mod, models_mod


@pytest.fixture(autouse=True)
def _mock_qdrant_module():
    """Inject a fake qdrant_client module for all tests."""
    mod, models_mod = _build_mock_qdrant_module()
    with patch.dict(sys.modules, {"qdrant_client": mod, "qdrant_client.models": models_mod}):
        yield


@pytest.fixture()
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = False
    client.upsert.return_value = None
    client.retrieve.return_value = []
    return client


@pytest.fixture()
def indexer(mock_qdrant_client):
    settings = EmbeddingSettings()
    idx = QdrantIndexer(settings)
    idx._client = mock_qdrant_client
    return idx


class TestInit:
    def test_creates_with_settings(self) -> None:
        settings = EmbeddingSettings()
        idx = QdrantIndexer(settings)
        assert idx._settings is settings
        assert idx._client is None


class TestMissingDependency:
    def test_raises_without_qdrant_client(self) -> None:
        idx = QdrantIndexer(EmbeddingSettings())
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(EmbedderNotAvailableError, match="qdrant-client"),
        ):
            idx._ensure_client()


class TestEnsureCollections:
    def test_creates_both_collections(self, indexer, mock_qdrant_client) -> None:
        indexer.ensure_collections()
        assert mock_qdrant_client.create_collection.call_count == 2

    def test_skips_existing_collections(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.collection_exists.return_value = True
        indexer.ensure_collections()
        mock_qdrant_client.create_collection.assert_not_called()

    def test_wraps_errors(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.collection_exists.side_effect = RuntimeError("connection refused")
        with pytest.raises(CollectionCreationError):
            indexer.ensure_collections()


class TestUpsertChunks:
    @pytest.mark.asyncio
    async def test_upserts_with_dual_vectors(self, indexer, mock_qdrant_client) -> None:
        chunk = _make_chunk()
        full_emb = np.random.randn(768).astype(np.float32)
        fast_emb = np.random.randn(64).astype(np.float32)
        sparse = SparseVector(indices=[0, 1], values=[1.5, 0.8])

        count = await indexer.upsert_chunks([chunk], [full_emb], [fast_emb], [sparse])
        assert count == 1
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_zero(self, indexer) -> None:
        count = await indexer.upsert_chunks([], [], [], [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_chunks(self, indexer, mock_qdrant_client) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk(f"text {i}", doc_id) for i in range(3)]
        fulls = [np.random.randn(768).astype(np.float32) for _ in range(3)]
        fasts = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        sparses = [SparseVector(indices=[0], values=[1.0]) for _ in range(3)]

        count = await indexer.upsert_chunks(chunks, fulls, fasts, sparses)
        assert count == 3

    @pytest.mark.asyncio
    async def test_upsert_error_raises(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.upsert.side_effect = RuntimeError("network error")
        chunk = _make_chunk()
        with pytest.raises(IndexingError, match="network error"):
            await indexer.upsert_chunks(
                [chunk],
                [np.zeros(768)],
                [np.zeros(64)],
                [SparseVector(indices=[], values=[])],
            )

    @pytest.mark.asyncio
    async def test_empty_sparse_vector(self, indexer, mock_qdrant_client) -> None:
        """Chunks with empty sparse vectors should still upsert."""
        chunk = _make_chunk()
        count = await indexer.upsert_chunks(
            [chunk],
            [np.zeros(768)],
            [np.zeros(64)],
            [SparseVector(indices=[], values=[])],
        )
        assert count == 1


class TestUpsertQuimQuestions:
    @pytest.mark.asyncio
    async def test_upserts_questions(self, indexer, mock_qdrant_client) -> None:
        doc_id = uuid4()
        chunk_id = uuid4()
        quim = QuIMDocument(
            document_id=doc_id,
            entries=[
                QuIMEntry(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    questions=["Q1?", "Q2?"],
                )
            ],
        )
        embeddings = [np.random.randn(768).astype(np.float32) for _ in range(2)]

        count = await indexer.upsert_quim_questions(quim, embeddings)
        assert count == 2
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_quim_returns_zero(self, indexer) -> None:
        quim = QuIMDocument(document_id=uuid4(), entries=[])
        count = await indexer.upsert_quim_questions(quim, [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_upsert_error_raises(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.upsert.side_effect = RuntimeError("timeout")
        doc_id = uuid4()
        quim = QuIMDocument(
            document_id=doc_id,
            entries=[
                QuIMEntry(
                    chunk_id=uuid4(),
                    document_id=doc_id,
                    questions=["Q?"],
                )
            ],
        )
        with pytest.raises(IndexingError, match="timeout"):
            await indexer.upsert_quim_questions(quim, [np.zeros(768)])


class TestChunkExists:
    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.retrieve.return_value = []
        assert await indexer.chunk_exists(uuid4()) is False

    @pytest.mark.asyncio
    async def test_returns_true_when_found(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.retrieve.return_value = [MagicMock()]
        assert await indexer.chunk_exists(uuid4()) is True

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, indexer, mock_qdrant_client) -> None:
        mock_qdrant_client.retrieve.side_effect = RuntimeError("error")
        assert await indexer.chunk_exists(uuid4()) is False


class TestBuildPayload:
    def test_returns_dict(self, indexer) -> None:
        chunk = _make_chunk()
        payload = indexer._build_chunk_payload(chunk)
        assert isinstance(payload, dict)
        assert "text" in payload
        assert "document_type" in payload

    def test_contains_chunk_id(self, indexer) -> None:
        chunk = _make_chunk()
        payload = indexer._build_chunk_payload(chunk)
        assert payload["id"] == str(chunk.id)
