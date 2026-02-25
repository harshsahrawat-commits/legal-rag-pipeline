"""Integration tests for the embedding pipeline with all mocks."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.acquisition._models import DocumentType
from src.chunking._models import (
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    IngestionMetadata,
    LegalChunk,
    ParentDocumentInfo,
    SourceInfo,
    StatuteMetadata,
)
from src.embedding._models import EmbeddingConfig, EmbeddingSettings, SparseVector
from src.embedding._sparse import BM25SparseEncoder
from src.embedding.pipeline import EmbeddingPipeline
from src.enrichment._models import QuIMDocument, QuIMEntry


def _build_mock_qdrant_module():
    mod = ModuleType("qdrant_client")
    models_mod = ModuleType("qdrant_client.models")
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
def _mock_qdrant():
    mod, models_mod = _build_mock_qdrant_module()
    with patch.dict(sys.modules, {"qdrant_client": mod, "qdrant_client.models": models_mod}):
        yield


def _make_chunk(
    text: str,
    chunk_index: int = 0,
    doc_id=None,
    parent_id=None,
    chunk_id=None,
) -> LegalChunk:
    now = datetime.now(UTC)
    c = LegalChunk(
        document_id=doc_id or uuid4(),
        text=text,
        contextualized_text=f"Indian Contract Act — {text}",
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
        statute=StatuteMetadata(act_name="Indian Contract Act"),
        content=ContentMetadata(),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="test",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            contextualized=True,
        ),
        parent_info=ParentDocumentInfo(parent_chunk_id=parent_id),
    )
    if chunk_id:
        c.id = chunk_id
    return c


def _write_data(chunks, path, quim_doc=None, parsed_doc=None, parsed_path=None):
    """Write enriched chunks, optional QuIM sidecar and parsed doc."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in chunks]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if quim_doc:
        quim_path = path.with_suffix(".quim.json")
        quim_path.write_text(quim_doc.model_dump_json(indent=2), encoding="utf-8")

    if parsed_doc and parsed_path:
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(parsed_doc.model_dump_json(indent=2), encoding="utf-8")


class TestLateChunkingE2E:
    @pytest.mark.asyncio
    async def test_embeds_all_chunks(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        enriched_dir = tmp_path / "enriched"
        doc_id = uuid4()
        chunks = [_make_chunk(f"Section {i} text", i, doc_id) for i in range(3)]
        path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"
        _write_data(chunks, path)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        result = await pipeline.run()

        assert result.documents_indexed == 1
        assert result.chunks_embedded == 3
        mock_qdrant_client.upsert.assert_called_once()


class TestBM25E2E:
    def test_sparse_encoding_pipeline(self) -> None:
        """BM25 encoder produces valid sparse vectors from contextualized text."""
        texts = [
            "Indian Contract Act — Section 10. All agreements are contracts.",
            "Indian Contract Act — Section 11. Every person is competent.",
        ]
        encoder = BM25SparseEncoder()
        encoder.build_vocabulary(texts)

        for text in texts:
            sparse = encoder.encode(text)
            assert isinstance(sparse, SparseVector)
            assert len(sparse.indices) > 0
            assert all(v > 0 for v in sparse.values)


class TestQuIME2E:
    @pytest.mark.asyncio
    async def test_embeds_quim_questions(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        enriched_dir = tmp_path / "enriched"
        doc_id = uuid4()
        chunks = [_make_chunk("Section 10 text", 0, doc_id)]
        path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"

        quim = QuIMDocument(
            document_id=doc_id,
            entries=[
                QuIMEntry(
                    chunk_id=chunks[0].id,
                    document_id=doc_id,
                    questions=["What is Section 10?", "Who can contract?"],
                )
            ],
        )
        _write_data(chunks, path, quim_doc=quim)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        result = await pipeline.run()

        assert result.quim_questions_embedded == 2
        # Qdrant upsert called twice: once for chunks, once for QuIM
        assert mock_qdrant_client.upsert.call_count == 2


class TestRedisParentE2E:
    @pytest.mark.asyncio
    async def test_stores_parent_text(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        enriched_dir = tmp_path / "enriched"
        doc_id = uuid4()
        parent_id = uuid4()
        parent = _make_chunk("Parent section text", 0, doc_id, chunk_id=parent_id)
        child = _make_chunk("Child sub-section", 1, doc_id, parent_id=parent_id)
        path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"
        _write_data([parent, child], path)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        result = await pipeline.run()

        assert result.parent_entries_stored == 1
        mock_redis_client.set.assert_called_once()
        key = mock_redis_client.set.call_args[0][0]
        assert key == f"parent:{parent_id}"


class TestLateCunkedFlagE2E:
    @pytest.mark.asyncio
    async def test_flag_set_on_all_chunks(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        enriched_dir = tmp_path / "enriched"
        doc_id = uuid4()
        chunks = [_make_chunk(f"text {i}", i, doc_id) for i in range(2)]
        path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"
        _write_data(chunks, path)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        await pipeline.run()

        data = json.loads(path.read_text(encoding="utf-8"))
        for chunk_data in data:
            assert chunk_data["ingestion"]["late_chunked"] is True


class TestIdempotencyE2E:
    @pytest.mark.asyncio
    async def test_skips_already_indexed(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        mock_qdrant_client.retrieve.return_value = [MagicMock()]  # chunk exists
        enriched_dir = tmp_path / "enriched"
        doc_id = uuid4()
        chunks = [_make_chunk("text", 0, doc_id)]
        path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"
        _write_data(chunks, path)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        result = await pipeline.run()

        assert result.documents_skipped == 1
        assert result.documents_indexed == 0
        mock_qdrant_client.upsert.assert_not_called()


class TestErrorIsolationE2E:
    @pytest.mark.asyncio
    async def test_bad_doc_doesnt_block_others(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        enriched_dir = tmp_path / "enriched"
        source_dir = enriched_dir / "indian_kanoon"
        source_dir.mkdir(parents=True, exist_ok=True)

        # Bad file
        bad_path = source_dir / "bad.json"
        bad_path.write_text("{{{invalid", encoding="utf-8")

        # Good file
        doc_id = uuid4()
        good_chunks = [_make_chunk("good text", 0, doc_id)]
        good_path = source_dir / f"{doc_id}.json"
        _write_data(good_chunks, good_path)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        result = await pipeline.run()

        assert result.documents_found == 2
        assert result.documents_failed == 1
        assert result.documents_indexed == 1


class TestMultiDocumentE2E:
    @pytest.mark.asyncio
    async def test_processes_multiple_docs(
        self,
        tmp_path,
        mock_embedder,
        mock_qdrant_client,
        mock_redis_client,
    ) -> None:
        enriched_dir = tmp_path / "enriched"
        for i in range(3):
            doc_id = uuid4()
            chunks = [_make_chunk(f"text {i}", 0, doc_id)]
            path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"
            _write_data(chunks, path)

        config = EmbeddingConfig(
            settings=EmbeddingSettings(input_dir=enriched_dir, parsed_dir=tmp_path / "parsed")
        )
        pipeline = EmbeddingPipeline(config=config)
        pipeline._embedder = mock_embedder
        pipeline._indexer._client = mock_qdrant_client
        pipeline._redis._client = mock_redis_client

        result = await pipeline.run()

        assert result.documents_found == 3
        assert result.documents_indexed == 3
        assert result.chunks_embedded == 3
