"""Tests for EmbeddingPipeline orchestrator."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — used at runtime in fixtures
from unittest.mock import AsyncMock, MagicMock
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
    StatuteMetadata,
)
from src.embedding._models import EmbeddingConfig, EmbeddingSettings
from src.embedding.pipeline import EmbeddingPipeline


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
        statute=StatuteMetadata(act_name="Test Act"),
        content=ContentMetadata(),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="test",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
    )


def _write_enriched_chunks(chunks: list[LegalChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in chunks]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture()
def enriched_dir(tmp_path: Path) -> Path:
    return tmp_path / "enriched"


@pytest.fixture()
def parsed_dir(tmp_path: Path) -> Path:
    return tmp_path / "parsed"


@pytest.fixture()
def settings(enriched_dir: Path, parsed_dir: Path) -> EmbeddingSettings:
    return EmbeddingSettings(
        input_dir=enriched_dir,
        parsed_dir=parsed_dir,
    )


@pytest.fixture()
def config(settings: EmbeddingSettings) -> EmbeddingConfig:
    return EmbeddingConfig(settings=settings)


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.load_model = MagicMock()
    embedder.embed_document_late_chunking = MagicMock(
        side_effect=lambda text, chunks: [np.random.randn(768).astype(np.float32) for _ in chunks]
    )
    embedder.embed_texts = MagicMock(
        side_effect=lambda texts: [np.random.randn(768).astype(np.float32) for _ in texts]
    )
    embedder.matryoshka_slice = MagicMock(
        side_effect=lambda e: np.random.randn(64).astype(np.float32)
    )
    return embedder


@pytest.fixture()
def mock_indexer():
    indexer = MagicMock()
    indexer.ensure_collections = MagicMock()
    indexer.upsert_chunks = AsyncMock(side_effect=lambda chunks, *a: len(chunks))
    indexer.upsert_quim_questions = AsyncMock(return_value=0)
    indexer.chunk_exists = AsyncMock(return_value=False)
    return indexer


@pytest.fixture()
def mock_redis():
    store = MagicMock()
    store.store_parents = AsyncMock(return_value=0)
    return store


@pytest.fixture()
def pipeline(config, mock_embedder, mock_indexer, mock_redis):
    p = EmbeddingPipeline(config=config)
    p._embedder = mock_embedder
    p._indexer = mock_indexer
    p._redis = mock_redis
    return p


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_discovers_enriched_files(self, pipeline, enriched_dir) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk("test text", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")

        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1

    @pytest.mark.asyncio
    async def test_excludes_quim_files(self, pipeline, enriched_dir) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk("test", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")
        # Write a quim sidecar — should be excluded
        (enriched_dir / "indian_kanoon" / f"{doc_id}.quim.json").write_text("{}", encoding="utf-8")

        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1

    @pytest.mark.asyncio
    async def test_empty_input_dir(self, pipeline) -> None:
        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 0

    @pytest.mark.asyncio
    async def test_source_filter(self, pipeline, enriched_dir) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk("test", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")
        _write_enriched_chunks(chunks, enriched_dir / "india_code" / f"{doc_id}.json")

        result = await pipeline.run(source_name="Indian Kanoon", dry_run=True)
        assert result.documents_found == 1

    @pytest.mark.asyncio
    async def test_unknown_source(self, pipeline) -> None:
        result = await pipeline.run(source_name="Unknown Source")
        assert result.documents_found == 0
        assert len(result.errors) == 1


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_no_processing(self, pipeline, enriched_dir, mock_embedder) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk("test", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")

        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1
        assert result.documents_indexed == 0
        mock_embedder.load_model.assert_not_called()


class TestProcessDocument:
    @pytest.mark.asyncio
    async def test_indexes_document(self, pipeline, enriched_dir, mock_indexer) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk("Section 10 text", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")

        result = await pipeline.run()
        assert result.documents_indexed == 1
        assert result.chunks_embedded == 1

    @pytest.mark.asyncio
    async def test_multiple_documents(self, pipeline, enriched_dir) -> None:
        for i in range(3):
            doc_id = uuid4()
            chunks = [_make_chunk(f"text {i}", doc_id=doc_id)]
            _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")

        result = await pipeline.run()
        assert result.documents_indexed == 3

    @pytest.mark.asyncio
    async def test_empty_document_skipped(self, pipeline, enriched_dir) -> None:
        path = enriched_dir / "indian_kanoon" / "empty.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")

        result = await pipeline.run()
        assert result.documents_skipped == 1


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_skips_existing_chunks(self, pipeline, enriched_dir, mock_indexer) -> None:
        mock_indexer.chunk_exists = AsyncMock(return_value=True)
        doc_id = uuid4()
        chunks = [_make_chunk("test", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")

        result = await pipeline.run()
        assert result.documents_skipped == 1
        assert result.documents_indexed == 0


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_load_error_isolates_document(self, pipeline, enriched_dir) -> None:
        path = enriched_dir / "indian_kanoon" / "bad.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invalid json{{{", encoding="utf-8")

        result = await pipeline.run()
        assert result.documents_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_embedding_error_isolates(self, pipeline, enriched_dir, mock_embedder) -> None:
        # Both paths must fail — pipeline falls back to embed_texts when no parsed doc
        mock_embedder.embed_document_late_chunking.side_effect = RuntimeError("GPU error")
        mock_embedder.embed_texts.side_effect = RuntimeError("GPU error")
        doc_id = uuid4()
        chunks = [_make_chunk("test", doc_id=doc_id)]
        _write_enriched_chunks(chunks, enriched_dir / "indian_kanoon" / f"{doc_id}.json")

        result = await pipeline.run()
        assert result.documents_failed == 1

    @pytest.mark.asyncio
    async def test_partial_failure(self, pipeline, enriched_dir, mock_embedder) -> None:
        """First doc fails, second succeeds."""
        call_count = [0]
        original_et = mock_embedder.embed_texts.side_effect

        def fail_et(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first doc fails")
            return original_et(texts)

        # No parsed doc → empty full_text → pipeline uses embed_texts for both
        mock_embedder.embed_texts.side_effect = fail_et

        for i in range(2):
            doc_id = uuid4()
            _write_enriched_chunks(
                [_make_chunk(f"text {i}", doc_id=doc_id)],
                enriched_dir / "indian_kanoon" / f"{doc_id}.json",
            )

        result = await pipeline.run()
        assert result.documents_failed == 1
        assert result.documents_indexed == 1


class TestLateCunkedFlag:
    @pytest.mark.asyncio
    async def test_updates_flag(self, pipeline, enriched_dir) -> None:
        doc_id = uuid4()
        chunks = [_make_chunk("test", doc_id=doc_id)]
        path = enriched_dir / "indian_kanoon" / f"{doc_id}.json"
        _write_enriched_chunks(chunks, path)

        await pipeline.run()

        # Re-read and verify the flag was set
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0]["ingestion"]["late_chunked"] is True


class TestResultCounts:
    @pytest.mark.asyncio
    async def test_result_has_timestamps(self, pipeline) -> None:
        result = await pipeline.run(dry_run=True)
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at
