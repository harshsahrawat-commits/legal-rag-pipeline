"""Tests for RedisParentStore with mocked async Redis client."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.acquisition._models import DocumentType
from src.chunking._models import (
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    CourtHierarchy,
    IngestionMetadata,
    JudgmentMetadata,
    LegalChunk,
    ParentDocumentInfo,
    SourceInfo,
    StatuteMetadata,
)
from src.embedding._exceptions import EmbedderNotAvailableError, RedisStoreError
from src.embedding._models import EmbeddingSettings
from src.embedding._redis_store import RedisParentStore


def _make_chunk(
    text: str = "test text",
    doc_id=None,
    chunk_id=None,
    parent_id=None,
    header_id=None,
    doc_type=DocumentType.STATUTE,
    chunk_type=ChunkType.STATUTORY_TEXT,
) -> LegalChunk:
    now = datetime.now(UTC)
    chunk = LegalChunk(
        document_id=doc_id or uuid4(),
        text=text,
        document_type=doc_type,
        chunk_type=chunk_type,
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
        parent_info=ParentDocumentInfo(
            parent_chunk_id=parent_id,
            judgment_header_chunk_id=header_id,
        ),
    )
    if chunk_id:
        chunk.id = chunk_id
    return chunk


@pytest.fixture()
def mock_redis():
    client = AsyncMock()
    client.set = AsyncMock(return_value=True)
    return client


@pytest.fixture()
def store(mock_redis):
    settings = EmbeddingSettings()
    s = RedisParentStore(settings)
    s._client = mock_redis
    return s


class TestInit:
    def test_creates_with_settings(self) -> None:
        settings = EmbeddingSettings()
        s = RedisParentStore(settings)
        assert s._settings is settings
        assert s._client is None


class TestMissingDependency:
    def test_raises_without_redis(self) -> None:
        s = RedisParentStore(EmbeddingSettings())
        with (
            patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}),
            pytest.raises(EmbedderNotAvailableError, match="redis"),
        ):
            s._ensure_client()


class TestStoreParents:
    @pytest.mark.asyncio
    async def test_stores_parent_chunks(self, store, mock_redis) -> None:
        parent_id = uuid4()
        parent = _make_chunk("parent text", chunk_id=parent_id)
        child = _make_chunk("child text", parent_id=parent_id)
        chunks = [parent, child]

        count = await store.store_parents(chunks)
        assert count == 1
        mock_redis.set.assert_called_once()
        key = mock_redis.set.call_args[0][0]
        assert key == f"parent:{parent_id}"

    @pytest.mark.asyncio
    async def test_stores_judgment_header(self, store, mock_redis) -> None:
        header_id = uuid4()
        header = _make_chunk(
            "Header chunk",
            chunk_id=header_id,
            doc_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.FACTS,
        )
        body = _make_chunk("Body chunk", header_id=header_id)
        chunks = [header, body]

        count = await store.store_parents(chunks)
        assert count == 1

    @pytest.mark.asyncio
    async def test_no_parents_returns_zero(self, store, mock_redis) -> None:
        chunks = [_make_chunk("standalone")]
        count = await store.store_parents(chunks)
        assert count == 0
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_references(self, store, mock_redis) -> None:
        parent_id = uuid4()
        parent = _make_chunk("parent", chunk_id=parent_id)
        child1 = _make_chunk("child1", parent_id=parent_id)
        child2 = _make_chunk("child2", parent_id=parent_id)
        chunks = [parent, child1, child2]

        count = await store.store_parents(chunks)
        assert count == 1  # parent stored only once

    @pytest.mark.asyncio
    async def test_referenced_id_not_in_chunks(self, store, mock_redis) -> None:
        """If parent_chunk_id references a chunk not in the list, skip it."""
        missing_id = uuid4()
        child = _make_chunk("orphan", parent_id=missing_id)
        count = await store.store_parents([child])
        assert count == 0

    @pytest.mark.asyncio
    async def test_redis_error_raises(self, store, mock_redis) -> None:
        mock_redis.set.side_effect = RuntimeError("connection lost")
        parent_id = uuid4()
        parent = _make_chunk("parent", chunk_id=parent_id)
        child = _make_chunk("child", parent_id=parent_id)
        with pytest.raises(RedisStoreError, match="connection lost"):
            await store.store_parents([parent, child])

    @pytest.mark.asyncio
    async def test_empty_chunks(self, store) -> None:
        count = await store.store_parents([])
        assert count == 0


class TestSerializeParent:
    def test_statute_chunk(self) -> None:
        now = datetime.now(UTC)
        chunk = LegalChunk(
            document_id=uuid4(),
            text="Section 10 text",
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
            statute=StatuteMetadata(act_name="IPC", section_number="302"),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        )
        result = json.loads(RedisParentStore._serialize_parent(chunk))
        assert result["act_name"] == "IPC"
        assert result["section_number"] == "302"
        assert result["document_type"] == "statute"

    def test_judgment_chunk(self) -> None:
        now = datetime.now(UTC)
        chunk = LegalChunk(
            document_id=uuid4(),
            text="Appeal dismissed",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.HOLDING,
            chunk_index=0,
            token_count=3,
            source=SourceInfo(
                url="https://example.com",
                source_name="test",
                scraped_at=now,
                last_verified=now,
            ),
            judgment=JudgmentMetadata(
                case_citation="AIR 2024 SC 1",
                court="Supreme Court",
                court_level=CourtHierarchy.SUPREME_COURT,
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL,
            ),
        )
        result = json.loads(RedisParentStore._serialize_parent(chunk))
        assert result["case_citation"] == "AIR 2024 SC 1"
        assert result["court"] == "Supreme Court"

    def test_minimal_chunk(self) -> None:
        now = datetime.now(UTC)
        chunk = LegalChunk(
            document_id=uuid4(),
            text="Some text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=2,
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
        result = json.loads(RedisParentStore._serialize_parent(chunk))
        assert "text" in result
        assert "act_name" not in result
        assert "case_citation" not in result


class TestKeyFormat:
    @pytest.mark.asyncio
    async def test_default_prefix(self, mock_redis) -> None:
        settings = EmbeddingSettings(redis_key_prefix="parent:")
        s = RedisParentStore(settings)
        s._client = mock_redis

        parent_id = uuid4()
        parent = _make_chunk("text", chunk_id=parent_id)
        child = _make_chunk("child", parent_id=parent_id)
        await s.store_parents([parent, child])

        key = mock_redis.set.call_args[0][0]
        assert key.startswith("parent:")

    @pytest.mark.asyncio
    async def test_custom_prefix(self, mock_redis) -> None:
        settings = EmbeddingSettings(redis_key_prefix="legal:")
        s = RedisParentStore(settings)
        s._client = mock_redis

        parent_id = uuid4()
        parent = _make_chunk("text", chunk_id=parent_id)
        child = _make_chunk("child", parent_id=parent_id)
        await s.store_parents([parent, child])

        key = mock_redis.set.call_args[0][0]
        assert key.startswith("legal:")
