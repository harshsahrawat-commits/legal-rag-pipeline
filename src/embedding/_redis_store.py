"""Redis parent text store for retrieval-time context expansion.

Stores the text and key metadata of parent chunks and judgment headers
so that retrieval can expand to the broader context of a matched chunk.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.embedding._exceptions import EmbedderNotAvailableError, RedisStoreError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.embedding._models import EmbeddingSettings

_log = get_logger(__name__)


class RedisParentStore:
    """Stores parent chunk text in Redis for retrieval-time expansion."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._client = None

    async def store_parents(self, chunks: list[LegalChunk]) -> int:
        """Store parent chunks and judgment headers in Redis.

        Identifies all chunks referenced as parent_chunk_id or
        judgment_header_chunk_id by other chunks, then stores their
        text and key metadata.

        Returns:
            Number of entries stored.
        """
        self._ensure_client()

        # Collect IDs that are referenced as parents or judgment headers
        parent_ids = set()
        header_ids = set()
        for chunk in chunks:
            if chunk.parent_info.parent_chunk_id is not None:
                parent_ids.add(chunk.parent_info.parent_chunk_id)
            if chunk.parent_info.judgment_header_chunk_id is not None:
                header_ids.add(chunk.parent_info.judgment_header_chunk_id)

        # Build lookup of chunk ID -> chunk
        chunk_map = {chunk.id: chunk for chunk in chunks}

        # Store all referenced parent/header chunks
        stored = 0
        all_refs = parent_ids | header_ids
        for ref_id in all_refs:
            if ref_id not in chunk_map:
                continue
            chunk = chunk_map[ref_id]
            key = f"{self._settings.redis_key_prefix}{chunk.id}"
            value = self._serialize_parent(chunk)
            try:
                await self._client.set(key, value)
                stored += 1
            except Exception as exc:
                msg = f"Failed to store parent chunk {chunk.id}: {exc}"
                raise RedisStoreError(msg) from exc

        if stored > 0:
            _log.info("parents_stored", count=stored)
        return stored

    @staticmethod
    def _serialize_parent(chunk: LegalChunk) -> str:
        """Serialize a parent chunk to a focused JSON string."""
        data = {
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "text": chunk.text,
            "document_type": chunk.document_type.value,
            "chunk_type": chunk.chunk_type.value,
        }
        if chunk.statute:
            data["act_name"] = chunk.statute.act_name
            data["section_number"] = chunk.statute.section_number
        if chunk.judgment:
            data["case_citation"] = chunk.judgment.case_citation
            data["court"] = chunk.judgment.court
        return json.dumps(data)

    def _ensure_client(self) -> None:
        """Lazy-initialize the async Redis client."""
        if self._client is not None:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            msg = "redis is required. Install with: pip install redis[hiredis]"
            raise EmbedderNotAvailableError(msg) from exc

        self._client = aioredis.from_url(self._settings.redis_url)
