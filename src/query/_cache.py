"""Semantic Query Cache backed by Qdrant (similarity) + Redis (response storage).

Caches RAG responses by query embedding similarity. On cache hit, the stored
response is returned without running the full retrieval + generation pipeline.
Supports targeted invalidation when underlying legislation changes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.query._exceptions import CacheError
from src.query._models import CacheResult
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.query._models import QuerySettings

_log = get_logger(__name__)


class SemanticQueryCache:
    """Qdrant-backed embedding similarity cache with Redis response storage.

    Qdrant stores query embeddings with metadata payloads. Redis stores the
    full serialized response keyed by a cache_key. On lookup, Qdrant finds
    similar queries; if the best match exceeds the similarity threshold, the
    response is fetched from Redis.

    All external dependencies (qdrant_client, redis) are lazy-imported so
    the module loads without them installed.
    """

    def __init__(self, settings: QuerySettings) -> None:
        self._settings = settings
        self._qdrant = None
        self._redis = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Check whether caching is enabled and required deps are importable."""
        if not self._settings.cache_enabled:
            return False
        try:
            import qdrant_client  # noqa: F401
            import redis  # noqa: F401
        except ImportError:
            return False
        return True

    def get(self, query_embedding: list[float]) -> CacheResult:
        """Look up a cached response by query embedding similarity.

        Returns a CacheResult with hit=True if a sufficiently similar query
        was found AND its response still exists in Redis. Otherwise hit=False.
        """
        if not self.is_available:
            return CacheResult(hit=False)

        try:
            self._ensure_clients()

            results = self._qdrant.search(
                collection_name=self._settings.cache_collection,
                query_vector=query_embedding,
                limit=1,
                score_threshold=self._settings.cache_similarity_threshold,
            )

            if not results:
                return CacheResult(hit=False)

            top = results[0]
            payload = top.payload or {}
            cache_key = payload.get("cache_key", "")
            similarity = top.score

            # Fetch response from Redis
            raw = self._redis.get(cache_key)
            if raw is None:
                # Redis entry expired — clean up orphaned Qdrant point
                _log.info("cache_redis_expired", cache_key=cache_key)
                try:
                    self._qdrant.delete(
                        collection_name=self._settings.cache_collection,
                        points_selector=[top.id],
                    )
                except Exception:
                    _log.warning("cache_orphan_cleanup_failed", point_id=top.id)
                return CacheResult(hit=False)

            response = json.loads(raw)
            _log.info(
                "cache_hit",
                cache_key=cache_key,
                similarity=similarity,
            )
            return CacheResult(
                hit=True,
                response=response,
                similarity=similarity,
                cache_key=cache_key,
            )
        except Exception as exc:
            _log.warning("cache_get_error", error=str(exc))
            return CacheResult(hit=False)

    def set(
        self,
        query_text: str,
        query_embedding: list[float],
        response: dict[str, Any],
        acts_cited: list[str] | None = None,
    ) -> str:
        """Store a query response in the cache.

        Args:
            query_text: The original query string.
            query_embedding: The query's embedding vector.
            response: The full response dict to cache.
            acts_cited: Optional list of act names cited in the response.

        Returns:
            The generated cache_key.

        Raises:
            CacheError: If storing fails irrecoverably.
        """
        self._ensure_clients()

        from qdrant_client.models import PointStruct

        cache_key = f"{self._settings.cache_redis_prefix}{uuid4()}"
        point_id = str(uuid4())
        now = datetime.now(UTC).isoformat()

        try:
            # Store response in Redis with TTL
            serialized = json.dumps(response)
            self._redis.setex(
                cache_key,
                self._settings.cache_ttl_seconds,
                serialized,
            )

            # Upsert embedding + metadata into Qdrant
            payload: dict[str, Any] = {
                "cache_key": cache_key,
                "original_query": query_text,
                "acts_cited": acts_cited or [],
                "cached_at": now,
            }

            point = PointStruct(
                id=point_id,
                vector=query_embedding,
                payload=payload,
            )

            self._qdrant.upsert(
                collection_name=self._settings.cache_collection,
                points=[point],
            )

            _log.info("cache_set", cache_key=cache_key, query=query_text[:80])
            return cache_key
        except Exception as exc:
            _log.warning("cache_set_error", error=str(exc))
            msg = f"Failed to store cache entry: {exc}"
            raise CacheError(msg) from exc

    def invalidate_for_act(self, act_name: str) -> int:
        """Invalidate all cached entries that cite a given act.

        Used when legislation is amended or replaced (e.g. IPC -> BNS).

        Returns:
            Number of entries invalidated.
        """
        if not self.is_available:
            return 0

        try:
            self._ensure_clients()
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
            )

            act_filter = Filter(
                must=[
                    FieldCondition(
                        key="acts_cited",
                        match=MatchValue(value=act_name),
                    ),
                ],
            )

            count = 0
            offset = None
            while True:
                scroll_kwargs: dict[str, Any] = {
                    "collection_name": self._settings.cache_collection,
                    "scroll_filter": act_filter,
                    "limit": 100,
                }
                if offset is not None:
                    scroll_kwargs["offset"] = offset

                points, next_offset = self._qdrant.scroll(**scroll_kwargs)

                point_ids = []
                for point in points:
                    payload = point.payload or {}
                    cache_key = payload.get("cache_key", "")
                    if cache_key:
                        self._redis.delete(cache_key)
                    point_ids.append(point.id)

                if point_ids:
                    self._qdrant.delete(
                        collection_name=self._settings.cache_collection,
                        points_selector=point_ids,
                    )
                count += len(point_ids)

                if next_offset is None:
                    break
                offset = next_offset

            _log.info("cache_invalidated_for_act", act_name=act_name, count=count)
            return count
        except Exception as exc:
            _log.warning("cache_invalidate_act_error", act_name=act_name, error=str(exc))
            return 0

    def invalidate_by_key(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry by its key.

        Deletes from both Redis and Qdrant.

        Returns:
            True if the entry was found and deleted, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            self._ensure_clients()
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
            )

            # Delete from Redis
            self._redis.delete(cache_key)

            # Find and delete from Qdrant
            key_filter = Filter(
                must=[
                    FieldCondition(
                        key="cache_key",
                        match=MatchValue(value=cache_key),
                    ),
                ],
            )

            points, _ = self._qdrant.scroll(
                collection_name=self._settings.cache_collection,
                scroll_filter=key_filter,
                limit=1,
            )

            if not points:
                return False

            self._qdrant.delete(
                collection_name=self._settings.cache_collection,
                points_selector=[points[0].id],
            )

            _log.info("cache_invalidated_by_key", cache_key=cache_key)
            return True
        except Exception as exc:
            _log.warning("cache_invalidate_key_error", cache_key=cache_key, error=str(exc))
            return False

    def clear(self) -> int:
        """Delete all entries from the cache.

        Returns:
            Number of entries deleted.
        """
        if not self.is_available:
            return 0

        try:
            self._ensure_clients()

            # Count existing points
            count = 0
            offset = None
            point_ids = []
            while True:
                scroll_kwargs: dict[str, Any] = {
                    "collection_name": self._settings.cache_collection,
                    "limit": 100,
                }
                if offset is not None:
                    scroll_kwargs["offset"] = offset

                points, next_offset = self._qdrant.scroll(**scroll_kwargs)

                for point in points:
                    payload = point.payload or {}
                    cache_key = payload.get("cache_key", "")
                    if cache_key:
                        self._redis.delete(cache_key)
                    point_ids.append(point.id)
                    count += 1

                if next_offset is None:
                    break
                offset = next_offset

            # Delete all points from Qdrant
            if point_ids:
                self._qdrant.delete(
                    collection_name=self._settings.cache_collection,
                    points_selector=point_ids,
                )

            _log.info("cache_cleared", count=count)
            return count
        except Exception as exc:
            _log.warning("cache_clear_error", error=str(exc))
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_clients(self) -> None:
        """Lazy-initialize Qdrant and Redis clients, creating the collection if needed."""
        if self._qdrant is not None and self._redis is not None:
            return

        if self._qdrant is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                msg = "qdrant-client is required. Install with: pip install qdrant-client"
                raise CacheError(msg) from exc

            self._qdrant = QdrantClient(
                host=self._settings.qdrant_host,
                port=self._settings.qdrant_port,
            )
            self._ensure_collection()

        if self._redis is None:
            try:
                import redis as redis_lib
            except ImportError as exc:
                msg = "redis is required. Install with: pip install redis"
                raise CacheError(msg) from exc

            self._redis = redis_lib.Redis.from_url(self._settings.redis_url)

    def _ensure_collection(self) -> None:
        """Create the query_cache collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        name = self._settings.cache_collection
        if self._qdrant.collection_exists(name):
            _log.debug("cache_collection_exists", name=name)
            return

        self._qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self._settings.embedding_dim,
                distance=Distance.COSINE,
            ),
            on_disk_payload=True,
        )
        _log.info("cache_collection_created", name=name)
