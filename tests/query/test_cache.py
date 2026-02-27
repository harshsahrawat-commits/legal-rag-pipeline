"""Tests for SemanticQueryCache with mocked Qdrant and Redis."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.query._exceptions import CacheError
from src.query._models import CacheResult, QuerySettings

# ---------------------------------------------------------------------------
# Fake module builders (qdrant_client + redis may not be installed)
# ---------------------------------------------------------------------------


def _build_mock_qdrant_module():
    """Build a fake qdrant_client module with model classes."""
    mod = ModuleType("qdrant_client")
    models_mod = ModuleType("qdrant_client.models")

    models_mod.PointStruct = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.VectorParams = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.Distance = MagicMock()
    models_mod.Distance.COSINE = "Cosine"
    models_mod.FieldCondition = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.Filter = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.MatchValue = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    mod.models = models_mod
    mod.QdrantClient = MagicMock
    return mod, models_mod


def _build_mock_redis_module():
    """Build a fake redis module."""
    mod = ModuleType("redis")
    mod.Redis = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_modules():
    """Inject fake qdrant_client and redis modules for all tests."""
    qdrant_mod, qdrant_models = _build_mock_qdrant_module()
    redis_mod = _build_mock_redis_module()
    with patch.dict(
        sys.modules,
        {
            "qdrant_client": qdrant_mod,
            "qdrant_client.models": qdrant_models,
            "redis": redis_mod,
        },
    ):
        yield


@pytest.fixture
def settings() -> QuerySettings:
    return QuerySettings()


@pytest.fixture
def disabled_settings() -> QuerySettings:
    return QuerySettings(cache_enabled=False)


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = False
    client.search.return_value = []
    client.scroll.return_value = ([], None)
    client.upsert.return_value = None
    client.delete.return_value = None
    client.create_collection.return_value = None
    return client


@pytest.fixture
def mock_redis_client():
    client = MagicMock()
    client.get.return_value = None
    client.setex.return_value = True
    client.delete.return_value = 1
    return client


@pytest.fixture
def cache(settings, mock_qdrant_client, mock_redis_client):
    """A SemanticQueryCache with pre-injected mock clients."""
    from src.query._cache import SemanticQueryCache

    c = SemanticQueryCache(settings)
    c._qdrant = mock_qdrant_client
    c._redis = mock_redis_client
    return c


@pytest.fixture
def sample_embedding() -> list[float]:
    return [0.1] * 768


@pytest.fixture
def sample_response() -> dict:
    return {"answer": "Section 302 of IPC deals with murder.", "sources": ["IPC"]}


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_true_when_enabled_and_deps_available(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        assert cache.is_available is True

    def test_false_when_disabled(self, disabled_settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(disabled_settings)
        assert cache.is_available is False

    def test_false_when_qdrant_missing(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        with patch.dict(sys.modules, {"qdrant_client": None}):
            assert cache.is_available is False

    def test_false_when_redis_missing(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        with patch.dict(sys.modules, {"redis": None}):
            assert cache.is_available is False

    def test_false_when_both_missing(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        with patch.dict(sys.modules, {"qdrant_client": None, "redis": None}):
            assert cache.is_available is False


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_miss_no_results(self, cache, mock_qdrant_client, sample_embedding) -> None:
        mock_qdrant_client.search.return_value = []
        result = cache.get(sample_embedding)
        assert result.hit is False
        assert result.response is None

    def test_hit_above_threshold(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding, sample_response
    ) -> None:
        point = MagicMock()
        point.payload = {"cache_key": "qcache:abc-123"}
        point.score = 0.95
        point.id = "pt-1"
        mock_qdrant_client.search.return_value = [point]
        mock_redis_client.get.return_value = json.dumps(sample_response)

        result = cache.get(sample_embedding)
        assert result.hit is True
        assert result.response == sample_response
        assert result.similarity == 0.95
        assert result.cache_key == "qcache:abc-123"

    def test_miss_redis_expired(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding
    ) -> None:
        """Qdrant hit but Redis entry expired — clean up orphan and miss."""
        point = MagicMock()
        point.payload = {"cache_key": "qcache:expired-key"}
        point.score = 0.96
        point.id = "pt-expired"
        mock_qdrant_client.search.return_value = [point]
        mock_redis_client.get.return_value = None

        result = cache.get(sample_embedding)
        assert result.hit is False
        # Should attempt to delete the orphaned Qdrant point
        mock_qdrant_client.delete.assert_called_once()

    def test_miss_redis_expired_cleanup_fails(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding
    ) -> None:
        """Qdrant hit, Redis expired, and orphan cleanup also fails — still returns miss."""
        point = MagicMock()
        point.payload = {"cache_key": "qcache:orphan"}
        point.score = 0.97
        point.id = "pt-orphan"
        mock_qdrant_client.search.return_value = [point]
        mock_redis_client.get.return_value = None
        mock_qdrant_client.delete.side_effect = RuntimeError("delete failed")

        result = cache.get(sample_embedding)
        assert result.hit is False

    def test_qdrant_error_graceful(self, cache, mock_qdrant_client, sample_embedding) -> None:
        mock_qdrant_client.search.side_effect = RuntimeError("connection refused")
        result = cache.get(sample_embedding)
        assert result.hit is False

    def test_redis_error_graceful(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding
    ) -> None:
        point = MagicMock()
        point.payload = {"cache_key": "qcache:redis-err"}
        point.score = 0.93
        point.id = "pt-re"
        mock_qdrant_client.search.return_value = [point]
        mock_redis_client.get.side_effect = RuntimeError("redis timeout")

        result = cache.get(sample_embedding)
        assert result.hit is False

    def test_returns_miss_when_not_available(self, disabled_settings, sample_embedding) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(disabled_settings)
        result = cache.get(sample_embedding)
        assert result.hit is False

    def test_empty_payload(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding
    ) -> None:
        """Point with empty/None payload returns miss (no cache_key)."""
        point = MagicMock()
        point.payload = None
        point.score = 0.99
        point.id = "pt-no-payload"
        mock_qdrant_client.search.return_value = [point]
        mock_redis_client.get.return_value = None  # empty cache_key

        result = cache.get(sample_embedding)
        assert result.hit is False

    def test_search_called_with_correct_params(
        self, cache, mock_qdrant_client, sample_embedding
    ) -> None:
        cache.get(sample_embedding)
        mock_qdrant_client.search.assert_called_once_with(
            collection_name="query_cache",
            query_vector=sample_embedding,
            limit=1,
            score_threshold=0.92,
        )


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------


class TestSet:
    def test_stores_in_both_backends(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding, sample_response
    ) -> None:
        key = cache.set("What is murder?", sample_embedding, sample_response)

        assert key.startswith("qcache:")
        mock_redis_client.setex.assert_called_once()
        mock_qdrant_client.upsert.assert_called_once()

    def test_returns_cache_key(self, cache, sample_embedding, sample_response) -> None:
        key = cache.set("What is Section 302?", sample_embedding, sample_response)
        assert isinstance(key, str)
        assert key.startswith("qcache:")

    def test_with_acts_cited(
        self, cache, mock_qdrant_client, sample_embedding, sample_response
    ) -> None:
        cache.set(
            "What is murder?",
            sample_embedding,
            sample_response,
            acts_cited=["IPC", "CrPC"],
        )
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        point = points[0]
        assert point.payload["acts_cited"] == ["IPC", "CrPC"]

    def test_without_acts_cited(
        self, cache, mock_qdrant_client, sample_embedding, sample_response
    ) -> None:
        cache.set("What is murder?", sample_embedding, sample_response)
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        point = points[0]
        assert point.payload["acts_cited"] == []

    def test_redis_ttl_matches_settings(
        self, cache, mock_redis_client, sample_embedding, sample_response
    ) -> None:
        cache.set("query", sample_embedding, sample_response)
        call_args = mock_redis_client.setex.call_args
        # setex(name, time, value)
        ttl = call_args[0][1]
        assert ttl == 86400

    def test_redis_stores_json(
        self, cache, mock_redis_client, sample_embedding, sample_response
    ) -> None:
        cache.set("query", sample_embedding, sample_response)
        call_args = mock_redis_client.setex.call_args
        stored_value = call_args[0][2]
        assert json.loads(stored_value) == sample_response

    def test_qdrant_payload_has_required_fields(
        self, cache, mock_qdrant_client, sample_embedding, sample_response
    ) -> None:
        cache.set("murder law India", sample_embedding, sample_response)
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        payload = points[0].payload
        assert "cache_key" in payload
        assert "original_query" in payload
        assert "acts_cited" in payload
        assert "cached_at" in payload
        assert payload["original_query"] == "murder law India"

    def test_redis_error_raises_cache_error(
        self, cache, mock_redis_client, sample_embedding, sample_response
    ) -> None:
        mock_redis_client.setex.side_effect = RuntimeError("redis write failed")
        with pytest.raises(CacheError, match="Failed to store cache entry"):
            cache.set("query", sample_embedding, sample_response)

    def test_qdrant_error_raises_cache_error(
        self, cache, mock_qdrant_client, mock_redis_client, sample_embedding, sample_response
    ) -> None:
        mock_qdrant_client.upsert.side_effect = RuntimeError("qdrant write failed")
        with pytest.raises(CacheError, match="Failed to store cache entry"):
            cache.set("query", sample_embedding, sample_response)

    def test_empty_response_dict(self, cache, sample_embedding) -> None:
        key = cache.set("query", sample_embedding, {})
        assert key.startswith("qcache:")

    def test_very_long_query_text(self, cache, sample_embedding, sample_response) -> None:
        long_query = "a" * 10000
        key = cache.set(long_query, sample_embedding, sample_response)
        assert key.startswith("qcache:")


# ---------------------------------------------------------------------------
# invalidate_for_act()
# ---------------------------------------------------------------------------


class TestInvalidateForAct:
    def test_deletes_matching_entries(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        point1 = MagicMock()
        point1.payload = {"cache_key": "qcache:p1"}
        point1.id = "id-1"
        point2 = MagicMock()
        point2.payload = {"cache_key": "qcache:p2"}
        point2.id = "id-2"

        mock_qdrant_client.scroll.return_value = ([point1, point2], None)

        count = cache.invalidate_for_act("IPC")
        assert count == 2
        assert mock_redis_client.delete.call_count == 2
        # Batch delete: one call with both point IDs
        assert mock_qdrant_client.delete.call_count == 1
        mock_qdrant_client.delete.assert_called_once_with(
            collection_name=cache._settings.cache_collection,
            points_selector=["id-1", "id-2"],
        )

    def test_no_matches_returns_zero(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.scroll.return_value = ([], None)
        count = cache.invalidate_for_act("NonExistent Act")
        assert count == 0

    def test_paginated_scroll(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        """Handles paginated Qdrant scroll results."""
        point1 = MagicMock()
        point1.payload = {"cache_key": "qcache:page1"}
        point1.id = "id-page1"
        point2 = MagicMock()
        point2.payload = {"cache_key": "qcache:page2"}
        point2.id = "id-page2"

        mock_qdrant_client.scroll.side_effect = [
            ([point1], "next-offset"),
            ([point2], None),
        ]

        count = cache.invalidate_for_act("CrPC")
        assert count == 2

    def test_qdrant_error_graceful(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.scroll.side_effect = RuntimeError("scroll failed")
        count = cache.invalidate_for_act("IPC")
        assert count == 0

    def test_returns_zero_when_not_available(self, disabled_settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(disabled_settings)
        count = cache.invalidate_for_act("IPC")
        assert count == 0

    def test_point_without_cache_key(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        """Point with empty cache_key — still deletes from Qdrant, skip Redis."""
        point = MagicMock()
        point.payload = {"cache_key": ""}
        point.id = "id-nocachekey"
        mock_qdrant_client.scroll.return_value = ([point], None)

        count = cache.invalidate_for_act("IPC")
        assert count == 1
        # Redis delete not called for empty key
        mock_redis_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# invalidate_by_key()
# ---------------------------------------------------------------------------


class TestInvalidateByKey:
    def test_deletes_specific_entry(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        point = MagicMock()
        point.id = "pt-specific"
        mock_qdrant_client.scroll.return_value = ([point], None)

        result = cache.invalidate_by_key("qcache:abc-123")
        assert result is True
        mock_redis_client.delete.assert_called_once_with("qcache:abc-123")
        mock_qdrant_client.delete.assert_called_once()

    def test_not_found_returns_false(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.scroll.return_value = ([], None)
        result = cache.invalidate_by_key("qcache:nonexistent")
        assert result is False

    def test_qdrant_error_graceful(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.scroll.side_effect = RuntimeError("scroll failed")
        result = cache.invalidate_by_key("qcache:err")
        assert result is False

    def test_returns_false_when_not_available(self, disabled_settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(disabled_settings)
        result = cache.invalidate_by_key("qcache:x")
        assert result is False

    def test_redis_delete_called_before_qdrant_scroll(
        self, cache, mock_qdrant_client, mock_redis_client
    ) -> None:
        """Redis key is deleted even if Qdrant scroll returns nothing."""
        mock_qdrant_client.scroll.return_value = ([], None)
        cache.invalidate_by_key("qcache:redis-only")
        mock_redis_client.delete.assert_called_once_with("qcache:redis-only")


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    def test_deletes_all_entries(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        point1 = MagicMock()
        point1.payload = {"cache_key": "qcache:c1"}
        point1.id = "id-c1"
        point2 = MagicMock()
        point2.payload = {"cache_key": "qcache:c2"}
        point2.id = "id-c2"
        mock_qdrant_client.scroll.return_value = ([point1, point2], None)

        count = cache.clear()
        assert count == 2
        assert mock_redis_client.delete.call_count == 2

    def test_empty_cache_returns_zero(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.scroll.return_value = ([], None)
        count = cache.clear()
        assert count == 0

    def test_paginated_clear(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        point1 = MagicMock()
        point1.payload = {"cache_key": "qcache:p1"}
        point1.id = "id-p1"
        point2 = MagicMock()
        point2.payload = {"cache_key": "qcache:p2"}
        point2.id = "id-p2"

        mock_qdrant_client.scroll.side_effect = [
            ([point1], "next"),
            ([point2], None),
        ]

        count = cache.clear()
        assert count == 2

    def test_qdrant_error_graceful(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.scroll.side_effect = RuntimeError("scroll failed")
        count = cache.clear()
        assert count == 0

    def test_returns_zero_when_not_available(self, disabled_settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(disabled_settings)
        count = cache.clear()
        assert count == 0

    def test_batch_delete_qdrant(self, cache, mock_qdrant_client, mock_redis_client) -> None:
        """Verifies that Qdrant delete is called with all point IDs."""
        point1 = MagicMock()
        point1.payload = {"cache_key": "qcache:b1"}
        point1.id = "id-b1"
        point2 = MagicMock()
        point2.payload = {"cache_key": "qcache:b2"}
        point2.id = "id-b2"
        mock_qdrant_client.scroll.return_value = ([point1, point2], None)

        cache.clear()
        # Should be called once with both IDs
        mock_qdrant_client.delete.assert_called_once_with(
            collection_name="query_cache",
            points_selector=["id-b1", "id-b2"],
        )


# ---------------------------------------------------------------------------
# _ensure_clients()
# ---------------------------------------------------------------------------


class TestEnsureClients:
    def test_lazy_init_creates_both_clients(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        assert cache._qdrant is None
        assert cache._redis is None

        # _ensure_clients will use the mocked modules from autouse fixture
        cache._ensure_clients()
        assert cache._qdrant is not None
        assert cache._redis is not None

    def test_skips_if_already_initialized(
        self, cache, mock_qdrant_client, mock_redis_client
    ) -> None:
        """Does not re-create clients if already set."""
        cache._ensure_clients()
        # Clients should remain the same objects
        assert cache._qdrant is mock_qdrant_client
        assert cache._redis is mock_redis_client

    def test_raises_cache_error_without_qdrant(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        with (
            patch.dict(sys.modules, {"qdrant_client": None}),
            pytest.raises(CacheError, match="qdrant-client"),
        ):
            cache._ensure_clients()

    def test_raises_cache_error_without_redis(self, settings) -> None:
        from src.query._cache import SemanticQueryCache

        cache = SemanticQueryCache(settings)
        with (
            patch.dict(sys.modules, {"redis": None}),
            pytest.raises(CacheError, match="redis"),
        ):
            cache._ensure_clients()


# ---------------------------------------------------------------------------
# _ensure_collection()
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    def test_creates_collection_if_not_exists(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.collection_exists.return_value = False
        cache._ensure_collection()
        mock_qdrant_client.create_collection.assert_called_once()

    def test_skips_if_collection_exists(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.collection_exists.return_value = True
        cache._ensure_collection()
        mock_qdrant_client.create_collection.assert_not_called()

    def test_collection_name_from_settings(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.collection_exists.return_value = False
        cache._ensure_collection()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs.get("collection_name") == "query_cache"

    def test_on_disk_payload_enabled(self, cache, mock_qdrant_client) -> None:
        mock_qdrant_client.collection_exists.return_value = False
        cache._ensure_collection()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs.get("on_disk_payload") is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_embedding_get(self, cache) -> None:
        result = cache.get([])
        # Should work (Qdrant will just return no results)
        assert isinstance(result, CacheResult)

    def test_empty_embedding_set(self, cache) -> None:
        key = cache.set("query", [], {"answer": "x"})
        assert key.startswith("qcache:")

    def test_large_response_set(self, cache, sample_embedding) -> None:
        large_resp = {"answer": "a" * 100000, "sources": list(range(1000))}
        key = cache.set("query", sample_embedding, large_resp)
        assert key.startswith("qcache:")

    def test_special_characters_in_query(self, cache, sample_embedding, sample_response) -> None:
        query = 'Section 302 "IPC" — murder vs §culpable homicide'
        key = cache.set(query, sample_embedding, sample_response)
        assert key.startswith("qcache:")

    def test_cache_result_model_defaults(self) -> None:
        result = CacheResult()
        assert result.hit is False
        assert result.response is None
        assert result.similarity == 0.0
        assert result.cache_key is None

    def test_get_returns_cache_result_type(self, cache, sample_embedding) -> None:
        result = cache.get(sample_embedding)
        assert isinstance(result, CacheResult)

    def test_set_unique_keys(self, cache, sample_embedding, sample_response) -> None:
        """Each set() call produces a unique cache key."""
        key1 = cache.set("query1", sample_embedding, sample_response)
        key2 = cache.set("query2", sample_embedding, sample_response)
        assert key1 != key2

    def test_none_acts_cited_becomes_empty_list(
        self, cache, mock_qdrant_client, sample_embedding, sample_response
    ) -> None:
        cache.set("query", sample_embedding, sample_response, acts_cited=None)
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args[1].get("points")
        assert points[0].payload["acts_cited"] == []

    def test_custom_threshold_setting(self, mock_qdrant_client, mock_redis_client) -> None:
        from src.query._cache import SemanticQueryCache

        custom = QuerySettings(cache_similarity_threshold=0.80)
        cache = SemanticQueryCache(custom)
        cache._qdrant = mock_qdrant_client
        cache._redis = mock_redis_client

        cache.get([0.1] * 768)
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("score_threshold") == 0.80

    def test_custom_ttl_setting(self, mock_qdrant_client, mock_redis_client) -> None:
        from src.query._cache import SemanticQueryCache

        custom = QuerySettings(cache_ttl_seconds=7200)
        cache = SemanticQueryCache(custom)
        cache._qdrant = mock_qdrant_client
        cache._redis = mock_redis_client

        cache.set("q", [0.1] * 768, {"a": 1})
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == 7200

    def test_custom_collection_name(self, mock_qdrant_client, mock_redis_client) -> None:
        from src.query._cache import SemanticQueryCache

        custom = QuerySettings(cache_collection="my_cache")
        cache = SemanticQueryCache(custom)
        cache._qdrant = mock_qdrant_client
        cache._redis = mock_redis_client

        cache.get([0.1] * 768)
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs.get("collection_name") == "my_cache"

    def test_custom_redis_prefix(self, mock_qdrant_client, mock_redis_client) -> None:
        from src.query._cache import SemanticQueryCache

        custom = QuerySettings(cache_redis_prefix="custom:")
        cache = SemanticQueryCache(custom)
        cache._qdrant = mock_qdrant_client
        cache._redis = mock_redis_client

        key = cache.set("q", [0.1] * 768, {"a": 1})
        assert key.startswith("custom:")
