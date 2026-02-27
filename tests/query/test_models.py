"""Tests for query intelligence Pydantic models."""

from __future__ import annotations

from src.query._models import (
    CacheEntry,
    CacheResult,
    HyDEResult,
    QueryConfig,
    QueryIntelligenceResult,
    QuerySettings,
    RouterResult,
)
from src.retrieval._models import QueryRoute


class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_basic_construction(self) -> None:
        entry = CacheEntry(
            query_text="What is Section 302 IPC?",
            cache_key="cache:abc123",
            response={"answer": "Section 302 deals with murder"},
        )
        assert entry.query_text == "What is Section 302 IPC?"
        assert entry.cache_key == "cache:abc123"
        assert entry.response["answer"] == "Section 302 deals with murder"
        assert entry.acts_cited == []
        assert entry.ttl_seconds == 86400

    def test_with_acts_cited(self) -> None:
        entry = CacheEntry(
            query_text="test",
            cache_key="cache:x",
            response={"answer": "test"},
            acts_cited=["Indian Penal Code", "Code of Criminal Procedure"],
        )
        assert len(entry.acts_cited) == 2

    def test_cached_at_default(self) -> None:
        entry = CacheEntry(
            query_text="test",
            cache_key="cache:x",
            response={},
        )
        assert entry.cached_at is not None
        assert entry.cached_at.tzinfo is not None

    def test_serialization_round_trip(self) -> None:
        entry = CacheEntry(
            query_text="test query",
            cache_key="cache:123",
            response={"key": "value", "nested": {"a": 1}},
            acts_cited=["IPC"],
            ttl_seconds=7200,
        )
        json_str = entry.model_dump_json()
        restored = CacheEntry.model_validate_json(json_str)
        assert restored.query_text == entry.query_text
        assert restored.cache_key == entry.cache_key
        assert restored.response == entry.response
        assert restored.acts_cited == entry.acts_cited
        assert restored.ttl_seconds == 7200


class TestCacheResult:
    """Tests for CacheResult model."""

    def test_default_is_miss(self) -> None:
        result = CacheResult()
        assert result.hit is False
        assert result.response is None
        assert result.similarity == 0.0
        assert result.cache_key is None

    def test_hit_result(self) -> None:
        result = CacheResult(
            hit=True,
            response={"answer": "Section 302 deals with murder"},
            similarity=0.95,
            cache_key="cache:abc",
        )
        assert result.hit is True
        assert result.similarity == 0.95

    def test_miss_result(self) -> None:
        result = CacheResult(hit=False, similarity=0.87)
        assert result.hit is False
        assert result.response is None


class TestRouterResult:
    """Tests for RouterResult model."""

    def test_default(self) -> None:
        result = RouterResult()
        assert result.route == QueryRoute.STANDARD
        assert result.confidence == 0.5
        assert result.signals == []

    def test_simple_route(self) -> None:
        result = RouterResult(
            route=QueryRoute.SIMPLE,
            confidence=1.0,
            signals=["pattern:section_lookup"],
        )
        assert result.route == QueryRoute.SIMPLE
        assert result.confidence == 1.0
        assert "pattern:section_lookup" in result.signals

    def test_analytical_route(self) -> None:
        result = RouterResult(
            route=QueryRoute.ANALYTICAL,
            confidence=0.9,
            signals=["signal:compare", "signal:multi_act"],
        )
        assert result.route == QueryRoute.ANALYTICAL
        assert len(result.signals) == 2

    def test_all_routes(self) -> None:
        for route in QueryRoute:
            result = RouterResult(route=route)
            assert result.route == route


class TestHyDEResult:
    """Tests for HyDEResult model."""

    def test_not_generated(self) -> None:
        result = HyDEResult()
        assert result.generated is False
        assert result.hypothetical_text is None
        assert result.hyde_embedding is None

    def test_generated(self) -> None:
        result = HyDEResult(
            hypothetical_text="Under the Indian Penal Code, Section 302...",
            hyde_embedding=[0.1] * 768,
            generated=True,
        )
        assert result.generated is True
        assert len(result.hyde_embedding) == 768


class TestQueryIntelligenceResult:
    """Tests for QueryIntelligenceResult model."""

    def test_basic(self) -> None:
        result = QueryIntelligenceResult(query_text="test query")
        assert result.query_text == "test query"
        assert result.route == QueryRoute.STANDARD
        assert result.cache_hit is False
        assert result.hyde_generated is False
        assert result.timings == {}
        assert result.errors == []

    def test_cache_hit_result(self) -> None:
        result = QueryIntelligenceResult(
            query_text="test",
            route=QueryRoute.STANDARD,
            cache_hit=True,
            cache_response={"answer": "cached answer"},
            timings={"cache_ms": 5.0},
        )
        assert result.cache_hit is True
        assert result.cache_response is not None

    def test_full_result(self) -> None:
        result = QueryIntelligenceResult(
            query_text="Compare Delhi and Mumbai rent control",
            route=QueryRoute.COMPLEX,
            cache_hit=False,
            hyde_generated=True,
            timings={"embed_ms": 20.0, "cache_ms": 5.0, "route_ms": 1.0, "hyde_ms": 300.0},
            errors=[],
        )
        assert result.route == QueryRoute.COMPLEX
        assert result.hyde_generated is True
        assert len(result.timings) == 4

    def test_with_errors(self) -> None:
        result = QueryIntelligenceResult(
            query_text="test",
            errors=["Cache unavailable", "HyDE timeout"],
        )
        assert len(result.errors) == 2

    def test_serialization(self) -> None:
        result = QueryIntelligenceResult(
            query_text="test",
            route=QueryRoute.ANALYTICAL,
            cache_hit=False,
            hyde_generated=True,
            timings={"total_ms": 350.0},
        )
        json_str = result.model_dump_json()
        restored = QueryIntelligenceResult.model_validate_json(json_str)
        assert restored.route == QueryRoute.ANALYTICAL
        assert restored.hyde_generated is True


class TestQuerySettings:
    """Tests for QuerySettings model."""

    def test_defaults(self) -> None:
        settings = QuerySettings()
        assert settings.cache_enabled is True
        assert settings.cache_similarity_threshold == 0.92
        assert settings.cache_ttl_seconds == 86400
        assert settings.cache_short_ttl_seconds == 3600
        assert settings.cache_collection == "query_cache"
        assert settings.cache_redis_prefix == "qcache:"
        assert settings.router_version == "v1_rule_based"
        assert settings.hyde_enabled is True
        assert settings.hyde_model == "claude-haiku"
        assert settings.hyde_max_tokens == 200
        assert settings.hyde_routes == ["complex", "analytical"]
        assert settings.embedding_model == "BAAI/bge-m3"
        assert settings.embedding_dim == 768
        assert settings.matryoshka_dim == 64
        assert settings.device == "cpu"
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.redis_url == "redis://localhost:6379/0"

    def test_custom_values(self) -> None:
        settings = QuerySettings(
            cache_enabled=False,
            cache_similarity_threshold=0.85,
            hyde_enabled=False,
            device="cuda",
        )
        assert settings.cache_enabled is False
        assert settings.cache_similarity_threshold == 0.85
        assert settings.hyde_enabled is False
        assert settings.device == "cuda"


class TestQueryConfig:
    """Tests for QueryConfig model."""

    def test_default(self) -> None:
        config = QueryConfig()
        assert isinstance(config.settings, QuerySettings)

    def test_with_settings(self) -> None:
        config = QueryConfig(settings=QuerySettings(cache_enabled=False))
        assert config.settings.cache_enabled is False

    def test_from_dict(self) -> None:
        data = {"settings": {"cache_enabled": False, "hyde_enabled": False}}
        config = QueryConfig.model_validate(data)
        assert config.settings.cache_enabled is False
        assert config.settings.hyde_enabled is False
