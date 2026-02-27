"""Tests for QueryIntelligenceLayer orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from src.query._models import (
    CacheResult,
    HyDEResult,
    QueryIntelligenceResult,
    QuerySettings,
)
from src.query.pipeline import QueryIntelligenceLayer
from src.retrieval._models import QueryRoute

# --- Fixtures ---


@pytest.fixture
def settings() -> QuerySettings:
    return QuerySettings()


@pytest.fixture
def layer(settings: QuerySettings) -> QueryIntelligenceLayer:
    return QueryIntelligenceLayer(settings)


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Embedder that returns deterministic embeddings."""
    embedder = MagicMock()
    embedder.embed_texts.return_value = [np.ones(768, dtype=np.float32)]
    embedder.matryoshka_slice.return_value = np.ones(64, dtype=np.float32)
    return embedder


# --- process() tests ---


class TestProcess:
    async def test_basic_process_no_embedder(self, layer: QueryIntelligenceLayer) -> None:
        """Without an embedder, still classifies the route."""
        qi_result, _rq = await layer.process("What is the punishment for cheating?")
        assert isinstance(qi_result, QueryIntelligenceResult)
        assert qi_result.query_text == "What is the punishment for cheating?"
        assert qi_result.cache_hit is False
        assert "embed_ms" in qi_result.timings
        assert "cache_ms" in qi_result.timings
        assert "route_ms" in qi_result.timings

    async def test_process_with_embedder(
        self, layer: QueryIntelligenceLayer, mock_embedder: MagicMock
    ) -> None:
        _qi_result, rq = await layer.process("What is Section 302?", embedder=mock_embedder)
        assert rq.query_embedding is not None
        assert len(rq.query_embedding) == 768
        assert rq.text == "What is Section 302?"

    async def test_route_classification_simple(
        self, layer: QueryIntelligenceLayer, mock_embedder: MagicMock
    ) -> None:
        qi_result, rq = await layer.process(
            "What does Section 420 IPC say?", embedder=mock_embedder
        )
        assert qi_result.route == QueryRoute.SIMPLE
        assert rq.route == QueryRoute.SIMPLE

    async def test_route_classification_standard(
        self, layer: QueryIntelligenceLayer, mock_embedder: MagicMock
    ) -> None:
        qi_result, _rq = await layer.process(
            "Can a landlord evict a tenant for personal use?", embedder=mock_embedder
        )
        assert qi_result.route == QueryRoute.STANDARD

    async def test_cache_hit_returns_early(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        layer = QueryIntelligenceLayer(settings)
        cached_response = {"answer": "Section 302 deals with murder"}

        from src.query._cache import SemanticQueryCache

        with (
            patch.object(
                layer._cache,
                "get",
                return_value=CacheResult(
                    hit=True,
                    response=cached_response,
                    similarity=0.95,
                    cache_key="cache:abc",
                ),
            ),
            patch.object(
                SemanticQueryCache, "is_available", new_callable=PropertyMock, return_value=True
            ),
        ):
            qi_result, _rq = await layer.process("What is Section 302?", embedder=mock_embedder)

        assert qi_result.cache_hit is True
        assert qi_result.cache_response == cached_response
        # Route and HyDE should not have run (no route_ms, no hyde_ms)
        assert "route_ms" not in qi_result.timings
        assert "hyde_ms" not in qi_result.timings

    async def test_hyde_generated_for_complex(self, settings: QuerySettings) -> None:
        layer = QueryIntelligenceLayer(settings)
        mock_emb = MagicMock()
        mock_emb.embed_texts.return_value = [np.ones(768, dtype=np.float32)]
        mock_emb.matryoshka_slice.return_value = np.ones(64, dtype=np.float32)

        hyde_result = HyDEResult(
            hypothetical_text="Under the DV Act and IPC...",
            hyde_embedding=[0.5] * 768,
            generated=True,
        )
        with patch.object(layer._hyde, "maybe_generate", return_value=hyde_result):
            qi_result, rq = await layer.process(
                "What is the interplay between Section 498A IPC and the DV Act?",
                embedder=mock_emb,
            )

        assert qi_result.hyde_generated is True
        # HyDE embedding should replace the query embedding
        assert rq.query_embedding == [0.5] * 768
        assert rq.hyde_text == "Under the DV Act and IPC..."

    async def test_hyde_not_generated_for_standard(
        self, layer: QueryIntelligenceLayer, mock_embedder: MagicMock
    ) -> None:
        qi_result, rq = await layer.process(
            "What is the punishment for cheating?", embedder=mock_embedder
        )
        assert qi_result.hyde_generated is False
        assert rq.hyde_text is None

    async def test_router_error_falls_back_to_standard(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        layer = QueryIntelligenceLayer(settings)
        with patch.object(layer._router, "classify", side_effect=RuntimeError("router crashed")):
            qi_result, _rq = await layer.process("test query", embedder=mock_embedder)

        assert qi_result.route == QueryRoute.STANDARD
        assert len(qi_result.errors) > 0
        assert "Router failed" in qi_result.errors[0]

    async def test_timings_populated(
        self, layer: QueryIntelligenceLayer, mock_embedder: MagicMock
    ) -> None:
        qi_result, _ = await layer.process("test query", embedder=mock_embedder)
        assert "embed_ms" in qi_result.timings
        assert "cache_ms" in qi_result.timings
        assert "route_ms" in qi_result.timings
        assert "hyde_ms" in qi_result.timings
        assert all(v >= 0 for v in qi_result.timings.values())


# --- store_response() tests ---


class TestStoreResponse:
    async def test_store_when_cache_available(self, settings: QuerySettings) -> None:
        from src.query._cache import SemanticQueryCache

        layer = QueryIntelligenceLayer(settings)
        with (
            patch.object(
                SemanticQueryCache, "is_available", new_callable=PropertyMock, return_value=True
            ),
            patch.object(layer._cache, "set", return_value="cache:123"),
        ):
            key = await layer.store_response(
                query_text="test",
                query_embedding=[0.1] * 768,
                response={"answer": "test answer"},
                acts_cited=["IPC"],
            )
        assert key == "cache:123"

    async def test_store_returns_none_without_embedding(
        self, layer: QueryIntelligenceLayer
    ) -> None:
        key = await layer.store_response(
            query_text="test",
            query_embedding=None,
            response={"answer": "test"},
        )
        assert key is None

    async def test_store_returns_none_when_unavailable(self, settings: QuerySettings) -> None:
        from src.query._cache import SemanticQueryCache

        layer = QueryIntelligenceLayer(settings)
        with patch.object(
            SemanticQueryCache, "is_available", new_callable=PropertyMock, return_value=False
        ):
            key = await layer.store_response(
                query_text="test",
                query_embedding=[0.1] * 768,
                response={"answer": "test"},
            )
        assert key is None

    async def test_store_error_graceful(self, settings: QuerySettings) -> None:
        from src.query._cache import SemanticQueryCache

        layer = QueryIntelligenceLayer(settings)
        with (
            patch.object(
                SemanticQueryCache, "is_available", new_callable=PropertyMock, return_value=True
            ),
            patch.object(layer._cache, "set", side_effect=RuntimeError("Redis down")),
        ):
            key = await layer.store_response(
                query_text="test",
                query_embedding=[0.1] * 768,
                response={"answer": "test"},
            )
        assert key is None


# --- invalidate_for_act() tests ---


class TestInvalidateForAct:
    async def test_invalidation_delegates(self, settings: QuerySettings) -> None:
        layer = QueryIntelligenceLayer(settings)
        with patch.object(layer._cache, "invalidate_for_act", return_value=3):
            count = await layer.invalidate_for_act("Indian Penal Code")
        assert count == 3

    async def test_invalidation_error_returns_zero(self, settings: QuerySettings) -> None:
        layer = QueryIntelligenceLayer(settings)
        with patch.object(layer._cache, "invalidate_for_act", side_effect=RuntimeError("error")):
            count = await layer.invalidate_for_act("IPC")
        assert count == 0


# --- from_config() tests ---


class TestFromConfig:
    def test_from_default_config(self) -> None:
        layer = QueryIntelligenceLayer.from_config()
        assert layer._settings.cache_enabled is True

    def test_from_missing_config_uses_defaults(self) -> None:
        layer = QueryIntelligenceLayer.from_config("/nonexistent/path.yaml")
        assert layer._settings.cache_enabled is True

    def test_from_custom_config(self, tmp_path: object) -> None:
        from pathlib import Path

        config_file = Path(str(tmp_path)) / "custom.yaml"
        config_file.write_text(
            "settings:\n  cache_enabled: false\n  hyde_enabled: false\n",
            encoding="utf-8",
        )
        layer = QueryIntelligenceLayer.from_config(str(config_file))
        assert layer._settings.cache_enabled is False
        assert layer._settings.hyde_enabled is False


# --- _embed_query helper ---


class TestEmbedQuery:
    def test_returns_embedding(
        self, layer: QueryIntelligenceLayer, mock_embedder: MagicMock
    ) -> None:
        result = layer._embed_query("test query", mock_embedder)
        assert result is not None
        assert len(result) == 768

    def test_returns_none_without_embedder(self, layer: QueryIntelligenceLayer) -> None:
        result = layer._embed_query("test query", None)
        assert result is None

    def test_returns_none_on_error(self, layer: QueryIntelligenceLayer) -> None:
        bad_emb = MagicMock()
        bad_emb.embed_texts.side_effect = RuntimeError("OOM")
        result = layer._embed_query("test query", bad_emb)
        assert result is None


# --- _matryoshka_slice helper ---


class TestMatryoshkaSlice:
    def test_slices_embedding(self, mock_embedder: MagicMock) -> None:
        full = [1.0] * 768
        result = QueryIntelligenceLayer._matryoshka_slice(full, mock_embedder)
        assert result is not None
        assert len(result) == 64

    def test_returns_none_without_embedding(self, mock_embedder: MagicMock) -> None:
        result = QueryIntelligenceLayer._matryoshka_slice(None, mock_embedder)
        assert result is None

    def test_returns_none_without_embedder(self) -> None:
        result = QueryIntelligenceLayer._matryoshka_slice([1.0] * 768, None)
        assert result is None

    def test_fallback_truncation(self) -> None:
        """If matryoshka_slice method fails, falls back to simple [:64]."""
        bad_emb = MagicMock()
        bad_emb.matryoshka_slice.side_effect = RuntimeError("no method")
        full = list(range(768))
        result = QueryIntelligenceLayer._matryoshka_slice(full, bad_emb)
        assert result is not None
        assert len(result) == 64
        assert result == list(range(64))
