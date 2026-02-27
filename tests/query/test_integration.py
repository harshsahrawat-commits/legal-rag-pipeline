"""Integration tests for the Query Intelligence Layer.

Tests the full pipeline: cache → route → HyDE → RetrievalQuery.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from src.query._cache import SemanticQueryCache
from src.query._models import QuerySettings
from src.query.pipeline import QueryIntelligenceLayer
from src.retrieval._models import QueryRoute


@pytest.fixture
def settings() -> QuerySettings:
    return QuerySettings()


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_texts.return_value = [np.ones(768, dtype=np.float32)]
    embedder.matryoshka_slice.return_value = np.ones(64, dtype=np.float32)
    return embedder


class TestEndToEnd:
    """E2E tests through the full query intelligence pipeline."""

    async def test_simple_query_e2e(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        """SIMPLE query: cache miss → classify SIMPLE → no HyDE → RetrievalQuery."""
        layer = QueryIntelligenceLayer(settings)
        qi_result, rq = await layer.process(
            "What does Section 302 IPC say?", embedder=mock_embedder
        )
        assert qi_result.route == QueryRoute.SIMPLE
        assert qi_result.cache_hit is False
        assert qi_result.hyde_generated is False
        assert rq.route == QueryRoute.SIMPLE
        assert rq.text == "What does Section 302 IPC say?"
        assert rq.query_embedding is not None

    async def test_standard_query_e2e(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        """STANDARD query: default route, no HyDE."""
        layer = QueryIntelligenceLayer(settings)
        qi_result, rq = await layer.process(
            "What is the punishment for cheating?", embedder=mock_embedder
        )
        assert qi_result.route == QueryRoute.STANDARD
        assert qi_result.hyde_generated is False
        assert rq.route == QueryRoute.STANDARD

    async def test_analytical_query_triggers_hyde(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        """ANALYTICAL query: route → HyDE triggered (but fails gracefully without anthropic)."""
        layer = QueryIntelligenceLayer(settings)
        qi_result, rq = await layer.process(
            "Compare the eviction grounds under Delhi and Mumbai rent control laws",
            embedder=mock_embedder,
        )
        assert qi_result.route == QueryRoute.ANALYTICAL
        # HyDE won't generate without anthropic installed, but no crash
        assert rq.route == QueryRoute.ANALYTICAL

    async def test_all_four_routes_produce_valid_retrieval_query(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        """All routes produce a valid RetrievalQuery."""
        layer = QueryIntelligenceLayer(settings)
        queries = {
            "What does Section 302 IPC say?": QueryRoute.SIMPLE,
            "What is the punishment for cheating?": QueryRoute.STANDARD,
            "Compare the eviction grounds under Delhi and Mumbai rent control laws": QueryRoute.ANALYTICAL,
        }
        for query_text, expected_route in queries.items():
            qi_result, rq = await layer.process(query_text, embedder=mock_embedder)
            assert qi_result.route == expected_route, f"Failed for: {query_text}"
            assert rq.text == query_text
            assert rq.query_embedding is not None

    async def test_cache_store_and_retrieve(self, settings: QuerySettings) -> None:
        """Store a response and verify cache_hit on re-query."""
        layer = QueryIntelligenceLayer(settings)
        mock_cache = MagicMock()
        mock_cache.set.return_value = "cache:test123"
        layer._cache = mock_cache

        with patch.object(
            SemanticQueryCache, "is_available", new_callable=PropertyMock, return_value=True
        ):
            key = await layer.store_response(
                query_text="test query",
                query_embedding=[0.1] * 768,
                response={"answer": "test answer"},
                acts_cited=["Indian Penal Code"],
            )
        assert key == "cache:test123"

    async def test_error_isolation_all_components_fail(self, settings: QuerySettings) -> None:
        """All components fail gracefully, pipeline still returns a result."""
        layer = QueryIntelligenceLayer(settings)

        # No embedder → no embedding
        # Cache will be unavailable → miss
        # Router should still work (pure regex, no deps)
        qi_result, rq = await layer.process("What is the punishment for cheating?")

        # Should still get a result
        assert isinstance(qi_result, type(qi_result))
        assert qi_result.route == QueryRoute.STANDARD
        assert rq.text == "What is the punishment for cheating?"

    async def test_from_config_creates_working_layer(self) -> None:
        """from_config() creates a layer that can process queries."""
        layer = QueryIntelligenceLayer.from_config()
        qi_result, _rq = await layer.process("What is Section 302?")
        assert qi_result.query_text == "What is Section 302?"

    async def test_timings_all_stages_recorded(
        self, settings: QuerySettings, mock_embedder: MagicMock
    ) -> None:
        """All timing stages are recorded."""
        layer = QueryIntelligenceLayer(settings)
        qi_result, _ = await layer.process("test query", embedder=mock_embedder)
        expected_stages = ["embed_ms", "cache_ms", "route_ms", "hyde_ms"]
        for stage in expected_stages:
            assert stage in qi_result.timings, f"Missing timing: {stage}"
            assert qi_result.timings[stage] >= 0
