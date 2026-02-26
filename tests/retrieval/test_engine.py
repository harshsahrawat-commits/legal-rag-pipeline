"""Tests for RetrievalEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval._engine import RetrievalEngine
from src.retrieval._models import (
    FusedChunk,
    QueryRoute,
    RetrievalQuery,
    RetrievalSettings,
    ScoredChunk,
)


@pytest.fixture()
def settings() -> RetrievalSettings:
    return RetrievalSettings()


@pytest.fixture()
def engine(settings: RetrievalSettings) -> RetrievalEngine:
    return RetrievalEngine(settings)


def _make_scored(chunk_id: str, channel: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=score,
        channel=channel,
        payload={"parent_info": {"parent_chunk_id": None, "judgment_header_chunk_id": None}},
    )


def _make_fused(chunk_id: str, rrf_score: float = 0.03) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        rrf_score=rrf_score,
        channels=["dense"],
        payload={"parent_info": {"parent_chunk_id": None, "judgment_header_chunk_id": None}},
    )


class TestRetrieveStandardRoute:
    """STANDARD route: dense + BM25 + QuIM + fusion + rerank + expand."""

    @pytest.mark.asyncio
    async def test_full_standard_flow(self, engine: RetrievalEngine) -> None:
        """Engine calls all channels, fuses, reranks, expands."""
        query = RetrievalQuery(
            text="Section 420 IPC",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1, 5],
            sparse_values=[0.5, 0.3],
            route=QueryRoute.STANDARD,
        )

        dense_results = [_make_scored("c1", "dense", 0.95)]
        bm25_results = [_make_scored("c2", "bm25", 3.2)]
        quim_results = [_make_scored("c3", "quim", 0.80)]
        fused_chunks = [_make_fused("c1"), _make_fused("c2")]

        engine._dense.search = MagicMock(return_value=dense_results)
        engine._sparse.search = MagicMock(return_value=bm25_results)
        engine._quim.search = MagicMock(return_value=quim_results)
        engine._fusion.fuse = MagicMock(return_value=fused_chunks)
        engine._reranker._model = None  # not loaded → skip reranking
        engine._expander.expand = AsyncMock(
            return_value=[
                MagicMock(chunk_id="c1", chunk_text="text", total_tokens=50),
                MagicMock(chunk_id="c2", chunk_text="text", total_tokens=40),
            ]
        )

        result = await engine.retrieve(query)

        assert result.route == QueryRoute.STANDARD
        assert "dense" in result.search_channels_used
        assert "bm25" in result.search_channels_used
        assert "quim" in result.search_channels_used
        assert "graph" not in result.search_channels_used
        assert len(result.chunks) == 2
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_channel_error_isolated(self, engine: RetrievalEngine) -> None:
        """If one channel fails, others still return results."""
        query = RetrievalQuery(
            text="test",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1],
            sparse_values=[0.5],
        )

        engine._dense.search = MagicMock(side_effect=Exception("Qdrant down"))
        engine._sparse.search = MagicMock(return_value=[_make_scored("c1", "bm25")])
        engine._quim.search = MagicMock(return_value=[])
        engine._fusion.fuse = MagicMock(return_value=[_make_fused("c1")])
        engine._reranker._model = None
        engine._expander.expand = AsyncMock(
            return_value=[
                MagicMock(chunk_id="c1", chunk_text="text", total_tokens=50),
            ]
        )

        result = await engine.retrieve(query)

        assert len(result.errors) >= 1
        assert "Dense search failed" in result.errors[0]
        assert len(result.chunks) == 1  # BM25 still worked

    @pytest.mark.asyncio
    async def test_timings_populated(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="test",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
        )
        engine._dense.search = MagicMock(return_value=[])
        engine._quim.search = MagicMock(return_value=[])
        engine._fusion.fuse = MagicMock(return_value=[])
        engine._reranker._model = None
        engine._expander.expand = AsyncMock(return_value=[])

        result = await engine.retrieve(query)

        assert "embed_ms" in result.timings
        assert "dense_ms" in result.timings
        assert "fusion_ms" in result.timings


class TestRetrieveSimpleRoute:
    """SIMPLE route: KG direct query only."""

    @pytest.mark.asyncio
    async def test_simple_route_uses_kg(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="What is Section 420 of Indian Penal Code?",
            route=QueryRoute.SIMPLE,
        )

        with patch.object(engine, "_kg_direct_query", new_callable=AsyncMock) as mock_kg:
            mock_kg.return_value = {"found": True, "is_in_force": False}
            result = await engine.retrieve(query)

        assert result.route == QueryRoute.SIMPLE
        assert result.kg_direct_answer is not None
        assert result.kg_direct_answer["found"] is True
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_simple_route_no_vector_search(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(text="Section 420 IPC", route=QueryRoute.SIMPLE)
        engine._dense.search = MagicMock()

        with patch.object(engine, "_kg_direct_query", new_callable=AsyncMock) as mock_kg:
            mock_kg.return_value = None
            await engine.retrieve(query)

        engine._dense.search.assert_not_called()


class TestRetrieveComplexRoute:
    """COMPLEX route: includes graph search."""

    @pytest.mark.asyncio
    async def test_complex_includes_graph(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="Section 302 IPC",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            route=QueryRoute.COMPLEX,
        )

        engine._dense.search = MagicMock(return_value=[])
        engine._quim.search = MagicMock(return_value=[])
        engine._graph.search = AsyncMock(return_value=[_make_scored("c1", "graph")])
        engine._fusion.fuse = MagicMock(return_value=[_make_fused("c1")])
        engine._reranker._model = None
        engine._expander.expand = AsyncMock(
            return_value=[
                MagicMock(chunk_id="c1", chunk_text="text", total_tokens=50),
            ]
        )

        result = await engine.retrieve(query)

        assert "graph" in result.search_channels_used
        engine._graph.search.assert_called_once()


class TestHybridSearch:
    """hybrid_search for Phase 8 per-claim retrieval."""

    @pytest.mark.asyncio
    async def test_returns_scored_chunks(self, engine: RetrievalEngine) -> None:
        engine._dense.search = MagicMock(return_value=[_make_scored("c1", "dense")])
        engine._sparse.search = MagicMock(return_value=[_make_scored("c2", "bm25")])
        engine._fusion.fuse = MagicMock(
            return_value=[
                _make_fused("c1", 0.032),
                _make_fused("c2", 0.016),
            ]
        )

        # Need embeddings
        engine._embedder = MagicMock()
        import numpy as np

        engine._embedder.embed_texts.return_value = [np.zeros(768)]
        engine._embedder.matryoshka_slice.return_value = np.zeros(64)
        engine._bm25 = MagicMock()
        engine._bm25.encode.return_value = MagicMock(indices=[1], values=[0.5])

        results = await engine.hybrid_search("Section 420 cheating", top_k=5)

        assert len(results) == 2
        assert all(isinstance(r, ScoredChunk) for r in results)
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_models(self, engine: RetrievalEngine) -> None:
        """No embedder loaded → can't search → empty results."""
        results = await engine.hybrid_search("test", top_k=5)
        assert results == []


class TestPrepareQueryVectors:
    """_prepare_query_vectors embedding computation."""

    def test_passthrough_when_provided(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="test",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1],
            sparse_values=[0.5],
        )
        full, fast, idx, val = engine._prepare_query_vectors(query)
        assert full == [0.1] * 768
        assert fast == [0.2] * 64
        assert idx == [1]
        assert val == [0.5]

    def test_computes_when_embedder_available(self, engine: RetrievalEngine) -> None:
        import numpy as np

        engine._embedder = MagicMock()
        engine._embedder.embed_texts.return_value = [np.ones(768)]
        engine._embedder.matryoshka_slice.return_value = np.ones(64) * 0.5

        query = RetrievalQuery(text="test")
        full, fast, _idx, _val = engine._prepare_query_vectors(query)

        assert full is not None
        assert len(full) == 768
        assert fast is not None
        assert len(fast) == 64

    def test_none_when_no_embedder(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(text="test")
        full, fast, idx, val = engine._prepare_query_vectors(query)
        assert full is None
        assert fast is None
        assert idx is None
        assert val is None


class TestLoadModels:
    """load_models loads embedding, reranker, and BM25 vocab."""

    @pytest.mark.slow
    def test_load_models_logs_warnings_on_failure(self, engine: RetrievalEngine) -> None:
        """load_models does not raise even if all loads fail."""
        # All imports will fail since we're in test environment without real models
        engine.load_models()
        # No exception raised — each failure is logged and isolated

    def test_load_models_with_mocked_deps(self, engine: RetrievalEngine) -> None:
        """load_models works when dependencies are mocked."""
        from src.retrieval._exceptions import RerankerNotAvailableError

        engine._reranker.load_model = MagicMock(side_effect=RerankerNotAvailableError("no torch"))
        engine.load_models()
        # No exception — each component failure is isolated


class TestClose:
    """Engine resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_calls_graph_close(self, engine: RetrievalEngine) -> None:
        engine._graph.close = AsyncMock()
        await engine.close()
        engine._graph.close.assert_called_once()
