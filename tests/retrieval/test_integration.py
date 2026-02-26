"""Integration tests for the retrieval module.

Tests the full pipeline end-to-end with all components mocked but wired together,
validating that data flows correctly through search → fusion → rerank → expand → FLARE.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval._engine import RetrievalEngine
from src.retrieval._models import (
    ExpandedContext,
    FusedChunk,
    QueryRoute,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalSettings,
    ScoredChunk,
)
from src.retrieval.pipeline import RetrievalPipeline

# --- Helpers ---


def _make_scored(chunk_id: str, channel: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=score,
        channel=channel,
        payload={
            "id": chunk_id,
            "text": f"Text for {chunk_id}",
            "parent_info": {
                "parent_chunk_id": None,
                "judgment_header_chunk_id": None,
            },
        },
    )


def _make_fused(chunk_id: str, rrf_score: float = 0.03) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        rrf_score=rrf_score,
        channels=["dense", "bm25"],
        payload={
            "id": chunk_id,
            "parent_info": {
                "parent_chunk_id": None,
                "judgment_header_chunk_id": None,
            },
        },
    )


def _make_expanded(chunk_id: str) -> ExpandedContext:
    return ExpandedContext(
        chunk_id=chunk_id,
        chunk_text=f"Expanded text for {chunk_id}",
        relevance_score=0.9,
        total_tokens=20,
        metadata={"id": chunk_id},
    )


def _setup_engine_mocks(engine: RetrievalEngine) -> None:
    """Configure all engine components with reasonable mock defaults."""
    # Search channels
    engine._dense.search = MagicMock(
        return_value=[
            _make_scored("c1", "dense", 0.95),
            _make_scored("c2", "dense", 0.88),
        ]
    )
    engine._sparse.search = MagicMock(
        return_value=[
            _make_scored("c1", "bm25", 3.2),
            _make_scored("c3", "bm25", 2.1),
        ]
    )
    engine._quim.search = MagicMock(return_value=[_make_scored("c4", "quim", 0.80)])
    engine._graph.search = AsyncMock(return_value=[_make_scored("c5", "graph", 1.0)])

    # Fusion — overlapping c1 from dense+bm25
    engine._fusion.fuse = MagicMock(
        return_value=[
            _make_fused("c1", 0.032),
            _make_fused("c2", 0.016),
            _make_fused("c3", 0.015),
        ]
    )

    # Reranker — not loaded (skip)
    engine._reranker._model = None

    # Expander
    engine._expander.expand = AsyncMock(
        return_value=[_make_expanded("c1"), _make_expanded("c2"), _make_expanded("c3")]
    )

    # FLARE — not available by default
    engine._flare._settings.flare_enabled = False


@pytest.fixture()
def settings() -> RetrievalSettings:
    return RetrievalSettings()


@pytest.fixture()
def engine(settings: RetrievalSettings) -> RetrievalEngine:
    e = RetrievalEngine(settings)
    _setup_engine_mocks(e)
    return e


# ===================================================================
# E2E route tests
# ===================================================================


class TestSimpleRouteE2E:
    """SIMPLE route: KG direct query only, no vector search."""

    @pytest.mark.asyncio
    async def test_simple_returns_kg_answer(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="What is Section 420 of Indian Penal Code?",
            route=QueryRoute.SIMPLE,
        )

        with patch.object(engine, "_kg_direct_query", new_callable=AsyncMock) as mock_kg:
            mock_kg.return_value = {
                "section": "420",
                "is_in_force": False,
                "replaced_by": "BNS 420",
            }
            result = await engine.retrieve(query)

        assert result.route == QueryRoute.SIMPLE
        assert result.kg_direct_answer is not None
        assert result.kg_direct_answer["section"] == "420"
        assert result.chunks == []
        assert result.finished_at is not None
        # No vector search was called
        engine._dense.search.assert_not_called()
        engine._sparse.search.assert_not_called()
        engine._quim.search.assert_not_called()


class TestStandardRouteE2E:
    """STANDARD route: dense + BM25 + QuIM + fusion + expand."""

    @pytest.mark.asyncio
    async def test_standard_flow(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="What is cheating under Section 420?",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1, 5, 10],
            sparse_values=[0.5, 0.3, 0.2],
            route=QueryRoute.STANDARD,
        )

        result = await engine.retrieve(query)

        assert result.route == QueryRoute.STANDARD
        assert "dense" in result.search_channels_used
        assert "bm25" in result.search_channels_used
        assert "quim" in result.search_channels_used
        assert "graph" not in result.search_channels_used
        assert len(result.chunks) == 3
        assert result.total_context_tokens == 60  # 20 * 3
        assert result.finished_at is not None
        assert result.elapsed_ms > 0
        # Graph search NOT called for STANDARD
        engine._graph.search.assert_not_called()


class TestComplexRouteE2E:
    """COMPLEX route: all channels including graph."""

    @pytest.mark.asyncio
    async def test_complex_includes_graph(self, engine: RetrievalEngine) -> None:
        query = RetrievalQuery(
            text="Section 302 of Indian Penal Code",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1],
            sparse_values=[0.5],
            route=QueryRoute.COMPLEX,
        )

        result = await engine.retrieve(query)

        assert result.route == QueryRoute.COMPLEX
        assert "graph" in result.search_channels_used
        engine._graph.search.assert_called_once()


class TestAnalyticalRouteE2E:
    """ANALYTICAL route: all channels + FLARE."""

    @pytest.mark.asyncio
    async def test_analytical_with_flare_disabled(self, engine: RetrievalEngine) -> None:
        """ANALYTICAL without FLARE runs like COMPLEX."""
        query = RetrievalQuery(
            text="Analyze the evolution of Section 302 IPC jurisprudence",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1],
            sparse_values=[0.5],
            route=QueryRoute.ANALYTICAL,
        )

        result = await engine.retrieve(query)

        assert result.route == QueryRoute.ANALYTICAL
        assert "graph" in result.search_channels_used
        assert result.flare_retrievals == 0

    @pytest.mark.asyncio
    async def test_analytical_with_flare_enabled(self, engine: RetrievalEngine) -> None:
        """ANALYTICAL with FLARE triggers active re-retrieval."""
        engine._flare._settings.flare_enabled = True
        engine._flare._settings.flare_segment_tokens = 5

        # Set up FLARE mock
        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            # Manually set FLARE client
            responses = ["[0.2]", '["follow-up about IPC 302"]']
            call_count = 0

            async def multi_response(**kwargs):
                nonlocal call_count
                text = responses[min(call_count, len(responses) - 1)]
                call_count += 1
                block = MagicMock()
                block.text = text
                resp = MagicMock()
                resp.content = [block]
                return resp

            engine._flare._client = MagicMock()
            engine._flare._client.messages.create = AsyncMock(side_effect=multi_response)

            # hybrid_search will be called for re-retrieval
            with patch.object(engine, "hybrid_search", new_callable=AsyncMock) as mock_hs:
                mock_hs.return_value = [_make_scored("c_new", "dense", 0.85)]

                query = RetrievalQuery(
                    text="Analyze the evolution of Section 302 IPC",
                    query_embedding=[0.1] * 768,
                    query_embedding_fast=[0.2] * 64,
                    sparse_indices=[1],
                    sparse_values=[0.5],
                    route=QueryRoute.ANALYTICAL,
                )

                result = await engine.retrieve(query)

        assert result.route == QueryRoute.ANALYTICAL
        assert result.flare_retrievals >= 1
        assert "flare_ms" in result.timings


# ===================================================================
# hybrid_search for Phase 8
# ===================================================================


class TestHybridSearchPhase8:
    """Simulates Phase 8 using engine.hybrid_search for per-claim verification."""

    @pytest.mark.asyncio
    async def test_phase8_claim_verification(self, engine: RetrievalEngine) -> None:
        """Phase 8 GenGround calls hybrid_search(claim) for each claim."""
        import numpy as np

        engine._embedder = MagicMock()
        engine._embedder.embed_texts.return_value = [np.zeros(768)]
        engine._embedder.matryoshka_slice.return_value = np.zeros(64)
        engine._bm25 = MagicMock()
        engine._bm25.encode.return_value = MagicMock(indices=[1, 2], values=[0.5, 0.3])

        claims = [
            "Section 420 IPC requires mens rea",
            "The punishment for cheating is 7 years",
            "This section was replaced by BNS 2023",
        ]

        for claim in claims:
            results = await engine.hybrid_search(claim, top_k=3)
            assert isinstance(results, list)
            assert all(isinstance(r, ScoredChunk) for r in results)


# ===================================================================
# Error isolation
# ===================================================================


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_all_channels_fail_returns_empty(self, engine: RetrievalEngine) -> None:
        """If every search channel fails, result has errors but no crash."""
        engine._dense.search = MagicMock(side_effect=Exception("Qdrant down"))
        engine._sparse.search = MagicMock(side_effect=Exception("BM25 down"))
        engine._quim.search = MagicMock(side_effect=Exception("QuIM down"))
        engine._fusion.fuse = MagicMock(return_value=[])
        engine._expander.expand = AsyncMock(return_value=[])

        query = RetrievalQuery(
            text="test",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1],
            sparse_values=[0.5],
        )

        result = await engine.retrieve(query)

        assert len(result.errors) >= 3
        assert result.chunks == []
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_expand_failure_uses_fallback(self, engine: RetrievalEngine) -> None:
        """If expansion fails, chunks are built from fused data."""
        engine._expander.expand = AsyncMock(side_effect=Exception("Redis down"))

        query = RetrievalQuery(
            text="test",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1],
            sparse_values=[0.5],
        )

        result = await engine.retrieve(query)

        assert any("Context expansion failed" in e for e in result.errors)
        # Fallback creates ExpandedContext from FusedChunk data
        assert len(result.chunks) > 0


# ===================================================================
# Multi-query batch via Pipeline
# ===================================================================


class TestBatchPipeline:
    @pytest.mark.asyncio
    async def test_multiple_queries_batch(self) -> None:
        """Pipeline processes multiple queries, each getting independent results."""
        config = RetrievalConfig(settings=RetrievalSettings())
        pipeline = RetrievalPipeline(config=config)

        call_count = 0

        async def mock_retrieve(query: RetrievalQuery) -> RetrievalResult:
            nonlocal call_count
            call_count += 1
            return RetrievalResult(
                query_text=query.text,
                route=query.route,
                chunks=[_make_expanded(f"c{call_count}")],
                total_context_tokens=20,
            )

        pipeline._engine.retrieve = AsyncMock(side_effect=mock_retrieve)
        pipeline._engine.close = AsyncMock()

        results = await pipeline.run(
            queries=["Query 1", "Query 2", "Query 3"],
            load_models=False,
        )

        assert len(results) == 3
        assert results[0].query_text == "Query 1"
        assert results[1].query_text == "Query 2"
        assert results[2].query_text == "Query 3"
        # Each got distinct chunks
        assert results[0].chunks[0].chunk_id != results[1].chunks[0].chunk_id

    @pytest.mark.asyncio
    async def test_pipeline_query_error_isolation(self) -> None:
        """One failing query does not prevent others from completing."""
        config = RetrievalConfig(settings=RetrievalSettings())
        pipeline = RetrievalPipeline(config=config)

        async def mock_retrieve(query: RetrievalQuery) -> RetrievalResult:
            if "fail" in query.text:
                raise RuntimeError("Intentional failure")
            return RetrievalResult(query_text=query.text)

        pipeline._engine.retrieve = AsyncMock(side_effect=mock_retrieve)
        pipeline._engine.close = AsyncMock()

        results = await pipeline.run(
            queries=["good query", "fail query", "another good"],
            load_models=False,
        )

        assert len(results) == 3
        assert results[0].errors == []
        assert "Intentional failure" in results[1].errors[0]
        assert results[2].errors == []
