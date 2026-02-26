"""Tests for retrieval data models."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

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


class TestQueryRoute:
    """QueryRoute enum values."""

    def test_simple(self) -> None:
        assert QueryRoute.SIMPLE == "simple"

    def test_standard(self) -> None:
        assert QueryRoute.STANDARD == "standard"

    def test_complex(self) -> None:
        assert QueryRoute.COMPLEX == "complex"

    def test_analytical(self) -> None:
        assert QueryRoute.ANALYTICAL == "analytical"

    def test_all_values(self) -> None:
        assert len(QueryRoute) == 4


class TestRetrievalQuery:
    """RetrievalQuery validation and defaults."""

    def test_minimal_query(self) -> None:
        q = RetrievalQuery(text="What is IPC?")
        assert q.text == "What is IPC?"
        assert q.query_embedding is None
        assert q.route == QueryRoute.STANDARD
        assert q.max_results == 20
        assert q.max_context_tokens == 30_000

    def test_full_query(self) -> None:
        q = RetrievalQuery(
            text="Section 498A IPC",
            query_embedding=[0.1] * 768,
            query_embedding_fast=[0.2] * 64,
            sparse_indices=[1, 5, 10],
            sparse_values=[0.5, 0.3, 0.8],
            route=QueryRoute.COMPLEX,
            hyde_text="A hypothetical answer about Section 498A...",
            metadata_filters={"document_type": "statute"},
            reference_date=date(2024, 7, 1),
            max_results=10,
            max_context_tokens=20_000,
        )
        assert len(q.query_embedding) == 768
        assert q.route == QueryRoute.COMPLEX
        assert q.reference_date == date(2024, 7, 1)
        assert q.max_results == 10

    def test_query_requires_text(self) -> None:
        with pytest.raises(ValueError, match="text"):
            RetrievalQuery.model_validate({})

    def test_default_route_is_standard(self) -> None:
        q = RetrievalQuery(text="test")
        assert q.route == QueryRoute.STANDARD

    def test_hyde_text_optional(self) -> None:
        q = RetrievalQuery(text="test")
        assert q.hyde_text is None


class TestScoredChunk:
    """ScoredChunk intermediate model."""

    def test_basic_creation(self) -> None:
        sc = ScoredChunk(
            chunk_id="c1",
            text="Some text",
            score=0.95,
            channel="dense",
        )
        assert sc.chunk_id == "c1"
        assert sc.score == 0.95
        assert sc.channel == "dense"
        assert sc.payload == {}

    def test_with_payload(self) -> None:
        sc = ScoredChunk(
            chunk_id="c1",
            text="text",
            score=0.8,
            channel="bm25",
            payload={"document_type": "statute"},
        )
        assert sc.payload["document_type"] == "statute"

    def test_contextualized_text(self) -> None:
        sc = ScoredChunk(
            chunk_id="c1",
            text="original",
            contextualized_text="This section of IPC... original",
            score=0.9,
            channel="dense",
        )
        assert sc.contextualized_text is not None


class TestFusedChunk:
    """FusedChunk post-RRF model."""

    def test_basic_creation(self) -> None:
        fc = FusedChunk(
            chunk_id="c1",
            text="text",
            rrf_score=0.032,
            channels=["dense", "bm25"],
        )
        assert fc.rrf_score == 0.032
        assert len(fc.channels) == 2
        assert fc.rerank_score is None

    def test_with_rerank_score(self) -> None:
        fc = FusedChunk(
            chunk_id="c1",
            text="text",
            rrf_score=0.032,
            rerank_score=0.95,
            channels=["dense"],
        )
        assert fc.rerank_score == 0.95


class TestExpandedContext:
    """ExpandedContext final output model."""

    def test_basic_creation(self) -> None:
        ec = ExpandedContext(
            chunk_id="c1",
            chunk_text="Section 420 IPC",
            total_tokens=50,
        )
        assert ec.parent_text is None
        assert ec.judgment_header_text is None
        assert ec.relevance_score == 0.0

    def test_with_parent_context(self) -> None:
        ec = ExpandedContext(
            chunk_id="c1",
            chunk_text="Section 420 IPC",
            parent_text="Chapter XVII deals with...",
            relevance_score=0.95,
            total_tokens=200,
        )
        assert ec.parent_text is not None
        assert ec.total_tokens == 200

    def test_with_judgment_header(self) -> None:
        ec = ExpandedContext(
            chunk_id="c1",
            chunk_text="The court held...",
            judgment_header_text="State v. Accused, 2024 SCC 123",
            total_tokens=100,
        )
        assert ec.judgment_header_text is not None


class TestRetrievalResult:
    """RetrievalResult model."""

    def test_default_values(self) -> None:
        r = RetrievalResult()
        assert r.query_text == ""
        assert r.route == QueryRoute.STANDARD
        assert r.chunks == []
        assert r.errors == []
        assert r.flare_retrievals == 0
        assert r.started_at is not None
        assert r.finished_at is None

    def test_elapsed_ms_none_when_not_finished(self) -> None:
        r = RetrievalResult()
        assert r.elapsed_ms == 0.0

    def test_elapsed_ms_computed(self) -> None:
        now = datetime.now(UTC)
        r = RetrievalResult(
            started_at=now,
            finished_at=now + timedelta(milliseconds=500),
        )
        assert abs(r.elapsed_ms - 500.0) < 1.0

    def test_with_chunks(self) -> None:
        ec = ExpandedContext(chunk_id="c1", chunk_text="text", total_tokens=50)
        r = RetrievalResult(
            query_text="test query",
            chunks=[ec],
            total_context_tokens=50,
            search_channels_used=["dense", "bm25"],
        )
        assert len(r.chunks) == 1
        assert r.total_context_tokens == 50

    def test_with_errors(self) -> None:
        r = RetrievalResult(errors=["BM25 search failed", "Graph unavailable"])
        assert len(r.errors) == 2

    def test_with_timings(self) -> None:
        r = RetrievalResult(
            timings={
                "dense_ms": 25.0,
                "bm25_ms": 50.0,
                "fusion_ms": 5.0,
                "rerank_ms": 30.0,
            }
        )
        assert r.timings["dense_ms"] == 25.0


class TestRetrievalSettings:
    """RetrievalSettings defaults and overrides."""

    def test_defaults(self) -> None:
        s = RetrievalSettings()
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.chunks_collection == "legal_chunks"
        assert s.quim_collection == "quim_questions"
        assert s.embedding_dim == 768
        assert s.matryoshka_dim == 64
        assert s.dense_fast_top_k == 1000
        assert s.dense_full_top_k == 100
        assert s.bm25_top_k == 100
        assert s.quim_top_k == 50
        assert s.rrf_k == 60
        assert s.fused_top_k == 150
        assert s.rerank_top_k == 20
        assert s.max_context_tokens == 30_000
        assert s.flare_enabled is True
        assert s.flare_max_retrievals == 5
        assert s.bm25_vocab_path is None

    def test_override_values(self) -> None:
        s = RetrievalSettings(
            qdrant_host="remote-host",
            qdrant_port=6334,
            dense_fast_top_k=500,
            rerank_top_k=10,
            flare_enabled=False,
        )
        assert s.qdrant_host == "remote-host"
        assert s.dense_fast_top_k == 500
        assert s.flare_enabled is False


class TestRetrievalConfig:
    """RetrievalConfig root model."""

    def test_default_config(self) -> None:
        c = RetrievalConfig()
        assert isinstance(c.settings, RetrievalSettings)

    def test_config_with_override(self) -> None:
        c = RetrievalConfig(settings=RetrievalSettings(rerank_top_k=10))
        assert c.settings.rerank_top_k == 10
