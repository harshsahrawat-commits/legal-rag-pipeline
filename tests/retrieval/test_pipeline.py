"""Tests for RetrievalPipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.retrieval._models import RetrievalConfig, RetrievalResult, RetrievalSettings
from src.retrieval.pipeline import RetrievalPipeline


@pytest.fixture()
def pipeline() -> RetrievalPipeline:
    config = RetrievalConfig(settings=RetrievalSettings())
    return RetrievalPipeline(config=config)


class TestCollectQueries:
    """_collect_queries gathers queries from all sources."""

    def test_from_list(self) -> None:
        result = RetrievalPipeline._collect_queries(
            queries=["q1", "q2"],
            queries_file=None,
            interactive=False,
        )
        assert result == ["q1", "q2"]

    def test_from_json_file_list(self, tmp_path: Path) -> None:
        f = tmp_path / "queries.json"
        f.write_text(json.dumps(["q1", "q2", "q3"]))
        result = RetrievalPipeline._collect_queries(
            queries=None,
            queries_file=f,
            interactive=False,
        )
        assert result == ["q1", "q2", "q3"]

    def test_from_json_file_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "queries.json"
        f.write_text(json.dumps({"queries": ["q1", "q2"]}))
        result = RetrievalPipeline._collect_queries(
            queries=None,
            queries_file=f,
            interactive=False,
        )
        assert result == ["q1", "q2"]

    def test_combined_sources(self, tmp_path: Path) -> None:
        f = tmp_path / "queries.json"
        f.write_text(json.dumps(["file_q"]))
        result = RetrievalPipeline._collect_queries(
            queries=["list_q"],
            queries_file=f,
            interactive=False,
        )
        assert result == ["list_q", "file_q"]

    def test_empty_inputs(self) -> None:
        result = RetrievalPipeline._collect_queries(
            queries=None,
            queries_file=None,
            interactive=False,
        )
        assert result == []

    def test_invalid_file_handled(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json")
        result = RetrievalPipeline._collect_queries(
            queries=None,
            queries_file=f,
            interactive=False,
        )
        assert result == []


class TestPipelineRun:
    """Pipeline.run processes queries through the engine."""

    @pytest.mark.asyncio
    async def test_processes_query_list(self, pipeline: RetrievalPipeline) -> None:
        mock_result = RetrievalResult(query_text="q1")
        pipeline._engine.retrieve = AsyncMock(return_value=mock_result)
        pipeline._engine.close = AsyncMock()

        results = await pipeline.run(queries=["q1", "q2"], load_models=False)

        assert len(results) == 2
        assert pipeline._engine.retrieve.call_count == 2

    @pytest.mark.asyncio
    async def test_dry_run_skips_search(self, pipeline: RetrievalPipeline) -> None:
        pipeline._engine.retrieve = AsyncMock()
        pipeline._engine.close = AsyncMock()

        results = await pipeline.run(queries=["q1"], dry_run=True)

        assert len(results) == 1
        assert results[0].query_text == "q1"
        pipeline._engine.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_query_error_isolation(self, pipeline: RetrievalPipeline) -> None:
        async def mock_retrieve(query):
            if query.text == "bad":
                raise RuntimeError("boom")
            return RetrievalResult(query_text=query.text)

        pipeline._engine.retrieve = AsyncMock(side_effect=mock_retrieve)
        pipeline._engine.close = AsyncMock()

        results = await pipeline.run(queries=["good", "bad", "good2"], load_models=False)

        assert len(results) == 3
        assert results[0].errors == []
        assert len(results[1].errors) == 1
        assert "boom" in results[1].errors[0]
        assert results[2].errors == []

    @pytest.mark.asyncio
    async def test_loads_from_file(self, pipeline: RetrievalPipeline, tmp_path: Path) -> None:
        f = tmp_path / "q.json"
        f.write_text(json.dumps(["from_file"]))

        pipeline._engine.retrieve = AsyncMock(return_value=RetrievalResult(query_text="from_file"))
        pipeline._engine.close = AsyncMock()

        results = await pipeline.run(queries_file=f, load_models=False)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_engine_accessible(self, pipeline: RetrievalPipeline) -> None:
        """Pipeline exposes engine for Phase 8 reuse."""
        assert pipeline.engine is pipeline._engine


class TestPipelineInit:
    """Pipeline initialization from config."""

    def test_from_config_object(self) -> None:
        config = RetrievalConfig(settings=RetrievalSettings(rerank_top_k=5))
        p = RetrievalPipeline(config=config)
        assert p._settings.rerank_top_k == 5

    def test_from_config_path(self) -> None:
        p = RetrievalPipeline(config_path=Path("configs/retrieval.yaml"))
        assert p._settings.qdrant_host == "localhost"
