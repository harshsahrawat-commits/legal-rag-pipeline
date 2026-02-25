"""Tests for KnowledgeGraphPipeline."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge_graph._models import KGConfig, KGSettings
from src.knowledge_graph.pipeline import KnowledgeGraphPipeline


@pytest.fixture()
def tmp_enriched_dir(tmp_path: Path, statute_chunk) -> Path:
    """Create a temp enriched directory with a sample document."""
    source_dir = tmp_path / "data" / "enriched" / "indian_kanoon"
    source_dir.mkdir(parents=True)
    doc_path = source_dir / "doc1.json"
    doc_path.write_text(
        json.dumps([statute_chunk.model_dump(mode="json")]),
        encoding="utf-8",
    )
    return tmp_path / "data" / "enriched"


@pytest.fixture()
def config(tmp_enriched_dir) -> KGConfig:
    return KGConfig(settings=KGSettings(input_dir=tmp_enriched_dir))


@pytest.fixture()
def mock_neo4j_pipeline(config):
    """Pipeline with mocked Neo4j client."""
    pipeline = KnowledgeGraphPipeline(config=config)
    mock_client = AsyncMock()
    mock_client.setup_schema = AsyncMock()
    mock_client.merge_act = AsyncMock()
    mock_client.merge_section = AsyncMock()
    mock_client.merge_section_version = AsyncMock()
    mock_client.merge_judgment = AsyncMock()
    mock_client.merge_amendment = AsyncMock()
    mock_client.merge_legal_concept = AsyncMock()
    mock_client.merge_court = AsyncMock()
    mock_client.merge_judge = AsyncMock()
    mock_client.create_relationship = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.run_query = AsyncMock(return_value=[])
    mock_client.session = MagicMock()
    pipeline._client = mock_client
    pipeline._integrity_checker._client = mock_client
    return pipeline


class TestPipelineDiscovery:
    @pytest.mark.asyncio
    async def test_discovers_files(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run(dry_run=True)
        assert result.documents_found == 1

    @pytest.mark.asyncio
    async def test_dry_run_no_processing(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run(dry_run=True)
        assert result.documents_ingested == 0
        mock_neo4j_pipeline._client.setup_schema.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_source(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run(source_name="Unknown Source")
        assert len(result.errors) == 1
        assert "Unknown source" in result.errors[0]

    @pytest.mark.asyncio
    async def test_source_filter(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run(source_name="Indian Kanoon", dry_run=True)
        assert result.documents_found == 1

    @pytest.mark.asyncio
    async def test_empty_input_dir(self, tmp_path) -> None:
        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "nonexistent"))
        pipeline = KnowledgeGraphPipeline(config=config)
        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 0


class TestPipelineIngestion:
    @pytest.mark.asyncio
    async def test_ingests_statute(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run(skip_integrity=True)
        assert result.documents_ingested == 1
        assert result.nodes_created > 0
        assert result.relationships_created > 0

    @pytest.mark.asyncio
    async def test_schema_setup_called(self, mock_neo4j_pipeline) -> None:
        await mock_neo4j_pipeline.run(skip_integrity=True)
        mock_neo4j_pipeline._client.setup_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_methods_called(self, mock_neo4j_pipeline) -> None:
        await mock_neo4j_pipeline.run(skip_integrity=True)
        # statute_chunk has act, section, section_version, amendment
        mock_neo4j_pipeline._client.merge_act.assert_called()
        mock_neo4j_pipeline._client.merge_section.assert_called()
        mock_neo4j_pipeline._client.merge_section_version.assert_called()

    @pytest.mark.asyncio
    async def test_error_isolation(self, mock_neo4j_pipeline) -> None:
        """Single document failure doesn't crash pipeline."""
        mock_neo4j_pipeline._client.merge_act.side_effect = RuntimeError("db error")
        result = await mock_neo4j_pipeline.run(skip_integrity=True)
        assert result.documents_failed == 1
        assert result.documents_ingested == 0

    @pytest.mark.asyncio
    async def test_empty_chunks_skipped(self, tmp_path) -> None:
        source_dir = tmp_path / "data" / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        doc_path = source_dir / "empty.json"
        doc_path.write_text("[]", encoding="utf-8")

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "data" / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        mock_client = AsyncMock()
        mock_client.setup_schema = AsyncMock()
        mock_client.run_query = AsyncMock(return_value=[])
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=True)
        assert result.documents_skipped == 1


class TestPipelineIntegrity:
    @pytest.mark.asyncio
    async def test_integrity_runs_by_default(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run()
        assert result.integrity_passed is True  # no violations from empty responses

    @pytest.mark.asyncio
    async def test_skip_integrity(self, mock_neo4j_pipeline) -> None:
        result = await mock_neo4j_pipeline.run(skip_integrity=True)
        assert result.integrity_passed is None

    @pytest.mark.asyncio
    async def test_integrity_failure_recorded(self, mock_neo4j_pipeline) -> None:
        """When integrity fails, violations are recorded in errors."""
        call_count = 0

        async def _mock_query(cypher, params=None):
            nonlocal call_count
            call_count += 1
            # First integrity query (section_versions) returns a violation
            if call_count == 1:
                return [{"act": "Test", "section": "1"}]
            return []

        mock_neo4j_pipeline._client.run_query = AsyncMock(side_effect=_mock_query)
        result = await mock_neo4j_pipeline.run()
        assert result.integrity_passed is False
        assert any("Integrity" in e for e in result.errors)


class TestPipelineClose:
    @pytest.mark.asyncio
    async def test_close(self, mock_neo4j_pipeline) -> None:
        await mock_neo4j_pipeline.close()
        mock_neo4j_pipeline._client.close.assert_called_once()
