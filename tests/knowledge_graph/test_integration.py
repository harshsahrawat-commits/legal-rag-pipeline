"""Integration tests for the knowledge graph module.

End-to-end tests with mocked Neo4j: statute ingestion, judgment ingestion,
idempotency, and mixed document handling.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge_graph._models import KGConfig, KGSettings
from src.knowledge_graph.pipeline import KnowledgeGraphPipeline


def _write_chunks(path: Path, chunks: list) -> None:
    path.write_text(
        json.dumps([c.model_dump(mode="json") for c in chunks]),
        encoding="utf-8",
    )


@pytest.fixture()
def mock_client():
    client = AsyncMock()
    client.setup_schema = AsyncMock()
    client.merge_act = AsyncMock()
    client.merge_section = AsyncMock()
    client.merge_section_version = AsyncMock()
    client.merge_judgment = AsyncMock()
    client.merge_amendment = AsyncMock()
    client.merge_legal_concept = AsyncMock()
    client.merge_court = AsyncMock()
    client.merge_judge = AsyncMock()
    client.create_relationship = AsyncMock()
    client.close = AsyncMock()
    client.run_query = AsyncMock(return_value=[])
    client.session = MagicMock()
    return client


class TestStatuteIngestion:
    @pytest.mark.asyncio
    async def test_full_statute_pipeline(
        self, tmp_path, statute_chunk, definition_chunk, mock_client
    ) -> None:
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "ipc.json", [statute_chunk, definition_chunk])

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=True)

        assert result.documents_ingested == 1
        assert result.nodes_created > 0
        assert result.relationships_created > 0
        mock_client.merge_act.assert_called()
        mock_client.merge_section.assert_called()
        mock_client.merge_legal_concept.assert_called()


class TestJudgmentIngestion:
    @pytest.mark.asyncio
    async def test_full_judgment_pipeline(self, tmp_path, judgment_chunk, mock_client) -> None:
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "judgment.json", [judgment_chunk])

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=True)

        assert result.documents_ingested == 1
        mock_client.merge_judgment.assert_called()
        mock_client.merge_court.assert_called()
        mock_client.merge_judge.assert_called()
        mock_client.create_relationship.assert_called()


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_second_run_same_result(self, tmp_path, statute_chunk, mock_client) -> None:
        """MERGE-based ops: running twice should succeed without errors."""
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "ipc.json", [statute_chunk])

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result1 = await pipeline.run(skip_integrity=True)
        result2 = await pipeline.run(skip_integrity=True)

        assert result1.documents_ingested == 1
        assert result2.documents_ingested == 1
        assert result1.errors == []
        assert result2.errors == []


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_one_doc_fails_other_succeeds(
        self, tmp_path, statute_chunk, judgment_chunk, mock_client
    ) -> None:
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "aaa_bad.json", [statute_chunk])
        _write_chunks(source_dir / "zzz_good.json", [judgment_chunk])

        # Fail on merge_act (statute) but succeed on judgment
        call_count = 0

        async def _merge_act_fail(node):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "connection timeout"
                raise RuntimeError(msg)

        mock_client.merge_act = AsyncMock(side_effect=_merge_act_fail)

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=True)
        assert result.documents_failed == 1
        assert result.documents_ingested == 1  # judgment succeeds


class TestMixedDocuments:
    @pytest.mark.asyncio
    async def test_statute_and_judgment_together(
        self, tmp_path, statute_chunk, judgment_chunk, mock_client
    ) -> None:
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "statute.json", [statute_chunk])
        _write_chunks(source_dir / "judgment.json", [judgment_chunk])

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=True)
        assert result.documents_ingested == 2
        assert result.documents_failed == 0


class TestRepealedStatute:
    @pytest.mark.asyncio
    async def test_repealed_creates_repeals_relationship(
        self, tmp_path, repealed_statute_chunk, mock_client
    ) -> None:
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "crpc.json", [repealed_statute_chunk])

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=True)
        assert result.documents_ingested == 1
        # Should have REPEALS relationship
        rel_calls = mock_client.create_relationship.call_args_list
        rel_types = [call.args[0].rel_type for call in rel_calls]
        assert "REPEALS" in rel_types


class TestIntegrityIntegration:
    @pytest.mark.asyncio
    async def test_integrity_runs_after_ingestion(
        self, tmp_path, statute_chunk, mock_client
    ) -> None:
        source_dir = tmp_path / "enriched" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        _write_chunks(source_dir / "ipc.json", [statute_chunk])

        config = KGConfig(settings=KGSettings(input_dir=tmp_path / "enriched"))
        pipeline = KnowledgeGraphPipeline(config=config)
        pipeline._client = mock_client
        pipeline._integrity_checker._client = mock_client

        result = await pipeline.run(skip_integrity=False)
        assert result.integrity_passed is True
        # run_query should have been called for integrity checks
        assert mock_client.run_query.call_count >= 4
