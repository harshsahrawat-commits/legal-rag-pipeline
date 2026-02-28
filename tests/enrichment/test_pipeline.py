from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime in function bodies
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from src.enrichment._models import EnrichmentConfig, EnrichmentSettings
from src.enrichment.pipeline import EnrichmentPipeline, _load_chunks, _resolve_source_filter
from tests.enrichment.conftest import make_mock_provider

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.parsing._models import ParsedDocument

_CTX_RESPONSE = "This chunk is from Section 10 of the Indian Contract Act."
_QUIM_RESPONSE = "What is the penalty for breach?\nWho can enter a contract?"


def _write_chunk_file(
    tmp_path: Path,
    source: str,
    filename: str,
    chunks: list[LegalChunk],
) -> Path:
    """Helper to write chunks to a file matching the pipeline's expected layout."""
    source_dir = tmp_path / "chunks" / source
    source_dir.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in chunks]
    path = source_dir / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_parsed_file(
    tmp_path: Path,
    source: str,
    filename: str,
    parsed_doc: ParsedDocument,
) -> Path:
    parsed_dir = tmp_path / "parsed" / source
    parsed_dir.mkdir(parents=True, exist_ok=True)
    path = parsed_dir / filename
    path.write_text(parsed_doc.model_dump_json(indent=2), encoding="utf-8")
    return path


def _make_pipeline(
    tmp_path: Path,
    client_response: str = _CTX_RESPONSE,
) -> EnrichmentPipeline:
    settings = EnrichmentSettings(
        input_dir=tmp_path / "chunks",
        output_dir=tmp_path / "enriched",
        parsed_dir=tmp_path / "parsed",
        concurrency=2,
    )
    config = EnrichmentConfig(settings=settings)
    pipeline = EnrichmentPipeline(config=config)
    # Mock both enrichers' LLM providers
    pipeline._contextual._provider = make_mock_provider(client_response)
    pipeline._quim._provider = make_mock_provider(_QUIM_RESPONSE)
    return pipeline


class TestDiscovery:
    async def test_empty_input_dir(self, tmp_path: Path):
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()
        assert result.documents_found == 0

    async def test_discovers_chunk_files(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1

    async def test_excludes_quim_files(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        # Also write a quim file — should be excluded from discovery
        quim_dir = tmp_path / "chunks" / "indian_kanoon"
        (quim_dir / "doc1.quim.json").write_text("{}", encoding="utf-8")
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1

    async def test_source_filter(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_chunk_file(tmp_path, "india_code", "doc2.json", sample_statute_chunks)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(source_name="indian kanoon", dry_run=True)
        assert result.documents_found == 1


class TestDryRun:
    async def test_dry_run_does_not_enrich(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1
        assert result.documents_enriched == 0
        # No output files should exist
        enriched_dir = tmp_path / "enriched"
        assert not enriched_dir.exists() or not list(enriched_dir.glob("**/*.json"))


class TestUnknownSource:
    async def test_unknown_source_returns_error(self, tmp_path: Path):
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(source_name="nonexistent")
        assert len(result.errors) == 1
        assert "Unknown source" in result.errors[0]


class TestContextualStage:
    async def test_contextual_only(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(stage="contextual_retrieval")
        assert result.documents_enriched == 1
        assert result.chunks_contextualized == 3

    async def test_contextualized_flag_in_output(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="contextual_retrieval")

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        assert output_path.exists()
        enriched = _load_chunks(output_path)
        for chunk in enriched:
            assert chunk.ingestion.contextualized is True
            assert chunk.contextualized_text is not None


class TestQuIMStage:
    async def test_quim_only(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(stage="quim_rag")
        assert result.documents_enriched == 1
        assert result.chunks_quim_generated > 0

    async def test_quim_file_created(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="quim_rag")

        quim_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.quim.json"
        assert quim_path.exists()

        from src.enrichment._models import QuIMDocument

        quim_doc = QuIMDocument.model_validate_json(quim_path.read_text(encoding="utf-8"))
        assert len(quim_doc.entries) == 3


class TestBothStages:
    async def test_both_stages(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()
        assert result.documents_enriched == 1
        assert result.chunks_contextualized == 3
        assert result.chunks_quim_generated > 0


class TestIdempotency:
    async def test_skips_when_output_exists(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)

        # First run
        r1 = await pipeline.run()
        assert r1.documents_enriched == 1

        # Second run — should skip
        r2 = await pipeline.run()
        assert r2.documents_skipped == 1
        assert r2.documents_enriched == 0

    async def test_contextual_idempotency(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)

        r1 = await pipeline.run(stage="contextual_retrieval")
        assert r1.documents_enriched == 1

        r2 = await pipeline.run(stage="contextual_retrieval")
        assert r2.documents_skipped == 1


class TestErrorHandling:
    async def test_corrupt_chunk_file(self, tmp_path: Path):
        source_dir = tmp_path / "chunks" / "indian_kanoon"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "bad.json").write_text("not valid json", encoding="utf-8")

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()
        assert result.documents_failed == 1
        assert len(result.errors) == 1

    async def test_missing_parsed_doc_still_enriches(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
    ):
        """If parsed doc is missing, enrichers run with empty context."""
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        # Do NOT write parsed doc
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(stage="contextual_retrieval")
        assert result.documents_enriched == 1

    async def test_per_chunk_failure_isolated(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Per-chunk LLM failures are isolated — document still completes."""
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)

        # All LLM calls fail — but per-chunk isolation means the document
        # still "completes" with zero chunks contextualized
        pipeline._contextual._provider.acomplete = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )
        result = await pipeline.run(stage="contextual_retrieval")
        # Document is enriched (no document-level failure), but no chunks contextualized
        assert result.documents_enriched == 1
        assert result.chunks_contextualized == 0


class TestMultipleDocuments:
    async def test_multiple_docs(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_judgment_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_chunk_file(tmp_path, "indian_kanoon", "doc2.json", sample_judgment_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc2.json", sample_parsed_doc)

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()
        assert result.documents_found == 2
        assert result.documents_enriched == 2


class TestResultCounts:
    async def test_accurate_counts(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _write_chunk_file(tmp_path, "indian_kanoon", "doc1.json", sample_statute_chunks)
        _write_parsed_file(tmp_path, "indian_kanoon", "doc1.json", sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()
        assert result.documents_found == 1
        assert result.documents_enriched == 1
        assert result.documents_skipped == 0
        assert result.documents_failed == 0
        assert result.finished_at is not None


class TestResolveSourceFilter:
    def test_none_returns_none(self):
        assert _resolve_source_filter(None) is None

    def test_valid_source(self):
        from src.acquisition._models import SourceType

        assert _resolve_source_filter("Indian Kanoon") == SourceType.INDIAN_KANOON
        assert _resolve_source_filter("india code") == SourceType.INDIA_CODE

    def test_invalid_source(self):
        assert _resolve_source_filter("nonexistent") is None
