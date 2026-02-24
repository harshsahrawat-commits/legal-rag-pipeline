from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import pytest

from src.chunking._models import ChunkingConfig, ChunkingSettings, LegalChunk
from src.chunking.pipeline import (
    ChunkingPipeline,
    _assign_chunk_indices,
    _assign_sibling_ids,
    _save_chunks,
)

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument


@pytest.fixture()
def pipeline_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create input/output directories for pipeline tests."""
    input_dir = tmp_path / "parsed"
    output_dir = tmp_path / "chunks"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


@pytest.fixture()
def pipeline_config(pipeline_dirs: tuple[Path, Path]) -> ChunkingConfig:
    input_dir, output_dir = pipeline_dirs
    return ChunkingConfig(
        settings=ChunkingSettings(input_dir=input_dir, output_dir=output_dir),
    )


def _write_parsed_doc(doc: ParsedDocument, input_dir: Path, source_name: str = "indian_kanoon"):
    source_dir = input_dir / source_name
    source_dir.mkdir(exist_ok=True)
    doc_path = source_dir / f"{doc.document_id}.json"
    doc_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc_path


class TestAssignChunkIndices:
    def test_sequential_indices(self, sample_statute_doc: ParsedDocument):
        from src.chunking._models import ChunkingSettings
        from src.chunking._token_counter import TokenCounter
        from src.chunking.chunkers._page_level import PageLevelChunker

        chunker = PageLevelChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        _assign_chunk_indices(chunks)
        for idx, c in enumerate(chunks):
            assert c.chunk_index == idx

    def test_empty_list(self):
        _assign_chunk_indices([])  # should not raise


class TestAssignSiblingIds:
    def test_sibling_ids_window(self, sample_statute_doc: ParsedDocument):
        from src.chunking._models import ChunkingSettings
        from src.chunking._token_counter import TokenCounter
        from src.chunking.chunkers._statute_boundary import StatuteBoundaryChunker

        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        _assign_sibling_ids(chunks, window=2)

        if len(chunks) >= 3:
            # Middle chunk should have up to 4 siblings (2 before, 2 after)
            middle = chunks[len(chunks) // 2]
            assert len(middle.parent_info.sibling_chunk_ids) > 0
            assert middle.id not in middle.parent_info.sibling_chunk_ids

        # First chunk has no preceding siblings
        first = chunks[0]
        preceding = [sid for sid in first.parent_info.sibling_chunk_ids if sid != chunks[0].id]
        assert all(sid in [c.id for c in chunks] for sid in preceding)

    def test_single_chunk_no_siblings(self):
        from datetime import UTC, datetime
        from uuid import uuid4

        from src.acquisition._models import DocumentType
        from src.chunking._models import (
            ChunkStrategy,
            ChunkType,
            ContentMetadata,
            IngestionMetadata,
            SourceInfo,
        )

        now = datetime.now(UTC)
        chunk = LegalChunk(
            document_id=uuid4(),
            text="only chunk",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=2,
            source=SourceInfo(url="u", source_name="s", scraped_at=now, last_verified=now),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="p", chunk_strategy=ChunkStrategy.PAGE_LEVEL
            ),
        )
        _assign_sibling_ids([chunk])
        assert chunk.parent_info.sibling_chunk_ids == []


class TestSaveChunks:
    def test_save_and_reload(self, tmp_path: Path, sample_statute_doc: ParsedDocument):
        from src.chunking._models import ChunkingSettings
        from src.chunking._token_counter import TokenCounter
        from src.chunking.chunkers._statute_boundary import StatuteBoundaryChunker

        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        _assign_chunk_indices(chunks)

        output = tmp_path / "output.json"
        _save_chunks(chunks, output)

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data) == len(chunks)

        # Validate round-trip
        reloaded = [LegalChunk.model_validate(c) for c in data]
        assert len(reloaded) == len(chunks)
        assert reloaded[0].id == chunks[0].id


class TestPipelineRun:
    async def test_empty_input_dir(self, pipeline_config: ChunkingConfig):
        pipeline = ChunkingPipeline(config=pipeline_config)
        result = await pipeline.run()
        assert result.documents_found == 0
        assert result.documents_chunked == 0

    async def test_dry_run(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        input_dir, _ = pipeline_dirs
        _write_parsed_doc(sample_statute_doc, input_dir)
        pipeline = ChunkingPipeline(config=pipeline_config)
        result = await pipeline.run(dry_run=True)
        assert result.documents_found == 1
        assert result.documents_chunked == 0

    async def test_chunks_statute(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        input_dir, output_dir = pipeline_dirs
        _write_parsed_doc(sample_statute_doc, input_dir)
        pipeline = ChunkingPipeline(config=pipeline_config)
        result = await pipeline.run()
        assert result.documents_chunked == 1
        assert result.chunks_created > 0
        assert result.errors == []

        # Verify output file
        output_file = output_dir / "indian_kanoon" / f"{sample_statute_doc.document_id}.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text(encoding="utf-8"))
        chunks = [LegalChunk.model_validate(c) for c in data]
        assert len(chunks) == result.chunks_created

    async def test_chunks_judgment(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        input_dir, _ = pipeline_dirs
        _write_parsed_doc(sample_judgment_doc, input_dir)
        pipeline = ChunkingPipeline(config=pipeline_config)
        result = await pipeline.run()
        assert result.documents_chunked == 1
        assert result.chunks_created >= 6  # header + 5 sections

    async def test_idempotency(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        input_dir, _ = pipeline_dirs
        _write_parsed_doc(sample_statute_doc, input_dir)
        pipeline = ChunkingPipeline(config=pipeline_config)

        # First run
        r1 = await pipeline.run()
        assert r1.documents_chunked == 1

        # Second run — should skip
        r2 = await pipeline.run()
        assert r2.documents_chunked == 0
        assert r2.documents_skipped == 1

    async def test_source_filter(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        input_dir, _ = pipeline_dirs
        _write_parsed_doc(sample_statute_doc, input_dir, source_name="indian_kanoon")
        pipeline = ChunkingPipeline(config=pipeline_config)

        result = await pipeline.run(source_name="India Code")
        assert result.documents_found == 0

    async def test_unknown_source(self, pipeline_config: ChunkingConfig):
        pipeline = ChunkingPipeline(config=pipeline_config)
        result = await pipeline.run(source_name="Unknown Source")
        assert len(result.errors) == 1
        assert "Unknown source" in result.errors[0]

    async def test_corrupt_json_error_isolation(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
    ):
        input_dir, _ = pipeline_dirs
        source_dir = input_dir / "indian_kanoon"
        source_dir.mkdir()
        corrupt = source_dir / "bad.json"
        corrupt.write_text("{invalid json", encoding="utf-8")
        pipeline = ChunkingPipeline(config=pipeline_config)
        result = await pipeline.run()
        assert result.documents_failed == 1
        assert result.documents_chunked == 0

    async def test_chunk_indices_sequential(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        input_dir, output_dir = pipeline_dirs
        _write_parsed_doc(sample_statute_doc, input_dir)
        pipeline = ChunkingPipeline(config=pipeline_config)
        await pipeline.run()

        output_file = output_dir / "indian_kanoon" / f"{sample_statute_doc.document_id}.json"
        data = json.loads(output_file.read_text(encoding="utf-8"))
        chunks = [LegalChunk.model_validate(c) for c in data]
        for idx, c in enumerate(chunks):
            assert c.chunk_index == idx

    async def test_sibling_ids_populated(
        self,
        pipeline_config: ChunkingConfig,
        pipeline_dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        input_dir, output_dir = pipeline_dirs
        _write_parsed_doc(sample_statute_doc, input_dir)
        pipeline = ChunkingPipeline(config=pipeline_config)
        await pipeline.run()

        output_file = output_dir / "indian_kanoon" / f"{sample_statute_doc.document_id}.json"
        data = json.loads(output_file.read_text(encoding="utf-8"))
        chunks = [LegalChunk.model_validate(c) for c in data]
        if len(chunks) > 1:
            # Middle chunks should have siblings
            mid = chunks[len(chunks) // 2]
            assert len(mid.parent_info.sibling_chunk_ids) > 0
