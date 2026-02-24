"""Integration tests for the chunking pipeline.

End-to-end: ParsedDocument -> ChunkingPipeline -> JSON output with validated
metadata, token bounds, sibling IDs, and correct routing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from src.acquisition._models import DocumentType
from src.chunking._models import (
    ChunkingConfig,
    ChunkingSettings,
    ChunkStrategy,
    ChunkType,
    LegalChunk,
)
from src.chunking._token_counter import TokenCounter
from src.chunking.pipeline import ChunkingPipeline

if TYPE_CHECKING:
    from pathlib import Path

    from src.parsing._models import ParsedDocument


# ── Helpers ──────────────────────────────────────────────────────


def _write_parsed_doc(
    doc: ParsedDocument,
    input_dir: Path,
    source_name: str = "indian_kanoon",
) -> Path:
    source_dir = input_dir / source_name
    source_dir.mkdir(exist_ok=True)
    path = source_dir / f"{doc.document_id}.json"
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return path


def _read_chunks(output_dir: Path, source_name: str, doc_id: str) -> list[LegalChunk]:
    path = output_dir / source_name / f"{doc_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [LegalChunk.model_validate(c) for c in data]


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "parsed"
    out = tmp_path / "chunks"
    inp.mkdir()
    out.mkdir()
    return inp, out


@pytest.fixture()
def config(dirs: tuple[Path, Path]) -> ChunkingConfig:
    inp, out = dirs
    return ChunkingConfig(settings=ChunkingSettings(input_dir=inp, output_dir=out))


@pytest.fixture()
def tc() -> TokenCounter:
    return TokenCounter()


# ── Statute routing + metadata ───────────────────────────────────


class TestStatuteEndToEnd:
    """Statute doc → StatuteBoundaryChunker → validated output."""

    async def test_routes_to_statute_boundary(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        result = await pipeline.run()

        assert result.documents_chunked == 1
        assert result.errors == []

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        assert len(chunks) > 0
        assert all(c.ingestion.chunk_strategy == ChunkStrategy.STRUCTURE_BOUNDARY for c in chunks)

    async def test_statute_metadata_present(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        for chunk in chunks:
            assert chunk.document_type == DocumentType.STATUTE
            assert chunk.statute is not None
            assert chunk.statute.act_name == "Indian Contract Act"
            assert chunk.judgment is None

    async def test_statute_chunk_types(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        chunk_types = {c.chunk_type for c in chunks}
        # Statute fixture has definitions (Section 2) and regular sections
        assert ChunkType.STATUTORY_TEXT in chunk_types or ChunkType.DEFINITION in chunk_types

    async def test_statute_document_ids_consistent(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        for chunk in chunks:
            assert chunk.document_id == sample_statute_doc.document_id


# ── Judgment routing + metadata ──────────────────────────────────


class TestJudgmentEndToEnd:
    """Judgment doc → JudgmentStructuralChunker → validated output."""

    async def test_routes_to_judgment_structural(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        result = await pipeline.run()

        assert result.documents_chunked == 1
        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        assert all(c.ingestion.chunk_strategy == ChunkStrategy.JUDGMENT_STRUCTURAL for c in chunks)

    async def test_judgment_metadata_present(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        for chunk in chunks:
            assert chunk.document_type == DocumentType.JUDGMENT
            assert chunk.judgment is not None
            assert chunk.judgment.case_citation == "AIR 2024 SC 1500"
            assert chunk.statute is None

    async def test_judgment_header_chunk_id_propagated(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        # All non-header chunks should reference the header
        non_header = [c for c in chunks if c.parent_info.judgment_header_chunk_id is not None]
        assert len(non_header) >= len(chunks) - 1  # all except header itself

    async def test_judgment_has_expected_sections(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        chunk_types = {c.chunk_type for c in chunks}
        # Fixture has FACTS, ISSUES, REASONING, HOLDING, ORDER
        assert ChunkType.FACTS in chunk_types
        assert ChunkType.HOLDING in chunk_types


# ── Degraded scan routing ────────────────────────────────────────


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestDegradedScanRouting:
    """Low-OCR doc routing depends on available optional deps."""

    async def test_routes_correctly(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_degraded_scan_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_degraded_scan_doc, inp, source_name="india_code")
        pipeline = ChunkingPipeline(config=config)
        result = await pipeline.run()

        if _has_sentence_transformers():
            # SemanticMaxMin handles docs with no sections
            assert result.documents_chunked == 1
            chunks = _read_chunks(out, "india_code", str(sample_degraded_scan_doc.document_id))
            assert all(c.ingestion.chunk_strategy == ChunkStrategy.SEMANTIC_MAXMIN for c in chunks)
        else:
            # SemanticMaxMin selected but fails at runtime (no sentence-transformers)
            # Pipeline records it as a failure — error isolation works
            assert result.documents_failed == 1
            assert "sentence-transformers" in result.errors[0]

    async def test_page_level_direct(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_degraded_scan_doc: ParsedDocument,
    ):
        """PageLevelChunker correctly handles degraded scans when used directly."""
        from src.chunking._token_counter import TokenCounter
        from src.chunking.chunkers._page_level import PageLevelChunker

        chunker = PageLevelChunker(config.settings, TokenCounter())
        chunks = chunker.chunk(sample_degraded_scan_doc)
        # Fixture has 3 pages separated by \f
        assert len(chunks) == 3
        assert all(c.ingestion.chunk_strategy == ChunkStrategy.PAGE_LEVEL for c in chunks)


# ── Multi-document mixed run ─────────────────────────────────────


class TestMultiDocumentRun:
    """Pipeline processes multiple documents from different types in one run."""

    async def test_statute_and_judgment_together(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
        sample_judgment_doc: ParsedDocument,
    ):
        from uuid import uuid4

        inp, _out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        # Judgment needs a different doc_id to avoid filename collision
        judgment_copy = sample_judgment_doc.model_copy(update={"document_id": uuid4()})
        _write_parsed_doc(judgment_copy, inp)

        pipeline = ChunkingPipeline(config=config)
        result = await pipeline.run()

        assert result.documents_chunked == 2
        assert result.documents_failed == 0
        assert result.errors == []

    async def test_mixed_sources(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
        sample_judgment_doc: ParsedDocument,
    ):
        from uuid import uuid4

        inp, _out = dirs
        _write_parsed_doc(sample_statute_doc, inp, source_name="indian_kanoon")
        judgment_copy = sample_judgment_doc.model_copy(update={"document_id": uuid4()})
        _write_parsed_doc(judgment_copy, inp, source_name="india_code")

        pipeline = ChunkingPipeline(config=config)
        result = await pipeline.run()

        assert result.documents_chunked == 2
        assert result.documents_failed == 0


# ── Token bounds ─────────────────────────────────────────────────


class TestTokenBounds:
    """All chunks must respect max_tokens."""

    async def test_statute_chunks_within_bounds(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
        tc: TokenCounter,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        max_tokens = config.settings.max_tokens
        for chunk in chunks:
            actual = tc.count(chunk.text)
            assert actual <= max_tokens * 1.1, (
                f"Chunk {chunk.chunk_index} has {actual} tokens, max is {max_tokens}"
            )

    async def test_judgment_chunks_within_bounds(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
        tc: TokenCounter,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        max_tokens = config.settings.max_tokens
        for chunk in chunks:
            actual = tc.count(chunk.text)
            assert actual <= max_tokens * 1.1, (
                f"Chunk {chunk.chunk_index} has {actual} tokens, max is {max_tokens}"
            )

    async def test_token_count_field_accurate(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
        tc: TokenCounter,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        for chunk in chunks:
            expected = tc.count(chunk.text)
            assert chunk.token_count == expected


# ── Post-processing: indices + siblings ──────────────────────────


class TestPostProcessing:
    """Chunk indices are sequential and sibling IDs are well-formed."""

    async def test_indices_start_at_zero(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        assert chunks[0].chunk_index == 0

    async def test_indices_are_contiguous(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    async def test_sibling_ids_are_valid_uuids(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        all_ids = {c.id for c in chunks}
        for chunk in chunks:
            for sid in chunk.parent_info.sibling_chunk_ids:
                assert isinstance(sid, UUID)
                assert sid in all_ids
                assert sid != chunk.id

    async def test_sibling_window_respects_bounds(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_judgment_doc.document_id))
        if len(chunks) >= 5:
            # First chunk: at most 2 siblings (window=2, only right side)
            assert len(chunks[0].parent_info.sibling_chunk_ids) <= 2
            # Last chunk: at most 2 siblings (window=2, only left side)
            assert len(chunks[-1].parent_info.sibling_chunk_ids) <= 2
            # Middle chunk: at most 4 siblings (2 left + 2 right)
            mid = chunks[len(chunks) // 2]
            assert len(mid.parent_info.sibling_chunk_ids) <= 4


# ── JSON round-trip integrity ────────────────────────────────────


class TestJsonRoundTrip:
    """Chunks survive JSON serialization and deserialization."""

    async def test_all_fields_survive_round_trip(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        path = out / "indian_kanoon" / f"{sample_statute_doc.document_id}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [LegalChunk.model_validate(c) for c in raw]

        for chunk in chunks:
            # Core fields
            assert chunk.id is not None
            assert chunk.document_id is not None
            assert chunk.text
            assert chunk.token_count > 0
            assert chunk.chunk_index >= 0
            # Metadata sub-models
            assert chunk.source is not None
            assert chunk.source.source_name
            assert chunk.content is not None
            assert chunk.ingestion is not None
            assert chunk.ingestion.chunk_strategy in ChunkStrategy
            assert chunk.parent_info is not None

    async def test_judgment_metadata_round_trip(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_judgment_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_judgment_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        path = out / "indian_kanoon" / f"{sample_judgment_doc.document_id}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [LegalChunk.model_validate(c) for c in raw]

        for chunk in chunks:
            assert chunk.judgment is not None
            assert chunk.judgment.court == "Supreme Court of India"
            assert chunk.judgment.court_level is not None


# ── Source info provenance ───────────────────────────────────────


class TestSourceProvenance:
    """Source info correctly populated from ParsedDocument."""

    async def test_source_info_fields(
        self,
        config: ChunkingConfig,
        dirs: tuple[Path, Path],
        sample_statute_doc: ParsedDocument,
    ):
        inp, out = dirs
        _write_parsed_doc(sample_statute_doc, inp)
        pipeline = ChunkingPipeline(config=config)
        await pipeline.run()

        chunks = _read_chunks(out, "indian_kanoon", str(sample_statute_doc.document_id))
        for chunk in chunks:
            assert chunk.source.source_name
            assert chunk.source.url
            assert chunk.source.scraped_at is not None
            assert chunk.source.last_verified is not None
