"""End-to-end integration tests for the enrichment pipeline.

All LLM calls are mocked — no real API calls.
Tests validate data flow integrity, output format, and cross-stage behavior.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime
from typing import TYPE_CHECKING

from src.chunking._models import LegalChunk
from src.enrichment._models import EnrichmentConfig, EnrichmentSettings, QuIMDocument
from src.enrichment.pipeline import EnrichmentPipeline
from tests.enrichment.conftest import make_mock_provider

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument

_CTX_RESPONSE = (
    "This chunk is from the Indian Contract Act, Section 10, dealing with valid contracts."
)
_QUIM_RESPONSE = (
    "What constitutes a valid contract under Section 10?\n"
    "Who is competent to enter into a contract?\n"
    "What does free consent mean in the context of contracts?"
)


def _setup_test_files(
    tmp_path: Path,
    chunks: list[LegalChunk],
    parsed_doc: ParsedDocument,
    source: str = "indian_kanoon",
    filename: str = "doc1.json",
) -> tuple[Path, Path]:
    """Write chunk and parsed files, return (chunk_path, parsed_path)."""
    # Chunks
    chunk_dir = tmp_path / "chunks" / source
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / filename
    data = [c.model_dump(mode="json") for c in chunks]
    chunk_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Parsed doc
    parsed_dir = tmp_path / "parsed" / source
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = parsed_dir / filename
    parsed_path.write_text(parsed_doc.model_dump_json(indent=2), encoding="utf-8")

    return chunk_path, parsed_path


def _make_pipeline(tmp_path: Path) -> EnrichmentPipeline:
    settings = EnrichmentSettings(
        input_dir=tmp_path / "chunks",
        output_dir=tmp_path / "enriched",
        parsed_dir=tmp_path / "parsed",
        concurrency=2,
    )
    config = EnrichmentConfig(settings=settings)
    pipeline = EnrichmentPipeline(config=config)
    pipeline._contextual._provider = make_mock_provider(_CTX_RESPONSE)
    pipeline._quim._provider = make_mock_provider(_QUIM_RESPONSE)
    return pipeline


class TestContextualRetrievalEndToEnd:
    async def test_statute_chunks_all_contextualized(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(stage="contextual_retrieval")
        assert result.chunks_contextualized == 3

    async def test_contextualized_text_format_in_output(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="contextual_retrieval")

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        enriched = [
            LegalChunk.model_validate(c)
            for c in json.loads(output_path.read_text(encoding="utf-8"))
        ]
        for chunk in enriched:
            assert chunk.contextualized_text is not None
            assert chunk.contextualized_text.startswith(_CTX_RESPONSE)
            assert "\n\n" in chunk.contextualized_text
            # Original text preserved at the end
            assert chunk.contextualized_text.endswith(chunk.text)

    async def test_ingestion_flag_in_saved_file(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="contextual_retrieval")

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        enriched = [
            LegalChunk.model_validate(c)
            for c in json.loads(output_path.read_text(encoding="utf-8"))
        ]
        for chunk in enriched:
            assert chunk.ingestion.contextualized is True

    async def test_chunk_ids_preserved(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        original_ids = [c.id for c in sample_statute_chunks]
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="contextual_retrieval")

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        enriched = [
            LegalChunk.model_validate(c)
            for c in json.loads(output_path.read_text(encoding="utf-8"))
        ]
        enriched_ids = [c.id for c in enriched]
        assert enriched_ids == original_ids

    async def test_json_round_trip_preserves_fields(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="contextual_retrieval")

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        enriched = [
            LegalChunk.model_validate(c)
            for c in json.loads(output_path.read_text(encoding="utf-8"))
        ]
        for chunk in enriched:
            # Core fields survived serialization
            assert chunk.document_type is not None
            assert chunk.chunk_type is not None
            assert chunk.source is not None
            assert chunk.content is not None
            assert chunk.ingestion is not None


class TestQuIMEndToEnd:
    async def test_quim_file_created(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="quim_rag")

        quim_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.quim.json"
        assert quim_path.exists()

    async def test_quim_entries_match_chunk_ids(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="quim_rag")

        quim_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.quim.json"
        quim_doc = QuIMDocument.model_validate_json(quim_path.read_text(encoding="utf-8"))

        entry_chunk_ids = {e.chunk_id for e in quim_doc.entries}
        expected_ids = {c.id for c in sample_statute_chunks}
        assert entry_chunk_ids == expected_ids

    async def test_quim_count_on_chunks(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="quim_rag")

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        enriched = [
            LegalChunk.model_validate(c)
            for c in json.loads(output_path.read_text(encoding="utf-8"))
        ]
        for chunk in enriched:
            assert chunk.ingestion.quim_questions == 3

    async def test_quim_questions_are_strings(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run(stage="quim_rag")

        quim_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.quim.json"
        quim_doc = QuIMDocument.model_validate_json(quim_path.read_text(encoding="utf-8"))
        for entry in quim_doc.entries:
            for q in entry.questions:
                assert isinstance(q, str)
                assert len(q) > 0


class TestBothStagesEndToEnd:
    async def test_both_stages_in_single_run(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.chunks_contextualized == 3
        assert result.chunks_quim_generated > 0

        output_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.json"
        enriched = [
            LegalChunk.model_validate(c)
            for c in json.loads(output_path.read_text(encoding="utf-8"))
        ]
        for chunk in enriched:
            assert chunk.ingestion.contextualized is True
            assert chunk.ingestion.quim_questions > 0

    async def test_quim_file_also_created(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)
        await pipeline.run()

        quim_path = tmp_path / "enriched" / "indian_kanoon" / "doc1.quim.json"
        assert quim_path.exists()


class TestIdempotency:
    async def test_second_run_skips(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)

        r1 = await pipeline.run()
        assert r1.documents_enriched == 1

        r2 = await pipeline.run()
        assert r2.documents_skipped == 1
        assert r2.documents_enriched == 0

    async def test_quim_only_idempotent(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)
        pipeline = _make_pipeline(tmp_path)

        r1 = await pipeline.run(stage="quim_rag")
        assert r1.documents_enriched == 1

        r2 = await pipeline.run(stage="quim_rag")
        assert r2.documents_skipped == 1


class TestErrorIsolation:
    async def test_corrupt_input_does_not_abort_pipeline(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """One bad file + one good file: pipeline processes the good one."""
        # Good file
        _setup_test_files(tmp_path, sample_statute_chunks, sample_parsed_doc)

        # Bad file
        bad_dir = tmp_path / "chunks" / "indian_kanoon"
        (bad_dir / "bad.json").write_text("not json", encoding="utf-8")

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()
        assert result.documents_found == 2
        assert result.documents_enriched == 1
        assert result.documents_failed == 1

    async def test_missing_parsed_doc_uses_empty_context(
        self,
        tmp_path: Path,
        sample_statute_chunks: list[LegalChunk],
    ):
        """Enrichment works even without parsed doc (empty context)."""
        # Write chunks but NOT parsed doc
        chunk_dir = tmp_path / "chunks" / "indian_kanoon"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        data = [c.model_dump(mode="json") for c in sample_statute_chunks]
        (chunk_dir / "doc1.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(stage="contextual_retrieval")
        assert result.documents_enriched == 1
