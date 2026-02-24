from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.acquisition._models import SourceType
from src.enrichment._models import (
    EnrichmentConfig,
    EnrichmentResult,
    EnrichmentSettings,
    QuIMDocument,
    QuIMEntry,
)

# --- QuIMEntry ---


class TestQuIMEntry:
    def test_create(self):
        chunk_id = uuid4()
        doc_id = uuid4()
        entry = QuIMEntry(
            chunk_id=chunk_id,
            document_id=doc_id,
            questions=["What is the penalty?", "When does this apply?"],
            model="claude-haiku-4-5-20251001",
        )
        assert entry.chunk_id == chunk_id
        assert entry.document_id == doc_id
        assert len(entry.questions) == 2
        assert entry.model == "claude-haiku-4-5-20251001"

    def test_defaults(self):
        entry = QuIMEntry(
            chunk_id=uuid4(),
            document_id=uuid4(),
            questions=[],
        )
        assert entry.model == ""
        assert entry.generated_at is not None

    def test_json_round_trip(self):
        entry = QuIMEntry(
            chunk_id=uuid4(),
            document_id=uuid4(),
            questions=["Q1?", "Q2?", "Q3?"],
            model="test-model",
        )
        json_str = entry.model_dump_json()
        restored = QuIMEntry.model_validate_json(json_str)
        assert restored.chunk_id == entry.chunk_id
        assert restored.questions == entry.questions
        assert restored.model == entry.model

    def test_many_questions(self):
        entry = QuIMEntry(
            chunk_id=uuid4(),
            document_id=uuid4(),
            questions=[f"Question {i}?" for i in range(10)],
        )
        assert len(entry.questions) == 10


# --- QuIMDocument ---


class TestQuIMDocument:
    def test_create_empty(self):
        doc_id = uuid4()
        doc = QuIMDocument(document_id=doc_id, model="test-model")
        assert doc.document_id == doc_id
        assert doc.entries == []
        assert doc.model == "test-model"

    def test_create_with_entries(self):
        doc_id = uuid4()
        entries = [
            QuIMEntry(
                chunk_id=uuid4(),
                document_id=doc_id,
                questions=["Q1?"],
            ),
            QuIMEntry(
                chunk_id=uuid4(),
                document_id=doc_id,
                questions=["Q2?", "Q3?"],
            ),
        ]
        doc = QuIMDocument(document_id=doc_id, entries=entries)
        assert len(doc.entries) == 2

    def test_defaults(self):
        doc = QuIMDocument(document_id=uuid4())
        assert doc.entries == []
        assert doc.model == ""
        assert doc.generated_at is not None

    def test_json_round_trip(self):
        doc_id = uuid4()
        doc = QuIMDocument(
            document_id=doc_id,
            entries=[
                QuIMEntry(
                    chunk_id=uuid4(),
                    document_id=doc_id,
                    questions=["Q1?", "Q2?"],
                    model="test",
                ),
            ],
            model="test",
        )
        json_str = doc.model_dump_json()
        restored = QuIMDocument.model_validate_json(json_str)
        assert restored.document_id == doc.document_id
        assert len(restored.entries) == 1
        assert restored.entries[0].questions == ["Q1?", "Q2?"]

    def test_entry_document_ids_not_enforced(self):
        """QuIMDocument doesn't enforce matching doc IDs (pipeline responsibility)."""
        doc = QuIMDocument(
            document_id=uuid4(),
            entries=[
                QuIMEntry(
                    chunk_id=uuid4(),
                    document_id=uuid4(),  # Different from parent
                    questions=["Q?"],
                ),
            ],
        )
        assert len(doc.entries) == 1


# --- EnrichmentSettings ---


class TestEnrichmentSettings:
    def test_defaults(self):
        s = EnrichmentSettings()
        assert s.input_dir == Path("data/chunks")
        assert s.output_dir == Path("data/enriched")
        assert s.parsed_dir == Path("data/parsed")
        assert s.model == "claude-haiku-4-5-20251001"
        assert s.max_tokens_response == 512
        assert s.concurrency == 5
        assert s.quim_questions_per_chunk == 5
        assert s.context_window_tokens == 180_000
        assert s.document_window_overlap_tokens == 500
        assert s.skip_manual_review_chunks is False

    def test_override_all(self):
        s = EnrichmentSettings(
            input_dir=Path("/tmp/in"),
            output_dir=Path("/tmp/out"),
            parsed_dir=Path("/tmp/parsed"),
            model="claude-sonnet-4-6-20250514",
            max_tokens_response=1024,
            concurrency=10,
            quim_questions_per_chunk=3,
            context_window_tokens=100_000,
            document_window_overlap_tokens=1000,
            skip_manual_review_chunks=True,
        )
        assert s.model == "claude-sonnet-4-6-20250514"
        assert s.concurrency == 10
        assert s.quim_questions_per_chunk == 3
        assert s.skip_manual_review_chunks is True

    def test_partial_override(self):
        s = EnrichmentSettings(concurrency=20)
        assert s.concurrency == 20
        assert s.model == "claude-haiku-4-5-20251001"  # Default preserved

    def test_paths_are_path_objects(self):
        s = EnrichmentSettings(input_dir="custom/input")
        assert isinstance(s.input_dir, Path)
        assert s.input_dir == Path("custom/input")


# --- EnrichmentConfig ---


class TestEnrichmentConfig:
    def test_default_settings(self):
        cfg = EnrichmentConfig()
        assert cfg.settings.model == "claude-haiku-4-5-20251001"
        assert cfg.settings.concurrency == 5

    def test_custom_settings(self):
        cfg = EnrichmentConfig(settings=EnrichmentSettings(concurrency=20))
        assert cfg.settings.concurrency == 20

    def test_from_dict(self):
        cfg = EnrichmentConfig.model_validate(
            {"settings": {"model": "test-model", "concurrency": 3}}
        )
        assert cfg.settings.model == "test-model"
        assert cfg.settings.concurrency == 3

    def test_empty_dict_uses_defaults(self):
        cfg = EnrichmentConfig.model_validate({})
        assert cfg.settings.max_tokens_response == 512


# --- EnrichmentResult ---


class TestEnrichmentResult:
    def test_defaults(self):
        r = EnrichmentResult()
        assert r.source_type is None
        assert r.documents_found == 0
        assert r.documents_enriched == 0
        assert r.documents_skipped == 0
        assert r.documents_failed == 0
        assert r.chunks_contextualized == 0
        assert r.chunks_quim_generated == 0
        assert r.errors == []
        assert r.finished_at is None
        assert r.started_at is not None

    def test_with_values(self):
        r = EnrichmentResult(
            source_type=SourceType.INDIAN_KANOON,
            documents_found=10,
            documents_enriched=8,
            documents_skipped=1,
            documents_failed=1,
            chunks_contextualized=150,
            chunks_quim_generated=140,
            errors=["Failed: doc1"],
        )
        assert r.documents_enriched == 8
        assert r.chunks_contextualized == 150
        assert r.chunks_quim_generated == 140
        assert len(r.errors) == 1

    def test_json_round_trip(self):
        r = EnrichmentResult(
            documents_found=5,
            documents_enriched=3,
            chunks_contextualized=20,
            finished_at=datetime.now(UTC),
        )
        json_str = r.model_dump_json()
        restored = EnrichmentResult.model_validate_json(json_str)
        assert restored.documents_found == 5
        assert restored.chunks_contextualized == 20
        assert restored.finished_at is not None

    def test_errors_accumulate(self):
        r = EnrichmentResult()
        r.errors.append("error 1")
        r.errors.append("error 2")
        assert len(r.errors) == 2
