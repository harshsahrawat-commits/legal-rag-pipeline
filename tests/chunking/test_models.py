from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from src.acquisition._models import DocumentType, SourceType
from src.chunking._models import (
    AmendmentRecord,
    ChunkingConfig,
    ChunkingResult,
    ChunkingSettings,
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    CourtHierarchy,
    IngestionMetadata,
    JudgmentMetadata,
    LegalChunk,
    ParentDocumentInfo,
    SourceInfo,
    StatuteMetadata,
    TemporalStatus,
)


class TestEnums:
    def test_chunk_type_values(self):
        assert ChunkType.STATUTORY_TEXT == "statutory_text"
        assert ChunkType.DEFINITION == "definition"
        assert ChunkType.RAPTOR_SUMMARY == "raptor_summary"

    def test_chunk_strategy_values(self):
        assert ChunkStrategy.STRUCTURE_BOUNDARY == "structure_boundary"
        assert ChunkStrategy.PAGE_LEVEL == "page_level"
        assert ChunkStrategy.QUIM == "quim"

    def test_temporal_status_values(self):
        assert TemporalStatus.IN_FORCE == "in_force"
        assert TemporalStatus.SUPERSEDED == "superseded"

    def test_court_hierarchy_ordering(self):
        assert CourtHierarchy.SUPREME_COURT < CourtHierarchy.HIGH_COURT
        assert CourtHierarchy.HIGH_COURT < CourtHierarchy.DISTRICT_COURT
        assert CourtHierarchy.DISTRICT_COURT < CourtHierarchy.TRIBUNAL
        assert CourtHierarchy.TRIBUNAL < CourtHierarchy.QUASI_JUDICIAL

    def test_court_hierarchy_is_int(self):
        assert CourtHierarchy.SUPREME_COURT == 1
        assert isinstance(CourtHierarchy.SUPREME_COURT.value, int)


class TestSourceInfo:
    def test_create(self):
        now = datetime.now(UTC)
        si = SourceInfo(
            url="https://example.com", source_name="Test", scraped_at=now, last_verified=now
        )
        assert si.url == "https://example.com"
        assert si.source_name == "Test"


class TestStatuteMetadata:
    def test_defaults(self):
        sm = StatuteMetadata(act_name="IPC")
        assert sm.is_in_force is True
        assert sm.temporal_status == TemporalStatus.IN_FORCE
        assert sm.amendment_history == []

    def test_with_amendment(self):
        am = AmendmentRecord(
            amending_act="Criminal Law Amendment Act, 2013",
            date=date(2013, 4, 2),
            nature="substitution",
        )
        sm = StatuteMetadata(act_name="IPC", amendment_history=[am])
        assert len(sm.amendment_history) == 1
        assert sm.amendment_history[0].nature == "substitution"


class TestJudgmentMetadata:
    def test_create(self):
        jm = JudgmentMetadata(
            case_citation="AIR 2024 SC 1500",
            court="Supreme Court of India",
            court_level=CourtHierarchy.SUPREME_COURT,
        )
        assert jm.case_citation == "AIR 2024 SC 1500"
        assert jm.court_level == CourtHierarchy.SUPREME_COURT
        assert jm.is_overruled is False

    def test_defaults(self):
        jm = JudgmentMetadata(
            case_citation="test",
            court="test",
            court_level=CourtHierarchy.HIGH_COURT,
        )
        assert jm.alt_citations == []
        assert jm.judge_names == []
        assert jm.followed_in == []


class TestContentMetadata:
    def test_defaults(self):
        cm = ContentMetadata()
        assert cm.language == "en"
        assert cm.has_hindi is False
        assert cm.sections_cited == []

    def test_with_citations(self):
        cm = ContentMetadata(
            sections_cited=["Section 302 IPC"],
            acts_cited=["Indian Penal Code, 1860"],
            cases_cited=["AIR 1962 SC 605"],
        )
        assert len(cm.sections_cited) == 1


class TestIngestionMetadata:
    def test_create(self):
        im = IngestionMetadata(
            ingested_at=datetime.now(UTC),
            parser="docling_pdf",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        )
        assert im.contextualized is False
        assert im.requires_manual_review is False
        assert im.raptor_level is None


class TestParentDocumentInfo:
    def test_defaults(self):
        pdi = ParentDocumentInfo()
        assert pdi.parent_chunk_id is None
        assert pdi.sibling_chunk_ids == []
        assert pdi.judgment_header_chunk_id is None


class TestLegalChunk:
    def test_create_statute_chunk(self):
        now = datetime.now(UTC)
        doc_id = uuid4()
        chunk = LegalChunk(
            document_id=doc_id,
            text="Section 10. All agreements are contracts...",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=42,
            source=SourceInfo(
                url="https://example.com",
                source_name="Indian Kanoon",
                scraped_at=now,
                last_verified=now,
            ),
            statute=StatuteMetadata(act_name="Indian Contract Act"),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="html_indian_kanoon",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        )
        assert chunk.document_type == DocumentType.STATUTE
        assert chunk.statute is not None
        assert chunk.judgment is None
        assert isinstance(chunk.id, UUID)

    def test_create_judgment_chunk(self):
        now = datetime.now(UTC)
        chunk = LegalChunk(
            document_id=uuid4(),
            text="For the foregoing reasons, we hold...",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.HOLDING,
            chunk_index=3,
            token_count=25,
            source=SourceInfo(
                url="https://example.com",
                source_name="Indian Kanoon",
                scraped_at=now,
                last_verified=now,
            ),
            judgment=JudgmentMetadata(
                case_citation="AIR 2024 SC 1500",
                court="Supreme Court of India",
                court_level=CourtHierarchy.SUPREME_COURT,
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="html_indian_kanoon",
                chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL,
            ),
        )
        assert chunk.judgment is not None
        assert chunk.statute is None

    def test_json_round_trip(self):
        now = datetime.now(UTC)
        chunk = LegalChunk(
            document_id=uuid4(),
            text="Test chunk text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=3,
            source=SourceInfo(
                url="https://example.com",
                source_name="Test",
                scraped_at=now,
                last_verified=now,
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.PAGE_LEVEL,
            ),
        )
        json_str = chunk.model_dump_json()
        restored = LegalChunk.model_validate_json(json_str)
        assert restored.id == chunk.id
        assert restored.text == chunk.text
        assert restored.chunk_type == ChunkType.STATUTORY_TEXT

    def test_auto_generates_uuid(self):
        now = datetime.now(UTC)
        c1 = LegalChunk(
            document_id=uuid4(),
            text="a",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=1,
            source=SourceInfo(url="u", source_name="s", scraped_at=now, last_verified=now),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="p", chunk_strategy=ChunkStrategy.PAGE_LEVEL
            ),
        )
        c2 = LegalChunk(
            document_id=uuid4(),
            text="b",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=1,
            token_count=1,
            source=SourceInfo(url="u", source_name="s", scraped_at=now, last_verified=now),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="p", chunk_strategy=ChunkStrategy.PAGE_LEVEL
            ),
        )
        assert c1.id != c2.id


class TestChunkingSettings:
    def test_defaults(self):
        s = ChunkingSettings()
        assert s.max_tokens == 1500
        assert s.ocr_confidence_threshold == 0.80
        assert s.min_section_count_statute == 3
        assert s.similarity_threshold == 0.75

    def test_override(self):
        s = ChunkingSettings(max_tokens=1000, overlap_tokens=100)
        assert s.max_tokens == 1000
        assert s.overlap_tokens == 100


class TestChunkingConfig:
    def test_default_settings(self):
        cfg = ChunkingConfig()
        assert cfg.settings.max_tokens == 1500


class TestChunkingResult:
    def test_defaults(self):
        r = ChunkingResult()
        assert r.documents_found == 0
        assert r.chunks_created == 0
        assert r.errors == []
        assert r.finished_at is None

    def test_with_values(self):
        r = ChunkingResult(
            source_type=SourceType.INDIAN_KANOON,
            documents_found=10,
            documents_chunked=8,
            documents_skipped=1,
            documents_failed=1,
            chunks_created=120,
            errors=["Failed: doc1"],
        )
        assert r.documents_chunked == 8
        assert r.chunks_created == 120
