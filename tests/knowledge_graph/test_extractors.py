"""Tests for EntityExtractor."""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from src.acquisition._models import DocumentType
from src.chunking._models import (
    AmendmentRecord,
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    CourtHierarchy,
    IngestionMetadata,
    JudgmentMetadata,
    LegalChunk,
    SourceInfo,
    StatuteMetadata,
    TemporalStatus,
)
from src.knowledge_graph._extractors import EntityExtractor


@pytest.fixture()
def extractor() -> EntityExtractor:
    return EntityExtractor()


@pytest.fixture()
def now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture()
def source_info(now) -> SourceInfo:
    return SourceInfo(
        url="https://example.com",
        source_name="test",
        scraped_at=now,
        last_verified=now,
    )


def _make_statute_chunk(
    source_info,
    now,
    act_name="Indian Penal Code",
    act_number="Act No. 45 of 1860",
    section_number="302",
    text="Murder punishment text",
    chunk_type=ChunkType.STATUTORY_TEXT,
    temporal_status=TemporalStatus.IN_FORCE,
    is_in_force=True,
    amendment_history=None,
    legal_concepts=None,
) -> LegalChunk:
    return LegalChunk(
        document_id=uuid4(),
        text=text,
        document_type=DocumentType.STATUTE,
        chunk_type=chunk_type,
        chunk_index=0,
        token_count=5,
        source=source_info,
        statute=StatuteMetadata(
            act_name=act_name,
            act_number=act_number,
            section_number=section_number,
            chapter="XVI",
            date_enacted=date(1860, 10, 6),
            date_effective=date(1860, 10, 6),
            is_in_force=is_in_force,
            temporal_status=temporal_status,
            amendment_history=amendment_history or [],
        ),
        content=ContentMetadata(
            legal_concepts=legal_concepts or [],
        ),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="test",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
    )


def _make_judgment_chunk(
    source_info,
    now,
    citation="(2024) 1 SCC 100",
    court="Supreme Court of India",
    court_level=CourtHierarchy.SUPREME_COURT,
    judge_names=None,
    is_overruled=False,
    distinguished_in=None,
) -> LegalChunk:
    return LegalChunk(
        document_id=uuid4(),
        text="Judgment text here",
        document_type=DocumentType.JUDGMENT,
        chunk_type=ChunkType.HOLDING,
        chunk_index=0,
        token_count=4,
        source=source_info,
        judgment=JudgmentMetadata(
            case_citation=citation,
            court=court,
            court_level=court_level,
            bench_type="Division Bench",
            bench_strength=2,
            judge_names=judge_names or ["Justice A"],
            date_decided=date(2024, 1, 15),
            case_type="Criminal Appeal",
            parties_petitioner="State",
            parties_respondent="Accused",
            is_overruled=is_overruled,
            distinguished_in=distinguished_in or [],
        ),
        content=ContentMetadata(),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="test",
            chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL,
        ),
    )


class TestExtractFromChunk:
    def test_routes_statute(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now)
        entities = extractor.extract_from_chunk(chunk)
        assert len(entities.acts) == 1
        assert entities.acts[0].name == "Indian Penal Code"

    def test_routes_judgment(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(source_info, now)
        entities = extractor.extract_from_chunk(chunk)
        assert len(entities.judgments) == 1
        assert entities.judgments[0].citation == "(2024) 1 SCC 100"

    def test_unknown_type_returns_empty(self, extractor, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="notification text",
            document_type=DocumentType.NOTIFICATION,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=3,
            source=source_info,
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.PAGE_LEVEL,
            ),
        )
        entities = extractor.extract_from_chunk(chunk)
        assert entities.acts == []
        assert entities.judgments == []


class TestExtractFromStatuteChunk:
    def test_extracts_act(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now)
        entities = extractor.extract_from_statute_chunk(chunk)
        act = entities.acts[0]
        assert act.name == "Indian Penal Code"
        assert act.number == "Act No. 45 of 1860"
        assert act.year == 1860
        assert act.status == "in_force"

    def test_extracts_section(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now)
        entities = extractor.extract_from_statute_chunk(chunk)
        section = entities.sections[0]
        assert section.number == "302"
        assert section.parent_act == "Indian Penal Code"
        assert section.chapter == "XVI"
        assert section.is_in_force is True
        assert section.chunk_id == chunk.id

    def test_extracts_section_version(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now)
        entities = extractor.extract_from_statute_chunk(chunk)
        sv = entities.section_versions[0]
        assert sv.version_id.startswith("Indian Penal Code:302:")
        assert len(sv.text_hash) == 64  # SHA256 hex
        assert sv.effective_from == date(1860, 10, 6)

    def test_extracts_amendments(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(
            source_info,
            now,
            amendment_history=[
                AmendmentRecord(
                    amending_act="CLA 2013",
                    date=date(2013, 4, 2),
                    nature="substitution",
                    gazette_ref="GSR 123(E)",
                ),
            ],
        )
        entities = extractor.extract_from_statute_chunk(chunk)
        assert len(entities.amendments) == 1
        assert entities.amendments[0].amending_act == "CLA 2013"
        assert entities.amendments[0].nature == "substitution"

    def test_no_section_number_skips_section(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now, section_number=None)
        entities = extractor.extract_from_statute_chunk(chunk)
        assert len(entities.acts) == 1
        assert len(entities.sections) == 0
        assert len(entities.section_versions) == 0

    def test_no_statute_metadata_returns_empty(self, extractor, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="some text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=2,
            source=source_info,
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        )
        entities = extractor.extract_from_statute_chunk(chunk)
        assert entities.acts == []

    def test_repealed_act_status(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(
            source_info,
            now,
            temporal_status=TemporalStatus.REPEALED,
            is_in_force=False,
        )
        entities = extractor.extract_from_statute_chunk(chunk)
        assert entities.acts[0].status == "repealed"

    def test_partially_repealed_status(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(
            source_info,
            now,
            temporal_status=TemporalStatus.PARTIALLY_REPEALED,
        )
        entities = extractor.extract_from_statute_chunk(chunk)
        assert entities.acts[0].status == "partially_repealed"

    def test_definition_chunk_extracts_concepts(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(
            source_info,
            now,
            section_number="2",
            chunk_type=ChunkType.DEFINITION,
            legal_concepts=["mens rea", "actus reus"],
        )
        entities = extractor.extract_from_statute_chunk(chunk)
        assert len(entities.legal_concepts) == 2
        assert entities.legal_concepts[0].name == "mens rea"
        assert entities.legal_concepts[0].definition_source == "Section 2, Indian Penal Code"


class TestExtractFromJudgmentChunk:
    def test_extracts_judgment(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(source_info, now)
        entities = extractor.extract_from_judgment_chunk(chunk)
        j = entities.judgments[0]
        assert j.citation == "(2024) 1 SCC 100"
        assert j.court == "Supreme Court of India"
        assert j.court_level == 1
        assert j.status == "good_law"

    def test_extracts_court(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(source_info, now)
        entities = extractor.extract_from_judgment_chunk(chunk)
        court = entities.courts[0]
        assert court.name == "Supreme Court of India"
        assert court.hierarchy_level == 1

    def test_extracts_judges(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(source_info, now, judge_names=["Justice A", "Justice B"])
        entities = extractor.extract_from_judgment_chunk(chunk)
        assert len(entities.judges) == 2
        assert entities.judges[0].name == "Justice A"
        assert entities.judges[0].courts_served == ["Supreme Court of India"]

    def test_overruled_status(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(source_info, now, is_overruled=True)
        entities = extractor.extract_from_judgment_chunk(chunk)
        assert entities.judgments[0].status == "overruled"

    def test_distinguished_status(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(source_info, now, distinguished_in=["(2025) 1 SCC 50"])
        entities = extractor.extract_from_judgment_chunk(chunk)
        assert entities.judgments[0].status == "distinguished"

    def test_no_judgment_metadata_returns_empty(self, extractor, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="text",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.HOLDING,
            chunk_index=0,
            token_count=1,
            source=source_info,
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL,
            ),
        )
        entities = extractor.extract_from_judgment_chunk(chunk)
        assert entities.judgments == []

    def test_high_court_with_state(self, extractor, source_info, now) -> None:
        chunk = _make_judgment_chunk(
            source_info,
            now,
            court="Bombay High Court",
            court_level=CourtHierarchy.HIGH_COURT,
        )
        entities = extractor.extract_from_judgment_chunk(chunk)
        assert entities.courts[0].hierarchy_level == 2


class TestExtractLegalConcepts:
    def test_definition_chunk(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(
            source_info,
            now,
            section_number="2",
            chunk_type=ChunkType.DEFINITION,
            legal_concepts=["bail", "cognizable offence"],
        )
        concepts = extractor.extract_legal_concepts(chunk)
        assert len(concepts) == 2
        assert concepts[0].name == "bail"

    def test_non_definition_returns_empty(self, extractor, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now, legal_concepts=["something"])
        concepts = extractor.extract_legal_concepts(chunk)
        assert concepts == []

    def test_no_statute_meta_source_is_none(self, extractor, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.DEFINITION,
            chunk_index=0,
            token_count=1,
            source=source_info,
            content=ContentMetadata(legal_concepts=["term"]),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="test",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        )
        concepts = extractor.extract_legal_concepts(chunk)
        assert len(concepts) == 1
        assert concepts[0].definition_source is None


class TestExtractYear:
    def test_from_act_number(self) -> None:
        assert EntityExtractor._extract_year("Act No. 45 of 1860", None) == 1860

    def test_from_act_number_short(self) -> None:
        assert EntityExtractor._extract_year("2023", None) == 2023

    def test_from_date_enacted(self) -> None:
        assert EntityExtractor._extract_year(None, date(2023, 12, 25)) == 2023

    def test_prefers_act_number(self) -> None:
        assert EntityExtractor._extract_year("Act of 1860", date(2023, 1, 1)) == 1860

    def test_none_when_no_info(self) -> None:
        assert EntityExtractor._extract_year(None, None) is None

    def test_no_year_in_string(self) -> None:
        assert EntityExtractor._extract_year("NoYear", None) is None


class TestComputeTextHash:
    def test_deterministic(self) -> None:
        h1 = EntityExtractor._compute_text_hash("hello")
        h2 = EntityExtractor._compute_text_hash("hello")
        assert h1 == h2

    def test_different_texts_differ(self) -> None:
        h1 = EntityExtractor._compute_text_hash("hello")
        h2 = EntityExtractor._compute_text_hash("world")
        assert h1 != h2

    def test_sha256_length(self) -> None:
        h = EntityExtractor._compute_text_hash("test")
        assert len(h) == 64


class TestDetermineActStatus:
    def test_in_force(self, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now, temporal_status=TemporalStatus.IN_FORCE)
        assert EntityExtractor._determine_act_status(chunk.statute) == "in_force"

    def test_repealed(self, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now, temporal_status=TemporalStatus.REPEALED)
        assert EntityExtractor._determine_act_status(chunk.statute) == "repealed"

    def test_amended_is_in_force(self, source_info, now) -> None:
        chunk = _make_statute_chunk(source_info, now, temporal_status=TemporalStatus.AMENDED)
        assert EntityExtractor._determine_act_status(chunk.statute) == "in_force"
