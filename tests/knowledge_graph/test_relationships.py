"""Tests for RelationshipBuilder."""

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
)
from src.knowledge_graph._extractors import EntityExtractor
from src.knowledge_graph._models import ExtractedEntities
from src.knowledge_graph._relationships import RelationshipBuilder


@pytest.fixture()
def builder() -> RelationshipBuilder:
    return RelationshipBuilder()


@pytest.fixture()
def extractor() -> EntityExtractor:
    return EntityExtractor()


@pytest.fixture()
def now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture()
def source_info(now) -> SourceInfo:
    return SourceInfo(
        url="https://example.com", source_name="test", scraped_at=now, last_verified=now
    )


class TestBuildFromChunk:
    def test_routes_statute(self, builder, extractor, source_info, now, statute_chunk) -> None:
        entities = extractor.extract_from_chunk(statute_chunk)
        rels = builder.build_from_chunk(statute_chunk, entities)
        rel_types = {r.rel_type for r in rels}
        assert "CONTAINS" in rel_types

    def test_routes_judgment(self, builder, extractor, source_info, now, judgment_chunk) -> None:
        entities = extractor.extract_from_chunk(judgment_chunk)
        rels = builder.build_from_chunk(judgment_chunk, entities)
        rel_types = {r.rel_type for r in rels}
        assert "FILED_IN" in rel_types

    def test_unknown_type_returns_empty(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="notification",
            document_type=DocumentType.NOTIFICATION,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=1,
            source=source_info,
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.PAGE_LEVEL
            ),
        )
        rels = builder.build_from_chunk(chunk, ExtractedEntities())
        assert rels == []


class TestStatuteRelationships:
    def test_contains_relationship(self, builder, extractor, statute_chunk) -> None:
        entities = extractor.extract_from_chunk(statute_chunk)
        rels = builder.build_statute_relationships(statute_chunk, entities)
        contains = [r for r in rels if r.rel_type == "CONTAINS"]
        assert len(contains) == 1
        assert contains[0].from_key == {"name": "Indian Penal Code"}
        assert contains[0].to_key == {"parent_act": "Indian Penal Code", "number": "302"}

    def test_has_version_relationship(self, builder, extractor, statute_chunk) -> None:
        entities = extractor.extract_from_chunk(statute_chunk)
        rels = builder.build_statute_relationships(statute_chunk, entities)
        has_version = [r for r in rels if r.rel_type == "HAS_VERSION"]
        assert len(has_version) == 1
        assert has_version[0].from_label == "Section"
        assert has_version[0].to_label == "SectionVersion"

    def test_amendment_relationship(self, builder, extractor, statute_chunk) -> None:
        entities = extractor.extract_from_chunk(statute_chunk)
        rels = builder.build_statute_relationships(statute_chunk, entities)
        amends = [r for r in rels if r.rel_type == "AMENDS"]
        assert len(amends) == 1
        assert amends[0].from_label == "Amendment"

    def test_insertion_amendment_type(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="inserted text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=2,
            source=source_info,
            statute=StatuteMetadata(
                act_name="IPC",
                section_number="376A",
                amendment_history=[
                    AmendmentRecord(
                        amending_act="CLA 2013", date=date(2013, 4, 2), nature="insertion"
                    )
                ],
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY
            ),
        )
        extractor = EntityExtractor()
        entities = extractor.extract_from_chunk(chunk)
        rels = builder.build_statute_relationships(chunk, entities)
        inserts = [r for r in rels if r.rel_type == "INSERTS"]
        assert len(inserts) == 1

    def test_omission_amendment_type(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="omitted",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=1,
            source=source_info,
            statute=StatuteMetadata(
                act_name="IPC",
                section_number="377",
                amendment_history=[
                    AmendmentRecord(
                        amending_act="SC Order 2018", date=date(2018, 9, 6), nature="omission"
                    )
                ],
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY
            ),
        )
        extractor = EntityExtractor()
        entities = extractor.extract_from_chunk(chunk)
        rels = builder.build_statute_relationships(chunk, entities)
        omits = [r for r in rels if r.rel_type == "OMITS"]
        assert len(omits) == 1

    def test_repeals_relationship(self, builder, extractor, repealed_statute_chunk) -> None:
        entities = extractor.extract_from_chunk(repealed_statute_chunk)
        rels = builder.build_statute_relationships(repealed_statute_chunk, entities)
        repeals = [r for r in rels if r.rel_type == "REPEALS"]
        assert len(repeals) == 1
        assert repeals[0].from_key == {"name": "Bharatiya Nagarik Suraksha Sanhita"}
        assert repeals[0].to_key == {"name": "Code of Criminal Procedure"}

    def test_references_relationship(self, builder, extractor, statute_chunk) -> None:
        # statute_chunk has sections_cited=["Section 300, IPC"]
        entities = extractor.extract_from_chunk(statute_chunk)
        rels = builder.build_statute_relationships(statute_chunk, entities)
        refs = [r for r in rels if r.rel_type == "REFERENCES"]
        assert len(refs) == 1
        assert refs[0].to_key == {"parent_act": "Indian Penal Code", "number": "300"}

    def test_no_self_reference(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=1,
            source=source_info,
            statute=StatuteMetadata(act_name="Indian Penal Code", section_number="302"),
            content=ContentMetadata(sections_cited=["Section 302, IPC"]),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY
            ),
        )
        extractor = EntityExtractor()
        entities = extractor.extract_from_chunk(chunk)
        rels = builder.build_statute_relationships(chunk, entities)
        refs = [r for r in rels if r.rel_type == "REFERENCES"]
        assert refs == []

    def test_defines_relationship(self, builder, extractor, definition_chunk) -> None:
        entities = extractor.extract_from_chunk(definition_chunk)
        rels = builder.build_statute_relationships(definition_chunk, entities)
        defines = [r for r in rels if r.rel_type == "DEFINES"]
        assert len(defines) == 2
        names = {d.to_key["name"] for d in defines}
        assert "mens rea" in names
        assert "actus reus" in names

    def test_no_statute_metadata_returns_empty(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="text",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=1,
            source=source_info,
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY
            ),
        )
        rels = builder.build_statute_relationships(chunk, ExtractedEntities())
        assert rels == []


class TestJudgmentRelationships:
    def test_filed_in(self, builder, extractor, judgment_chunk) -> None:
        entities = extractor.extract_from_chunk(judgment_chunk)
        rels = builder.build_judgment_relationships(judgment_chunk, entities)
        filed = [r for r in rels if r.rel_type == "FILED_IN"]
        assert len(filed) == 1
        assert filed[0].to_key == {"name": "Supreme Court of India"}

    def test_decided_by(self, builder, extractor, judgment_chunk) -> None:
        entities = extractor.extract_from_chunk(judgment_chunk)
        rels = builder.build_judgment_relationships(judgment_chunk, entities)
        decided = [r for r in rels if r.rel_type == "DECIDED_BY"]
        assert len(decided) == 2

    def test_cites_section(self, builder, extractor, judgment_chunk) -> None:
        entities = extractor.extract_from_chunk(judgment_chunk)
        rels = builder.build_judgment_relationships(judgment_chunk, entities)
        cites = [r for r in rels if r.rel_type == "CITES_SECTION"]
        assert len(cites) == 2  # Section 302 and Section 300

    def test_interprets(self, builder, extractor, judgment_chunk) -> None:
        entities = extractor.extract_from_chunk(judgment_chunk)
        rels = builder.build_judgment_relationships(judgment_chunk, entities)
        interprets = [r for r in rels if r.rel_type == "INTERPRETS"]
        assert len(interprets) == 2

    def test_cites_case(self, builder, extractor, judgment_chunk) -> None:
        entities = extractor.extract_from_chunk(judgment_chunk)
        rels = builder.build_judgment_relationships(judgment_chunk, entities)
        cites = [r for r in rels if r.rel_type == "CITES_CASE"]
        assert len(cites) == 1
        assert cites[0].to_key == {"citation": "(2020) 3 SCC 200"}

    def test_overrules(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="text",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.HOLDING,
            chunk_index=0,
            token_count=1,
            source=source_info,
            judgment=JudgmentMetadata(
                case_citation="(2024) 1 SCC 100",
                court="SC",
                court_level=CourtHierarchy.SUPREME_COURT,
                is_overruled=True,
                overruled_by="(2025) 1 SCC 50",
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL
            ),
        )
        extractor = EntityExtractor()
        entities = extractor.extract_from_chunk(chunk)
        rels = builder.build_judgment_relationships(chunk, entities)
        overrules = [r for r in rels if r.rel_type == "OVERRULES"]
        assert len(overrules) == 1
        assert overrules[0].from_key == {"citation": "(2025) 1 SCC 50"}
        assert overrules[0].to_key == {"citation": "(2024) 1 SCC 100"}

    def test_follows_and_distinguishes(self, builder, source_info, now) -> None:
        chunk = LegalChunk(
            document_id=uuid4(),
            text="text",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.HOLDING,
            chunk_index=0,
            token_count=1,
            source=source_info,
            judgment=JudgmentMetadata(
                case_citation="(2024) 1 SCC 100",
                court="SC",
                court_level=CourtHierarchy.SUPREME_COURT,
                followed_in=["(2025) 2 SCC 10"],
                distinguished_in=["(2025) 3 SCC 20"],
            ),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL
            ),
        )
        extractor = EntityExtractor()
        entities = extractor.extract_from_chunk(chunk)
        rels = builder.build_judgment_relationships(chunk, entities)
        follows = [r for r in rels if r.rel_type == "FOLLOWS"]
        assert len(follows) == 1
        distinguishes = [r for r in rels if r.rel_type == "DISTINGUISHES"]
        assert len(distinguishes) == 1

    def test_no_judgment_metadata_returns_empty(self, builder, source_info, now) -> None:
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
                ingested_at=now, parser="test", chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL
            ),
        )
        rels = builder.build_judgment_relationships(chunk, ExtractedEntities())
        assert rels == []


class TestParseSectionRef:
    def test_comma_separated(self) -> None:
        result = RelationshipBuilder._parse_section_ref("Section 302, IPC")
        assert result == ("Indian Penal Code", "302")

    def test_of_the(self) -> None:
        result = RelationshipBuilder._parse_section_ref("Section 10 of the Indian Contract Act")
        assert result == ("Indian Contract Act", "10")

    def test_of_without_the(self) -> None:
        result = RelationshipBuilder._parse_section_ref("Section 5 of Indian Evidence Act")
        assert result == ("Indian Evidence Act", "5")

    def test_alphanumeric_section(self) -> None:
        result = RelationshipBuilder._parse_section_ref("Section 376A, IPC")
        assert result == ("Indian Penal Code", "376A")

    def test_abbreviation_expansion(self) -> None:
        result = RelationshipBuilder._parse_section_ref("Section 161, CrPC")
        assert result == ("Code of Criminal Procedure", "161")

    def test_unparseable_returns_none(self) -> None:
        assert RelationshipBuilder._parse_section_ref("random text") is None

    def test_empty_returns_none(self) -> None:
        assert RelationshipBuilder._parse_section_ref("") is None


class TestAmendmentRelType:
    def test_substitution(self) -> None:
        assert RelationshipBuilder._amendment_rel_type("substitution") == "AMENDS"

    def test_insertion(self) -> None:
        assert RelationshipBuilder._amendment_rel_type("insertion") == "INSERTS"

    def test_omission(self) -> None:
        assert RelationshipBuilder._amendment_rel_type("omission") == "OMITS"

    def test_repeal(self) -> None:
        assert RelationshipBuilder._amendment_rel_type("repeal") == "OMITS"

    def test_default(self) -> None:
        assert RelationshipBuilder._amendment_rel_type("modification") == "AMENDS"
