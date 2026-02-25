"""Shared fixtures for knowledge graph tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock
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
from src.knowledge_graph._models import KGSettings


@pytest.fixture()
def kg_settings() -> KGSettings:
    return KGSettings()


@pytest.fixture()
def now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture()
def sample_source_info(now: datetime) -> SourceInfo:
    return SourceInfo(
        url="https://example.com/test",
        source_name="Indian Kanoon",
        scraped_at=now,
        last_verified=now,
    )


@pytest.fixture()
def sample_doc_id():
    return uuid4()


@pytest.fixture()
def statute_chunk(now: datetime, sample_source_info: SourceInfo, sample_doc_id) -> LegalChunk:
    """A statute chunk with full metadata."""
    return LegalChunk(
        document_id=sample_doc_id,
        text="Whoever commits murder shall be punished with death or imprisonment for life.",
        document_type=DocumentType.STATUTE,
        chunk_type=ChunkType.STATUTORY_TEXT,
        chunk_index=0,
        token_count=14,
        source=sample_source_info,
        statute=StatuteMetadata(
            act_name="Indian Penal Code",
            act_number="Act No. 45 of 1860",
            section_number="302",
            chapter="XVI",
            date_enacted=date(1860, 10, 6),
            date_effective=date(1860, 10, 6),
            is_in_force=True,
            temporal_status=TemporalStatus.IN_FORCE,
            amendment_history=[
                AmendmentRecord(
                    amending_act="Criminal Law Amendment Act, 2013",
                    date=date(2013, 4, 2),
                    nature="substitution",
                    gazette_ref="GSR 123(E)",
                ),
            ],
        ),
        content=ContentMetadata(
            sections_cited=["Section 300, IPC"],
            legal_concepts=[],
        ),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="html_indian_kanoon",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
    )


@pytest.fixture()
def definition_chunk(now: datetime, sample_source_info: SourceInfo, sample_doc_id) -> LegalChunk:
    """A definition chunk with legal concepts."""
    return LegalChunk(
        document_id=sample_doc_id,
        text="'Mens rea' means guilty mind. 'Actus reus' means guilty act.",
        document_type=DocumentType.STATUTE,
        chunk_type=ChunkType.DEFINITION,
        chunk_index=1,
        token_count=12,
        source=sample_source_info,
        statute=StatuteMetadata(
            act_name="Indian Penal Code",
            act_number="Act No. 45 of 1860",
            section_number="2",
        ),
        content=ContentMetadata(
            legal_concepts=["mens rea", "actus reus"],
        ),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="html_indian_kanoon",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
    )


@pytest.fixture()
def judgment_chunk(now: datetime, sample_source_info: SourceInfo, sample_doc_id) -> LegalChunk:
    """A judgment chunk with full metadata."""
    return LegalChunk(
        document_id=sample_doc_id,
        text="This Court holds that the conviction under Section 302 is upheld.",
        document_type=DocumentType.JUDGMENT,
        chunk_type=ChunkType.HOLDING,
        chunk_index=0,
        token_count=12,
        source=sample_source_info,
        judgment=JudgmentMetadata(
            case_citation="(2024) 1 SCC 100",
            alt_citations=["AIR 2024 SC 50"],
            court="Supreme Court of India",
            court_level=CourtHierarchy.SUPREME_COURT,
            bench_type="Division Bench",
            bench_strength=2,
            judge_names=["Justice D.Y. Chandrachud", "Justice J.B. Pardiwala"],
            date_decided=date(2024, 1, 15),
            case_type="Criminal Appeal",
            parties_petitioner="State of Maharashtra",
            parties_respondent="Ram Kumar",
        ),
        content=ContentMetadata(
            sections_cited=["Section 302, IPC", "Section 300, IPC"],
            cases_cited=["(2020) 3 SCC 200"],
        ),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="html_indian_kanoon",
            chunk_strategy=ChunkStrategy.JUDGMENT_STRUCTURAL,
        ),
    )


@pytest.fixture()
def repealed_statute_chunk(
    now: datetime, sample_source_info: SourceInfo, sample_doc_id
) -> LegalChunk:
    """A statute chunk from a repealed act."""
    return LegalChunk(
        document_id=sample_doc_id,
        text="This section stands repealed.",
        document_type=DocumentType.STATUTE,
        chunk_type=ChunkType.STATUTORY_TEXT,
        chunk_index=0,
        token_count=5,
        source=sample_source_info,
        statute=StatuteMetadata(
            act_name="Code of Criminal Procedure",
            section_number="154",
            is_in_force=False,
            temporal_status=TemporalStatus.REPEALED,
            date_repealed=date(2024, 7, 1),
            repealed_by="Bharatiya Nagarik Suraksha Sanhita",
        ),
        content=ContentMetadata(),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="html_indian_kanoon",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
    )


@pytest.fixture()
def mock_neo4j_session():
    """Mock Neo4j async session."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=AsyncMock(data=AsyncMock(return_value=[])))
    session.execute_write = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture()
def mock_neo4j_driver(mock_neo4j_session):
    """Mock Neo4j async driver."""
    driver = AsyncMock()
    driver.session = MagicMock(return_value=mock_neo4j_session)
    driver.close = AsyncMock()
    return driver
