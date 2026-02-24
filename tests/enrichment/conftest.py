from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._models import (
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
from src.enrichment._models import EnrichmentSettings
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)


@pytest.fixture()
def enrichment_settings() -> EnrichmentSettings:
    return EnrichmentSettings(concurrency=2)


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
def sample_statute_chunks(now: datetime, sample_source_info: SourceInfo) -> list[LegalChunk]:
    """Three statute chunks for testing enrichment."""
    doc_id = uuid4()
    return [
        LegalChunk(
            document_id=doc_id,
            text="Section 10. All agreements are contracts which are made by free consent of parties.",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=15,
            source=sample_source_info,
            statute=StatuteMetadata(act_name="Indian Contract Act", section_number="10"),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="html_indian_kanoon",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        ),
        LegalChunk(
            document_id=doc_id,
            text="Section 11. Every person is competent to contract who is of the age of majority.",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=1,
            token_count=16,
            source=sample_source_info,
            statute=StatuteMetadata(act_name="Indian Contract Act", section_number="11"),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="html_indian_kanoon",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        ),
        LegalChunk(
            document_id=doc_id,
            text="Section 12. A person is said to be of sound mind for the purpose of making a contract.",
            document_type=DocumentType.STATUTE,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=2,
            token_count=18,
            source=sample_source_info,
            statute=StatuteMetadata(act_name="Indian Contract Act", section_number="12"),
            content=ContentMetadata(),
            ingestion=IngestionMetadata(
                ingested_at=now,
                parser="html_indian_kanoon",
                chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
            ),
        ),
    ]


@pytest.fixture()
def sample_judgment_chunks(now: datetime, sample_source_info: SourceInfo) -> list[LegalChunk]:
    """Two judgment chunks for testing enrichment."""
    doc_id = uuid4()
    return [
        LegalChunk(
            document_id=doc_id,
            text="The appellant filed a criminal appeal challenging conviction under Section 302 IPC.",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.FACTS,
            chunk_index=0,
            token_count=14,
            source=sample_source_info,
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
        ),
        LegalChunk(
            document_id=doc_id,
            text="For the foregoing reasons, the appeal is dismissed.",
            document_type=DocumentType.JUDGMENT,
            chunk_type=ChunkType.HOLDING,
            chunk_index=1,
            token_count=10,
            source=sample_source_info,
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
        ),
    ]


@pytest.fixture()
def sample_parsed_doc() -> ParsedDocument:
    """Minimal ParsedDocument with raw_text for context."""
    return ParsedDocument(
        document_id=uuid4(),
        source_type=SourceType.INDIAN_KANOON,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.HTML,
        raw_text=(
            "Indian Contract Act, 1872\n\n"
            "Section 10. All agreements are contracts which are made by free consent.\n\n"
            "Section 11. Every person is competent to contract who is of the age of majority.\n\n"
            "Section 12. A person is said to be of sound mind for making a contract."
        ),
        sections=[
            ParsedSection(id="s10", level=SectionLevel.SECTION, number="10", text="Section 10."),
            ParsedSection(id="s11", level=SectionLevel.SECTION, number="11", text="Section 11."),
            ParsedSection(id="s12", level=SectionLevel.SECTION, number="12", text="Section 12."),
        ],
        title="Indian Contract Act",
        act_name="Indian Contract Act",
        parser_used=ParserType.HTML_INDIAN_KANOON,
        quality=QualityReport(overall_score=0.9, passed=True),
        raw_content_path="data/raw/test.html",
    )


def make_mock_async_anthropic(response_text: str) -> MagicMock:
    """Create a mock AsyncAnthropic client that returns predefined text."""
    client = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    response = MagicMock()
    response.content = [content_block]
    client.messages.create = AsyncMock(return_value=response)
    return client
