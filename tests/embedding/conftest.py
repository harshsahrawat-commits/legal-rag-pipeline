"""Shared fixtures for embedding tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._models import (
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    IngestionMetadata,
    LegalChunk,
    SourceInfo,
    StatuteMetadata,
)
from src.embedding._models import EmbeddingSettings
from src.enrichment._models import QuIMDocument, QuIMEntry
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)


@pytest.fixture()
def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings()


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
def sample_doc_id() -> object:
    return uuid4()


@pytest.fixture()
def sample_enriched_chunks(
    now: datetime,
    sample_source_info: SourceInfo,
    sample_doc_id,
) -> list[LegalChunk]:
    """Three enriched statute chunks with contextualized_text."""
    return [
        LegalChunk(
            document_id=sample_doc_id,
            text="Section 10. All agreements are contracts which are made by free consent.",
            contextualized_text=(
                "Indian Contract Act, 1872 — Section 10. "
                "All agreements are contracts which are made by free consent."
            ),
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
                contextualized=True,
            ),
        ),
        LegalChunk(
            document_id=sample_doc_id,
            text="Section 11. Every person is competent to contract who is of the age of majority.",
            contextualized_text=(
                "Indian Contract Act, 1872 — Section 11. "
                "Every person is competent to contract who is of the age of majority."
            ),
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
                contextualized=True,
            ),
        ),
        LegalChunk(
            document_id=sample_doc_id,
            text="Section 12. A person is said to be of sound mind for the purpose of making a contract.",
            contextualized_text=(
                "Indian Contract Act, 1872 — Section 12. "
                "A person is said to be of sound mind for the purpose of making a contract."
            ),
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
                contextualized=True,
            ),
        ),
    ]


@pytest.fixture()
def sample_parsed_doc(sample_doc_id) -> ParsedDocument:
    """Corresponding ParsedDocument with full text."""
    return ParsedDocument(
        document_id=sample_doc_id,
        source_type=SourceType.INDIAN_KANOON,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.HTML,
        raw_text=(
            "Indian Contract Act, 1872\n\n"
            "Section 10. All agreements are contracts which are made by free consent.\n\n"
            "Section 11. Every person is competent to contract who is of the age of majority.\n\n"
            "Section 12. A person is said to be of sound mind for the purpose of making a contract."
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


@pytest.fixture()
def sample_quim_doc(sample_doc_id, sample_enriched_chunks) -> QuIMDocument:
    """QuIM document with questions for the sample chunks."""
    return QuIMDocument(
        document_id=sample_doc_id,
        entries=[
            QuIMEntry(
                chunk_id=sample_enriched_chunks[0].id,
                document_id=sample_doc_id,
                questions=[
                    "What makes an agreement a valid contract?",
                    "When is consent considered free?",
                ],
            ),
            QuIMEntry(
                chunk_id=sample_enriched_chunks[1].id,
                document_id=sample_doc_id,
                questions=[
                    "Who is competent to enter a contract?",
                ],
            ),
        ],
    )


def make_fake_embedding(dim: int = 768, seed: int = 42) -> np.ndarray:
    """Create a deterministic fake embedding."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


@pytest.fixture()
def mock_embedder():
    """Mock LateChunkingEmbedder with deterministic output."""
    embedder = MagicMock()
    embedder.load_model = MagicMock()
    embedder.embed_document_late_chunking = MagicMock(
        side_effect=lambda text, chunks: [
            make_fake_embedding(768, seed=i) for i in range(len(chunks))
        ]
    )
    embedder.embed_texts = MagicMock(
        side_effect=lambda texts: [
            make_fake_embedding(768, seed=100 + i) for i in range(len(texts))
        ]
    )
    embedder.matryoshka_slice = MagicMock(
        side_effect=lambda e: make_fake_embedding(64, seed=hash(e.tobytes()) % 10000)
    )
    return embedder


@pytest.fixture()
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = False
    client.upsert.return_value = None
    client.retrieve.return_value = []
    return client


@pytest.fixture()
def mock_redis_client():
    client = AsyncMock()
    client.set = AsyncMock(return_value=True)
    return client
