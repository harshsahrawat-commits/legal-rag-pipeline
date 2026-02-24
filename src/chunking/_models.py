from __future__ import annotations

from datetime import UTC, date, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.acquisition._models import DocumentType, SourceType  # noqa: TC001

# --- Enums ---


class ChunkType(StrEnum):
    """What kind of content this chunk represents."""

    STATUTORY_TEXT = "statutory_text"
    DEFINITION = "definition"
    PROVISO = "proviso"
    EXPLANATION = "explanation"
    SCHEDULE_ENTRY = "schedule_entry"
    FACTS = "facts"
    ISSUES = "issues"
    REASONING = "reasoning"
    HOLDING = "holding"
    ORDER = "order"
    DISSENT = "dissent"
    OBITER = "obiter"
    RAPTOR_SUMMARY = "raptor_summary"


class ChunkStrategy(StrEnum):
    """Which chunking strategy produced this chunk."""

    STRUCTURE_BOUNDARY = "structure_boundary"
    JUDGMENT_STRUCTURAL = "judgment_structural"
    RECURSIVE_SEMANTIC = "recursive_semantic"
    PROPOSITION = "proposition"
    SEMANTIC_MAXMIN = "semantic_maxmin"
    PAGE_LEVEL = "page_level"
    RAPTOR = "raptor"
    QUIM = "quim"


class TemporalStatus(StrEnum):
    """Temporal validity of a statute or section."""

    IN_FORCE = "in_force"
    REPEALED = "repealed"
    PARTIALLY_REPEALED = "partially_repealed"
    AMENDED = "amended"
    SUPERSEDED = "superseded"


class CourtHierarchy(IntEnum):
    """Indian court hierarchy by precedent authority."""

    SUPREME_COURT = 1
    HIGH_COURT = 2
    DISTRICT_COURT = 3
    TRIBUNAL = 4
    QUASI_JUDICIAL = 5


# --- Sub-models ---


class SourceInfo(BaseModel):
    """Provenance information for a chunk."""

    url: str
    source_name: str
    scraped_at: datetime
    last_verified: datetime


class AmendmentRecord(BaseModel):
    """Record of a single amendment to a statute."""

    amending_act: str
    date: date
    nature: str
    gazette_ref: str | None = None


class StatuteMetadata(BaseModel):
    """Metadata specific to statute chunks."""

    act_name: str
    act_number: str | None = None
    section_number: str | None = None
    chapter: str | None = None
    part: str | None = None
    schedule: str | None = None
    date_enacted: date | None = None
    date_effective: date | None = None
    date_repealed: date | None = None
    repealed_by: str | None = None
    replaced_by_section: str | None = None
    is_in_force: bool = True
    temporal_status: TemporalStatus = TemporalStatus.IN_FORCE
    amendment_history: list[AmendmentRecord] = Field(default_factory=list)


class JudgmentMetadata(BaseModel):
    """Metadata specific to judgment chunks."""

    case_citation: str
    alt_citations: list[str] = Field(default_factory=list)
    court: str
    court_level: CourtHierarchy
    bench_type: str | None = None
    bench_strength: int | None = None
    judge_names: list[str] = Field(default_factory=list)
    date_decided: date | None = None
    case_type: str | None = None
    parties_petitioner: str | None = None
    parties_respondent: str | None = None
    is_overruled: bool = False
    overruled_by: str | None = None
    followed_in: list[str] = Field(default_factory=list)
    distinguished_in: list[str] = Field(default_factory=list)


class ContentMetadata(BaseModel):
    """Extracted legal content references within a chunk."""

    sections_cited: list[str] = Field(default_factory=list)
    acts_cited: list[str] = Field(default_factory=list)
    cases_cited: list[str] = Field(default_factory=list)
    legal_concepts: list[str] = Field(default_factory=list)
    language: str = "en"
    has_hindi: bool = False


class IngestionMetadata(BaseModel):
    """Processing provenance for a chunk."""

    ingested_at: datetime
    parser: str
    ocr_confidence: float | None = None
    quality_score: float | None = None
    chunk_strategy: ChunkStrategy
    contextualized: bool = False
    late_chunked: bool = False
    quim_questions: int = 0
    raptor_summary: bool = False
    raptor_level: int | None = None
    requires_manual_review: bool = False


class ParentDocumentInfo(BaseModel):
    """Parent-child relationships for retrieval-time context expansion."""

    parent_chunk_id: UUID | None = None
    sibling_chunk_ids: list[UUID] = Field(default_factory=list)
    judgment_header_chunk_id: UUID | None = None


# --- Primary output model ---


class LegalChunk(BaseModel):
    """The core data unit of the entire pipeline. Every chunk is this."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str
    contextualized_text: str | None = None

    document_type: DocumentType
    chunk_type: ChunkType
    chunk_index: int
    token_count: int

    source: SourceInfo
    statute: StatuteMetadata | None = None
    judgment: JudgmentMetadata | None = None
    content: ContentMetadata
    ingestion: IngestionMetadata
    parent_info: ParentDocumentInfo = Field(default_factory=ParentDocumentInfo)


# --- Config models ---


class ChunkingSettings(BaseModel):
    """Chunking-specific settings from configs/chunking.yaml."""

    input_dir: Path = Path("data/parsed")
    output_dir: Path = Path("data/chunks")

    max_tokens: int = 1500
    overlap_tokens: int = 150
    ocr_confidence_threshold: float = 0.80
    min_section_count_statute: int = 3
    min_section_count_judgment: int = 2

    similarity_threshold: float = 0.75
    semantic_percentile: float = 0.25
    min_chunk_tokens: int = 50
    page_separator: str = "\f"

    proposition_model: str = "claude-haiku-4-5-20251001"
    proposition_max_tokens_response: int = 4096

    embedding_model: str = "BAAI/bge-m3"


class ChunkingConfig(BaseModel):
    """Root model for configs/chunking.yaml."""

    settings: ChunkingSettings = Field(default_factory=ChunkingSettings)


class ChunkingResult(BaseModel):
    """Summary of a chunking pipeline run."""

    source_type: SourceType | None = None
    documents_found: int = 0
    documents_chunked: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
