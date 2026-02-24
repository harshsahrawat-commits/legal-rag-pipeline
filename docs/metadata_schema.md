# Metadata Schema — Pydantic Models

Every chunk in the system is a `LegalChunk` instance. Never use raw dicts.

## Core Models

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import date, datetime
from enum import Enum
from uuid import UUID, uuid4

class DocumentType(str, Enum):
    STATUTE = "statute"
    JUDGMENT = "judgment"
    NOTIFICATION = "notification"
    CIRCULAR = "circular"
    ORDER = "order"
    REPORT = "report"
    SCHEDULE = "schedule"

class TemporalStatus(str, Enum):
    IN_FORCE = "in_force"
    REPEALED = "repealed"
    PARTIALLY_REPEALED = "partially_repealed"
    AMENDED = "amended"             # Latest version ingested
    SUPERSEDED = "superseded"       # Replaced by new legislation

class ChunkType(str, Enum):
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

class ChunkStrategy(str, Enum):
    """Which chunking strategy produced this chunk."""
    STRUCTURE_BOUNDARY = "structure_boundary"    # Statute sections
    JUDGMENT_STRUCTURAL = "judgment_structural"  # Judgment headings
    RECURSIVE_SEMANTIC = "recursive_semantic"    # RSC hybrid fallback
    PROPOSITION = "proposition"                  # LLM-decomposed definitions
    SEMANTIC_MAXMIN = "semantic_maxmin"           # Max-Min semantic fallback
    PAGE_LEVEL = "page_level"                    # Page-per-chunk for degraded scans/schedules
    RAPTOR = "raptor"                            # RAPTOR summary tree nodes
    QUIM = "quim"                                # QuIM-RAG generated questions

class CourtHierarchy(int, Enum):
    SUPREME_COURT = 1
    HIGH_COURT = 2
    DISTRICT_COURT = 3
    TRIBUNAL = 4
    QUASI_JUDICIAL = 5

class SourceInfo(BaseModel):
    url: str
    source_name: str                # e.g., "Indian Kanoon", "India Code"
    scraped_at: datetime
    last_verified: datetime

class StatuteMetadata(BaseModel):
    act_name: str
    act_number: str | None = None   # e.g., "Act No. 45 of 1860"
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
    amendment_history: list[AmendmentRecord] = []

class AmendmentRecord(BaseModel):
    amending_act: str
    date: date
    nature: str                     # "substitution", "insertion", "omission"
    gazette_ref: str | None = None

class JudgmentMetadata(BaseModel):
    case_citation: str              # Primary citation e.g., "AIR 2023 SC 1234"
    alt_citations: list[str] = []
    court: str
    court_level: CourtHierarchy
    bench_type: str | None = None   # "Division Bench", "Full Bench", "Constitution Bench"
    bench_strength: int | None = None
    judge_names: list[str] = []
    date_decided: date | None = None
    case_type: str | None = None    # "Criminal Appeal", "Writ Petition", etc.
    parties_petitioner: str | None = None
    parties_respondent: str | None = None
    is_overruled: bool = False
    overruled_by: str | None = None
    followed_in: list[str] = []
    distinguished_in: list[str] = []

class ContentMetadata(BaseModel):
    sections_cited: list[str] = []
    acts_cited: list[str] = []
    cases_cited: list[str] = []
    legal_concepts: list[str] = []  # NER-extracted: "murder", "anticipatory bail", etc.
    language: str = "en"
    has_hindi: bool = False

class IngestionMetadata(BaseModel):
    ingested_at: datetime
    parser: str                     # "docling_v2", "llamaparse", "tesseract"
    ocr_confidence: float | None = None
    quality_score: float | None = None
    chunk_strategy: ChunkStrategy   # Which chunking strategy produced this chunk
    contextualized: bool = False
    late_chunked: bool = False
    quim_questions: int = 0
    raptor_summary: bool = False
    raptor_level: int | None = None # 0=Act summary, 1=chapter, 2=section (base)
    requires_manual_review: bool = False  # True for degraded scans

class ParentDocumentInfo(BaseModel):
    """Parent-child relationships for retrieval-time context expansion.

    At retrieval time, the system fetches parent text from Redis to give
    the LLM surrounding context (e.g., full section when sub-section matched).
    Parent text is stored in Redis (~200MB for 100K parents at ~2KB avg).
    """
    parent_chunk_id: UUID | None = None
    """For statutes: full section if this is a sub-section, or full chapter if this is a section.
    For judgments: full issue section if this is reasoning."""

    sibling_chunk_ids: list[UUID] = []
    """+-2 adjacent chunks from the same document for windowed context."""

    judgment_header_chunk_id: UUID | None = None
    """For judgment chunks: always points to the header/metadata chunk.
    At retrieval time, judgment header is always included for case context."""

class LegalChunk(BaseModel):
    """The core data unit of the entire pipeline. Every chunk is this."""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str                       # Original chunk text
    contextualized_text: str | None = None  # After Contextual Retrieval

    document_type: DocumentType
    chunk_type: ChunkType
    chunk_index: int                # Position within document
    token_count: int

    source: SourceInfo
    statute: StatuteMetadata | None = None
    judgment: JudgmentMetadata | None = None
    content: ContentMetadata
    ingestion: IngestionMetadata
    parent_info: ParentDocumentInfo = Field(default_factory=ParentDocumentInfo)

    # Embeddings stored separately in Qdrant, linked by chunk id
    # Qdrant stores dual named vectors: "fast" (64-dim) + "full" (768-dim)
    # QuIM questions stored separately in quim_questions collection, linked by chunk id
```

## Usage Rules

1. **Always construct via model, never from raw dict.** Pydantic validates types + constraints.
2. **statute and judgment are mutually exclusive.** A chunk is from a statute OR a judgment.
3. **content.sections_cited must be verified** against the knowledge graph during Phase 7.
4. **temporal_status must be refreshed** whenever an amendment is ingested.
5. **Serialize to JSON** for storage: `chunk.model_dump_json()`. Deserialize: `LegalChunk.model_validate_json(data)`.
6. **chunk_strategy must always be set.** Every chunk must declare which strategy produced it.
7. **parent_info.parent_chunk_id** should be populated at chunking time for all sub-section and reasoning chunks. The parent text itself is stored in Redis (key = parent_chunk_id, value = parent chunk text + metadata).
8. **parent_info.judgment_header_chunk_id** must be set for every judgment chunk. At retrieval time, the header is always fetched alongside matched judgment chunks.
