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
    chunk_strategy: str             # "structure_boundary", "semantic", "raptor"
    contextualized: bool = False
    late_chunked: bool = False
    quim_questions: int = 0
    raptor_summary: bool = False

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
    
    # Embeddings stored separately in Qdrant, linked by chunk id
    # QuIM questions stored separately, linked by chunk id
```

## Usage Rules

1. **Always construct via model, never from raw dict.** Pydantic validates types + constraints.
2. **statute and judgment are mutually exclusive.** A chunk is from a statute OR a judgment.
3. **content.sections_cited must be verified** against the knowledge graph during Phase 7.
4. **temporal_status must be refreshed** whenever an amendment is ingested.
5. **Serialize to JSON** for storage: `chunk.model_dump_json()`. Deserialize: `LegalChunk.model_validate_json(data)`.
