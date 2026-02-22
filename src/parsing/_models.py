from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.acquisition._models import ContentFormat, DocumentType, SourceType  # noqa: TC001

# --- Enums ---


class SectionLevel(StrEnum):
    """Hierarchy level in an Indian legal document."""

    # Statute levels
    PREAMBLE = "preamble"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUB_SECTION = "sub_section"
    CLAUSE = "clause"
    SUB_CLAUSE = "sub_clause"
    PROVISO = "proviso"
    EXPLANATION = "explanation"
    DEFINITION = "definition"
    SCHEDULE = "schedule"

    # Judgment levels
    HEADER = "header"
    FACTS = "facts"
    ISSUES = "issues"
    REASONING = "reasoning"
    HOLDING = "holding"
    ORDER = "order"
    DISSENT = "dissent"
    OBITER = "obiter"

    # Generic
    PARAGRAPH = "paragraph"


class ParserType(StrEnum):
    """Which parser produced the output."""

    DOCLING_PDF = "docling_pdf"
    PYMUPDF_PDF = "pymupdf_pdf"
    HTML_INDIAN_KANOON = "html_indian_kanoon"
    HTML_INDIA_CODE = "html_india_code"


# --- Structural models ---


class ParsedSection(BaseModel):
    """One structural node in a parsed document hierarchy.

    Sections form a tree: a Chapter contains Sections, which contain
    Sub-sections, Clauses, Provisos, and Explanations.
    """

    id: str
    level: SectionLevel
    number: str | None = None
    title: str | None = None
    text: str
    children: list[ParsedSection] = Field(default_factory=list)
    parent_id: str | None = None
    token_count: int = 0
    page_numbers: list[int] = Field(default_factory=list)


class ParsedTable(BaseModel):
    """A table extracted from the document."""

    id: str
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    page_number: int | None = None
    section_id: str | None = None
    row_count: int = 0
    col_count: int = 0


# --- Quality models ---


class QualityCheckResult(BaseModel):
    """Result of a single quality check."""

    check_name: str
    passed: bool
    score: float
    details: str = ""


class QualityReport(BaseModel):
    """Aggregated quality validation results for a parsed document."""

    overall_score: float
    passed: bool
    checks: list[QualityCheckResult] = Field(default_factory=list)
    flagged_for_review: bool = False


# --- Primary output model ---


class ParsedDocument(BaseModel):
    """Primary output of Phase 2. One per document.

    Saved as ``data/parsed/{source}/{doc_id}.json``.
    Phase 3 (Chunking) reads these to build chunks.
    """

    document_id: UUID = Field(default_factory=uuid4)
    source_type: SourceType
    document_type: DocumentType
    content_format: ContentFormat

    # Core content
    raw_text: str
    sections: list[ParsedSection] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)

    # Metadata (enriched from Phase 1 preliminary metadata)
    title: str | None = None
    act_name: str | None = None
    act_number: str | None = None
    year: int | None = None
    date: str | None = None
    court: str | None = None
    case_citation: str | None = None
    parties: str | None = None
    page_count: int | None = None

    # Parsing provenance
    parser_used: ParserType
    ocr_applied: bool = False
    ocr_confidence: float | None = None
    parsing_duration_seconds: float = 0.0

    # Quality
    quality: QualityReport

    # Lineage
    raw_content_path: str
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Config models ---


class ParsingSettings(BaseModel):
    """Parsing-specific settings from configs/parsing.yaml."""

    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/parsed")
    pdf_cache_dir: Path = Path("data/cache/pdf")

    prefer_docling: bool = True
    ocr_languages: list[str] = Field(default_factory=lambda: ["eng", "hin"])
    ocr_confidence_threshold: float = 0.85

    min_text_completeness: float = 0.5
    download_timeout_seconds: int = 120
    max_pdf_size_mb: int = 100
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    )


class ParsingConfig(BaseModel):
    """Root model for configs/parsing.yaml."""

    settings: ParsingSettings = Field(default_factory=ParsingSettings)
