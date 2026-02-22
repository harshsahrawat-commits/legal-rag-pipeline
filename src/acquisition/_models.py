from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# --- Enums ---


class DocumentType(StrEnum):
    """Type of legal document. Matches docs/metadata_schema.md."""

    STATUTE = "statute"
    JUDGMENT = "judgment"
    NOTIFICATION = "notification"
    CIRCULAR = "circular"
    ORDER = "order"
    REPORT = "report"
    SCHEDULE = "schedule"


class SourceType(StrEnum):
    """Supported acquisition sources."""

    INDIAN_KANOON = "indian_kanoon"
    INDIA_CODE = "india_code"


class ContentFormat(StrEnum):
    """Format of the raw downloaded content."""

    HTML = "html"
    PDF = "pdf"


class FlagType(StrEnum):
    """Types of issues detected during legal review."""

    SCANNED_PDF = "scanned_pdf"
    REGIONAL_LANGUAGE = "regional_language"
    CORRUPT_CONTENT = "corrupt_content"
    SMALL_CONTENT = "small_content"
    MISSING_METADATA = "missing_metadata"
    ENCODING_ISSUE = "encoding_issue"


class FlagSeverity(StrEnum):
    """Severity of a document flag."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# --- Config models ---


class ScrapeConfig(BaseModel):
    """Per-source scraping configuration."""

    seed_queries: list[str] = Field(default_factory=list)
    seed_act_ids: list[str] = Field(default_factory=list)
    max_pages_per_query: int = 5
    max_documents: int = 100


class SourceDefinition(BaseModel):
    """One entry in the source registry (sources.yaml)."""

    name: str
    source_type: SourceType
    base_url: str
    enabled: bool = True
    rate_limit_requests_per_second: float = 0.5
    request_timeout_seconds: int = 30
    max_retries: int = 3
    scrape_config: ScrapeConfig = Field(default_factory=ScrapeConfig)


class GlobalAcquisitionSettings(BaseModel):
    """Top-level settings for the acquisition module."""

    output_dir: Path = Path("data/raw")
    state_dir: Path = Path("data/state")
    concurrency: int = 2
    user_agent: str = "LegalRAGBot/0.1 (research; +https://github.com/legal-rag)"


class SourceRegistry(BaseModel):
    """Root model for configs/sources.yaml."""

    settings: GlobalAcquisitionSettings = Field(default_factory=GlobalAcquisitionSettings)
    sources: list[SourceDefinition]


# --- Crawl state models ---


class CrawlRecord(BaseModel):
    """State record for a single scraped URL."""

    url: str
    content_hash: str
    file_path: str
    scraped_at: datetime
    document_type: DocumentType | None = None


class CrawlState(BaseModel):
    """Full crawl state for a single source. Persisted to data/state/{source}.json."""

    source_type: SourceType
    last_run: datetime | None = None
    records: dict[str, CrawlRecord] = Field(default_factory=dict)


# --- Pipeline data models ---


class DiscoveredDocument(BaseModel):
    """Output of URL discovery — represents a URL to potentially download."""

    url: str
    source_type: SourceType
    is_new: bool = True
    content_hash_changed: bool = False


class ScrapedContent(BaseModel):
    """Raw download result from an HTTP request."""

    url: str
    content: str
    content_format: ContentFormat
    content_hash: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class PreliminaryMetadata(BaseModel):
    """Lightweight metadata extracted during acquisition (before full parsing)."""

    title: str | None = None
    act_name: str | None = None
    act_number: str | None = None
    year: int | None = None
    case_citation: str | None = None
    court: str | None = None
    date: str | None = None
    parties: str | None = None


class DocumentFlag(BaseModel):
    """An issue detected during legal review."""

    flag_type: FlagType
    message: str
    severity: FlagSeverity


class RawDocument(BaseModel):
    """Primary output of the acquisition phase. One per downloaded document.

    Serialized as a sidecar .meta.json file next to the raw content file.
    Phase 2 reads these to decide on parsing strategy.
    """

    document_id: UUID = Field(default_factory=uuid4)
    url: str
    source_type: SourceType
    content_format: ContentFormat
    raw_content_path: str
    document_type: DocumentType | None = None
    preliminary_metadata: PreliminaryMetadata = Field(default_factory=PreliminaryMetadata)
    flags: list[DocumentFlag] = Field(default_factory=list)
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = ""


class AcquisitionResult(BaseModel):
    """Summary of an acquisition pipeline run."""

    source_type: SourceType
    documents_discovered: int = 0
    documents_downloaded: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
