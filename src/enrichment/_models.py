from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID  # noqa: TC003 — Pydantic field type, needed at runtime

from pydantic import BaseModel, Field

from src.acquisition._models import SourceType  # noqa: TC001

# --- QuIM-RAG output models ---


class QuIMEntry(BaseModel):
    """Questions generated for a single chunk."""

    chunk_id: UUID
    document_id: UUID
    questions: list[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = ""


class QuIMDocument(BaseModel):
    """All QuIM entries for a single document. Saved as {doc_id}.quim.json."""

    document_id: UUID
    entries: list[QuIMEntry] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = ""


# --- Config models ---


class EnrichmentSettings(BaseModel):
    """Enrichment-specific settings from configs/enrichment.yaml."""

    input_dir: Path = Path("data/chunks")
    output_dir: Path = Path("data/enriched")
    parsed_dir: Path = Path("data/parsed")

    # LLM settings
    model: str = "claude-haiku-4-5-20251001"
    max_tokens_response: int = 512
    concurrency: int = 5

    # QuIM-RAG settings
    quim_questions_per_chunk: int = 5

    # Document windowing (for docs exceeding Haiku's context)
    context_window_tokens: int = 180_000
    document_window_overlap_tokens: int = 500

    # Behavior
    skip_manual_review_chunks: bool = False


class EnrichmentConfig(BaseModel):
    """Root model for configs/enrichment.yaml."""

    settings: EnrichmentSettings = Field(default_factory=EnrichmentSettings)


# --- Pipeline result model ---


class EnrichmentResult(BaseModel):
    """Summary of an enrichment pipeline run."""

    source_type: SourceType | None = None
    documents_found: int = 0
    documents_enriched: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_contextualized: int = 0
    chunks_quim_generated: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
