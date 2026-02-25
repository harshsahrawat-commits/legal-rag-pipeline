from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.acquisition._models import SourceType  # noqa: TC001

# --- Sparse vector model ---


class SparseVector(BaseModel):
    """BM25 sparse vector: parallel lists of token indices and weights."""

    indices: list[int]
    values: list[float]


# --- Config models ---


class EmbeddingSettings(BaseModel):
    """Embedding-specific settings from configs/embedding.yaml."""

    # Directories
    input_dir: Path = Path("data/enriched")
    parsed_dir: Path = Path("data/parsed")

    # Model
    model_name_or_path: str = "BAAI/bge-m3"
    embedding_dim: int = 768
    matryoshka_dim: int = 64
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 8192

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chunks_collection: str = "legal_chunks"
    quim_collection: str = "quim_questions"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "parent:"

    # Windowing (for docs exceeding model max_length)
    window_overlap_tokens: int = 128

    # Behavior
    skip_existing: bool = True


class EmbeddingConfig(BaseModel):
    """Root model for configs/embedding.yaml."""

    settings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


# --- Pipeline result model ---


class EmbeddingResult(BaseModel):
    """Summary of an embedding pipeline run."""

    source_type: SourceType | None = None
    documents_found: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_embedded: int = 0
    quim_questions_embedded: int = 0
    parent_entries_stored: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
