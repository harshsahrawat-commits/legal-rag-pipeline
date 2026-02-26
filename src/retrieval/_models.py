"""Data models for the retrieval module.

Defines query input, intermediate scored/fused chunks, expanded context,
retrieval results, and configuration settings.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — Pydantic field type (bm25_vocab_path)
from typing import Any

from pydantic import BaseModel, Field

# --- Enums ---


class QueryRoute(StrEnum):
    """How the query should be routed through the retrieval pipeline."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"


# --- Query input ---


class RetrievalQuery(BaseModel):
    """Input to the retrieval engine.

    Phase 0 (Query Intelligence) will produce this when built.
    Phase 7 can also build it from raw text.
    """

    text: str
    query_embedding: list[float] | None = None
    query_embedding_fast: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
    route: QueryRoute = QueryRoute.STANDARD
    hyde_text: str | None = None
    metadata_filters: dict[str, Any] | None = None
    reference_date: date | None = None
    max_results: int = 20
    max_context_tokens: int = 30_000


# --- Intermediate results ---


class ScoredChunk(BaseModel):
    """A chunk with a retrieval score from one search channel."""

    chunk_id: str
    text: str
    contextualized_text: str | None = None
    score: float
    channel: str
    document_type: str | None = None
    chunk_type: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class FusedChunk(BaseModel):
    """Post-fusion chunk with combined score from multiple channels."""

    chunk_id: str
    text: str
    contextualized_text: str | None = None
    rrf_score: float
    rerank_score: float | None = None
    channels: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


# --- Final output ---


class ExpandedContext(BaseModel):
    """A chunk with its parent/sibling context for LLM consumption."""

    chunk_id: str
    chunk_text: str
    parent_text: str | None = None
    judgment_header_text: str | None = None
    relevance_score: float = 0.0
    total_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Final output of Phase 7 for one query."""

    query_text: str = ""
    route: QueryRoute = QueryRoute.STANDARD
    chunks: list[ExpandedContext] = Field(default_factory=list)
    total_context_tokens: int = 0
    search_channels_used: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    kg_direct_answer: dict[str, Any] | None = None
    flare_retrievals: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed time in milliseconds."""
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds() * 1000


# --- Config models ---


class RetrievalSettings(BaseModel):
    """Retrieval-specific settings from configs/retrieval.yaml."""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chunks_collection: str = "legal_chunks"
    quim_collection: str = "quim_questions"

    # Embedding model (for query embedding)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 768
    matryoshka_dim: int = 64
    device: str = "cpu"

    # Reranker model
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "parent:"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Search parameters
    dense_fast_top_k: int = 1000
    dense_full_top_k: int = 100
    bm25_top_k: int = 100
    quim_top_k: int = 50
    graph_max_hops: int = 2
    graph_max_results: int = 50

    # Fusion
    rrf_k: int = 60
    fused_top_k: int = 150

    # Reranking
    rerank_top_k: int = 20
    rerank_batch_size: int = 32

    # Context expansion
    max_context_tokens: int = 30_000
    include_judgment_headers: bool = True
    include_parent_chunks: bool = True

    # FLARE
    flare_enabled: bool = True
    flare_segment_tokens: int = 300
    flare_confidence_threshold: float = 0.5
    flare_max_retrievals: int = 5

    # BM25 vocabulary
    bm25_vocab_path: Path | None = None


class RetrievalConfig(BaseModel):
    """Root model for configs/retrieval.yaml."""

    settings: RetrievalSettings = Field(default_factory=RetrievalSettings)
