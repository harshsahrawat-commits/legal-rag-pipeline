"""Data models for the Query Intelligence Layer (Phase 0).

Defines cache entries, router results, HyDE results, configuration,
and the overall query intelligence result.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.retrieval._models import QueryRoute

# --- Cache models ---


class CacheEntry(BaseModel):
    """Stored record in the semantic query cache."""

    query_text: str
    cache_key: str
    response: dict[str, Any]
    acts_cited: list[str] = Field(default_factory=list)
    cached_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = 86400


class CacheResult(BaseModel):
    """Result of a cache lookup."""

    hit: bool = False
    response: dict[str, Any] | None = None
    similarity: float = 0.0
    cache_key: str | None = None


# --- Router models ---


class RouterResult(BaseModel):
    """Output of the query router classification."""

    route: QueryRoute = QueryRoute.STANDARD
    confidence: float = 0.5
    signals: list[str] = Field(default_factory=list)


# --- HyDE models ---


class HyDEResult(BaseModel):
    """Output of the Selective HyDE component."""

    hypothetical_text: str | None = None
    hyde_embedding: list[float] | None = None
    generated: bool = False


# --- Orchestrator output ---


class QueryIntelligenceResult(BaseModel):
    """Full output of the Query Intelligence Layer.

    Contains the classified route, cache status, HyDE status,
    and the built RetrievalQuery ready for Phase 7.
    """

    query_text: str
    route: QueryRoute = QueryRoute.STANDARD
    cache_hit: bool = False
    cache_response: dict[str, Any] | None = None
    hyde_generated: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# --- Configuration ---


class QuerySettings(BaseModel):
    """Query intelligence settings from configs/query.yaml."""

    # Cache
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 86400
    cache_short_ttl_seconds: int = 3600
    cache_collection: str = "query_cache"
    cache_redis_prefix: str = "qcache:"

    # Router
    router_version: str = "v1_rule_based"

    # HyDE
    hyde_enabled: bool = True
    hyde_model: str = "claude-haiku"
    hyde_max_tokens: int = 200
    hyde_routes: list[str] = Field(
        default_factory=lambda: ["complex", "analytical"],
    )

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 768
    matryoshka_dim: int = 64
    device: str = "cpu"

    # External services
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_url: str = "redis://localhost:6379/0"


class QueryConfig(BaseModel):
    """Root model for configs/query.yaml."""

    settings: QuerySettings = Field(default_factory=QuerySettings)
