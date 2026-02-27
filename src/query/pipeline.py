"""QueryIntelligenceLayer: orchestrator for Phase 0.

Combines Semantic Cache, Adaptive Router, and Selective HyDE
into a single pre-retrieval pipeline that produces a RetrievalQuery.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.query._cache import SemanticQueryCache
from src.query._hyde import SelectiveHyDE
from src.query._models import CacheResult, HyDEResult, QueryIntelligenceResult
from src.query._router import AdaptiveQueryRouter
from src.retrieval._models import QueryRoute, RetrievalQuery
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.query._models import QuerySettings

_log = get_logger(__name__)


class QueryIntelligenceLayer:
    """Pre-retrieval layer: cache → route → HyDE → build RetrievalQuery.

    Usage::

        layer = QueryIntelligenceLayer(settings)
        qi_result, retrieval_query = await layer.process("What is Section 302?")
        # qi_result contains timings, cache status, route
        # retrieval_query is ready for RetrievalEngine.retrieve()
    """

    def __init__(self, settings: QuerySettings) -> None:
        self._settings = settings
        self._cache = SemanticQueryCache(settings)
        self._router = AdaptiveQueryRouter(settings)
        self._hyde = SelectiveHyDE(settings)
        self._embedder: Any = None

    def load_embedder(self) -> None:
        """Load the embedding model for query embedding.

        Call once before processing queries. If loading fails,
        the layer will still work but won't embed queries
        (requires pre-embedded queries).
        """
        try:
            from src.embedding._embedder import LateChunkingEmbedder
            from src.embedding._models import EmbeddingSettings

            emb_settings = EmbeddingSettings(
                model_name_or_path=self._settings.embedding_model,
                embedding_dim=self._settings.embedding_dim,
                matryoshka_dim=self._settings.matryoshka_dim,
                device=self._settings.device,
            )
            self._embedder = LateChunkingEmbedder(emb_settings)
            self._embedder.load_model()
            _log.info("query_embedder_loaded")
        except Exception as exc:
            _log.warning("query_embedder_load_failed", error=str(exc))

    async def process(
        self,
        query_text: str,
        embedder: Any = None,
    ) -> tuple[QueryIntelligenceResult, RetrievalQuery]:
        """Process a query through the intelligence layer.

        Args:
            query_text: Raw user query.
            embedder: Optional embedder override. If None, uses self._embedder.

        Returns:
            Tuple of (QueryIntelligenceResult, RetrievalQuery).
            The RetrievalQuery is ready for Phase 7's RetrievalEngine.
        """
        emb = embedder or self._embedder
        timings: dict[str, float] = {}
        errors: list[str] = []

        # ---- 1. Embed query ----
        t0 = time.monotonic()
        query_embedding = self._embed_query(query_text, emb)
        query_embedding_fast = self._matryoshka_slice(query_embedding, emb)
        timings["embed_ms"] = (time.monotonic() - t0) * 1000

        # ---- 2. Check cache ----
        t0 = time.monotonic()
        cache_result = self._check_cache(query_embedding)
        timings["cache_ms"] = (time.monotonic() - t0) * 1000

        if cache_result.hit:
            _log.info(
                "cache_hit",
                similarity=cache_result.similarity,
                cache_key=cache_result.cache_key,
            )
            qi_result = QueryIntelligenceResult(
                query_text=query_text,
                cache_hit=True,
                cache_response=cache_result.response,
                timings=timings,
            )
            # Still build a minimal RetrievalQuery for consistency
            retrieval_query = RetrievalQuery(
                text=query_text,
                query_embedding=query_embedding,
                query_embedding_fast=query_embedding_fast,
            )
            return qi_result, retrieval_query

        # ---- 3. Classify route ----
        t0 = time.monotonic()
        try:
            router_result = self._router.classify(query_text)
        except Exception as exc:
            _log.warning("router_failed", error=str(exc))
            errors.append(f"Router failed: {exc}")
            from src.query._models import RouterResult

            router_result = RouterResult()  # defaults to STANDARD
        timings["route_ms"] = (time.monotonic() - t0) * 1000

        # ---- 4. Maybe HyDE ----
        t0 = time.monotonic()
        hyde_result = self._maybe_hyde(query_text, router_result.route, emb)
        timings["hyde_ms"] = (time.monotonic() - t0) * 1000

        # ---- 5. Build RetrievalQuery ----
        # If HyDE generated an embedding, use it for vector search
        effective_embedding = hyde_result.hyde_embedding or query_embedding

        retrieval_query = RetrievalQuery(
            text=query_text,
            query_embedding=effective_embedding,
            query_embedding_fast=self._matryoshka_slice(effective_embedding, emb),
            route=router_result.route,
            hyde_text=hyde_result.hypothetical_text,
        )

        qi_result = QueryIntelligenceResult(
            query_text=query_text,
            route=router_result.route,
            cache_hit=False,
            hyde_generated=hyde_result.generated,
            timings=timings,
            errors=errors,
        )

        _log.info(
            "query_intelligence_complete",
            route=router_result.route.value,
            cache_hit=False,
            hyde=hyde_result.generated,
            total_ms=sum(timings.values()),
        )

        return qi_result, retrieval_query

    async def store_response(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        response: dict[str, Any],
        acts_cited: list[str] | None = None,
    ) -> str | None:
        """Store a response in the cache after retrieval+generation.

        Args:
            query_text: Original query text.
            query_embedding: Query embedding vector.
            response: Response dict to cache.
            acts_cited: Acts cited in the response (for invalidation).

        Returns:
            Cache key if stored, None if cache unavailable.
        """
        if query_embedding is None:
            _log.debug("cache_store_skipped_no_embedding")
            return None

        if not self._cache.is_available:
            return None

        try:
            return self._cache.set(
                query_text=query_text,
                query_embedding=query_embedding,
                response=response,
                acts_cited=acts_cited,
            )
        except Exception as exc:
            _log.warning("cache_store_failed", error=str(exc))
            return None

    async def invalidate_for_act(self, act_name: str) -> int:
        """Invalidate cache entries citing the given act.

        Called when an amendment is detected by the acquisition pipeline.
        """
        try:
            return self._cache.invalidate_for_act(act_name)
        except Exception as exc:
            _log.warning("cache_invalidation_failed", error=str(exc))
            return 0

    @classmethod
    def from_config(
        cls,
        config_path: str | None = None,
    ) -> QueryIntelligenceLayer:
        """Create a QueryIntelligenceLayer from a config file.

        Args:
            config_path: Path to config YAML. Defaults to configs/query.yaml.
        """
        from pathlib import Path

        from src.query._config import load_query_config
        from src.query._models import QueryConfig

        try:
            path = Path(config_path) if config_path else None
            config = load_query_config(path)
        except Exception:
            config = QueryConfig()
            _log.warning("config_load_failed_using_defaults")

        return cls(config.settings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_query(
        self,
        query_text: str,
        embedder: Any,
    ) -> list[float] | None:
        """Embed a query text, returning the full-dim vector."""
        if embedder is None:
            return None
        try:
            embeddings = embedder.embed_texts([query_text])
            if embeddings and len(embeddings) > 0:
                return embeddings[0].tolist()
        except Exception as exc:
            _log.warning("query_embedding_failed", error=str(exc))
        return None

    @staticmethod
    def _matryoshka_slice(
        embedding: list[float] | None,
        embedder: Any,
    ) -> list[float] | None:
        """Slice a full embedding to Matryoshka fast dimension."""
        if embedding is None or embedder is None:
            return None
        try:
            import numpy as np

            full_arr = np.array(embedding, dtype=np.float32)
            fast_arr = embedder.matryoshka_slice(full_arr)
            return fast_arr.tolist()
        except Exception:
            # Fallback: simple truncation
            try:
                return embedding[:64]
            except Exception:
                return None

    def _check_cache(self, query_embedding: list[float] | None) -> CacheResult:
        """Check the semantic cache for a hit."""
        from src.query._models import CacheResult

        if query_embedding is None:
            return CacheResult(hit=False)
        if not self._cache.is_available:
            return CacheResult(hit=False)
        try:
            return self._cache.get(query_embedding)
        except Exception as exc:
            _log.warning("cache_check_failed", error=str(exc))
            return CacheResult(hit=False)

    def _maybe_hyde(
        self,
        query_text: str,
        route: QueryRoute,
        embedder: Any,
    ) -> HyDEResult:
        """Run Selective HyDE if applicable."""
        from src.query._models import HyDEResult

        try:
            return self._hyde.maybe_generate(query_text, route, embedder=embedder)
        except Exception as exc:
            _log.warning("hyde_failed", error=str(exc))
            return HyDEResult(generated=False)
