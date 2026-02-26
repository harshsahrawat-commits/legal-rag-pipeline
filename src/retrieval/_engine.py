"""RetrievalEngine: reusable single-query handler.

Called by RetrievalPipeline for batch/interactive use,
and by Phase 8 (Hallucination Mitigation) via hybrid_search().
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.retrieval._exceptions import FLAREError, RerankerNotAvailableError
from src.retrieval._expander import ParentDocumentExpander
from src.retrieval._flare import FLAREActiveRetriever
from src.retrieval._fusion import ReciprocalRankFusion
from src.retrieval._models import (
    ExpandedContext,
    QueryRoute,
    RetrievalResult,
    ScoredChunk,
)
from src.retrieval._reranker import CrossEncoderReranker
from src.retrieval._searchers import (
    DenseSearcher,
    GraphSearcher,
    QuIMSearcher,
    SparseSearcher,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.retrieval._models import RetrievalQuery, RetrievalSettings

_log = get_logger(__name__)


class RetrievalEngine:
    """Reusable query handler for the retrieval pipeline.

    Orchestrates multi-channel search, fusion, reranking, and context
    expansion for a single query.  Also exposes ``hybrid_search()``
    for lightweight per-claim retrieval (Phase 8).
    """

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings

        # Search channels
        self._dense = DenseSearcher(settings)
        self._sparse = SparseSearcher(settings)
        self._quim = QuIMSearcher(settings)
        self._graph = GraphSearcher(settings)

        # Post-processing
        self._fusion = ReciprocalRankFusion(k=settings.rrf_k)
        self._reranker = CrossEncoderReranker(settings)
        self._expander = ParentDocumentExpander(settings)
        self._flare = FLAREActiveRetriever(settings)

        # Embedding model (lazy)
        self._embedder: Any = None
        self._bm25: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load embedding model, reranker, and BM25 vocabulary.

        Call once before processing queries.  Each component that fails
        to load is logged but does not prevent others from loading.
        """
        # Embedding model
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
            _log.info("embedder_loaded")
        except Exception as exc:
            _log.warning("embedder_load_failed", error=str(exc))

        # Cross-encoder reranker
        try:
            self._reranker.load_model()
        except RerankerNotAvailableError:
            _log.warning("reranker_not_available")
        except Exception as exc:
            _log.warning("reranker_load_failed", error=str(exc))

        # BM25 vocabulary
        if self._settings.bm25_vocab_path is not None:
            try:
                from src.embedding._sparse import BM25SparseEncoder

                self._bm25 = BM25SparseEncoder.load_vocabulary(
                    self._settings.bm25_vocab_path,
                )
                _log.info("bm25_vocab_loaded")
            except Exception as exc:
                _log.warning("bm25_vocab_load_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Full retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Full retrieval pipeline for one query.

        Steps:
        1. Embed query if embeddings not provided
        2. Route: SIMPLE goes to KG only
        3. Search channels (parallel-ready, currently sequential)
        4. Fuse via RRF
        5. Rerank with cross-encoder
        6. Expand context from Redis
        """
        result = RetrievalResult(
            query_text=query.text,
            route=query.route,
            started_at=datetime.now(UTC),
        )
        timings: dict[str, float] = {}

        # ---- SIMPLE route: KG direct query only ----
        if query.route == QueryRoute.SIMPLE:
            t0 = time.monotonic()
            kg_answer = await self._kg_direct_query(query.text)
            timings["kg_direct_ms"] = (time.monotonic() - t0) * 1000
            result.kg_direct_answer = kg_answer
            result.search_channels_used = ["graph"]
            result.timings = timings
            result.finished_at = datetime.now(UTC)
            return result

        # ---- Embed query if needed ----
        t0 = time.monotonic()
        emb_full, emb_fast, sparse_idx, sparse_val = self._prepare_query_vectors(query)
        timings["embed_ms"] = (time.monotonic() - t0) * 1000

        # ---- Search channels ----
        channel_results: dict[str, list[ScoredChunk]] = {}

        # Dense search
        if emb_full is not None and emb_fast is not None:
            t0 = time.monotonic()
            try:
                dense_hits = self._dense.search(
                    embedding_full=emb_full,
                    embedding_fast=emb_fast,
                    top_k_fast=self._settings.dense_fast_top_k,
                    top_k_full=self._settings.dense_full_top_k,
                )
                channel_results["dense"] = dense_hits
            except Exception as exc:
                _log.warning("dense_search_failed", error=str(exc))
                result.errors.append(f"Dense search failed: {exc}")
            timings["dense_ms"] = (time.monotonic() - t0) * 1000

        # BM25 search
        if sparse_idx is not None and sparse_val is not None and len(sparse_idx) > 0:
            t0 = time.monotonic()
            try:
                bm25_hits = self._sparse.search(
                    sparse_indices=sparse_idx,
                    sparse_values=sparse_val,
                    top_k=self._settings.bm25_top_k,
                )
                channel_results["bm25"] = bm25_hits
            except Exception as exc:
                _log.warning("bm25_search_failed", error=str(exc))
                result.errors.append(f"BM25 search failed: {exc}")
            timings["bm25_ms"] = (time.monotonic() - t0) * 1000

        # QuIM search
        if emb_full is not None:
            t0 = time.monotonic()
            try:
                quim_hits = self._quim.search(
                    embedding=emb_full,
                    top_k=self._settings.quim_top_k,
                )
                channel_results["quim"] = quim_hits
            except Exception as exc:
                _log.warning("quim_search_failed", error=str(exc))
                result.errors.append(f"QuIM search failed: {exc}")
            timings["quim_ms"] = (time.monotonic() - t0) * 1000

        # Graph search (COMPLEX / ANALYTICAL only)
        if query.route in {QueryRoute.COMPLEX, QueryRoute.ANALYTICAL}:
            t0 = time.monotonic()
            try:
                graph_hits = await self._graph.search(query.text)
                channel_results["graph"] = graph_hits
            except Exception as exc:
                _log.warning("graph_search_failed", error=str(exc))
                result.errors.append(f"Graph search failed: {exc}")
            timings["graph_ms"] = (time.monotonic() - t0) * 1000

        result.search_channels_used = list(channel_results.keys())

        # ---- Fuse ----
        t0 = time.monotonic()
        fused = self._fusion.fuse(channel_results, top_k=self._settings.fused_top_k)
        timings["fusion_ms"] = (time.monotonic() - t0) * 1000

        # ---- Rerank ----
        if self._reranker.is_loaded and fused:
            t0 = time.monotonic()
            try:
                fused = self._reranker.rerank(
                    query=query.text,
                    chunks=fused,
                    top_k=self._settings.rerank_top_k,
                )
            except Exception as exc:
                _log.warning("rerank_failed", error=str(exc))
                result.errors.append(f"Reranking failed: {exc}")
                # Fall back to RRF ordering, just truncate
                fused = fused[: self._settings.rerank_top_k]
            timings["rerank_ms"] = (time.monotonic() - t0) * 1000
        else:
            # No reranker available — truncate to rerank_top_k
            fused = fused[: self._settings.rerank_top_k]

        # ---- Expand context ----
        t0 = time.monotonic()
        try:
            expanded = await self._expander.expand(
                fused,
                max_context_tokens=query.max_context_tokens,
            )
        except Exception as exc:
            _log.warning("context_expansion_failed", error=str(exc))
            result.errors.append(f"Context expansion failed: {exc}")
            # Fallback: convert fused → expanded without parent text
            expanded = [
                ExpandedContext(
                    chunk_id=fc.chunk_id,
                    chunk_text=fc.contextualized_text or fc.text,
                    relevance_score=fc.rerank_score
                    if fc.rerank_score is not None
                    else fc.rrf_score,
                    metadata=fc.payload,
                )
                for fc in fused
            ]
        timings["expand_ms"] = (time.monotonic() - t0) * 1000

        # ---- FLARE (ANALYTICAL only) ----
        if query.route == QueryRoute.ANALYTICAL and self._flare.is_available:
            t0 = time.monotonic()
            try:
                expanded, flare_count = await self._flare.active_retrieve(
                    query=query.text,
                    initial_chunks=expanded,
                    search_fn=self.hybrid_search,
                )
                result.flare_retrievals = flare_count
            except FLAREError as exc:
                _log.warning("flare_failed", error=str(exc))
                result.errors.append(f"FLARE failed: {exc}")
            timings["flare_ms"] = (time.monotonic() - t0) * 1000

        # ---- Build result ----
        result.chunks = expanded
        result.total_context_tokens = sum(c.total_tokens for c in expanded)
        result.timings = timings
        result.finished_at = datetime.now(UTC)

        _log.info(
            "retrieval_complete",
            route=query.route,
            channels=result.search_channels_used,
            chunks_returned=len(expanded),
            elapsed_ms=result.elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Lightweight search for Phase 8
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        text: str,
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Lightweight search for Phase 8 GenGround per-claim retrieval.

        Runs dense + BM25 + RRF only (no QuIM, no graph, no rerank, no expand).
        """
        from src.retrieval._models import RetrievalQuery

        query = RetrievalQuery(text=text, max_results=top_k)
        emb_full, emb_fast, sparse_idx, sparse_val = self._prepare_query_vectors(query)

        channel_results: dict[str, list[ScoredChunk]] = {}

        # Dense
        if emb_full is not None and emb_fast is not None:
            try:
                channel_results["dense"] = self._dense.search(
                    embedding_full=emb_full,
                    embedding_fast=emb_fast,
                    top_k_fast=min(200, self._settings.dense_fast_top_k),
                    top_k_full=min(50, self._settings.dense_full_top_k),
                )
            except Exception as exc:
                _log.warning("hybrid_dense_failed", error=str(exc))

        # BM25
        if sparse_idx is not None and sparse_val is not None and len(sparse_idx) > 0:
            try:
                channel_results["bm25"] = self._sparse.search(
                    sparse_indices=sparse_idx,
                    sparse_values=sparse_val,
                    top_k=min(50, self._settings.bm25_top_k),
                )
            except Exception as exc:
                _log.warning("hybrid_bm25_failed", error=str(exc))

        if not channel_results:
            return []

        fused = self._fusion.fuse(channel_results, top_k=top_k)
        # Convert FusedChunks back to ScoredChunks for simple interface
        return [
            ScoredChunk(
                chunk_id=fc.chunk_id,
                text=fc.text,
                contextualized_text=fc.contextualized_text,
                score=fc.rrf_score,
                channel=",".join(fc.channels),
                payload=fc.payload,
            )
            for fc in fused
        ]

    # ------------------------------------------------------------------
    # KG direct query (SIMPLE route)
    # ------------------------------------------------------------------

    async def _kg_direct_query(self, query_text: str) -> dict[str, Any] | None:
        """Extract section ref from query and query KG directly."""
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references(query_text)
        if not refs:
            return None

        section, act = refs[0]
        try:
            self._graph._ensure_client()
            qb = self._graph._query_builder
            if qb is None:
                return None
            status = await qb.temporal_status(section=section, act=act)
            return status
        except Exception as exc:
            _log.warning("kg_direct_query_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_query_vectors(
        self,
        query: RetrievalQuery,
    ) -> tuple[
        list[float] | None,
        list[float] | None,
        list[int] | None,
        list[float] | None,
    ]:
        """Ensure query has full, fast, and sparse vectors.

        Returns (embedding_full, embedding_fast, sparse_indices, sparse_values).
        Any component may be None if the model/vocab is not available.
        """
        emb_full = query.query_embedding
        emb_fast = query.query_embedding_fast
        sparse_idx = query.sparse_indices
        sparse_val = query.sparse_values

        # Compute full embedding if not provided
        if emb_full is None and self._embedder is not None:
            try:
                embeddings = self._embedder.embed_texts([query.text])
                if embeddings:
                    emb_full = embeddings[0].tolist()
            except Exception as exc:
                _log.warning("query_embedding_failed", error=str(exc))

        # Compute fast (Matryoshka) embedding if not provided
        if emb_fast is None and emb_full is not None and self._embedder is not None:
            try:
                import numpy as np

                full_arr = np.array(emb_full, dtype=np.float32)
                fast_arr = self._embedder.matryoshka_slice(full_arr)
                emb_fast = fast_arr.tolist()
            except Exception as exc:
                _log.warning("matryoshka_slice_failed", error=str(exc))

        # Compute sparse vector if not provided
        if sparse_idx is None and self._bm25 is not None:
            try:
                sv = self._bm25.encode(query.text)
                sparse_idx = sv.indices
                sparse_val = sv.values
            except Exception as exc:
                _log.warning("bm25_encode_failed", error=str(exc))

        return emb_full, emb_fast, sparse_idx, sparse_val

    async def close(self) -> None:
        """Release resources held by search components."""
        await self._graph.close()
