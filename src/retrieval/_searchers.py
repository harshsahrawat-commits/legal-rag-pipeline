"""Searcher classes for multi-channel retrieval.

Provides four search channels:
- DenseSearcher: Matryoshka 2-stage funnel via Qdrant named vectors
- SparseSearcher: BM25 sparse vector search via Qdrant
- QuIMSearcher: Question-based search against QuIM question embeddings
- GraphSearcher: Knowledge-graph-backed section/judgment lookup
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.retrieval._exceptions import SearchError, SearchNotAvailableError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.retrieval._models import RetrievalSettings, ScoredChunk

_log = get_logger(__name__)

# --- Common helpers ---

_SECTION_PATTERNS: list[re.Pattern[str]] = [
    # "Section 420 of the Indian Penal Code"
    re.compile(
        r"[Ss]ection\s+(\d+[A-Za-z]*)\s+of\s+(?:the\s+)?(.+?)(?:\s*,|\s*\.|$)",
    ),
    # "S. 302 IPC" / "s.302 CrPC"
    re.compile(
        r"[Ss]\.\s*(\d+[A-Za-z]*)\s+(\S+)",
    ),
    # "Sec 10 Contract Act"
    re.compile(
        r"[Ss]ec\.?\s+(\d+[A-Za-z]*)\s+(?:of\s+(?:the\s+)?)?(.+?)(?:\s*,|\s*\.|$)",
    ),
]

# Common abbreviation → full act name
_ACT_ALIASES: dict[str, str] = {
    "ipc": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure",
    "bns": "Bharatiya Nyaya Sanhita",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "bsa": "Bharatiya Sakshya Adhiniyam",
}


def extract_section_references(text: str) -> list[tuple[str, str]]:
    """Extract (section_number, act_name) pairs from query text."""
    refs: list[tuple[str, str]] = []
    for pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            section = match.group(1).strip()
            act_raw = match.group(2).strip().rstrip(",.")
            # Resolve aliases
            act = _ACT_ALIASES.get(act_raw.lower(), act_raw)
            refs.append((section, act))
    return refs


def _point_to_scored_chunk(point: Any, channel: str) -> ScoredChunk:
    """Convert a Qdrant ScoredPoint to a ScoredChunk."""
    from src.retrieval._models import ScoredChunk

    payload = getattr(point, "payload", {}) or {}
    return ScoredChunk(
        chunk_id=payload.get("id", str(getattr(point, "id", ""))),
        text=payload.get("text", ""),
        contextualized_text=payload.get("contextualized_text"),
        score=float(getattr(point, "score", 0.0)),
        channel=channel,
        document_type=payload.get("document_type"),
        chunk_type=payload.get("chunk_type"),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# DenseSearcher
# ---------------------------------------------------------------------------


class DenseSearcher:
    """Matryoshka 2-stage funnel search against legal_chunks collection.

    Stage 1: fast 64-dim vector search for broad recall.
    Stage 2: full 768-dim vector re-search filtered to stage-1 IDs.
    """

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def search(
        self,
        embedding_full: list[float],
        embedding_fast: list[float],
        top_k_fast: int = 1000,
        top_k_full: int = 100,
    ) -> list[ScoredChunk]:
        """Execute 2-stage dense search.

        Args:
            embedding_full: 768-dim query embedding.
            embedding_fast: 64-dim Matryoshka query embedding.
            top_k_fast: Number of candidates from stage 1.
            top_k_full: Final top-k from stage 2.

        Returns:
            List of ScoredChunk with channel="dense".
        """
        self._ensure_client()

        try:
            # Stage 1 — fast vector
            stage1 = self._client.search(
                collection_name=self._settings.chunks_collection,
                query_vector=("fast", embedding_fast),
                limit=top_k_fast,
            )
        except Exception as exc:
            msg = f"Dense search stage-1 failed: {exc}"
            raise SearchError(msg) from exc

        if not stage1:
            return []

        # Collect IDs from stage 1
        candidate_ids = [str(getattr(pt, "id", "")) for pt in stage1]

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            id_filter = Filter(must=[FieldCondition(key="id", match=MatchAny(any=candidate_ids))])

            # Stage 2 — full vector with ID filter
            stage2 = self._client.search(
                collection_name=self._settings.chunks_collection,
                query_vector=("full", embedding_full),
                query_filter=id_filter,
                limit=top_k_full,
            )
        except SearchError:
            raise
        except Exception as exc:
            msg = f"Dense search stage-2 failed: {exc}"
            raise SearchError(msg) from exc

        return [_point_to_scored_chunk(pt, "dense") for pt in (stage2 or [])]

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            msg = "qdrant-client is required. Install with: pip install qdrant-client"
            raise SearchNotAvailableError(msg) from exc
        self._client = QdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
        )


# ---------------------------------------------------------------------------
# SparseSearcher
# ---------------------------------------------------------------------------


class SparseSearcher:
    """BM25 sparse vector search against legal_chunks collection."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def search(
        self,
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 100,
    ) -> list[ScoredChunk]:
        """Execute BM25 sparse search.

        Args:
            sparse_indices: Token indices from BM25 encoder.
            sparse_values: BM25 weights for each token.
            top_k: Number of results.

        Returns:
            List of ScoredChunk with channel="bm25".
        """
        self._ensure_client()

        if not sparse_indices:
            return []

        try:
            from qdrant_client.models import NamedSparseVector
            from qdrant_client.models import SparseVector as QdrantSparseVector

            results = self._client.search(
                collection_name=self._settings.chunks_collection,
                query_vector=NamedSparseVector(
                    name="bm25",
                    vector=QdrantSparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                ),
                limit=top_k,
            )
        except Exception as exc:
            msg = f"BM25 sparse search failed: {exc}"
            raise SearchError(msg) from exc

        return [_point_to_scored_chunk(pt, "bm25") for pt in (results or [])]

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            msg = "qdrant-client is required. Install with: pip install qdrant-client"
            raise SearchNotAvailableError(msg) from exc
        self._client = QdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
        )


# ---------------------------------------------------------------------------
# QuIMSearcher
# ---------------------------------------------------------------------------


class QuIMSearcher:
    """Search QuIM question embeddings for question-based retrieval."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def search(
        self,
        embedding: list[float],
        top_k: int = 50,
    ) -> list[ScoredChunk]:
        """Search quim_questions collection for matching questions.

        The payload contains ``source_chunk_id`` which maps back to the
        original chunk in legal_chunks.  We use that as the ScoredChunk's
        ``chunk_id`` so downstream fusion can de-duplicate correctly.

        Args:
            embedding: Query embedding (768-dim).
            top_k: Number of results.

        Returns:
            List of ScoredChunk with channel="quim".
        """
        from src.retrieval._models import ScoredChunk

        self._ensure_client()

        try:
            results = self._client.search(
                collection_name=self._settings.quim_collection,
                query_vector=embedding,
                limit=top_k,
            )
        except Exception as exc:
            msg = f"QuIM search failed: {exc}"
            raise SearchError(msg) from exc

        chunks: list[ScoredChunk] = []
        for pt in results or []:
            payload = getattr(pt, "payload", {}) or {}
            # Map back to the source chunk
            chunk_id = payload.get("source_chunk_id", str(getattr(pt, "id", "")))
            chunks.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    text=payload.get("question", ""),
                    score=float(getattr(pt, "score", 0.0)),
                    channel="quim",
                    payload=payload,
                )
            )
        return chunks

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            msg = "qdrant-client is required. Install with: pip install qdrant-client"
            raise SearchNotAvailableError(msg) from exc
        self._client = QdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
        )


# ---------------------------------------------------------------------------
# GraphSearcher
# ---------------------------------------------------------------------------


class GraphSearcher:
    """Knowledge-graph-backed section/judgment lookup.

    Parses the query for legal section references (e.g. "Section 302 IPC"),
    then queries Neo4j via QueryBuilder for citation traversals and
    hierarchy navigation.
    """

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._client: Any = None
        self._query_builder: Any = None

    async def search(
        self,
        query_text: str,
        *,
        max_results: int | None = None,
    ) -> list[ScoredChunk]:
        """Extract references from query and look them up in the KG.

        Args:
            query_text: Natural-language query.
            max_results: Cap on returned chunks (defaults to settings.graph_max_results).

        Returns:
            List of ScoredChunk with channel="graph", score=1.0.
        """
        from src.retrieval._models import ScoredChunk

        cap = max_results or self._settings.graph_max_results
        refs = extract_section_references(query_text)
        if not refs:
            _log.debug("graph_search_no_refs", query=query_text)
            return []

        await self._ensure_client_async()

        results: list[ScoredChunk] = []
        for section, act in refs:
            try:
                # Citation traversal — judgments citing/interpreting this section
                citations = await self._query_builder.citation_traversal(section=section, act=act)
                for row in citations or []:
                    results.append(
                        ScoredChunk(
                            chunk_id=row.get("citation", ""),
                            text=f"Judgment {row.get('citation', '')} cites s.{section} of {act}",
                            score=1.0,
                            channel="graph",
                            document_type="judgment",
                            payload=row,
                        )
                    )
            except Exception as exc:
                _log.warning(
                    "graph_citation_traversal_error",
                    section=section,
                    act=act,
                    error=str(exc),
                )

            try:
                # Hierarchy navigation — all sections under the act
                hierarchy = await self._query_builder.hierarchy_navigation(act=act)
                for row in hierarchy or []:
                    chunk_id = f"{act}:s.{row.get('number', '')}"
                    results.append(
                        ScoredChunk(
                            chunk_id=chunk_id,
                            text=f"s.{row.get('number', '')} of {act}",
                            score=1.0,
                            channel="graph",
                            document_type="statute",
                            payload=row,
                        )
                    )
            except Exception as exc:
                _log.warning(
                    "graph_hierarchy_error",
                    act=act,
                    error=str(exc),
                )

        # Deduplicate by chunk_id, keep first occurrence
        seen: set[str] = set()
        deduped: list[ScoredChunk] = []
        for chunk in results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduped.append(chunk)

        return deduped[:cap]

    async def _ensure_client_async(self) -> None:
        """Lazy-initialize Neo4j client and QueryBuilder."""
        if self._query_builder is not None:
            return

        try:
            from src.knowledge_graph._client import Neo4jClient
            from src.knowledge_graph._queries import QueryBuilder
        except ImportError as exc:
            msg = "neo4j is required. Install with: pip install neo4j"
            raise SearchNotAvailableError(msg) from exc

        from src.knowledge_graph._models import KGSettings

        kg_settings = KGSettings(
            neo4j_uri=self._settings.neo4j_uri,
            neo4j_user=self._settings.neo4j_user,
            neo4j_password=self._settings.neo4j_password,
            neo4j_database=self._settings.neo4j_database,
        )
        self._client = Neo4jClient(kg_settings)
        self._query_builder = QueryBuilder(self._client)

    async def close(self) -> None:
        """Close the underlying Neo4j driver."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._query_builder = None
