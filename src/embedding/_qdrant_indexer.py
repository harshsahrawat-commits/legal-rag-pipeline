"""Qdrant indexer for dual-vector + sparse BM25 storage.

Creates two collections:
- legal_chunks: named vectors "full" (768d) + "fast" (64d), sparse "bm25", full payload
- quim_questions: single 768d vector + source_chunk_id payload
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002 — used at runtime for .tolist()

from src.embedding._exceptions import (
    CollectionCreationError,
    EmbedderNotAvailableError,
    IndexingError,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from src.chunking._models import LegalChunk
    from src.embedding._models import EmbeddingSettings, SparseVector
    from src.enrichment._models import QuIMDocument

_log = get_logger(__name__)


class QdrantIndexer:
    """Manages Qdrant collections and upserts for the embedding pipeline."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._client = None

    def ensure_collections(self) -> None:
        """Create or verify legal_chunks and quim_questions collections."""
        self._ensure_client()

        try:
            self._create_chunks_collection()
            self._create_quim_collection()
        except Exception as exc:
            if isinstance(exc, CollectionCreationError):
                raise
            msg = f"Failed to ensure collections: {exc}"
            raise CollectionCreationError(msg) from exc

    async def upsert_chunks(
        self,
        chunks: list[LegalChunk],
        full_embeddings: list[np.ndarray],
        fast_embeddings: list[np.ndarray],
        sparse_vectors: list[SparseVector],
    ) -> int:
        """Upsert chunks with dual vectors and sparse BM25 into legal_chunks.

        Returns:
            Number of points upserted.
        """
        self._ensure_client()
        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector as QdrantSparseVector

        points = []
        for chunk, full_emb, fast_emb, sparse in zip(
            chunks,
            full_embeddings,
            fast_embeddings,
            sparse_vectors,
            strict=True,
        ):
            payload = self._build_chunk_payload(chunk)
            vectors = {
                "full": full_emb.tolist(),
                "fast": fast_emb.tolist(),
            }
            sparse_vectors_dict = {}
            if sparse.indices:
                sparse_vectors_dict["bm25"] = QdrantSparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                )

            point = PointStruct(
                id=str(chunk.id),
                vector={**vectors, **sparse_vectors_dict},
                payload=payload,
            )
            points.append(point)

        if not points:
            return 0

        try:
            self._client.upsert(
                collection_name=self._settings.chunks_collection,
                points=points,
            )
            _log.info("chunks_upserted", count=len(points))
            return len(points)
        except Exception as exc:
            msg = f"Failed to upsert {len(points)} chunks: {exc}"
            raise IndexingError(msg) from exc

    async def upsert_quim_questions(
        self,
        quim_doc: QuIMDocument,
        question_embeddings: list[np.ndarray],
    ) -> int:
        """Upsert QuIM questions with embeddings into quim_questions collection.

        Returns:
            Number of points upserted.
        """
        self._ensure_client()
        from qdrant_client.models import PointStruct

        points = []
        emb_idx = 0
        for entry in quim_doc.entries:
            for question in entry.questions:
                if emb_idx >= len(question_embeddings):
                    break
                point = PointStruct(
                    id=str(f"{entry.chunk_id}-q{emb_idx}"),
                    vector=question_embeddings[emb_idx].tolist(),
                    payload={
                        "source_chunk_id": str(entry.chunk_id),
                        "document_id": str(entry.document_id),
                        "question": question,
                    },
                )
                points.append(point)
                emb_idx += 1

        if not points:
            return 0

        try:
            self._client.upsert(
                collection_name=self._settings.quim_collection,
                points=points,
            )
            _log.info("quim_questions_upserted", count=len(points))
            return len(points)
        except Exception as exc:
            msg = f"Failed to upsert {len(points)} QuIM questions: {exc}"
            raise IndexingError(msg) from exc

    async def chunk_exists(self, chunk_id: UUID) -> bool:
        """Check if a chunk is already indexed (idempotency check)."""
        self._ensure_client()
        try:
            results = self._client.retrieve(
                collection_name=self._settings.chunks_collection,
                ids=[str(chunk_id)],
            )
            return len(results) > 0
        except Exception:
            return False

    def _build_chunk_payload(self, chunk: LegalChunk) -> dict:
        """Build the Qdrant payload dict from a LegalChunk."""
        return chunk.model_dump(mode="json")

    def _ensure_client(self) -> None:
        """Lazy-initialize the Qdrant client."""
        if self._client is not None:
            return

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            msg = "qdrant-client is required. Install with: pip install qdrant-client"
            raise EmbedderNotAvailableError(msg) from exc

        self._client = QdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
        )

    def _create_chunks_collection(self) -> None:
        """Create the legal_chunks collection with dual vectors + sparse."""
        from qdrant_client.models import (
            Distance,
            SparseVectorParams,
            VectorParams,
        )

        name = self._settings.chunks_collection
        if self._client.collection_exists(name):
            _log.debug("collection_exists", name=name)
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config={
                "full": VectorParams(
                    size=self._settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
                "fast": VectorParams(
                    size=self._settings.matryoshka_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(),
            },
        )
        _log.info("collection_created", name=name)

    def _create_quim_collection(self) -> None:
        """Create the quim_questions collection."""
        from qdrant_client.models import Distance, VectorParams

        name = self._settings.quim_collection
        if self._client.collection_exists(name):
            _log.debug("collection_exists", name=name)
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self._settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        _log.info("collection_created", name=name)
