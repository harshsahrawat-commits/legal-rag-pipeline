"""Embedding pipeline orchestrator.

Scans Phase 4 output (``data/enriched/{source}/``), loads each document's
enriched chunks, embeds via Late Chunking, indexes into Qdrant (dual vectors
+ BM25 sparse), embeds QuIM questions, and stores parent text in Redis.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.acquisition._models import SourceType
from src.embedding._config import load_embedding_config
from src.embedding._embedder import LateChunkingEmbedder
from src.embedding._models import EmbeddingConfig, EmbeddingResult
from src.embedding._qdrant_indexer import QdrantIndexer
from src.embedding._redis_store import RedisParentStore
from src.embedding._sparse import BM25SparseEncoder
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.enrichment._models import QuIMDocument
    from src.parsing._models import ParsedDocument

_log = get_logger(__name__)

_SOURCE_NAME_MAP: dict[str, SourceType] = {
    "indian kanoon": SourceType.INDIAN_KANOON,
    "india code": SourceType.INDIA_CODE,
}


class EmbeddingPipeline:
    """Orchestrates Phase 5: enriched chunks -> embeddings + indexes."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = load_embedding_config(config_path)
        self._settings = self._config.settings
        self._embedder = LateChunkingEmbedder(self._settings)
        self._indexer = QdrantIndexer(self._settings)
        self._redis = RedisParentStore(self._settings)
        self._sparse = BM25SparseEncoder()

    async def run(
        self,
        *,
        source_name: str | None = None,
        dry_run: bool = False,
    ) -> EmbeddingResult:
        """Run the embedding pipeline.

        Args:
            source_name: Limit to a specific source (e.g. 'Indian Kanoon').
            dry_run: Discover files but don't process them.
        """
        started = datetime.now(UTC)
        result = EmbeddingResult(started_at=started)

        source_filter = _resolve_source_filter(source_name)
        if source_name is not None and source_filter is None:
            _log.error("unknown_source_name", name=source_name)
            result.errors.append(f"Unknown source: {source_name}")
            result.finished_at = datetime.now(UTC)
            return result

        enriched_files = self._discover_enriched_files(source_filter)
        result.documents_found = len(enriched_files)

        _log.info(
            "pipeline_starting",
            documents_found=len(enriched_files),
            source_filter=source_name,
            dry_run=dry_run,
        )

        if dry_run:
            result.finished_at = datetime.now(UTC)
            return result

        # Load model and ensure collections
        self._embedder.load_model()
        self._indexer.ensure_collections()

        for enriched_path in enriched_files:
            await self._process_document(enriched_path, result)

        result.finished_at = datetime.now(UTC)
        _log.info(
            "pipeline_complete",
            indexed=result.documents_indexed,
            skipped=result.documents_skipped,
            failed=result.documents_failed,
            chunks_embedded=result.chunks_embedded,
            quim_embedded=result.quim_questions_embedded,
            parents_stored=result.parent_entries_stored,
        )
        return result

    def _discover_enriched_files(self, source_filter: SourceType | None) -> list[Path]:
        """Find all enriched chunk JSON files in the input directory."""
        input_dir = Path(self._settings.input_dir)
        if not input_dir.exists():
            _log.warning("input_dir_missing", path=str(input_dir))
            return []

        if source_filter is not None:
            source_dir = input_dir / source_filter.value
            if not source_dir.exists():
                return []
            return sorted(p for p in source_dir.glob("*.json") if not p.name.endswith(".quim.json"))

        return sorted(p for p in input_dir.glob("**/*.json") if not p.name.endswith(".quim.json"))

    async def _process_document(
        self,
        enriched_path: Path,
        result: EmbeddingResult,
    ) -> None:
        """Load, embed, and index one document's chunks."""
        # Load enriched chunks
        try:
            chunks = _load_chunks(enriched_path)
        except Exception as exc:
            _log.error("load_chunks_failed", path=str(enriched_path), error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Failed to load {enriched_path.name}: {exc}")
            return

        if not chunks:
            result.documents_skipped += 1
            return

        # Idempotency: check if first chunk already indexed
        if self._settings.skip_existing:
            try:
                if await self._indexer.chunk_exists(chunks[0].id):
                    _log.debug("already_indexed", path=str(enriched_path.name))
                    result.documents_skipped += 1
                    return
            except Exception:
                pass  # Proceed with indexing if check fails

        try:
            # Load parsed document for full text
            parsed_doc = self._load_parsed_doc(enriched_path)
            full_text = parsed_doc.raw_text or ""

            # Step 1: Late Chunking embedding
            if full_text:
                full_embeddings = self._embedder.embed_document_late_chunking(full_text, chunks)
            else:
                # Fallback: standard embedding per chunk
                full_embeddings = self._embedder.embed_texts([c.text for c in chunks])

            # Step 2: Matryoshka slice for fast vectors
            fast_embeddings = [self._embedder.matryoshka_slice(e) for e in full_embeddings]

            # Step 3: BM25 sparse encoding
            bm25_texts = [c.contextualized_text or c.text for c in chunks]
            self._sparse.build_vocabulary(bm25_texts)
            sparse_vectors = [self._sparse.encode(t) for t in bm25_texts]

            # Step 4: Qdrant upsert
            upserted = await self._indexer.upsert_chunks(
                chunks,
                full_embeddings,
                fast_embeddings,
                sparse_vectors,
            )
            result.chunks_embedded += upserted

            # Step 5: QuIM embedding + upsert
            quim_doc = self._load_quim_doc(enriched_path)
            if quim_doc is not None and quim_doc.entries:
                quim_count = await self._process_quim(quim_doc)
                result.quim_questions_embedded += quim_count

            # Step 6: Redis parent store
            parent_count = await self._redis.store_parents(chunks)
            result.parent_entries_stored += parent_count

            # Step 7: Update late_chunked flag
            self._update_late_chunked_flag(chunks, enriched_path)

            result.documents_indexed += 1
            _log.info(
                "document_indexed",
                path=str(enriched_path.name),
                chunks=len(chunks),
            )

        except Exception as exc:
            _log.error(
                "embedding_failed",
                path=str(enriched_path.name),
                error=str(exc),
            )
            result.documents_failed += 1
            result.errors.append(f"Failed to embed {enriched_path.name}: {exc}")

    async def _process_quim(self, quim_doc: QuIMDocument) -> int:
        """Embed and index QuIM questions."""
        all_questions = []
        for entry in quim_doc.entries:
            all_questions.extend(entry.questions)

        if not all_questions:
            return 0

        embeddings = self._embedder.embed_texts(all_questions)
        return await self._indexer.upsert_quim_questions(quim_doc, embeddings)

    def _load_parsed_doc(self, enriched_path: Path) -> ParsedDocument:
        """Load the corresponding ParsedDocument for full document text."""
        from src.acquisition._models import ContentFormat, DocumentType
        from src.parsing._models import ParsedDocument, ParserType, QualityReport

        relative = enriched_path.relative_to(self._settings.input_dir)
        parsed_path = Path(self._settings.parsed_dir) / relative

        if parsed_path.exists():
            try:
                raw = parsed_path.read_text(encoding="utf-8")
                return ParsedDocument.model_validate_json(raw)
            except Exception:
                _log.warning("parsed_doc_load_failed", path=str(parsed_path))

        _log.warning("parsed_doc_not_found", path=str(parsed_path))
        from src.acquisition._models import SourceType

        return ParsedDocument(
            source_type=SourceType.INDIAN_KANOON,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.HTML,
            raw_text="",
            parser_used=ParserType.HTML_INDIAN_KANOON,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path="",
        )

    def _load_quim_doc(self, enriched_path: Path) -> QuIMDocument | None:
        """Load the QuIM sidecar if it exists."""
        from src.enrichment._models import QuIMDocument

        quim_path = enriched_path.with_suffix(".quim.json")
        if not quim_path.exists():
            return None

        try:
            raw = quim_path.read_text(encoding="utf-8")
            return QuIMDocument.model_validate_json(raw)
        except Exception:
            _log.warning("quim_load_failed", path=str(quim_path))
            return None

    def _update_late_chunked_flag(
        self,
        chunks: list[LegalChunk],
        enriched_path: Path,
    ) -> None:
        """Set ingestion.late_chunked=True on all chunks and save back."""
        for chunk in chunks:
            chunk.ingestion.late_chunked = True
        data = [chunk.model_dump(mode="json") for chunk in chunks]
        enriched_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve_source_filter(source_name: str | None) -> SourceType | None:
    if source_name is None:
        return None
    return _SOURCE_NAME_MAP.get(source_name.lower().strip())


def _load_chunks(path: Path) -> list[LegalChunk]:
    from src.chunking._models import LegalChunk

    data = json.loads(path.read_text(encoding="utf-8"))
    return [LegalChunk.model_validate(c) for c in data]
