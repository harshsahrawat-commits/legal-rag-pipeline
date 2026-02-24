"""Chunking pipeline orchestrator.

Scans Phase 2 output (``data/parsed/{source}/``), loads each
``ParsedDocument``, routes to the appropriate chunker, and saves
``list[LegalChunk]`` JSON to ``data/chunks/{source}/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.acquisition._models import SourceType
from src.chunking._config import load_chunking_config
from src.chunking._models import ChunkingConfig, ChunkingResult, LegalChunk
from src.chunking._router import ChunkerRouter
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._judgment_structural import JudgmentStructuralChunker
from src.chunking.chunkers._page_level import PageLevelChunker
from src.chunking.chunkers._statute_boundary import StatuteBoundaryChunker
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument

_log = get_logger(__name__)

_SOURCE_NAME_MAP: dict[str, SourceType] = {
    "indian kanoon": SourceType.INDIAN_KANOON,
    "india code": SourceType.INDIA_CODE,
}


class ChunkingPipeline:
    """Orchestrates Phase 3: parsed documents -> chunks."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = load_chunking_config(config_path)
        self._settings = self._config.settings
        self._tc = TokenCounter()
        self._router = self._build_router()

    def _build_router(self) -> ChunkerRouter:
        """Register chunkers in priority order."""
        router = ChunkerRouter(self._settings, self._tc)
        router.register(StatuteBoundaryChunker(self._settings, self._tc))
        router.register(JudgmentStructuralChunker(self._settings, self._tc))
        # Optional chunkers registered here if available
        self._register_optional_chunkers(router)
        # PageLevel is always last (universal fallback)
        router.register(PageLevelChunker(self._settings, self._tc))
        return router

    def _register_optional_chunkers(self, router: ChunkerRouter) -> None:
        """Try to register optional chunkers (RSC, Semantic, Proposition)."""
        try:
            from src.chunking.chunkers._recursive_semantic import RecursiveSemanticChunker

            router.register(RecursiveSemanticChunker(self._settings, self._tc))
        except ImportError:
            _log.debug("rsc_chunker_unavailable")

        try:
            from src.chunking.chunkers._semantic_maxmin import SemanticMaxMinChunker

            router.register(SemanticMaxMinChunker(self._settings, self._tc))
        except ImportError:
            _log.debug("semantic_chunker_unavailable")

    async def run(
        self,
        *,
        source_name: str | None = None,
        dry_run: bool = False,
    ) -> ChunkingResult:
        """Run the chunking pipeline."""
        started = datetime.now(UTC)
        result = ChunkingResult(started_at=started)

        source_filter = _resolve_source_filter(source_name)
        if source_name is not None and source_filter is None:
            _log.error("unknown_source_name", name=source_name)
            result.errors.append(f"Unknown source: {source_name}")
            result.finished_at = datetime.now(UTC)
            return result

        parsed_files = self._discover_parsed_files(source_filter)
        result.documents_found = len(parsed_files)

        _log.info(
            "pipeline_starting",
            documents_found=len(parsed_files),
            source_filter=source_name,
            dry_run=dry_run,
        )

        if dry_run:
            result.finished_at = datetime.now(UTC)
            return result

        for parsed_path in parsed_files:
            await self._process_document(parsed_path, result)

        result.finished_at = datetime.now(UTC)
        _log.info(
            "pipeline_complete",
            chunked=result.documents_chunked,
            skipped=result.documents_skipped,
            failed=result.documents_failed,
            chunks_created=result.chunks_created,
        )
        return result

    def _discover_parsed_files(self, source_filter: SourceType | None) -> list[Path]:
        """Find all ``.json`` files in the parsed output directory."""
        input_dir = Path(self._settings.input_dir)
        if not input_dir.exists():
            _log.warning("input_dir_missing", path=str(input_dir))
            return []

        if source_filter is not None:
            source_dir = input_dir / source_filter.value
            if not source_dir.exists():
                return []
            return sorted(p for p in source_dir.glob("*.json") if not p.name.endswith(".meta.json"))

        return sorted(p for p in input_dir.glob("**/*.json") if not p.name.endswith(".meta.json"))

    async def _process_document(
        self,
        parsed_path: Path,
        result: ChunkingResult,
    ) -> None:
        """Load, chunk, post-process, and save one document."""
        try:
            doc = _load_parsed_document(parsed_path)
        except Exception as exc:
            _log.error("load_failed", path=str(parsed_path), error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Failed to load {parsed_path.name}: {exc}")
            return

        doc_id = str(doc.document_id)

        # Idempotency: skip if output already exists
        output_path = self._output_path_for(doc, parsed_path)
        if output_path.exists():
            _log.debug("already_chunked", doc_id=doc_id)
            result.documents_skipped += 1
            return

        try:
            chunker = self._router.select(doc)
            chunks = chunker.chunk(doc)
        except Exception as exc:
            _log.error("chunk_failed", doc_id=doc_id, error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Failed to chunk {doc_id}: {exc}")
            return

        # Post-processing: assign chunk_index and sibling_chunk_ids
        _assign_chunk_indices(chunks)
        _assign_sibling_ids(chunks)

        # Save
        _save_chunks(chunks, output_path)
        result.documents_chunked += 1
        result.chunks_created += len(chunks)
        _log.info(
            "document_chunked",
            doc_id=doc_id,
            strategy=chunker.strategy.value,
            chunks=len(chunks),
        )

    def _output_path_for(self, doc: ParsedDocument, parsed_path: Path) -> Path:
        """Compute the output path for chunks."""
        # Mirror source directory structure
        source_dir = parsed_path.parent.name
        output_dir = Path(self._settings.output_dir) / source_dir
        return output_dir / parsed_path.name


def _resolve_source_filter(source_name: str | None) -> SourceType | None:
    if source_name is None:
        return None
    return _SOURCE_NAME_MAP.get(source_name.lower().strip())


def _load_parsed_document(path: Path) -> ParsedDocument:
    from src.parsing._models import ParsedDocument

    raw_json = path.read_text(encoding="utf-8")
    return ParsedDocument.model_validate_json(raw_json)


def _assign_chunk_indices(chunks: list[LegalChunk]) -> None:
    """Assign sequential ``chunk_index`` starting from 0."""
    for idx, chunk in enumerate(chunks):
        chunk.chunk_index = idx


def _assign_sibling_ids(chunks: list[LegalChunk], window: int = 2) -> None:
    """Assign ``sibling_chunk_ids`` with a +-window around each chunk."""
    chunk_ids = [c.id for c in chunks]
    for idx, chunk in enumerate(chunks):
        start = max(0, idx - window)
        end = min(len(chunk_ids), idx + window + 1)
        siblings = [chunk_ids[i] for i in range(start, end) if i != idx]
        chunk.parent_info.sibling_chunk_ids = siblings


def _save_chunks(chunks: list[LegalChunk], output_path: Path) -> None:
    """Serialize and save chunks as a JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump(mode="json") for chunk in chunks]
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _log.debug("chunks_saved", path=str(output_path), count=len(chunks))
