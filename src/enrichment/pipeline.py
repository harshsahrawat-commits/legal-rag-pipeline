"""Enrichment pipeline orchestrator.

Scans Phase 3 output (``data/chunks/{source}/``), loads each document's
chunks and corresponding parsed document, runs enrichment stages
(contextual retrieval, QuIM-RAG), and saves enriched output to
``data/enriched/{source}/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.acquisition._models import SourceType
from src.enrichment._config import load_enrichment_config
from src.enrichment._models import EnrichmentConfig, EnrichmentResult
from src.enrichment.enrichers._contextual import ContextualRetrievalEnricher
from src.enrichment.enrichers._quim import QuIMRagEnricher
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


class EnrichmentPipeline:
    """Orchestrates Phase 4: chunks -> enriched chunks."""

    def __init__(
        self,
        config: EnrichmentConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = load_enrichment_config(config_path)
        self._settings = self._config.settings
        self._contextual = ContextualRetrievalEnricher(self._settings)
        self._quim = QuIMRagEnricher(self._settings)

    async def run(
        self,
        *,
        source_name: str | None = None,
        stage: str | None = None,
        dry_run: bool = False,
    ) -> EnrichmentResult:
        """Run the enrichment pipeline.

        Args:
            source_name: Limit to a specific source (e.g. 'indian kanoon').
            stage: Run only one stage: 'contextual_retrieval' or 'quim_rag'.
                   None runs both stages in order.
            dry_run: Discover files but don't process them.
        """
        started = datetime.now(UTC)
        result = EnrichmentResult(started_at=started)

        source_filter = _resolve_source_filter(source_name)
        if source_name is not None and source_filter is None:
            _log.error("unknown_source_name", name=source_name)
            result.errors.append(f"Unknown source: {source_name}")
            result.finished_at = datetime.now(UTC)
            return result

        chunk_files = self._discover_chunk_files(source_filter)
        result.documents_found = len(chunk_files)

        _log.info(
            "pipeline_starting",
            documents_found=len(chunk_files),
            source_filter=source_name,
            stage=stage,
            dry_run=dry_run,
        )

        if dry_run:
            result.finished_at = datetime.now(UTC)
            return result

        for chunk_path in chunk_files:
            await self._process_document(chunk_path, result, stage)

        result.finished_at = datetime.now(UTC)
        _log.info(
            "pipeline_complete",
            enriched=result.documents_enriched,
            skipped=result.documents_skipped,
            failed=result.documents_failed,
            contextualized=result.chunks_contextualized,
            quim_generated=result.chunks_quim_generated,
        )
        return result

    def _discover_chunk_files(self, source_filter: SourceType | None) -> list[Path]:
        """Find all chunk JSON files in the input directory."""
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
        chunk_path: Path,
        result: EnrichmentResult,
        stage: str | None,
    ) -> None:
        """Load, enrich, and save one document's chunks."""
        output_path = self._output_path_for(chunk_path)
        quim_path = self._quim_path_for(chunk_path)

        # Idempotency: skip if output already exists
        run_contextual = stage is None or stage == "contextual_retrieval"
        run_quim = stage is None or stage == "quim_rag"

        if run_contextual and run_quim and output_path.exists() and quim_path.exists():
            _log.debug("already_enriched", path=str(chunk_path))
            result.documents_skipped += 1
            return
        if stage == "contextual_retrieval" and output_path.exists():
            _log.debug("already_contextualized", path=str(chunk_path))
            result.documents_skipped += 1
            return
        if stage == "quim_rag" and quim_path.exists():
            _log.debug("already_quim", path=str(chunk_path))
            result.documents_skipped += 1
            return

        # Load chunks
        try:
            chunks = _load_chunks(chunk_path)
        except Exception as exc:
            _log.error("load_chunks_failed", path=str(chunk_path), error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Failed to load {chunk_path.name}: {exc}")
            return

        # Load parsed document (for full text context)
        parsed_doc = self._load_parsed_doc(chunk_path)

        try:
            # Stage 1: Contextual Retrieval
            if run_contextual:
                chunks = await self._contextual.enrich_document(chunks, parsed_doc)
                ctx_count = sum(1 for c in chunks if c.ingestion.contextualized)
                result.chunks_contextualized += ctx_count

            # Stage 2: QuIM-RAG
            quim_doc = None
            if run_quim:
                chunks = await self._quim.enrich_document(chunks, parsed_doc)
                quim_doc = self._quim.get_quim_document()
                quim_count = sum(c.ingestion.quim_questions for c in chunks)
                result.chunks_quim_generated += quim_count

            # Save enriched chunks
            _save_enriched_chunks(chunks, output_path)

            # Save QuIM document if generated
            if quim_doc is not None and quim_doc.entries:
                _save_quim_document(quim_doc, quim_path)

            result.documents_enriched += 1
            _log.info(
                "document_enriched",
                path=str(chunk_path.name),
                chunks=len(chunks),
            )

        except Exception as exc:
            _log.error(
                "enrichment_failed",
                path=str(chunk_path.name),
                error=str(exc),
            )
            result.documents_failed += 1
            result.errors.append(f"Failed to enrich {chunk_path.name}: {exc}")

    def _output_path_for(self, chunk_path: Path) -> Path:
        """Compute the enriched output path mirroring the source structure."""
        relative = chunk_path.relative_to(self._settings.input_dir)
        return Path(self._settings.output_dir) / relative

    def _quim_path_for(self, chunk_path: Path) -> Path:
        """Compute the QuIM sidecar path."""
        output = self._output_path_for(chunk_path)
        return output.with_suffix(".quim.json")

    def _load_parsed_doc(self, chunk_path: Path) -> ParsedDocument:
        """Load the corresponding ParsedDocument for full document text.

        Falls back to a minimal stub if the parsed file doesn't exist.
        """
        from src.acquisition._models import ContentFormat, DocumentType
        from src.parsing._models import ParsedDocument, ParserType, QualityReport

        relative = chunk_path.relative_to(self._settings.input_dir)
        parsed_path = Path(self._settings.parsed_dir) / relative

        if parsed_path.exists():
            try:
                raw = parsed_path.read_text(encoding="utf-8")
                return ParsedDocument.model_validate_json(raw)
            except Exception:
                _log.warning("parsed_doc_load_failed", path=str(parsed_path))

        _log.warning("parsed_doc_not_found", path=str(parsed_path))
        # Return a minimal stub so enrichers can still run (with empty context)
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


def _resolve_source_filter(source_name: str | None) -> SourceType | None:
    if source_name is None:
        return None
    return _SOURCE_NAME_MAP.get(source_name.lower().strip())


def _load_chunks(path: Path) -> list[LegalChunk]:
    from src.chunking._models import LegalChunk

    data = json.loads(path.read_text(encoding="utf-8"))
    return [LegalChunk.model_validate(c) for c in data]


def _save_enriched_chunks(chunks: list[LegalChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump(mode="json") for chunk in chunks]
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _log.debug("enriched_chunks_saved", path=str(output_path), count=len(chunks))


def _save_quim_document(quim_doc: QuIMDocument, quim_path: Path) -> None:
    quim_path.parent.mkdir(parents=True, exist_ok=True)
    quim_path.write_text(quim_doc.model_dump_json(indent=2), encoding="utf-8")
    _log.debug("quim_saved", path=str(quim_path))
