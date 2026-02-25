"""Knowledge graph pipeline orchestrator.

Reads enriched chunks from data/enriched/, extracts entities and relationships,
and upserts them into Neo4j. Runs post-ingestion integrity checks.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.acquisition._models import SourceType
from src.knowledge_graph._client import Neo4jClient
from src.knowledge_graph._config import load_kg_config
from src.knowledge_graph._extractors import EntityExtractor
from src.knowledge_graph._integrity import IntegrityChecker
from src.knowledge_graph._models import KGConfig, KGResult
from src.knowledge_graph._queries import QueryBuilder
from src.knowledge_graph._relationships import RelationshipBuilder
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk

_log = get_logger(__name__)

_SOURCE_NAME_MAP: dict[str, SourceType] = {
    "indian kanoon": SourceType.INDIAN_KANOON,
    "india code": SourceType.INDIA_CODE,
}


class KnowledgeGraphPipeline:
    """Orchestrates Phase 6: enriched chunks -> Neo4j knowledge graph."""

    def __init__(
        self,
        config: KGConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = load_kg_config(config_path)
        self._settings = self._config.settings
        self._client = Neo4jClient(self._settings)
        self._extractor = EntityExtractor()
        self._relationship_builder = RelationshipBuilder()
        self._integrity_checker = IntegrityChecker(self._client)
        self._query_builder = QueryBuilder(self._client)

    @property
    def client(self) -> Neo4jClient:
        """Expose the Neo4j client for external use."""
        return self._client

    @property
    def query_builder(self) -> QueryBuilder:
        """Expose query builder for downstream phases."""
        return self._query_builder

    async def run(
        self,
        *,
        source_name: str | None = None,
        dry_run: bool = False,
        skip_integrity: bool = False,
    ) -> KGResult:
        """Run the knowledge graph ingestion pipeline."""
        started = datetime.now(UTC)
        result = KGResult(started_at=started)

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

        # Setup schema
        await self._client.setup_schema()

        # Process each document
        for enriched_path in enriched_files:
            await self._process_document(enriched_path, result)

        # Run integrity checks
        if not skip_integrity and result.documents_ingested > 0:
            try:
                report = await self._integrity_checker.check_all()
                result.integrity_passed = report.passed
                if not report.passed:
                    for check in report.checks:
                        for violation in check.violations:
                            result.errors.append(f"Integrity: {violation}")
            except Exception as exc:
                _log.error("integrity_check_failed", error=str(exc))
                result.errors.append(f"Integrity check failed: {exc}")

        result.finished_at = datetime.now(UTC)
        _log.info(
            "pipeline_complete",
            ingested=result.documents_ingested,
            skipped=result.documents_skipped,
            failed=result.documents_failed,
            nodes=result.nodes_created,
            relationships=result.relationships_created,
        )
        return result

    async def close(self) -> None:
        """Shut down the Neo4j client."""
        await self._client.close()

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

    async def _process_document(self, enriched_path: Path, result: KGResult) -> None:
        """Load, extract, and ingest one document's entities and relationships."""
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

        try:
            doc_nodes = 0
            doc_rels = 0

            for chunk in chunks:
                # Extract entities
                entities = self._extractor.extract_from_chunk(chunk)

                # Merge nodes
                for act in entities.acts:
                    await self._client.merge_act(act)
                    doc_nodes += 1
                for section in entities.sections:
                    await self._client.merge_section(section)
                    doc_nodes += 1
                for sv in entities.section_versions:
                    await self._client.merge_section_version(sv)
                    doc_nodes += 1
                for judgment in entities.judgments:
                    await self._client.merge_judgment(judgment)
                    doc_nodes += 1
                for amendment in entities.amendments:
                    await self._client.merge_amendment(amendment)
                    doc_nodes += 1
                for concept in entities.legal_concepts:
                    await self._client.merge_legal_concept(concept)
                    doc_nodes += 1
                for court in entities.courts:
                    await self._client.merge_court(court)
                    doc_nodes += 1
                for judge in entities.judges:
                    await self._client.merge_judge(judge)
                    doc_nodes += 1

                # Build and merge relationships
                rels = self._relationship_builder.build_from_chunk(chunk, entities)
                for rel in rels:
                    await self._client.create_relationship(rel)
                    doc_rels += 1

            result.nodes_created += doc_nodes
            result.relationships_created += doc_rels
            result.documents_ingested += 1
            _log.info(
                "document_ingested",
                path=str(enriched_path.name),
                nodes=doc_nodes,
                relationships=doc_rels,
            )

        except Exception as exc:
            _log.error(
                "ingestion_failed",
                path=str(enriched_path.name),
                error=str(exc),
            )
            result.documents_failed += 1
            result.errors.append(f"Failed to ingest {enriched_path.name}: {exc}")


def _resolve_source_filter(source_name: str | None) -> SourceType | None:
    if source_name is None:
        return None
    return _SOURCE_NAME_MAP.get(source_name.lower().strip())


def _load_chunks(path: Path) -> list[LegalChunk]:
    from src.chunking._models import LegalChunk

    data = json.loads(path.read_text(encoding="utf-8"))
    return [LegalChunk.model_validate(c) for c in data]
