"""CLI entry point: python -m src.knowledge_graph.run

Usage:
    python -m src.knowledge_graph.run --source="Indian Kanoon" --console-log
    python -m src.knowledge_graph.run --dry-run
    python -m src.knowledge_graph.run --skip-integrity
    python -m src.knowledge_graph.run  # all sources
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.knowledge_graph.pipeline import KnowledgeGraphPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal RAG Knowledge Graph Pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Source name to ingest (e.g., "Indian Kanoon"). Omit for all.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents but don't ingest into Neo4j.",
    )
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="Skip post-ingestion integrity checks.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Use human-readable console logging instead of JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to knowledge_graph config YAML (default: configs/knowledge_graph.yaml).",
    )

    args = parser.parse_args()

    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    config_path = Path(args.config) if args.config else None
    pipeline = KnowledgeGraphPipeline(config_path=config_path)

    result = asyncio.run(
        pipeline.run(
            source_name=args.source,
            dry_run=args.dry_run,
            skip_integrity=args.skip_integrity,
        )
    )

    status = "OK" if not result.errors else "ERRORS"
    _log.info(
        "result_summary",
        status=status,
        found=result.documents_found,
        ingested=result.documents_ingested,
        skipped=result.documents_skipped,
        failed=result.documents_failed,
        nodes=result.nodes_created,
        relationships=result.relationships_created,
        integrity=result.integrity_passed,
        errors=result.errors,
    )

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
