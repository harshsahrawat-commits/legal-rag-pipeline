"""CLI entry point: python -m src.enrichment.run

Usage:
    python -m src.enrichment.run --source="Indian Kanoon" --console-log
    python -m src.enrichment.run --stage=contextual_retrieval --dry-run
    python -m src.enrichment.run --stage=quim_rag
    python -m src.enrichment.run  # all stages, all sources
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.enrichment.pipeline import EnrichmentPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal RAG Enrichment Pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Source name to enrich (e.g., "Indian Kanoon"). Omit for all.',
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["contextual_retrieval", "quim_rag"],
        help="Run only one enrichment stage. Omit for all stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents but don't enrich or write output.",
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
        help="Path to enrichment config YAML (default: configs/enrichment.yaml).",
    )

    args = parser.parse_args()

    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    config_path = Path(args.config) if args.config else None
    pipeline = EnrichmentPipeline(config_path=config_path)
    result = asyncio.run(
        pipeline.run(
            source_name=args.source,
            stage=args.stage,
            dry_run=args.dry_run,
        )
    )

    status = "OK" if not result.errors else "ERRORS"
    _log.info(
        "result_summary",
        status=status,
        found=result.documents_found,
        enriched=result.documents_enriched,
        skipped=result.documents_skipped,
        failed=result.documents_failed,
        contextualized=result.chunks_contextualized,
        quim_generated=result.chunks_quim_generated,
        errors=result.errors,
    )

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
