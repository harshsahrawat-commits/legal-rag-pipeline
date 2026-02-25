"""CLI entry point: python -m src.embedding.run

Usage:
    python -m src.embedding.run --source="Indian Kanoon" --console-log
    python -m src.embedding.run --dry-run
    python -m src.embedding.run --device=cuda
    python -m src.embedding.run  # all sources
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.embedding.pipeline import EmbeddingPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal RAG Embedding Pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Source name to embed (e.g., "Indian Kanoon"). Omit for all.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents but don't embed or index.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference (cpu, cuda). Overrides config.",
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
        help="Path to embedding config YAML (default: configs/embedding.yaml).",
    )

    args = parser.parse_args()

    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    config_path = Path(args.config) if args.config else None
    pipeline = EmbeddingPipeline(config_path=config_path)

    # Override device from CLI if specified
    if args.device:
        pipeline._settings.device = args.device

    result = asyncio.run(
        pipeline.run(
            source_name=args.source,
            dry_run=args.dry_run,
        )
    )

    status = "OK" if not result.errors else "ERRORS"
    _log.info(
        "result_summary",
        status=status,
        found=result.documents_found,
        indexed=result.documents_indexed,
        skipped=result.documents_skipped,
        failed=result.documents_failed,
        chunks_embedded=result.chunks_embedded,
        quim_embedded=result.quim_questions_embedded,
        parents_stored=result.parent_entries_stored,
        errors=result.errors,
    )

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
