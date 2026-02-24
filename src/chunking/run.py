"""CLI entry point: python -m src.chunking.run

Usage:
    python -m src.chunking.run --source="Indian Kanoon" --console-log
    python -m src.chunking.run --dry-run
    python -m src.chunking.run  # all sources
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.chunking.pipeline import ChunkingPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal RAG Chunking Pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Source name to chunk (e.g., "Indian Kanoon"). Omit for all.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents but don't chunk or write output.",
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
        help="Path to chunking config YAML (default: configs/chunking.yaml).",
    )

    args = parser.parse_args()

    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    config_path = Path(args.config) if args.config else None
    pipeline = ChunkingPipeline(config_path=config_path)
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
        chunked=result.documents_chunked,
        skipped=result.documents_skipped,
        failed=result.documents_failed,
        chunks_created=result.chunks_created,
        errors=result.errors,
    )

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
