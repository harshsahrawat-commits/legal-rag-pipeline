"""CLI entry point: python -m src.parsing.run

Usage:
    python -m src.parsing.run --source="India Code" --console-log
    python -m src.parsing.run --dry-run
    python -m src.parsing.run  # all sources
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.parsing.pipeline import ParsingPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal RAG Parsing Pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Source name to parse (e.g., "India Code"). Omit for all.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents but don't parse or write output.",
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
        help="Path to parsing config YAML (default: configs/parsing.yaml).",
    )

    args = parser.parse_args()

    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    config_path = Path(args.config) if args.config else None
    pipeline = ParsingPipeline(config_path=config_path)
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
        parsed=result.documents_parsed,
        skipped=result.documents_skipped,
        failed=result.documents_failed,
        errors=result.errors,
    )

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
