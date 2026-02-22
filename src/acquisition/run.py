"""CLI entry point: python -m src.acquisition.run

Usage:
    python -m src.acquisition.run --source="Indian Kanoon" --mode=incremental
    python -m src.acquisition.run --mode=full --dry-run
    python -m src.acquisition.run  # all sources, incremental
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.acquisition.pipeline import AcquisitionPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal RAG Acquisition Pipeline",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help='Source name to scrape (e.g., "Indian Kanoon"). Omit for all.',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental", "full"],
        default="incremental",
        help="incremental (skip known URLs) or full (re-download all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover URLs but don't download.",
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

    args = parser.parse_args()

    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    pipeline = AcquisitionPipeline()
    results = asyncio.run(
        pipeline.run(
            source_name=args.source,
            mode=args.mode,
            dry_run=args.dry_run,
        )
    )

    # Summary
    for r in results:
        status = "OK" if not r.errors else "ERRORS"
        _log.info(
            "result_summary",
            source=r.source_type.value,
            status=status,
            downloaded=r.documents_downloaded,
            failed=r.documents_failed,
            errors=r.errors,
        )

    # Exit code
    has_errors = any(r.errors for r in results)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
