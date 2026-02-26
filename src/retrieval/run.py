"""CLI entry point: python -m src.retrieval.run

Usage:
    python -m src.retrieval.run --query "What is Section 420 IPC?"
    python -m src.retrieval.run --queries-file data/eval/test_queries.json
    python -m src.retrieval.run --interactive
    python -m src.retrieval.run --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.retrieval.pipeline import RetrievalPipeline
from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    """Run the retrieval pipeline CLI."""
    parser = argparse.ArgumentParser(description="Legal RAG Retrieval Pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query text to retrieve for.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to JSON file with list of queries.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read queries from stdin (one per line).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without searching.",
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
        help="Path to retrieval config YAML.",
    )

    args = parser.parse_args()
    configure_logging(log_level=args.log_level, json_output=not args.console_log)

    config_path = Path(args.config) if args.config else None
    pipeline = RetrievalPipeline(config_path=config_path)

    queries = [args.query] if args.query else None
    queries_file = Path(args.queries_file) if args.queries_file else None

    results = asyncio.run(
        pipeline.run(
            queries=queries,
            queries_file=queries_file,
            interactive=args.interactive,
            dry_run=args.dry_run,
        )
    )

    # Print summary
    for i, r in enumerate(results):
        status = "OK" if not r.errors else "ERRORS"
        chunks = len(r.chunks)
        elapsed = f"{r.elapsed_ms:.0f}ms" if r.finished_at else "N/A"
        print(f"[{i}] {status} | {chunks} chunks | {elapsed} | {r.query_text[:60]}")
        for err in r.errors:
            print(f"     ERROR: {err}")


if __name__ == "__main__":
    main()
