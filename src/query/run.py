"""CLI entry point for the Query Intelligence Layer."""

from __future__ import annotations

import argparse
import asyncio
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="query-intelligence",
        description="Phase 0: Query Intelligence Layer — cache, routing, HyDE",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Process a single query",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Only show route classification (no cache/HyDE)",
    )
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics",
    )
    parser.add_argument(
        "--invalidate-act",
        type=str,
        help="Invalidate cache entries citing this act",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without side effects",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Use console (human-readable) logging instead of JSON",
    )
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    from pathlib import Path

    from src.query._config import load_query_config
    from src.query._router import AdaptiveQueryRouter
    from src.utils._logging import configure_logging, get_logger

    configure_logging(
        log_level=args.log_level,
        json_output=not args.console_log,
    )
    log = get_logger(__name__)

    config_path = Path(args.config) if args.config else None
    try:
        config = load_query_config(config_path)
    except Exception:
        # Fall back to defaults if config not found
        from src.query._models import QueryConfig

        config = QueryConfig()
        log.warning("config_not_found_using_defaults")

    settings = config.settings

    if args.classify_only and args.query:
        router = AdaptiveQueryRouter(settings)
        result = router.classify(args.query)
        print(f"Query:      {args.query}")
        print(f"Route:      {result.route.value}")
        print(f"Confidence: {result.confidence}")
        print(f"Signals:    {result.signals}")
        return

    if args.invalidate_act:
        if args.dry_run:
            print(f"Would invalidate cache for act: {args.invalidate_act}")
            return
        log.info("cache_invalidation_not_implemented_yet")
        return

    if args.cache_stats:
        log.info("cache_stats_not_implemented_yet")
        return

    if args.query:
        if args.dry_run:
            router = AdaptiveQueryRouter(settings)
            result = router.classify(args.query)
            print(f"Query:  {args.query}")
            print(f"Route:  {result.route.value}")
            print(f"HyDE:   {'yes' if result.route.value in settings.hyde_routes else 'no'}")
            print(f"Cache:  {'enabled' if settings.cache_enabled else 'disabled'}")
            return

        # Full processing
        from src.query.pipeline import QueryIntelligenceLayer

        layer = QueryIntelligenceLayer(settings)
        qi_result, _rq = asyncio.run(layer.process(args.query))
        print(f"Query:      {qi_result.query_text}")
        print(f"Route:      {qi_result.route.value}")
        print(f"Cache hit:  {qi_result.cache_hit}")
        print(f"HyDE:       {qi_result.hyde_generated}")
        print(f"Timings:    {qi_result.timings}")
        if qi_result.errors:
            print(f"Errors:     {qi_result.errors}")
        return

    parser.print_help()
    sys.exit(1)
