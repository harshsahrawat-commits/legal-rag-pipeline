"""CLI entrypoint for the evaluation module.

Usage:
    python -m src.evaluation --queries data/eval/test_queries.json --dry-run
    python -m src.evaluation --queries data/eval/test_queries.json --skip-ragas
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.utils._logging import configure_logging, get_logger

_log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Phase 9: Evaluation & Quality Assurance",
    )
    # Input modes
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--queries",
        type=Path,
        help="Path to test_queries.json for batch evaluation",
    )
    input_group.add_argument(
        "--query",
        type=str,
        help="Single query text to evaluate (requires --response)",
    )
    parser.add_argument(
        "--response",
        type=str,
        help="Response text for single query evaluation",
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        help="JSON file with retrieved contexts for single query",
    )

    # Mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset, skip evaluation",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS metrics (fast mode: legal/latency/QI only)",
    )
    parser.add_argument(
        "--human-generate",
        action="store_true",
        help="Generate human evaluation worksheets",
    )
    parser.add_argument(
        "--human-import",
        action="store_true",
        help="Import completed human scoresheets",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report from last evaluation results",
    )

    # Standard flags
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to evaluation config YAML (default: configs/evaluation.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Use human-readable console logging instead of JSON",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Execute the evaluation pipeline."""
    from src.evaluation._config import load_evaluation_config
    from src.evaluation._test_dataset import TestDatasetLoader

    config = load_evaluation_config(args.config)
    settings = config.settings

    # Dry-run: validate config + dataset only
    if args.dry_run:
        _log.info("evaluation_dry_run_start")
        if args.queries:
            loader = TestDatasetLoader(settings)
            dataset = loader.load(args.queries)
            _log.info(
                "dry_run_dataset_valid",
                query_count=len(dataset.queries),
                version=dataset.version,
            )
        _log.info("dry_run_complete")
        return 0

    # Batch evaluation mode
    if args.queries:
        from src.evaluation.pipeline import EvaluationPipeline

        loader = TestDatasetLoader(settings)
        dataset = loader.load(args.queries)
        inputs = loader.to_evaluation_inputs(dataset)

        if args.skip_ragas:
            settings.ragas_enabled = False

        pipeline = EvaluationPipeline(settings)
        result = await pipeline.evaluate(inputs)

        _log.info(
            "evaluation_complete",
            queries_evaluated=result.queries_evaluated,
            all_targets_met=result.all_targets_met,
            elapsed_ms=result.elapsed_ms,
            errors=result.errors,
        )
        return 0

    # Single query mode
    if args.query:
        from src.evaluation._models import EvaluationInput
        from src.evaluation.pipeline import EvaluationPipeline

        if not args.response:
            _log.error("single_query_requires_response")
            return 1

        contexts: list[str] = []
        if args.contexts and args.contexts.exists():
            import json

            contexts = json.loads(args.contexts.read_text(encoding="utf-8"))

        inp = EvaluationInput(
            query_id="cli_query",
            query_text=args.query,
            response_text=args.response,
            retrieved_contexts=contexts,
        )

        if args.skip_ragas:
            settings.ragas_enabled = False

        pipeline = EvaluationPipeline(settings)
        result = await pipeline.evaluate([inp])

        _log.info(
            "evaluation_complete",
            queries_evaluated=result.queries_evaluated,
            citation_accuracy=result.legal_metrics.citation_accuracy,
            temporal_accuracy=result.legal_metrics.temporal_accuracy,
            elapsed_ms=result.elapsed_ms,
        )
        return 0

    _log.error("no_input_specified", hint="Use --queries or --query")
    return 1


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(
        log_level=args.log_level,
        json_output=not args.console_log,
    )
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)
