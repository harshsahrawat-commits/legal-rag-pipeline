"""CLI entrypoint for the hallucination mitigation module.

Usage:
    python -m src.hallucination --response "Section 420 IPC provides..."
    python -m src.hallucination --response-file response.txt --dry-run
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
        prog="hallucination",
        description="Phase 8: Hallucination Mitigation — verify LLM responses",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--response",
        type=str,
        help="Response text to verify (inline)",
    )
    input_group.add_argument(
        "--response-file",
        type=Path,
        help="Path to a file containing the response text",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to hallucination config YAML (default: configs/hallucination.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and input, but skip actual verification",
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
    """Execute the hallucination pipeline."""
    from src.hallucination._config import load_hallucination_config
    from src.hallucination._models import VerificationInput
    from src.hallucination.pipeline import HallucinationPipeline

    config = load_hallucination_config(args.config)

    # Read response text
    response_text = args.response or args.response_file.read_text(encoding="utf-8")

    if not response_text.strip():
        _log.error("empty_response_text")
        return 1

    _log.info(
        "hallucination_run_start",
        response_length=len(response_text),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        _log.info("dry_run_complete", response_preview=response_text[:200])
        return 0

    verification_input = VerificationInput(response_text=response_text)
    pipeline = HallucinationPipeline(config.settings)
    result = await pipeline.verify(verification_input)

    _log.info(
        "hallucination_run_complete",
        confidence=result.confidence.overall_score,
        citations_verified=result.summary.verified_citations,
        citations_total=result.summary.total_citations,
        temporal_warnings=result.summary.temporal_warnings,
        claims_total=result.summary.total_claims,
        claims_supported=result.summary.supported_claims,
        elapsed_ms=result.elapsed_ms,
        errors=result.errors,
    )
    return 0


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
