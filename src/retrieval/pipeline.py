"""RetrievalPipeline: batch/interactive orchestrator for Phase 7.

Wraps RetrievalEngine for CLI usage, batch evaluation, and interactive mode.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from src.retrieval._config import load_retrieval_config
from src.retrieval._engine import RetrievalEngine
from src.retrieval._models import RetrievalQuery, RetrievalResult
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from src.retrieval._models import RetrievalConfig

_log = get_logger(__name__)


class RetrievalPipeline:
    """Orchestrates query processing via RetrievalEngine.

    Supports three modes:
    - Batch: process a list of queries or a JSON file
    - Interactive: read queries from stdin
    - Single: process one query
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = load_retrieval_config(config_path)
        self._settings = self._config.settings
        self._engine = RetrievalEngine(self._settings)

    @property
    def engine(self) -> RetrievalEngine:
        """Expose the underlying engine (for Phase 8 reuse)."""
        return self._engine

    async def run(
        self,
        *,
        queries: list[str] | None = None,
        queries_file: Path | None = None,
        interactive: bool = False,
        dry_run: bool = False,
        load_models: bool = True,
    ) -> list[RetrievalResult]:
        """Run retrieval for one or more queries.

        Args:
            queries: List of query texts.
            queries_file: Path to JSON file with query list.
            interactive: Read queries from stdin (one per line).
            dry_run: Only report what would be done; don't search.
            load_models: Whether to load embedding/reranker models.

        Returns:
            List of RetrievalResults, one per query.
        """
        # Collect query texts
        query_texts = self._collect_queries(queries, queries_file, interactive)

        _log.info("pipeline_start", query_count=len(query_texts), dry_run=dry_run)

        if dry_run:
            _log.info("dry_run_complete", query_count=len(query_texts))
            return [RetrievalResult(query_text=qt) for qt in query_texts]

        # Load models (once, before processing)
        if load_models:
            self._engine.load_models()

        # Process queries
        results: list[RetrievalResult] = []
        for i, qt in enumerate(query_texts):
            _log.info("processing_query", index=i, text=qt[:80])
            try:
                query = RetrievalQuery(text=qt)
                result = await self._engine.retrieve(query)
                results.append(result)
            except Exception as exc:
                _log.error("query_failed", index=i, error=str(exc))
                results.append(
                    RetrievalResult(
                        query_text=qt,
                        errors=[str(exc)],
                    )
                )

        _log.info(
            "pipeline_complete",
            total=len(results),
            errors=sum(1 for r in results if r.errors),
        )

        import contextlib

        with contextlib.suppress(Exception):
            await self._engine.close()

        return results

    @staticmethod
    def _collect_queries(
        queries: list[str] | None,
        queries_file: Path | None,
        interactive: bool,
    ) -> list[str]:
        """Gather query texts from all input sources."""
        result: list[str] = []

        if queries:
            result.extend(queries)

        if queries_file is not None:
            try:
                raw = queries_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, list):
                    result.extend(str(q) for q in data)
                elif isinstance(data, dict) and "queries" in data:
                    result.extend(str(q) for q in data["queries"])
                else:
                    _log.warning("unknown_queries_file_format", path=str(queries_file))
            except Exception as exc:
                _log.error("queries_file_load_failed", error=str(exc))

        if interactive:
            print("Enter queries (one per line, Ctrl+D to finish):")
            for line in sys.stdin:
                stripped = line.strip()
                if stripped:
                    result.append(stripped)

        return result
