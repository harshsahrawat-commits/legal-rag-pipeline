from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.acquisition._config import load_source_registry
from src.acquisition._models import (
    AcquisitionResult,
    CrawlState,
    SourceDefinition,
    SourceRegistry,
    SourceType,
)
from src.acquisition._state import CrawlStateStore
from src.acquisition.agents._change_detection import ChangeDetectionAgent
from src.acquisition.agents._legal_review import LegalReviewAgent
from src.acquisition.agents._source_discovery import SourceDiscoveryAgent
from src.acquisition.scrapers._india_code import IndiaCodeScraper
from src.acquisition.scrapers._indian_kanoon import IndianKanoonScraper
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.acquisition.base_scraper import BaseScraper

_log = get_logger(__name__)

_SCRAPER_MAP: dict[SourceType, type[BaseScraper]] = {
    SourceType.INDIAN_KANOON: IndianKanoonScraper,
    SourceType.INDIA_CODE: IndiaCodeScraper,
}


class AcquisitionPipeline:
    """Orchestrates the full acquisition pipeline across all sources.

    Runs sources concurrently via asyncio, each with its own rate limiter.
    """

    def __init__(
        self,
        registry: SourceRegistry | None = None,
        config_path: Path | None = None,
    ) -> None:
        if registry is not None:
            self._registry = registry
        else:
            self._registry = load_source_registry(config_path)
        self._settings = self._registry.settings
        self._state_store = CrawlStateStore(self._settings.state_dir)
        self._discovery_agent = SourceDiscoveryAgent(self._registry)
        self._change_agent = ChangeDetectionAgent()
        self._review_agent = LegalReviewAgent()

    async def run(
        self,
        *,
        source_name: str | None = None,
        mode: str = "incremental",
        dry_run: bool = False,
    ) -> list[AcquisitionResult]:
        """Run the acquisition pipeline.

        Args:
            source_name: If given, only run for this source (by name).
            mode: "incremental" (default) skips known URLs; "full" re-downloads all.
            dry_run: If True, discover URLs but don't download.

        Returns:
            List of AcquisitionResult, one per source.
        """
        if source_name:
            source_def = self._discovery_agent.get_source_by_name(source_name)
            if source_def is None:
                _log.error("source_not_found", name=source_name)
                return []
            sources = [source_def]
        else:
            sources = self._discovery_agent.get_enabled_sources()

        if not sources:
            _log.warning("no_sources_to_process")
            return []

        _log.info(
            "pipeline_starting",
            source_count=len(sources),
            mode=mode,
            dry_run=dry_run,
        )

        # Run each source concurrently with a semaphore
        sem = asyncio.Semaphore(self._settings.concurrency)
        tasks = [self._run_source(source, sem, mode=mode, dry_run=dry_run) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: list[AcquisitionResult] = []
        for source, result in zip(sources, results, strict=False):
            if isinstance(result, Exception):
                _log.error(
                    "source_failed",
                    source=source.name,
                    error=str(result),
                )
                final_results.append(
                    AcquisitionResult(
                        source_type=source.source_type,
                        errors=[str(result)],
                    )
                )
            else:
                final_results.append(result)

        _log.info(
            "pipeline_complete",
            total_downloaded=sum(r.documents_downloaded for r in final_results),
            total_failed=sum(r.documents_failed for r in final_results),
        )
        return final_results

    async def _run_source(
        self,
        source_def: SourceDefinition,
        sem: asyncio.Semaphore,
        *,
        mode: str,
        dry_run: bool,
    ) -> AcquisitionResult:
        """Run acquisition for a single source."""
        async with sem:
            started = datetime.now(UTC)
            result = AcquisitionResult(
                source_type=source_def.source_type,
                started_at=started,
            )

            _log.info("source_starting", source=source_def.name)

            # Load state
            state = self._state_store.load(source_def.source_type)
            if mode == "full":
                state = CrawlState(source_type=source_def.source_type)

            # Create scraper
            scraper_cls = _SCRAPER_MAP.get(source_def.source_type)
            if scraper_cls is None:
                msg = f"No scraper for source type: {source_def.source_type}"
                result.errors.append(msg)
                result.finished_at = datetime.now(UTC)
                return result

            scraper = scraper_cls(
                source_def,
                self._state_store,
                self._settings.output_dir,
            )

            try:
                raw_docs = await scraper.run(state=state, dry_run=dry_run)

                # Post-scrape legal review
                for doc in raw_docs:
                    try:
                        content_path = Path(doc.raw_content_path)
                        if content_path.exists():
                            content = content_path.read_text(encoding="utf-8")
                            self._review_agent.review(doc, content)
                            # Re-save meta.json with updated flags
                            meta_path = content_path.with_suffix(".meta.json")
                            meta_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                    except Exception as exc:
                        _log.warning("review_failed", url=doc.url, error=str(exc))

                result.documents_downloaded = len(raw_docs)

                # Save state
                if not dry_run:
                    self._state_store.save(state)

            except Exception as exc:
                _log.error("source_error", source=source_def.name, error=str(exc))
                result.errors.append(str(exc))

            result.finished_at = datetime.now(UTC)
            _log.info(
                "source_complete",
                source=source_def.name,
                downloaded=result.documents_downloaded,
                errors=len(result.errors),
            )
            return result
