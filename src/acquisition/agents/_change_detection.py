from __future__ import annotations

from src.acquisition._models import CrawlState, DiscoveredDocument, SourceType
from src.utils._logging import get_logger

_log = get_logger(__name__)


class ChangeDetectionAgent:
    """Filters discovered URLs against crawl state to find new/changed documents."""

    def filter_urls(
        self,
        urls: list[str],
        state: CrawlState,
        source_type: SourceType,
    ) -> list[DiscoveredDocument]:
        """Compare URLs against crawl state and return only new or changed ones.

        Args:
            urls: Discovered URLs to check.
            state: Current crawl state for this source.
            source_type: The source type.

        Returns:
            List of DiscoveredDocument with is_new flag set.
        """
        results: list[DiscoveredDocument] = []

        for url in urls:
            existing = state.records.get(url)
            if existing is None:
                results.append(
                    DiscoveredDocument(
                        url=url,
                        source_type=source_type,
                        is_new=True,
                        content_hash_changed=False,
                    )
                )
            # If URL exists, it will be re-checked during scraping
            # (content hash comparison happens in BaseScraper.run)

        _log.info(
            "change_detection_complete",
            total_urls=len(urls),
            new_urls=len(results),
            skipped=len(urls) - len(results),
        )
        return results
