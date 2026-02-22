from __future__ import annotations

import abc
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from src.acquisition._exceptions import ScrapingError
from src.acquisition._http_client import HttpClient
from src.acquisition._models import (
    ContentFormat,
    CrawlRecord,
    CrawlState,
    DocumentFlag,
    DocumentType,
    PreliminaryMetadata,
    RawDocument,
    ScrapedContent,
    SourceDefinition,
    SourceType,
)
from src.acquisition._rate_limiter import AsyncRateLimiter
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from src.acquisition._state import CrawlStateStore

_log = get_logger(__name__)


class BaseScraper(abc.ABC):
    """Template method pattern for source scrapers.

    Subclasses implement 4 abstract methods:
    - discover_urls: yield URLs from search/listing pages
    - detect_content_format: determine if HTML or PDF
    - classify_document: assign DocumentType from content
    - extract_metadata: pull preliminary metadata from content
    """

    def __init__(
        self,
        source_def: SourceDefinition,
        state_store: CrawlStateStore,
        output_dir: Path,
        *,
        user_agent: str = "LegalRAGBot/0.1",
    ) -> None:
        self._source_def = source_def
        self._state_store = state_store
        self._output_dir = output_dir / source_def.source_type.value
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._user_agent = user_agent

    @property
    def source_type(self) -> SourceType:
        return self._source_def.source_type

    @property
    def source_def(self) -> SourceDefinition:
        return self._source_def

    @abc.abstractmethod
    async def discover_urls(self, client: HttpClient) -> list[str]:
        """Discover document URLs from search/listing pages."""

    @abc.abstractmethod
    def detect_content_format(self, url: str, content: str) -> ContentFormat:
        """Determine the content format from URL or content inspection."""

    @abc.abstractmethod
    def classify_document(self, url: str, content: str) -> DocumentType | None:
        """Classify the document type from URL patterns or content."""

    @abc.abstractmethod
    def extract_metadata(self, url: str, content: str) -> PreliminaryMetadata:
        """Extract lightweight metadata from raw content."""

    def detect_flags(self, content: str, content_format: ContentFormat) -> list[DocumentFlag]:
        """Detect issues with the content. Override for source-specific checks."""
        return []

    def _doc_id_from_url(self, url: str) -> str:
        """Extract a filesystem-safe document ID from a URL. Override if needed."""
        # Default: use last path segment, stripping slashes
        parts = url.rstrip("/").split("/")
        return parts[-1] if parts else str(uuid4())

    async def scrape_document(
        self,
        client: HttpClient,
        url: str,
    ) -> ScrapedContent:
        """Download a single document."""
        content_format = ContentFormat.HTML  # default, refined after download
        return await client.fetch(url, content_format=content_format)

    def _save_document(
        self,
        scraped: ScrapedContent,
        doc_type: DocumentType | None,
        metadata: PreliminaryMetadata,
        flags: list[DocumentFlag],
    ) -> RawDocument:
        """Save raw content and metadata sidecar to disk."""
        doc_id = self._doc_id_from_url(scraped.url)
        content_format = self.detect_content_format(scraped.url, scraped.content)
        ext = "pdf" if content_format == ContentFormat.PDF else "html"
        content_path = self._output_dir / f"{doc_id}.{ext}"
        meta_path = self._output_dir / f"{doc_id}.meta.json"

        content_path.write_text(scraped.content, encoding="utf-8")

        raw_doc = RawDocument(
            document_id=uuid4(),
            url=scraped.url,
            source_type=self.source_type,
            content_format=content_format,
            raw_content_path=str(content_path),
            document_type=doc_type,
            preliminary_metadata=metadata,
            flags=flags,
            scraped_at=datetime.now(UTC),
            content_hash=scraped.content_hash,
        )

        meta_path.write_text(raw_doc.model_dump_json(indent=2), encoding="utf-8")
        return raw_doc

    async def run(
        self,
        *,
        state: CrawlState,
        dry_run: bool = False,
    ) -> list[RawDocument]:
        """Execute the full scraper pipeline for this source.

        1. Discover URLs
        2. Filter against crawl state (skip known + unchanged)
        3. Scrape each new/changed URL
        4. Classify, extract metadata, detect flags
        5. Save to disk + update state

        Args:
            state: Current crawl state (loaded by caller).
            dry_run: If True, discover URLs but don't download.

        Returns:
            List of RawDocument objects for successfully scraped documents.
        """
        rate_limiter = AsyncRateLimiter(self._source_def.rate_limit_requests_per_second)
        results: list[RawDocument] = []

        async with HttpClient(
            rate_limiter=rate_limiter,
            user_agent=self._user_agent,
            timeout_seconds=self._source_def.request_timeout_seconds,
            max_retries=self._source_def.max_retries,
        ) as client:
            # Step 1: Discover
            _log.info("discovering_urls", source=self.source_type.value)
            urls = await self.discover_urls(client)
            _log.info("urls_discovered", source=self.source_type.value, count=len(urls))

            if dry_run:
                return results

            # Step 2-5: For each URL
            for url in urls:
                # Check state
                existing = state.records.get(url)
                try:
                    # Step 3: Scrape
                    scraped = await self.scrape_document(client, url)

                    # Skip if content unchanged
                    if existing and existing.content_hash == scraped.content_hash:
                        _log.debug("content_unchanged", url=url)
                        continue

                    # Step 4: Classify + extract
                    doc_type = self.classify_document(url, scraped.content)
                    metadata = self.extract_metadata(url, scraped.content)
                    content_format = self.detect_content_format(url, scraped.content)
                    flags = self.detect_flags(scraped.content, content_format)

                    # Step 5: Save
                    raw_doc = self._save_document(scraped, doc_type, metadata, flags)
                    results.append(raw_doc)

                    # Update state
                    record = CrawlRecord(
                        url=url,
                        content_hash=scraped.content_hash,
                        file_path=raw_doc.raw_content_path,
                        scraped_at=datetime.now(UTC),
                        document_type=doc_type,
                    )
                    self._state_store.upsert_record(state, record)

                    _log.info(
                        "document_scraped",
                        url=url,
                        doc_type=doc_type.value if doc_type else None,
                    )

                except ScrapingError as exc:
                    _log.warning("scrape_failed", url=url, error=str(exc))
                    continue

        return results
