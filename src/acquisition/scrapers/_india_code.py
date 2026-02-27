from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pydantic import BaseModel

from src.acquisition._models import (
    ContentFormat,
    DocumentFlag,
    DocumentType,
    FlagSeverity,
    FlagType,
    PreliminaryMetadata,
)
from src.acquisition.base_scraper import BaseScraper
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.acquisition._http_client import HttpClient

_log = get_logger(__name__)

_CENTRAL_ACTS_HANDLE = "123456789/1362"
_ITEMS_PER_PAGE = 100
_HANDLE_RE = re.compile(r"/handle/123456789/(\d+)")
_BITSTREAM_PDF_RE = re.compile(r"/bitstream/123456789/\d+/\d+/.+\.pdf", re.IGNORECASE)


class _BrowseEntry(BaseModel):
    """Metadata extracted from one row of the India Code browse listing."""

    handle: str
    detail_url: str
    short_title: str
    act_number: str | None = None
    enactment_date: str | None = None


class IndiaCodeScraper(BaseScraper):
    """Scraper for indiacode.nic.in — official Indian statute repository.

    Discovery works by paginating the DSpace browse listing for Central Acts.
    Each act's detail page HTML is saved as raw content, and the PDF bitstream
    URL is stored in ``PreliminaryMetadata.download_url`` for Phase 2 to fetch.
    """

    def __init__(self, source_def, state_store, output_dir, **kwargs) -> None:
        super().__init__(source_def, state_store, output_dir, **kwargs)
        self._browse_metadata: dict[str, _BrowseEntry] = {}

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def discover_urls(self, client: HttpClient) -> list[str]:
        """Paginate the Central Acts browse listing to discover act URLs."""
        self._browse_metadata.clear()

        base = self._source_def.base_url.rstrip("/")
        config = self._source_def.scrape_config
        max_pages = config.max_pages_per_query or 9

        urls: list[str] = []
        offset = 0

        for _ in range(max_pages):
            if len(urls) >= config.max_documents:
                break

            listing_url = (
                f"{base}/handle/{_CENTRAL_ACTS_HANDLE}/browse"
                f"?type=shorttitle&sort_by=3&order=ASC"
                f"&rpp={_ITEMS_PER_PAGE}&offset={offset}"
            )

            _log.info("fetching_browse_page", offset=offset, source="india_code")
            result = await client.fetch(listing_url)
            entries = self._parse_browse_listing(base, result.content)

            if not entries:
                _log.info("browse_listing_empty", offset=offset)
                break

            for entry in entries:
                if len(urls) >= config.max_documents:
                    break
                self._browse_metadata[entry.detail_url] = entry
                urls.append(entry.detail_url)

            offset += _ITEMS_PER_PAGE

        _log.info("discovery_complete", count=len(urls), source="india_code")
        return urls

    def detect_content_format(self, url: str, content: str) -> ContentFormat:
        """India Code detail pages are saved as HTML."""
        return ContentFormat.HTML

    def classify_document(self, url: str, content: str) -> DocumentType | None:
        """India Code hosts statutes. Check for schedules in the title."""
        entry = self._browse_metadata.get(url)
        if entry and re.search(r"\bschedule\b", entry.short_title, re.IGNORECASE):
            return DocumentType.SCHEDULE
        return DocumentType.STATUTE

    def extract_metadata(self, url: str, content: str) -> PreliminaryMetadata:
        """Extract metadata from browse listing cache and detail page HTML."""
        meta = PreliminaryMetadata()

        # Primary source: browse listing cache
        entry = self._browse_metadata.get(url)
        if entry:
            meta.title = entry.short_title
            meta.act_name = entry.short_title
            meta.act_number = entry.act_number
            meta.date = entry.enactment_date

            year_match = re.search(r"\b(\d{4})\b", entry.short_title)
            if year_match:
                meta.year = int(year_match.group(1))

        # Extract PDF URL from detail page HTML
        pdf_url = self._extract_pdf_url(content)
        if pdf_url:
            base = self._source_def.base_url.rstrip("/")
            meta.download_url = urljoin(base + "/", pdf_url)

        return meta

    def detect_flags(self, content: str, content_format: ContentFormat) -> list[DocumentFlag]:
        """Flag quality issues with the downloaded detail page."""
        flags: list[DocumentFlag] = []

        if len(content) < 500:
            flags.append(
                DocumentFlag(
                    flag_type=FlagType.SMALL_CONTENT,
                    message=f"Content only {len(content)} chars",
                    severity=FlagSeverity.WARNING,
                )
            )

        if not self._extract_pdf_url(content):
            flags.append(
                DocumentFlag(
                    flag_type=FlagType.MISSING_METADATA,
                    message="No PDF bitstream link found in detail page",
                    severity=FlagSeverity.WARNING,
                )
            )

        return flags

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _doc_id_from_url(self, url: str) -> str:
        """Extract DSpace handle ID with ic_ prefix."""
        match = _HANDLE_RE.search(url)
        if match:
            return f"ic_{match.group(1)}"
        return super()._doc_id_from_url(url)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_browse_listing(self, base_url: str, html: str) -> list[_BrowseEntry]:
        """Parse the server-rendered browse listing table.

        Real table structure (from Playwright probe)::

            <tr>
              <td>6-Oct-1860</td>
              <td><em>45</em></td>
              <td><strong>The Indian Penal Code, 1860</strong></td>
              <td><a href="/handle/123456789/1505?view_type=browse">View...</a></td>
            </tr>
        """
        soup = BeautifulSoup(html, "html.parser")
        entries: list[_BrowseEntry] = []

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            # Find the detail page link in the row
            link = row.find("a", href=_HANDLE_RE)
            if not link:
                continue

            href = link.get("href", "")
            handle_match = _HANDLE_RE.search(href)
            if not handle_match:
                continue

            handle = handle_match.group(1)
            detail_url = f"{base_url}{href}"

            # Extract text from cells
            enactment_date = cells[0].get_text(strip=True)
            act_number = cells[1].get_text(strip=True)
            short_title = cells[2].get_text(strip=True)

            entries.append(
                _BrowseEntry(
                    handle=handle,
                    detail_url=detail_url,
                    short_title=short_title,
                    act_number=act_number or None,
                    enactment_date=enactment_date or None,
                )
            )

        return entries

    def _extract_pdf_url(self, html: str) -> str | None:
        """Extract the English PDF bitstream URL from a detail page.

        The detail page contains ``<a>`` tags pointing to PDF bitstreams.
        We pick the first English PDF (skip Hindi/regional language PDFs).
        """
        soup = BeautifulSoup(html, "html.parser")

        for link in soup.find_all("a", href=_BITSTREAM_PDF_RE):
            href = link.get("href", "")
            link_text = link.get_text(strip=True)

            # Skip Hindi or regional language PDFs
            if _is_non_latin(link_text):
                continue

            return href

        return None


def _is_non_latin(text: str) -> bool:
    """Check if the majority of alphabetic characters are non-Latin (Hindi, etc.)."""
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    non_latin = sum(1 for c in alpha_chars if ord(c) > 0x024F)
    return non_latin / len(alpha_chars) > 0.5
