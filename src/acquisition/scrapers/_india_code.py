from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

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


class IndiaCodeScraper(BaseScraper):
    """Scraper for indiacode.nic.in — official Indian statute repository."""

    async def discover_urls(self, client: HttpClient) -> list[str]:
        """Build URLs from seed act IDs in the config."""
        urls: list[str] = []
        base = self._source_def.base_url.rstrip("/")
        config = self._source_def.scrape_config

        for act_id in config.seed_act_ids:
            if len(urls) >= config.max_documents:
                break
            url = f"{base}/handle/123456789/{act_id}"
            urls.append(url)

        return urls

    def detect_content_format(self, url: str, content: str) -> ContentFormat:
        if url.endswith(".pdf") or content.startswith("%PDF"):
            return ContentFormat.PDF
        return ContentFormat.HTML

    def classify_document(self, url: str, content: str) -> DocumentType | None:
        """India Code is almost always statutes. Check for schedules."""
        soup = BeautifulSoup(content, "html.parser")
        title = self._get_title(soup)
        if title and re.search(r"\bschedule\b", title, re.IGNORECASE):
            return DocumentType.SCHEDULE
        return DocumentType.STATUTE

    def extract_metadata(self, url: str, content: str) -> PreliminaryMetadata:
        """Extract act name, number, year from India Code HTML."""
        soup = BeautifulSoup(content, "html.parser")
        meta = PreliminaryMetadata()

        # Act title
        title_div = soup.find("div", class_="actTitle")
        if title_div:
            meta.title = title_div.get_text(strip=True)
            meta.act_name = meta.title

        # Act number
        num_div = soup.find("div", class_="actNumber")
        if num_div:
            meta.act_number = num_div.get_text(strip=True)

        # Year from title or act number
        if meta.title:
            year_match = re.search(r"\b(\d{4})\b", meta.title)
            if year_match:
                meta.year = int(year_match.group(1))

        # Enactment date
        date_div = soup.find("div", class_="enactmentDate")
        if date_div:
            meta.date = date_div.get_text(strip=True).strip("[]")

        return meta

    def detect_flags(self, content: str, content_format: ContentFormat) -> list[DocumentFlag]:
        flags: list[DocumentFlag] = []
        if content_format == ContentFormat.PDF and not content.startswith("%PDF"):
            flags.append(
                DocumentFlag(
                    flag_type=FlagType.CORRUPT_CONTENT,
                    message="Expected PDF but content doesn't start with %PDF header",
                    severity=FlagSeverity.ERROR,
                )
            )
        if len(content) < 500:
            flags.append(
                DocumentFlag(
                    flag_type=FlagType.SMALL_CONTENT,
                    message=f"Content only {len(content)} chars",
                    severity=FlagSeverity.WARNING,
                )
            )
        return flags

    def _get_title(self, soup: BeautifulSoup) -> str | None:
        title_div = soup.find("div", class_="actTitle")
        if title_div:
            return title_div.get_text(strip=True)
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        return None

    def _doc_id_from_url(self, url: str) -> str:
        """Extract act ID from India Code URL."""
        # /handle/123456789/{act_id}
        parts = url.rstrip("/").split("/")
        return parts[-1] if parts else super()._doc_id_from_url(url)
