from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import quote_plus, urljoin

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

_JUDGMENT_PATTERN = re.compile(r"\bvs?\b\.?\s", re.IGNORECASE)
_STATUTE_PATTERN = re.compile(r"Section\s+\d+.*\b(in|of)\s+The\b", re.IGNORECASE)
_COURT_PATTERNS = {
    "Supreme Court of India": re.compile(r"Supreme\s+Court", re.IGNORECASE),
    "High Court": re.compile(r"High\s+Court", re.IGNORECASE),
    "District Court": re.compile(r"District\s+(Court|Judge)", re.IGNORECASE),
}

_DATE_PATTERN = re.compile(r"on\s+(\d{1,2}\s+\w+,?\s+\d{4})", re.IGNORECASE)


class IndianKanoonScraper(BaseScraper):
    """Scraper for indiankanoon.org — India's largest legal search engine."""

    async def discover_urls(self, client: HttpClient) -> list[str]:
        """Paginate search results for seed queries, extract /doc/{id}/ URLs."""
        urls: list[str] = []
        seen: set[str] = set()
        base = self._source_def.base_url.rstrip("/")
        config = self._source_def.scrape_config
        max_docs = config.max_documents

        for query in config.seed_queries:
            if len(urls) >= max_docs:
                break
            for page in range(config.max_pages_per_query):
                if len(urls) >= max_docs:
                    break

                search_url = f"{base}/search/?formInput={quote_plus(query)}&pagenum={page}"
                try:
                    result = await client.fetch(search_url)
                except Exception:
                    _log.warning("search_page_failed", query=query, page=page)
                    break

                page_urls = self._extract_doc_urls(base, result.content)
                if not page_urls:
                    break  # No more results

                for u in page_urls:
                    if u not in seen and len(urls) < max_docs:
                        seen.add(u)
                        urls.append(u)

        return urls

    def _extract_doc_urls(self, base_url: str, html: str) -> list[str]:
        """Extract /doc/{id}/ URLs from a search results page."""
        soup = BeautifulSoup(html, "html.parser")
        urls: list[str] = []

        for link in soup.select("a[href]"):
            href = link.get("href", "")
            if isinstance(href, list):
                href = href[0]
            if re.match(r"/doc/\d+/?", href):
                full_url = urljoin(base_url, href)
                if not full_url.endswith("/"):
                    full_url += "/"
                urls.append(full_url)

        return urls

    def detect_content_format(self, url: str, content: str) -> ContentFormat:
        return ContentFormat.HTML

    def classify_document(self, url: str, content: str) -> DocumentType | None:
        """Classify as STATUTE or JUDGMENT based on title patterns."""
        soup = BeautifulSoup(content, "html.parser")
        title = self._get_title(soup)
        if not title:
            return None

        if _STATUTE_PATTERN.search(title):
            return DocumentType.STATUTE

        if _JUDGMENT_PATTERN.search(title):
            return DocumentType.JUDGMENT

        # Fallback: check for court indicators in page
        bench_div = soup.find("div", class_="doc_bench")
        if bench_div and bench_div.get_text(strip=True):
            return DocumentType.JUDGMENT

        return None

    def extract_metadata(self, url: str, content: str) -> PreliminaryMetadata:
        """Extract title, court, citation, date from Indian Kanoon HTML."""
        soup = BeautifulSoup(content, "html.parser")
        title = self._get_title(soup)
        meta = PreliminaryMetadata(title=title)

        # Court
        bench_div = soup.find("div", class_="doc_bench")
        if bench_div:
            meta.court = bench_div.get_text(strip=True)

        # Citation
        cite_div = soup.find("div", class_="doc_citations")
        if cite_div:
            meta.case_citation = cite_div.get_text(strip=True)

        # Date from title
        if title:
            date_match = _DATE_PATTERN.search(title)
            if date_match:
                meta.date = date_match.group(1)

            # Parties
            if _JUDGMENT_PATTERN.search(title):
                # "State Of Maharashtra vs Bandishala on 15 March, 2024"
                parts = re.split(r"\bvs?\b\.?\s", title, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    meta.parties = title.split(" on ")[0] if " on " in title else title

            # Act name for statutes
            if _STATUTE_PATTERN.search(title):
                act_match = re.search(
                    r"(?:in|of)\s+The\s+(.+?)(?:\s*,\s*\d{4})?$", title, re.IGNORECASE
                )
                if act_match:
                    meta.act_name = act_match.group(1).strip()

        return meta

    def detect_flags(self, content: str, content_format: ContentFormat) -> list[DocumentFlag]:
        flags: list[DocumentFlag] = []
        if len(content) < 500:
            flags.append(
                DocumentFlag(
                    flag_type=FlagType.SMALL_CONTENT,
                    message=f"Content only {len(content)} chars",
                    severity=FlagSeverity.WARNING,
                )
            )
        # Check for non-ASCII heavy content (possible regional language)
        non_ascii = sum(1 for c in content if ord(c) > 127)
        if len(content) > 0 and non_ascii / len(content) > 0.3:
            flags.append(
                DocumentFlag(
                    flag_type=FlagType.REGIONAL_LANGUAGE,
                    message="High non-ASCII ratio suggests regional language content",
                    severity=FlagSeverity.INFO,
                )
            )
        return flags

    def _get_title(self, soup: BeautifulSoup) -> str | None:
        title_div = soup.find("div", class_="doc_title")
        if title_div:
            return title_div.get_text(strip=True)
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        return None

    def _doc_id_from_url(self, url: str) -> str:
        """Extract document ID from /doc/{id}/ URL."""
        match = re.search(r"/doc/(\d+)/?", url)
        return match.group(1) if match else super()._doc_id_from_url(url)
