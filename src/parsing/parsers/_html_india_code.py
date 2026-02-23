"""India Code HTML detail page parser.

Extracts metadata from India Code detail page HTML.  The actual statute
text lives in JS-rendered tabs or a linked PDF, so this parser returns
a ``ParsedDocument`` with metadata and minimal text content.  The
pipeline uses the extracted metadata and delegates to the Docling PDF
parser for the real content once the PDF is downloaded.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)
from src.parsing.parsers._base import BaseParser
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from src.acquisition._models import RawDocument

_log = get_logger(__name__)

# Year pattern in act titles like "The Information Technology Act, 2000"
_YEAR_RE = re.compile(r"\b(\d{4})\b")

# Metadata field labels in the detail page table
_FIELD_MAP: dict[str, str] = {
    "act number": "act_number",
    "enactment date": "date",
    "act year": "year",
    "short title": "act_name",
    "ministry": "ministry",
    "department": "department",
}


class IndiaCodeHtmlParser(BaseParser):
    """Parses India Code detail page HTML for metadata extraction.

    India Code detail pages (DSpace) contain a metadata table with act
    number, enactment date, year, etc.  The actual statute text is either
    in a JS-rendered tab or a linked PDF.  This parser extracts metadata
    and returns a minimal ``ParsedDocument``; the pipeline then downloads
    and parses the PDF for full content.
    """

    @property
    def parser_type(self) -> ParserType:
        return ParserType.HTML_INDIA_CODE

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return (
            raw_doc.source_type == SourceType.INDIA_CODE
            and raw_doc.content_format == ContentFormat.HTML
        )

    def parse(
        self,
        content_path: Path,
        raw_doc: RawDocument,
    ) -> ParsedDocument:
        start = time.monotonic()

        html = content_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        metadata = self._extract_metadata(soup, raw_doc)
        title = metadata.get("title") or metadata.get("act_name")
        raw_text = self._extract_visible_text(soup)

        sections: list[ParsedSection] = []
        if raw_text.strip():
            sections.append(
                ParsedSection(
                    id="ic_preamble",
                    level=SectionLevel.PREAMBLE,
                    title=title,
                    text=raw_text,
                    token_count=len(raw_text) // 4,
                )
            )

        elapsed = time.monotonic() - start

        # Determine year from metadata or title
        year = metadata.pop("year", None)
        if year is not None and not isinstance(year, int):
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        return ParsedDocument(
            document_id=raw_doc.document_id,
            source_type=raw_doc.source_type,
            document_type=raw_doc.document_type or DocumentType.STATUTE,
            content_format=ContentFormat.HTML,
            raw_text=raw_text,
            sections=sections,
            tables=[],
            title=metadata.get("title"),
            act_name=metadata.get("act_name"),
            act_number=metadata.get("act_number"),
            year=year,
            date=metadata.get("date"),
            parser_used=ParserType.HTML_INDIA_CODE,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path=str(content_path),
            parsing_duration_seconds=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(soup: BeautifulSoup, raw_doc: RawDocument) -> dict:
        """Extract metadata from the detail page structure.

        Sources (in priority order):
        1. ``table.itemDisplayTable`` — DSpace metadata table
        2. ``div.item-page-field-wrapper h2`` — act title heading
        3. ``<title>`` tag
        4. Phase 1 preliminary metadata (fallback)
        """
        meta: dict[str, str | int | None] = {}

        # 1) Title from item-page-field-wrapper heading
        wrapper = soup.find("div", class_="item-page-field-wrapper")
        if wrapper and isinstance(wrapper, Tag):
            h2 = wrapper.find("h2")
            if h2 and isinstance(h2, Tag):
                meta["title"] = h2.get_text(strip=True)
                meta["act_name"] = meta["title"]

        # 2) DSpace metadata table
        table = soup.find("table", class_="itemDisplayTable")
        if table and isinstance(table, Tag):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).rstrip(":").lower()
                value = cells[1].get_text(strip=True)
                if not value:
                    continue

                for pattern, field in _FIELD_MAP.items():
                    if pattern in label:
                        meta[field] = value
                        break

        # 3) Fallback: title tag
        if "title" not in meta:
            title_tag = soup.find("title")
            if title_tag and isinstance(title_tag, Tag):
                raw_title = title_tag.get_text(strip=True)
                # Strip "India Code: " prefix
                if raw_title.lower().startswith("india code:"):
                    raw_title = raw_title[len("india code:") :].strip()
                if raw_title:
                    meta["title"] = raw_title

        # 4) Year from title or act_name
        if "year" not in meta:
            title_text = meta.get("title") or meta.get("act_name") or ""
            if isinstance(title_text, str):
                year_match = _YEAR_RE.search(title_text)
                if year_match:
                    meta["year"] = int(year_match.group(1))

        # 5) Merge Phase 1 preliminary metadata as fallback
        pm = raw_doc.preliminary_metadata
        for field in ("title", "act_name", "act_number", "year", "date"):
            val = getattr(pm, field, None)
            if val is not None and field not in meta:
                meta[field] = val

        return meta

    @staticmethod
    def _extract_visible_text(soup: BeautifulSoup) -> str:
        """Extract whatever visible text exists in the detail page.

        India Code detail pages are mostly JS-rendered, so visible text
        is limited to the title, metadata labels, and PDF link text.
        """
        parts: list[str] = []

        # Title heading
        wrapper = soup.find("div", class_="item-page-field-wrapper")
        if wrapper and isinstance(wrapper, Tag):
            text = wrapper.get_text(strip=True)
            if text:
                parts.append(text)

        # Metadata table values
        table = soup.find("table", class_="itemDisplayTable")
        if table and isinstance(table, Tag):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if label and value:
                        parts.append(f"{label} {value}")

        return "\n".join(parts)
