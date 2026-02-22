"""Indian Kanoon HTML parser.

Parses Indian Kanoon HTML pages (judgments and statutes) into
structured ``ParsedDocument`` objects with hierarchical sections.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.parsing._exceptions import DocumentStructureError
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

# ---------------------------------------------------------------------------
# Compiled patterns — statute structure
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"^(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+)?)\b[.\s]*(.*)",
    re.IGNORECASE,
)
_CHAPTER_RE = re.compile(
    r"^CHAPTER\s+([IVXLCDM]+|\d+)\s*[-\u2014\u2013:.]?\s*(.*)",
    re.IGNORECASE,
)
_ACT_NUMBER_RE = re.compile(
    r"\(Act\s+No\.?\s*(\d+)\s+of\s+(\d{4})\)",
    re.IGNORECASE,
)
_ACT_DATE_RE = re.compile(r"\[(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})\]")
_SUBSECTION_RE = re.compile(r"^\((\d+)\)")
_CLAUSE_RE = re.compile(r"^\(([a-z])\)")
_PROVISO_RE = re.compile(r"^Provided\s+that", re.IGNORECASE)
_EXPLANATION_RE = re.compile(r"^Explanation\.?", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Judgment section markers (case-insensitive substring matching)
# ---------------------------------------------------------------------------
_FACTS_MARKERS = [
    "facts of the case",
    "brief facts",
    "factual matrix",
    "the case of the prosecution",
]
_ISSUES_MARKERS = [
    "issues for consideration",
    "questions of law",
    "points for determination",
]
_REASONING_MARKERS = [
    "analysis and reasoning",
    "discussion",
    "we have carefully considered",
]
_HOLDING_MARKERS = [
    "holding",
    "in view of the above",
    "for the foregoing reasons",
    "we hold that",
]
_DISSENT_MARKERS = ["dissenting", "i am unable to agree"]
_ORDER_EXACT = "order"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class IndianKanoonHtmlParser(BaseParser):
    """Parses Indian Kanoon HTML into structured ``ParsedDocument``.

    Handles two document types:

    * **Judgments** — flat sections: HEADER, FACTS, ISSUES, REASONING,
      HOLDING, ORDER (optionally DISSENT / OBITER).
    * **Statutes** — hierarchical: PREAMBLE → CHAPTER → SECTION →
      CLAUSE / PROVISO / EXPLANATION children.
    """

    # -- BaseParser interface ------------------------------------------------

    @property
    def parser_type(self) -> ParserType:
        return ParserType.HTML_INDIAN_KANOON

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return (
            raw_doc.source_type == SourceType.INDIAN_KANOON
            and raw_doc.content_format == ContentFormat.HTML
        )

    def parse(
        self,
        content_path: Path,
        raw_doc: RawDocument,
    ) -> ParsedDocument:
        start = time.monotonic()

        html = self._read_html(content_path)
        if not html.strip():
            raise DocumentStructureError(f"Empty content in {content_path}")

        soup = BeautifulSoup(html, "html.parser")
        doc_type = self._detect_document_type(soup, raw_doc)
        metadata = self._extract_metadata(soup, raw_doc)

        judgments_div = soup.find("div", class_="judgments")
        if judgments_div is None or not isinstance(judgments_div, Tag):
            judgments_div = soup.find("body")
            if judgments_div is None or not judgments_div.get_text(strip=True):
                raise DocumentStructureError(
                    f"No content found in {content_path}"
                )
            _log.warning("missing_judgments_div", path=str(content_path))

        raw_text = judgments_div.get_text(separator="\n", strip=True)
        paragraphs: list[Tag] = [
            p for p in judgments_div.find_all("p") if isinstance(p, Tag)
        ]

        if doc_type == DocumentType.JUDGMENT:
            sections = self._parse_judgment(paragraphs)
        elif doc_type == DocumentType.STATUTE:
            sections = self._parse_statute(paragraphs)
        else:
            sections = [
                ParsedSection(
                    id="para_1",
                    level=SectionLevel.PARAGRAPH,
                    text=raw_text,
                    token_count=self._estimate_tokens(raw_text),
                )
            ]

        elapsed = time.monotonic() - start

        return ParsedDocument(
            document_id=raw_doc.document_id,
            source_type=raw_doc.source_type,
            document_type=doc_type,
            content_format=ContentFormat.HTML,
            raw_text=raw_text,
            sections=sections,
            tables=[],
            parser_used=ParserType.HTML_INDIAN_KANOON,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path=str(content_path),
            parsing_duration_seconds=round(elapsed, 3),
            **metadata,
        )

    # -- File I/O ------------------------------------------------------------

    @staticmethod
    def _read_html(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    # -- Metadata extraction -------------------------------------------------

    @staticmethod
    def _extract_metadata(soup: BeautifulSoup, raw_doc: RawDocument) -> dict:
        meta: dict[str, str | int | None] = {}

        title_div = soup.find("div", class_="doc_title")
        if title_div and isinstance(title_div, Tag):
            title_text = title_div.get_text(strip=True)
            meta["title"] = title_text
            if re.search(r"\bvs?\b\.?\s", title_text, re.IGNORECASE):
                meta["parties"] = title_text

        bench_div = soup.find("div", class_="doc_bench")
        if bench_div and isinstance(bench_div, Tag):
            meta["court"] = bench_div.get_text(strip=True)

        cite_div = soup.find("div", class_="doc_citations")
        if cite_div and isinstance(cite_div, Tag):
            meta["case_citation"] = cite_div.get_text(strip=True)

        # Merge Phase 1 preliminary metadata for anything not yet set
        pm = raw_doc.preliminary_metadata
        for field in ("act_name", "year", "date", "court", "case_citation"):
            val = getattr(pm, field, None)
            if val is not None and field not in meta:
                meta[field] = val

        return meta

    # -- Document type detection ---------------------------------------------

    @staticmethod
    def _detect_document_type(
        soup: BeautifulSoup,
        raw_doc: RawDocument,
    ) -> DocumentType:
        if raw_doc.document_type is not None:
            return raw_doc.document_type

        judgments_div = soup.find("div", class_="judgments")
        if not judgments_div or not isinstance(judgments_div, Tag):
            return DocumentType.JUDGMENT

        text = judgments_div.get_text()
        # Use boundary-based patterns (no ^ anchor) for mid-text detection
        has_sections = bool(
            re.search(r"\bSection\s+\d+", text, re.IGNORECASE)
        )
        has_chapters = bool(
            re.search(r"\bCHAPTER\s+[IVXLCDM\d]+", text, re.IGNORECASE)
        )
        has_act_number = bool(_ACT_NUMBER_RE.search(text))

        if has_sections and (has_chapters or has_act_number):
            return DocumentType.STATUTE

        return DocumentType.JUDGMENT

    # -- Token estimation ----------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4 if text else 0

    # -- Judgment parsing ----------------------------------------------------

    def _parse_judgment(self, paragraphs: list[Tag]) -> list[ParsedSection]:
        # 1) Identify section boundaries
        boundaries: list[tuple[int, SectionLevel, str]] = []
        for i, p in enumerate(paragraphs):
            bold = p.find("b")
            if not bold or not isinstance(bold, Tag):
                continue
            # Only treat as heading when bold starts the paragraph
            bold_text = bold.get_text(strip=True)
            para_text = p.get_text(strip=True)
            if not para_text.startswith(bold_text):
                continue

            level = self._classify_judgment_heading(bold_text)
            if level is not None:
                boundaries.append((i, level, bold_text))

        # 2) No structural markers → single paragraph
        if not boundaries:
            all_text = "\n\n".join(
                p.get_text(strip=True) for p in paragraphs
            )
            if not all_text:
                return []
            return [
                ParsedSection(
                    id="jdg_para_1",
                    level=SectionLevel.PARAGRAPH,
                    text=all_text,
                    token_count=self._estimate_tokens(all_text),
                )
            ]

        sections: list[ParsedSection] = []

        # 3) Header: everything before first boundary
        if boundaries[0][0] > 0:
            header_text = "\n\n".join(
                p.get_text(strip=True) for p in paragraphs[: boundaries[0][0]]
            )
            if header_text:
                sections.append(
                    ParsedSection(
                        id="jdg_header",
                        level=SectionLevel.HEADER,
                        title="Header",
                        text=header_text,
                        token_count=self._estimate_tokens(header_text),
                    )
                )

        # 4) Sections between boundaries
        for k, (idx, level, heading) in enumerate(boundaries):
            end = boundaries[k + 1][0] if k + 1 < len(boundaries) else len(paragraphs)
            content_paragraphs = paragraphs[idx + 1 : end]
            text = "\n\n".join(
                p.get_text(strip=True) for p in content_paragraphs
            )
            sec_id = f"jdg_{level.value}"
            sections.append(
                ParsedSection(
                    id=sec_id,
                    level=level,
                    title=heading,
                    text=text,
                    token_count=self._estimate_tokens(text),
                )
            )

        return sections

    @staticmethod
    def _classify_judgment_heading(heading_text: str) -> SectionLevel | None:
        lower = heading_text.lower().strip()

        if any(m in lower for m in _FACTS_MARKERS):
            return SectionLevel.FACTS
        if any(m in lower for m in _ISSUES_MARKERS):
            return SectionLevel.ISSUES
        if any(m in lower for m in _REASONING_MARKERS):
            return SectionLevel.REASONING
        if any(m in lower for m in _HOLDING_MARKERS):
            return SectionLevel.HOLDING
        if any(m in lower for m in _DISSENT_MARKERS):
            return SectionLevel.DISSENT
        # ORDER must be an exact match to avoid false positives
        if lower == _ORDER_EXACT:
            return SectionLevel.ORDER
        return None

    # -- Statute parsing -----------------------------------------------------

    def _parse_statute(self, paragraphs: list[Tag]) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        current_chapter: ParsedSection | None = None
        current_section: ParsedSection | None = None

        preamble, start_idx = self._extract_preamble(paragraphs)
        if preamble is not None:
            sections.append(preamble)

        for p in paragraphs[start_idx:]:
            text = p.get_text(strip=True)
            if not text:
                continue

            bold = p.find("b")
            bold_text = bold.get_text(strip=True) if bold and isinstance(bold, Tag) else ""

            # -- Chapter heading -------------------------------------------------
            if bold_text:
                ch_match = _CHAPTER_RE.match(bold_text)
                if ch_match:
                    # Finalize current section → chapter
                    current_section = self._finalize_section(
                        current_section, current_chapter, sections
                    )
                    # Finalize current chapter → top-level
                    current_chapter = self._finalize_chapter(
                        current_chapter, sections
                    )
                    current_chapter = ParsedSection(
                        id=f"ch_{ch_match.group(1)}",
                        level=SectionLevel.CHAPTER,
                        number=ch_match.group(1),
                        title=ch_match.group(2).strip() if ch_match.group(2) else None,
                        text="",
                    )
                    continue

                # -- Section heading -------------------------------------------------
                sec_match = _SECTION_RE.match(bold_text)
                if sec_match:
                    current_section = self._finalize_section(
                        current_section, current_chapter, sections
                    )
                    sec_num = sec_match.group(1)
                    sec_title = sec_match.group(2).strip().rstrip(".")
                    parent_id = current_chapter.id if current_chapter else None
                    current_section = ParsedSection(
                        id=f"sec_{sec_num}",
                        level=SectionLevel.SECTION,
                        number=sec_num,
                        title=sec_title or None,
                        text="",
                        parent_id=parent_id,
                    )
                    continue

            # -- Child-level classification (proviso, explanation, clause) --------
            if current_section is not None:
                if _PROVISO_RE.match(text):
                    prov_idx = sum(
                        1
                        for c in current_section.children
                        if c.level == SectionLevel.PROVISO
                    ) + 1
                    current_section.children.append(
                        ParsedSection(
                            id=f"{current_section.id}_proviso_{prov_idx}",
                            level=SectionLevel.PROVISO,
                            text=text,
                            parent_id=current_section.id,
                            token_count=self._estimate_tokens(text),
                        )
                    )
                    continue

                if _EXPLANATION_RE.match(text):
                    exp_idx = sum(
                        1
                        for c in current_section.children
                        if c.level == SectionLevel.EXPLANATION
                    ) + 1
                    current_section.children.append(
                        ParsedSection(
                            id=f"{current_section.id}_explanation_{exp_idx}",
                            level=SectionLevel.EXPLANATION,
                            text=text,
                            parent_id=current_section.id,
                            token_count=self._estimate_tokens(text),
                        )
                    )
                    continue

                clause_match = _CLAUSE_RE.match(text)
                if clause_match:
                    current_section.children.append(
                        ParsedSection(
                            id=f"{current_section.id}_clause_{clause_match.group(1)}",
                            level=SectionLevel.CLAUSE,
                            number=clause_match.group(1),
                            text=text,
                            parent_id=current_section.id,
                            token_count=self._estimate_tokens(text),
                        )
                    )
                    continue

                subsec_match = _SUBSECTION_RE.match(text)
                if subsec_match:
                    current_section.children.append(
                        ParsedSection(
                            id=f"{current_section.id}_subsec_{subsec_match.group(1)}",
                            level=SectionLevel.SUB_SECTION,
                            number=subsec_match.group(1),
                            text=text,
                            parent_id=current_section.id,
                            token_count=self._estimate_tokens(text),
                        )
                    )
                    continue

            # -- Continuation text -----------------------------------------------
            if current_section is not None:
                if current_section.text:
                    current_section.text += "\n\n" + text
                else:
                    current_section.text = text
            elif current_chapter is not None:
                if current_chapter.text:
                    current_chapter.text += "\n\n" + text
                else:
                    current_chapter.text = text

        # Finalize remaining open nodes
        self._finalize_section(current_section, current_chapter, sections)
        self._finalize_chapter(current_chapter, sections)

        # Recompute token counts for sections/chapters that accumulated text
        for sec in sections:
            self._recompute_tokens(sec)

        return sections

    def _extract_preamble(
        self,
        paragraphs: list[Tag],
    ) -> tuple[ParsedSection | None, int]:
        """Extract preamble from initial paragraphs before CHAPTER/Section."""
        preamble_parts: list[str] = []
        end_idx = 0

        for i, p in enumerate(paragraphs):
            text = p.get_text(strip=True)
            if not text:
                continue

            bold = p.find("b")
            bold_text = bold.get_text(strip=True) if bold and isinstance(bold, Tag) else ""

            # Stop at first chapter or section heading
            if bold_text and (
                _CHAPTER_RE.match(bold_text) or _SECTION_RE.match(bold_text)
            ):
                break

            preamble_parts.append(text)
            end_idx = i + 1

        if not preamble_parts:
            return None, 0

        preamble_text = "\n\n".join(preamble_parts)
        return (
            ParsedSection(
                id="preamble",
                level=SectionLevel.PREAMBLE,
                text=preamble_text,
                token_count=self._estimate_tokens(preamble_text),
            ),
            end_idx,
        )

    @staticmethod
    def _finalize_section(
        section: ParsedSection | None,
        chapter: ParsedSection | None,
        top_level: list[ParsedSection],
    ) -> None:
        """Attach finished section to its chapter, or to top-level list."""
        if section is None:
            return None
        if chapter is not None:
            chapter.children.append(section)
        else:
            top_level.append(section)
        return None

    @staticmethod
    def _finalize_chapter(
        chapter: ParsedSection | None,
        top_level: list[ParsedSection],
    ) -> None:
        """Attach finished chapter to the top-level list."""
        if chapter is None:
            return None
        top_level.append(chapter)
        return None

    def _recompute_tokens(self, section: ParsedSection) -> None:
        """Recursively set token_count from text length."""
        section.token_count = self._estimate_tokens(section.text)
        for child in section.children:
            self._recompute_tokens(child)
