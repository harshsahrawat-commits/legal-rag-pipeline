"""Docling-based PDF parser for Indian legal documents.

Wraps IBM's Docling ``DocumentConverter`` to parse statute PDFs into
structured ``ParsedDocument`` objects.  Exports to markdown (which
preserves section headings), then walks the markdown to build a
``ParsedSection`` hierarchy using Indian statute regex patterns.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from src.acquisition._models import ContentFormat, DocumentType
from src.parsing._exceptions import DocumentStructureError, ParserNotAvailableError
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParsedTable,
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
# Statute structure patterns (reused from IK HTML parser / chunking guide)
# ---------------------------------------------------------------------------
_CHAPTER_RE = re.compile(
    r"^(?:#+\s*)?CHAPTER\s+([IVXLCDM]+|\d+)\s*[-\u2014\u2013:.]?\s*(.*)",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"^(?:#+\s*)?(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+)?)\b[.\s]*(.*)",
    re.IGNORECASE,
)
_SUBSECTION_RE = re.compile(r"^\((\d+)\)")
_CLAUSE_RE = re.compile(r"^\(([a-z])\)")
_PROVISO_RE = re.compile(r"^Provided\s+that", re.IGNORECASE)
_EXPLANATION_RE = re.compile(r"^Explanation\.?", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"^(?:#+\s*)?SCHEDULE", re.IGNORECASE)
_PART_RE = re.compile(
    r"^(?:#+\s*)?PART\s+([IVXLCDM]+|\d+)\s*[-\u2014\u2013:.]?\s*(.*)",
    re.IGNORECASE,
)

# Markdown heading pattern
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def _docling_available() -> bool:
    """Check if docling is importable."""
    try:
        import docling.document_converter  # noqa: F401

        return True
    except ImportError:
        return False


class DoclingPdfParser(BaseParser):
    """Parses PDF documents using IBM Docling.

    Converts PDFs to structured markdown via Docling's
    ``DocumentConverter``, then walks the markdown output to build
    a ``ParsedSection`` hierarchy using Indian statute patterns.

    Raises ``ParserNotAvailableError`` at parse time if docling is
    not installed.
    """

    @property
    def parser_type(self) -> ParserType:
        return ParserType.DOCLING_PDF

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return raw_doc.content_format == ContentFormat.PDF

    def parse(
        self,
        content_path: Path,
        raw_doc: RawDocument,
    ) -> ParsedDocument:
        if not _docling_available():
            raise ParserNotAvailableError(
                "docling is not installed. Install with: pip install docling"
            )

        start = time.monotonic()

        md_text, page_count, tables = self._convert_with_docling(content_path)

        if not md_text.strip():
            raise DocumentStructureError(f"Docling produced empty output for {content_path}")

        doc_type = raw_doc.document_type or DocumentType.STATUTE
        sections = self._parse_markdown_sections(md_text, doc_type)

        elapsed = time.monotonic() - start

        # Merge metadata from Phase 1
        pm = raw_doc.preliminary_metadata
        meta: dict[str, str | int | None] = {}
        for field in ("title", "act_name", "act_number", "year", "date"):
            val = getattr(pm, field, None)
            if val is not None:
                meta[field] = val

        return ParsedDocument(
            document_id=raw_doc.document_id,
            source_type=raw_doc.source_type,
            document_type=doc_type,
            content_format=ContentFormat.PDF,
            raw_text=md_text,
            sections=sections,
            tables=tables,
            page_count=page_count,
            parser_used=ParserType.DOCLING_PDF,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path=str(content_path),
            parsing_duration_seconds=round(elapsed, 3),
            **meta,
        )

    # ------------------------------------------------------------------
    # Docling conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_with_docling(
        pdf_path: Path,
    ) -> tuple[str, int | None, list[ParsedTable]]:
        """Run Docling on a PDF and return (markdown, page_count, tables)."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        md_text = doc.export_to_markdown()

        # Page count from Docling's internal page tracking
        page_count: int | None = None
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)

        # Extract tables
        tables: list[ParsedTable] = []
        if hasattr(doc, "tables"):
            for i, tbl in enumerate(doc.tables):
                parsed_tbl = _extract_table(tbl, i)
                if parsed_tbl is not None:
                    tables.append(parsed_tbl)

        return md_text, page_count, tables

    # ------------------------------------------------------------------
    # Markdown → ParsedSection hierarchy
    # ------------------------------------------------------------------

    def _parse_markdown_sections(
        self,
        md_text: str,
        doc_type: DocumentType,
    ) -> list[ParsedSection]:
        """Walk markdown lines and build a section tree."""
        lines = md_text.split("\n")
        if doc_type in (DocumentType.STATUTE, DocumentType.SCHEDULE):
            return self._parse_statute_markdown(lines)
        return self._parse_generic_markdown(lines)

    def _parse_statute_markdown(self, lines: list[str]) -> list[ParsedSection]:
        """Parse statute-structured markdown into hierarchical sections."""
        sections: list[ParsedSection] = []
        current_chapter: ParsedSection | None = None
        current_section: ParsedSection | None = None
        preamble_lines: list[str] = []
        in_preamble = True

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Strip markdown heading markers for pattern matching
            clean = _MD_HEADING_RE.sub(r"\2", stripped)

            # -- Chapter heading --
            ch_match = _CHAPTER_RE.match(clean)
            if ch_match:
                in_preamble = False
                current_section = _finalize_section(current_section, current_chapter, sections)
                current_chapter = _finalize_chapter(current_chapter, sections)
                current_chapter = ParsedSection(
                    id=f"ch_{ch_match.group(1)}",
                    level=SectionLevel.CHAPTER,
                    number=ch_match.group(1),
                    title=ch_match.group(2).strip() or None,
                    text="",
                )
                continue

            # -- Part heading --
            part_match = _PART_RE.match(clean)
            if part_match:
                in_preamble = False
                current_section = _finalize_section(current_section, current_chapter, sections)
                current_chapter = _finalize_chapter(current_chapter, sections)
                sections.append(
                    ParsedSection(
                        id=f"part_{part_match.group(1)}",
                        level=SectionLevel.PART,
                        number=part_match.group(1),
                        title=part_match.group(2).strip() or None,
                        text="",
                    )
                )
                continue

            # -- Schedule heading --
            if _SCHEDULE_RE.match(clean):
                in_preamble = False
                current_section = _finalize_section(current_section, current_chapter, sections)
                current_chapter = _finalize_chapter(current_chapter, sections)
                sched_idx = sum(1 for s in sections if s.level == SectionLevel.SCHEDULE) + 1
                current_chapter = ParsedSection(
                    id=f"schedule_{sched_idx}",
                    level=SectionLevel.SCHEDULE,
                    title=clean,
                    text="",
                )
                continue

            # -- Section heading --
            sec_match = _SECTION_RE.match(clean)
            if sec_match:
                in_preamble = False
                current_section = _finalize_section(current_section, current_chapter, sections)
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

            # -- Preamble accumulation --
            if in_preamble:
                preamble_lines.append(stripped)
                continue

            # -- Child-level classification --
            if current_section is not None:
                child = _try_classify_child(stripped, current_section)
                if child is not None:
                    current_section.children.append(child)
                    continue

            # -- Continuation text --
            if current_section is not None:
                _append_text(current_section, stripped)
            elif current_chapter is not None:
                _append_text(current_chapter, stripped)
            else:
                preamble_lines.append(stripped)

        # Finalize remaining open nodes
        _finalize_section(current_section, current_chapter, sections)
        _finalize_chapter(current_chapter, sections)

        # Prepend preamble
        if preamble_lines:
            preamble_text = "\n".join(preamble_lines)
            sections.insert(
                0,
                ParsedSection(
                    id="preamble",
                    level=SectionLevel.PREAMBLE,
                    text=preamble_text,
                    token_count=len(preamble_text) // 4,
                ),
            )

        # Recompute token counts
        for sec in sections:
            _recompute_tokens(sec)

        return sections

    def _parse_generic_markdown(self, lines: list[str]) -> list[ParsedSection]:
        """Parse non-statute markdown using heading hierarchy."""
        sections: list[ParsedSection] = []
        current_lines: list[str] = []
        current_heading: str | None = None
        sec_idx = 0

        for line in lines:
            stripped = line.strip()
            heading_match = _MD_HEADING_RE.match(stripped)
            if heading_match:
                # Flush previous section
                if current_lines:
                    sec_idx += 1
                    text = "\n".join(current_lines)
                    sections.append(
                        ParsedSection(
                            id=f"para_{sec_idx}",
                            level=SectionLevel.PARAGRAPH,
                            title=current_heading,
                            text=text,
                            token_count=len(text) // 4,
                        )
                    )
                    current_lines = []
                current_heading = heading_match.group(2)
            elif stripped:
                current_lines.append(stripped)

        # Flush last section
        if current_lines:
            sec_idx += 1
            text = "\n".join(current_lines)
            sections.append(
                ParsedSection(
                    id=f"para_{sec_idx}",
                    level=SectionLevel.PARAGRAPH,
                    title=current_heading,
                    text=text,
                    token_count=len(text) // 4,
                )
            )

        return sections


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _try_classify_child(text: str, parent: ParsedSection) -> ParsedSection | None:
    """Try to classify a line as a child element of the current section."""
    if _PROVISO_RE.match(text):
        idx = sum(1 for c in parent.children if c.level == SectionLevel.PROVISO) + 1
        return ParsedSection(
            id=f"{parent.id}_proviso_{idx}",
            level=SectionLevel.PROVISO,
            text=text,
            parent_id=parent.id,
            token_count=len(text) // 4,
        )

    if _EXPLANATION_RE.match(text):
        idx = sum(1 for c in parent.children if c.level == SectionLevel.EXPLANATION) + 1
        return ParsedSection(
            id=f"{parent.id}_explanation_{idx}",
            level=SectionLevel.EXPLANATION,
            text=text,
            parent_id=parent.id,
            token_count=len(text) // 4,
        )

    clause_match = _CLAUSE_RE.match(text)
    if clause_match:
        return ParsedSection(
            id=f"{parent.id}_clause_{clause_match.group(1)}",
            level=SectionLevel.CLAUSE,
            number=clause_match.group(1),
            text=text,
            parent_id=parent.id,
            token_count=len(text) // 4,
        )

    subsec_match = _SUBSECTION_RE.match(text)
    if subsec_match:
        return ParsedSection(
            id=f"{parent.id}_subsec_{subsec_match.group(1)}",
            level=SectionLevel.SUB_SECTION,
            number=subsec_match.group(1),
            text=text,
            parent_id=parent.id,
            token_count=len(text) // 4,
        )

    return None


def _finalize_section(
    section: ParsedSection | None,
    chapter: ParsedSection | None,
    top_level: list[ParsedSection],
) -> None:
    """Attach finished section to its chapter or top-level list."""
    if section is None:
        return None
    if chapter is not None:
        chapter.children.append(section)
    else:
        top_level.append(section)
    return None


def _finalize_chapter(
    chapter: ParsedSection | None,
    top_level: list[ParsedSection],
) -> None:
    """Attach finished chapter to the top-level list."""
    if chapter is None:
        return None
    top_level.append(chapter)
    return None


def _append_text(section: ParsedSection, line: str) -> None:
    """Append a line of text to a section."""
    if section.text:
        section.text += "\n" + line
    else:
        section.text = line


def _recompute_tokens(section: ParsedSection) -> None:
    """Recursively set token_count from text length."""
    section.token_count = len(section.text) // 4 if section.text else 0
    for child in section.children:
        _recompute_tokens(child)


def _extract_table(tbl: object, index: int) -> ParsedTable | None:
    """Extract a ParsedTable from a Docling table object."""
    try:
        # Docling TableItem has export_to_dataframe() or data attribute
        if hasattr(tbl, "export_to_dataframe"):
            df = tbl.export_to_dataframe()
            headers = list(df.columns)
            rows = [list(row) for row in df.values]
            return ParsedTable(
                id=f"table_{index + 1}",
                headers=[str(h) for h in headers],
                rows=[[str(cell) for cell in row] for row in rows],
                row_count=len(rows),
                col_count=len(headers),
            )
        if hasattr(tbl, "data") and hasattr(tbl.data, "table_cells"):
            # Fallback: raw cell data
            cells = tbl.data.table_cells
            if not cells:
                return None
            max_row = max(c.row_span[0] for c in cells) + 1
            max_col = max(c.col_span[0] for c in cells) + 1
            grid: list[list[str]] = [[""] * max_col for _ in range(max_row)]
            for cell in cells:
                grid[cell.row_span[0]][cell.col_span[0]] = cell.text
            return ParsedTable(
                id=f"table_{index + 1}",
                headers=grid[0] if grid else [],
                rows=grid[1:] if len(grid) > 1 else [],
                row_count=max(max_row - 1, 0),
                col_count=max_col,
            )
    except Exception:
        _log.warning("table_extraction_failed", table_index=index)
    return None
