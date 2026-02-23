from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.acquisition._models import ContentFormat, DocumentType, RawDocument, SourceType
from src.parsing._exceptions import DocumentStructureError
from src.parsing._models import (
    ParsedDocument,
    ParserType,
    SectionLevel,
)
from src.parsing.parsers._html_indian_kanoon import IndianKanoonHtmlParser

from .conftest import SAMPLE_JUDGMENT_HTML, SAMPLE_STATUTE_HTML

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_doc(
    doc_type: DocumentType | None = None,
    source: SourceType = SourceType.INDIAN_KANOON,
    fmt: ContentFormat = ContentFormat.HTML,
) -> RawDocument:
    return RawDocument(
        url="https://indiankanoon.org/doc/123/",
        source_type=source,
        content_format=fmt,
        raw_content_path="data/raw/indian_kanoon/doc_123.html",
        document_type=doc_type,
    )


def _write_html(tmp_path: Path, html: str, name: str = "doc.html") -> Path:
    p = tmp_path / name
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TestCanParse
# ---------------------------------------------------------------------------


class TestCanParse:
    def test_accepts_indian_kanoon_html(self):
        parser = IndianKanoonHtmlParser()
        raw = _make_raw_doc()
        assert parser.can_parse(raw) is True

    def test_rejects_india_code_html(self):
        parser = IndianKanoonHtmlParser()
        raw = _make_raw_doc(source=SourceType.INDIA_CODE)
        assert parser.can_parse(raw) is False

    def test_rejects_pdf_format(self):
        parser = IndianKanoonHtmlParser()
        raw = _make_raw_doc(fmt=ContentFormat.PDF)
        assert parser.can_parse(raw) is False


# ---------------------------------------------------------------------------
# TestParserType
# ---------------------------------------------------------------------------


class TestParserType:
    def test_parser_type_is_html_indian_kanoon(self):
        parser = IndianKanoonHtmlParser()
        assert parser.parser_type == ParserType.HTML_INDIAN_KANOON


# ---------------------------------------------------------------------------
# TestJudgmentParsing
# ---------------------------------------------------------------------------


class TestJudgmentParsing:
    def test_detects_judgment_sections(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        levels = [s.level for s in doc.sections]
        assert SectionLevel.HEADER in levels
        assert SectionLevel.FACTS in levels
        assert SectionLevel.ISSUES in levels
        assert SectionLevel.REASONING in levels
        assert SectionLevel.HOLDING in levels
        assert SectionLevel.ORDER in levels

    def test_header_section_contains_court_and_parties(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        header = next(s for s in doc.sections if s.level == SectionLevel.HEADER)
        assert "SUPREME COURT" in header.text
        assert "State of Maharashtra" in header.text

    def test_facts_section_text(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        facts = next(s for s in doc.sections if s.level == SectionLevel.FACTS)
        assert "appellant State of Maharashtra" in facts.text
        assert "Section 302" in facts.text

    def test_issues_section_text(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        issues = next(s for s in doc.sections if s.level == SectionLevel.ISSUES)
        assert "High Court was correct" in issues.text

    def test_reasoning_section_text(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        reasoning = next(s for s in doc.sections if s.level == SectionLevel.REASONING)
        assert "circumstantial evidence" in reasoning.text
        assert "Nanavati" in reasoning.text

    def test_holding_section_text(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        holding = next(s for s in doc.sections if s.level == SectionLevel.HOLDING)
        assert "appeal is dismissed" in holding.text

    def test_order_section_text(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        order = next(s for s in doc.sections if s.level == SectionLevel.ORDER)
        assert "No order as to costs" in order.text


# ---------------------------------------------------------------------------
# TestStatuteParsing
# ---------------------------------------------------------------------------


class TestStatuteParsing:
    def test_detects_statute_chapters(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        chapters = [s for s in doc.sections if s.level == SectionLevel.CHAPTER]
        assert len(chapters) == 2
        assert chapters[0].number == "I"
        assert chapters[1].number == "II"

    def test_detects_statute_sections(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        # Sections are children of chapters
        all_sections: list[ParsedDocument] = []
        for s in doc.sections:
            if s.level == SectionLevel.SECTION:
                all_sections.append(s)
            for child in s.children:
                if child.level == SectionLevel.SECTION:
                    all_sections.append(child)

        numbers = {s.number for s in all_sections}
        assert "1" in numbers
        assert "2" in numbers
        assert "10" in numbers
        assert "23" in numbers

    def test_section_2_has_clauses(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        sec2 = self._find_section(doc, "2")
        assert sec2 is not None
        clauses = [c for c in sec2.children if c.level == SectionLevel.CLAUSE]
        clause_nums = {c.number for c in clauses}
        assert "a" in clause_nums
        assert "b" in clause_nums

    def test_section_10_has_explanation(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        sec10 = self._find_section(doc, "10")
        assert sec10 is not None
        explanations = [c for c in sec10.children if c.level == SectionLevel.EXPLANATION]
        assert len(explanations) >= 1

    def test_section_10_has_proviso(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        sec10 = self._find_section(doc, "10")
        assert sec10 is not None
        provisos = [c for c in sec10.children if c.level == SectionLevel.PROVISO]
        assert len(provisos) >= 1

    def test_preamble_extracted(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        preamble = next((s for s in doc.sections if s.level == SectionLevel.PREAMBLE), None)
        assert preamble is not None
        assert "Act No. 9 of 1872" in preamble.text

    def test_chapter_contains_sections_as_children(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.STATUTE)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        ch1 = next(
            (s for s in doc.sections if s.level == SectionLevel.CHAPTER and s.number == "I"),
            None,
        )
        assert ch1 is not None
        section_children = [c for c in ch1.children if c.level == SectionLevel.SECTION]
        assert len(section_children) >= 2  # at least Section 1, 2

    @staticmethod
    def _find_section(doc: ParsedDocument, number: str):
        """Recursively find a SECTION with the given number."""
        for s in doc.sections:
            if s.level == SectionLevel.SECTION and s.number == number:
                return s
            for child in s.children:
                if child.level == SectionLevel.SECTION and child.number == number:
                    return child
        return None


# ---------------------------------------------------------------------------
# TestMetadataExtraction
# ---------------------------------------------------------------------------


class TestMetadataExtraction:
    def test_judgment_title_extracted(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert doc.title == "State Of Maharashtra vs Rajesh Kumar"

    def test_judgment_bench_extracted(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert doc.court is not None
        assert "Justice A.B. Sharma" in doc.court

    def test_judgment_citation_extracted(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert doc.case_citation is not None
        assert "AIR 2024 SC 1500" in doc.case_citation


# ---------------------------------------------------------------------------
# TestDocumentTypeDetection
# ---------------------------------------------------------------------------


class TestDocumentTypeDetection:
    def test_auto_detect_judgment(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=None)  # no type hint
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert doc.document_type == DocumentType.JUDGMENT

    def test_auto_detect_statute(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_STATUTE_HTML)
        raw = _make_raw_doc(doc_type=None)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert doc.document_type == DocumentType.STATUTE


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_file_raises_document_structure_error(self, tmp_path: Path):
        path = _write_html(tmp_path, "")
        raw = _make_raw_doc()
        with pytest.raises(DocumentStructureError, match="Empty content"):
            IndianKanoonHtmlParser().parse(path, raw)

    def test_no_judgments_div_falls_back_to_body(self, tmp_path: Path):
        html = "<html><body><p>Some legal text here.</p></body></html>"
        path = _write_html(tmp_path, html)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert doc.raw_text
        assert "Some legal text" in doc.raw_text

    def test_no_structural_markers_returns_paragraph(self, tmp_path: Path):
        html = """<html><body>
        <div class="judgments">
        <p>This is some legal text without any bold headings.</p>
        <p>Another paragraph of plain text.</p>
        </div></body></html>"""
        path = _write_html(tmp_path, html)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        assert len(doc.sections) == 1
        assert doc.sections[0].level == SectionLevel.PARAGRAPH

    def test_token_count_populated(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)
        for section in doc.sections:
            if section.text:
                assert section.token_count > 0


# ---------------------------------------------------------------------------
# TestOutputContract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_parsed_document_fields(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        assert doc.source_type == SourceType.INDIAN_KANOON
        assert doc.content_format == ContentFormat.HTML
        assert doc.parser_used == ParserType.HTML_INDIAN_KANOON
        assert doc.document_type == DocumentType.JUDGMENT
        assert len(doc.raw_text) > 0
        assert doc.parsing_duration_seconds >= 0

    def test_parsed_document_serializes_to_json(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_JUDGMENT_HTML)
        raw = _make_raw_doc(doc_type=DocumentType.JUDGMENT)
        doc = IndianKanoonHtmlParser().parse(path, raw)

        json_str = doc.model_dump_json()
        roundtripped = ParsedDocument.model_validate_json(json_str)
        assert roundtripped.document_id == doc.document_id
        assert len(roundtripped.sections) == len(doc.sections)
