from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    PreliminaryMetadata,
    RawDocument,
    SourceType,
)
from src.parsing._exceptions import DocumentStructureError, ParserNotAvailableError
from src.parsing._models import ParserType, SectionLevel
from src.parsing.parsers._docling_pdf import DoclingPdfParser

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Sample markdown outputs (simulating Docling export_to_markdown)
# ---------------------------------------------------------------------------

STATUTE_MARKDOWN = """\
THE ACADEMY OF SCIENTIFIC AND INNOVATIVE RESEARCH ACT, 2011

(Act No. 13 of 2011)

[6th February, 2012]

An Act to establish and incorporate an Academy of Scientific and Innovative Research.

## CHAPTER I - PRELIMINARY

## Section 1. Short title and commencement.

(1) This Act may be called the Academy of Scientific and Innovative Research Act, 2011.

(2) It shall come into force on such date as the Central Government may appoint.

## Section 2. Definitions.

In this Act, unless the context otherwise requires,—

(a) "Academy" means the Academy of Scientific and Innovative Research;

(b) "Council" means the Governing Council of the Academy;

Explanation. For the purposes of this section, "prescribed" means prescribed by rules.

## CHAPTER II - ESTABLISHMENT OF THE ACADEMY

## Section 3. Establishment of Academy.

There shall be established an Academy to be called the Academy of Scientific and Innovative Research.

Provided that the Academy shall be a body corporate.

## Section 4. Objects of the Academy.

The objects of the Academy shall be to create human resource in science and technology.
"""

GENERIC_MARKDOWN = """\
# Introduction

This is a notification from the Ministry of Law.

# Section A

Details about section A of the notification.

# Section B

Details about section B of the notification.
"""

EMPTY_MARKDOWN = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_doc(
    doc_type: DocumentType | None = DocumentType.STATUTE,
    source: SourceType = SourceType.INDIA_CODE,
    fmt: ContentFormat = ContentFormat.PDF,
    **meta_kwargs,
) -> RawDocument:
    return RawDocument(
        url="https://www.indiacode.nic.in/bitstream/123456789/2110/1/a2012-13.pdf",
        source_type=source,
        content_format=fmt,
        raw_content_path="data/cache/pdf/ic_2110.pdf",
        document_type=doc_type,
        preliminary_metadata=PreliminaryMetadata(**meta_kwargs),
    )


def _write_pdf(tmp_path: Path, name: str = "test.pdf") -> Path:
    """Write a dummy PDF file (content doesn't matter — we mock Docling)."""
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 dummy content")
    return p


def _patch_docling(md_text: str, page_count: int = 5):
    """Patch _convert_with_docling to return parsed markdown without Docling."""
    from src.parsing._models import ParsedTable

    tables: list[ParsedTable] = []

    return patch.object(
        DoclingPdfParser,
        "_convert_with_docling",
        staticmethod(lambda _path: (md_text, page_count, tables)),
    )


# ---------------------------------------------------------------------------
# TestCanParse
# ---------------------------------------------------------------------------


class TestCanParse:
    def test_accepts_pdf_format(self):
        parser = DoclingPdfParser()
        raw = _make_raw_doc(fmt=ContentFormat.PDF)
        assert parser.can_parse(raw) is True

    def test_rejects_html_format(self):
        parser = DoclingPdfParser()
        raw = _make_raw_doc(fmt=ContentFormat.HTML)
        assert parser.can_parse(raw) is False

    def test_accepts_any_source_with_pdf(self):
        parser = DoclingPdfParser()
        raw = _make_raw_doc(source=SourceType.INDIAN_KANOON, fmt=ContentFormat.PDF)
        assert parser.can_parse(raw) is True


# ---------------------------------------------------------------------------
# TestParserType
# ---------------------------------------------------------------------------


class TestParserType:
    def test_parser_type(self):
        assert DoclingPdfParser().parser_type == ParserType.DOCLING_PDF


# ---------------------------------------------------------------------------
# TestDoclingNotAvailable
# ---------------------------------------------------------------------------


class TestDoclingNotAvailable:
    def test_raises_when_docling_missing(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=False,
            ),
            pytest.raises(ParserNotAvailableError, match="docling is not installed"),
        ):
            parser.parse(pdf, raw)


# ---------------------------------------------------------------------------
# TestStatuteParsing
# ---------------------------------------------------------------------------


class TestStatuteParsing:
    def test_extracts_preamble(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        preamble = result.sections[0]
        assert preamble.level == SectionLevel.PREAMBLE
        assert "Academy of Scientific" in preamble.text

    def test_extracts_chapters(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        chapters = [s for s in result.sections if s.level == SectionLevel.CHAPTER]
        assert len(chapters) == 2
        assert chapters[0].id == "ch_I"
        assert chapters[1].id == "ch_II"

    def test_chapters_contain_sections(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        ch1 = next(s for s in result.sections if s.id == "ch_I")
        child_ids = [c.id for c in ch1.children]
        assert "sec_1" in child_ids
        assert "sec_2" in child_ids

    def test_section_has_subsections(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        ch1 = next(s for s in result.sections if s.id == "ch_I")
        sec1 = next(c for c in ch1.children if c.id == "sec_1")
        subsections = [c for c in sec1.children if c.level == SectionLevel.SUB_SECTION]
        assert len(subsections) == 2

    def test_section_has_clauses(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        ch1 = next(s for s in result.sections if s.id == "ch_I")
        sec2 = next(c for c in ch1.children if c.id == "sec_2")
        clauses = [c for c in sec2.children if c.level == SectionLevel.CLAUSE]
        assert len(clauses) == 2
        assert clauses[0].number == "a"
        assert clauses[1].number == "b"

    def test_section_has_explanation(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        ch1 = next(s for s in result.sections if s.id == "ch_I")
        sec2 = next(c for c in ch1.children if c.id == "sec_2")
        explanations = [c for c in sec2.children if c.level == SectionLevel.EXPLANATION]
        assert len(explanations) == 1

    def test_section_has_proviso(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        ch2 = next(s for s in result.sections if s.id == "ch_II")
        sec3 = next(c for c in ch2.children if c.id == "sec_3")
        provisos = [c for c in sec3.children if c.level == SectionLevel.PROVISO]
        assert len(provisos) == 1


# ---------------------------------------------------------------------------
# TestGenericParsing
# ---------------------------------------------------------------------------


class TestGenericParsing:
    def test_parses_generic_markdown_as_paragraphs(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc(doc_type=DocumentType.NOTIFICATION)

        with (
            _patch_docling(GENERIC_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        assert len(result.sections) >= 2
        assert all(s.level == SectionLevel.PARAGRAPH for s in result.sections)

    def test_generic_sections_have_headings(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc(doc_type=DocumentType.NOTIFICATION)

        with (
            _patch_docling(GENERIC_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        titles = [s.title for s in result.sections if s.title]
        assert "Introduction" in titles
        assert "Section A" in titles


# ---------------------------------------------------------------------------
# TestEmptyOutput
# ---------------------------------------------------------------------------


class TestEmptyOutput:
    def test_raises_on_empty_markdown(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(EMPTY_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
            pytest.raises(DocumentStructureError, match="empty output"),
        ):
            parser.parse(pdf, raw)


# ---------------------------------------------------------------------------
# TestOutputContract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_document_fields(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc(
            title="The Academy Act, 2011",
            act_name="The Academy Act, 2011",
            year=2011,
        )

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        assert result.source_type == SourceType.INDIA_CODE
        assert result.document_type == DocumentType.STATUTE
        assert result.content_format == ContentFormat.PDF
        assert result.parser_used == ParserType.DOCLING_PDF
        assert result.page_count == 5
        assert result.parsing_duration_seconds >= 0
        assert result.title == "The Academy Act, 2011"
        assert result.year == 2011
        assert result.quality.passed is False  # placeholder

    def test_raw_text_is_full_markdown(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        assert "CHAPTER I" in result.raw_text
        assert "Section 1" in result.raw_text

    def test_round_trips_through_json(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        json_str = result.model_dump_json()
        assert '"docling_pdf"' in json_str

    def test_token_counts_set(self, tmp_path: Path):
        parser = DoclingPdfParser()
        pdf = _write_pdf(tmp_path)
        raw = _make_raw_doc()

        with (
            _patch_docling(STATUTE_MARKDOWN),
            patch(
                "src.parsing.parsers._docling_pdf._docling_available",
                return_value=True,
            ),
        ):
            result = parser.parse(pdf, raw)

        for sec in result.sections:
            if sec.text:
                assert sec.token_count > 0
