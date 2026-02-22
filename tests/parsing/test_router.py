from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.acquisition._models import ContentFormat, DocumentType, RawDocument, SourceType
from src.parsing._exceptions import UnsupportedFormatError
from src.parsing._models import (
    ParsedDocument,
    ParserType,
    ParsingSettings,
    QualityReport,
)
from src.parsing._router import ParserRouter
from src.parsing.parsers._base import BaseParser

if TYPE_CHECKING:
    from pathlib import Path


# --- Stub parsers ---


class PdfParser(BaseParser):
    def parse(self, content_path: Path, raw_doc: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_type=raw_doc.source_type,
            document_type=DocumentType.STATUTE,
            content_format=raw_doc.content_format,
            raw_text="pdf content",
            parser_used=self.parser_type,
            quality=QualityReport(overall_score=1.0, passed=True),
            raw_content_path=str(content_path),
        )

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return raw_doc.content_format == ContentFormat.PDF

    @property
    def parser_type(self) -> ParserType:
        return ParserType.DOCLING_PDF


class HtmlIKParser(BaseParser):
    def parse(self, content_path: Path, raw_doc: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_type=raw_doc.source_type,
            document_type=DocumentType.JUDGMENT,
            content_format=raw_doc.content_format,
            raw_text="html content",
            parser_used=self.parser_type,
            quality=QualityReport(overall_score=1.0, passed=True),
            raw_content_path=str(content_path),
        )

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return (
            raw_doc.source_type == SourceType.INDIAN_KANOON
            and raw_doc.content_format == ContentFormat.HTML
        )

    @property
    def parser_type(self) -> ParserType:
        return ParserType.HTML_INDIAN_KANOON


class HtmlICParser(BaseParser):
    def parse(self, content_path: Path, raw_doc: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_type=raw_doc.source_type,
            document_type=DocumentType.STATUTE,
            content_format=raw_doc.content_format,
            raw_text="india code html",
            parser_used=self.parser_type,
            quality=QualityReport(overall_score=1.0, passed=True),
            raw_content_path=str(content_path),
        )

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return (
            raw_doc.source_type == SourceType.INDIA_CODE
            and raw_doc.content_format == ContentFormat.HTML
        )

    @property
    def parser_type(self) -> ParserType:
        return ParserType.HTML_INDIA_CODE


class NeverParser(BaseParser):
    """Parser that rejects everything — useful for priority testing."""

    def parse(self, content_path: Path, raw_doc: RawDocument) -> ParsedDocument:
        raise NotImplementedError

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return False

    @property
    def parser_type(self) -> ParserType:
        return ParserType.PYMUPDF_PDF


# --- Helpers ---


def _make_raw_doc(
    source: SourceType = SourceType.INDIA_CODE,
    fmt: ContentFormat = ContentFormat.PDF,
) -> RawDocument:
    return RawDocument(
        url="https://example.com/doc",
        source_type=source,
        content_format=fmt,
        raw_content_path="data/raw/test/doc.html",
    )


# --- Tests ---


class TestParserRouter:
    def test_register_adds_parser(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(PdfParser())
        assert len(router.available_parsers) == 1

    def test_register_multiple_parsers(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(PdfParser())
        router.register(HtmlIKParser())
        router.register(HtmlICParser())
        assert len(router.available_parsers) == 3

    def test_available_parsers_returns_copy(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(PdfParser())
        parsers = router.available_parsers
        parsers.clear()
        assert len(router.available_parsers) == 1

    def test_select_parser_matches_pdf(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(PdfParser())
        raw_doc = _make_raw_doc(fmt=ContentFormat.PDF)
        selected = router.select_parser(raw_doc)
        assert selected.parser_type == ParserType.DOCLING_PDF

    def test_select_parser_matches_ik_html(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(PdfParser())
        router.register(HtmlIKParser())
        raw_doc = _make_raw_doc(source=SourceType.INDIAN_KANOON, fmt=ContentFormat.HTML)
        selected = router.select_parser(raw_doc)
        assert selected.parser_type == ParserType.HTML_INDIAN_KANOON

    def test_select_parser_matches_ic_html(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(HtmlICParser())
        raw_doc = _make_raw_doc(source=SourceType.INDIA_CODE, fmt=ContentFormat.HTML)
        selected = router.select_parser(raw_doc)
        assert selected.parser_type == ParserType.HTML_INDIA_CODE

    def test_priority_first_match_wins(self, parsing_settings: ParsingSettings):
        """When multiple parsers can handle a doc, the first registered wins."""
        router = ParserRouter(parsing_settings)
        pdf1 = PdfParser()
        pdf2 = PdfParser()  # same type, both can_parse PDF
        router.register(pdf1)
        router.register(pdf2)
        raw_doc = _make_raw_doc(fmt=ContentFormat.PDF)
        selected = router.select_parser(raw_doc)
        assert selected is pdf1

    def test_skips_parsers_that_cannot_handle(self, parsing_settings: ParsingSettings):
        """A NeverParser is skipped; the next capable parser is selected."""
        router = ParserRouter(parsing_settings)
        router.register(NeverParser())
        router.register(PdfParser())
        raw_doc = _make_raw_doc(fmt=ContentFormat.PDF)
        selected = router.select_parser(raw_doc)
        assert selected.parser_type == ParserType.DOCLING_PDF

    def test_no_parser_raises_unsupported_format(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        router.register(PdfParser())
        raw_doc = _make_raw_doc(source=SourceType.INDIAN_KANOON, fmt=ContentFormat.HTML)
        with pytest.raises(UnsupportedFormatError, match="No parser available"):
            router.select_parser(raw_doc)

    def test_empty_router_raises_unsupported_format(self, parsing_settings: ParsingSettings):
        router = ParserRouter(parsing_settings)
        raw_doc = _make_raw_doc()
        with pytest.raises(UnsupportedFormatError):
            router.select_parser(raw_doc)
