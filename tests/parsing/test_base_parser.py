from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.acquisition._models import ContentFormat, DocumentType, RawDocument, SourceType
from src.parsing._models import (
    ParsedDocument,
    ParserType,
    QualityReport,
)
from src.parsing.parsers._base import BaseParser

if TYPE_CHECKING:
    from pathlib import Path


# --- Concrete stub for testing ---


class StubParser(BaseParser):
    """Minimal concrete parser for testing the ABC contract."""

    def parse(self, content_path: Path, raw_doc: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_type=raw_doc.source_type,
            document_type=DocumentType.STATUTE,
            content_format=raw_doc.content_format,
            raw_text="stub",
            parser_used=self.parser_type,
            quality=QualityReport(overall_score=1.0, passed=True),
            raw_content_path=str(content_path),
        )

    def can_parse(self, raw_doc: RawDocument) -> bool:
        return raw_doc.content_format == ContentFormat.PDF

    @property
    def parser_type(self) -> ParserType:
        return ParserType.DOCLING_PDF


# --- Tests ---


class TestBaseParserABC:
    """BaseParser is abstract and enforces the interface contract."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseParser()  # type: ignore[abstract]

    def test_subclass_missing_parse_raises(self):
        class BadParser(BaseParser):
            def can_parse(self, raw_doc):
                return True

            @property
            def parser_type(self):
                return ParserType.DOCLING_PDF

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_subclass_missing_can_parse_raises(self):
        class BadParser(BaseParser):
            def parse(self, content_path, raw_doc):
                return None

            @property
            def parser_type(self):
                return ParserType.DOCLING_PDF

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_subclass_missing_parser_type_raises(self):
        class BadParser(BaseParser):
            def parse(self, content_path, raw_doc):
                return None

            def can_parse(self, raw_doc):
                return True

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        parser = StubParser()
        assert isinstance(parser, BaseParser)

    def test_parser_type_returns_enum(self):
        parser = StubParser()
        assert parser.parser_type == ParserType.DOCLING_PDF

    def test_can_parse_returns_bool(self):
        parser = StubParser()
        raw_doc = RawDocument(
            url="https://example.com/test.pdf",
            source_type=SourceType.INDIA_CODE,
            content_format=ContentFormat.PDF,
            raw_content_path="data/raw/india_code/ic_100.html",
        )
        assert parser.can_parse(raw_doc) is True

    def test_can_parse_rejects_unsupported(self):
        parser = StubParser()
        raw_doc = RawDocument(
            url="https://example.com/test.html",
            source_type=SourceType.INDIAN_KANOON,
            content_format=ContentFormat.HTML,
            raw_content_path="data/raw/indian_kanoon/doc_1.html",
        )
        assert parser.can_parse(raw_doc) is False
