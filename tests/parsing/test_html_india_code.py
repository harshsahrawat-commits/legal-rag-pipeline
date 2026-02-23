from __future__ import annotations

from typing import TYPE_CHECKING

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    PreliminaryMetadata,
    RawDocument,
    SourceType,
)
from src.parsing._models import ParserType, SectionLevel
from src.parsing.parsers._html_india_code import IndiaCodeHtmlParser

from .conftest import SAMPLE_INDIA_CODE_HTML

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_doc(
    doc_type: DocumentType | None = DocumentType.STATUTE,
    source: SourceType = SourceType.INDIA_CODE,
    fmt: ContentFormat = ContentFormat.HTML,
    **meta_kwargs,
) -> RawDocument:
    return RawDocument(
        url="https://www.indiacode.nic.in/handle/123456789/2110",
        source_type=source,
        content_format=fmt,
        raw_content_path="data/raw/india_code/ic_2110.html",
        document_type=doc_type,
        preliminary_metadata=PreliminaryMetadata(**meta_kwargs),
    )


def _write_html(tmp_path: Path, html: str, name: str = "ic_detail.html") -> Path:
    p = tmp_path / name
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TestCanParse
# ---------------------------------------------------------------------------


class TestCanParse:
    def test_accepts_india_code_html(self):
        parser = IndiaCodeHtmlParser()
        raw = _make_raw_doc()
        assert parser.can_parse(raw) is True

    def test_rejects_indian_kanoon_html(self):
        parser = IndiaCodeHtmlParser()
        raw = _make_raw_doc(source=SourceType.INDIAN_KANOON)
        assert parser.can_parse(raw) is False

    def test_rejects_india_code_pdf(self):
        parser = IndiaCodeHtmlParser()
        raw = _make_raw_doc(fmt=ContentFormat.PDF)
        assert parser.can_parse(raw) is False


# ---------------------------------------------------------------------------
# TestParserType
# ---------------------------------------------------------------------------


class TestParserType:
    def test_parser_type(self):
        parser = IndiaCodeHtmlParser()
        assert parser.parser_type == ParserType.HTML_INDIA_CODE


# ---------------------------------------------------------------------------
# TestMetadataExtraction
# ---------------------------------------------------------------------------


class TestMetadataExtraction:
    def test_extracts_title_from_heading(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.title == "The Academy of Scientific and Innovative Research Act, 2011"

    def test_extracts_act_number_from_table(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.act_number == "13"

    def test_extracts_year_from_table(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.year == 2011

    def test_extracts_date_from_table(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.date == "6-Feb-2012"

    def test_extracts_act_name(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.act_name == "The Academy of Scientific and Innovative Research Act, 2011"

    def test_falls_back_to_preliminary_metadata(self, tmp_path: Path):
        """When HTML lacks metadata, Phase 1 preliminary metadata is used."""
        minimal_html = """\
<html>
<head><title>India Code: Some Act</title></head>
<body><div id="content"><p>No metadata table here.</p></div></body>
</html>
"""
        path = _write_html(tmp_path, minimal_html)
        raw = _make_raw_doc(
            title="The Some Act, 2020",
            act_name="The Some Act, 2020",
            act_number="42",
            year=2020,
            date="1-Jan-2020",
        )
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.act_name == "The Some Act, 2020"
        assert result.act_number == "42"
        assert result.year == 2020
        assert result.date == "1-Jan-2020"

    def test_extracts_year_from_title_when_no_table(self, tmp_path: Path):
        """Year is extracted from the title text when not in metadata table."""
        html = """\
<html>
<head><title>India Code: The Arms Act, 1959</title></head>
<body>
<div class="item-page-field-wrapper"><h2>The Arms Act, 1959</h2></div>
</body>
</html>
"""
        path = _write_html(tmp_path, html)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.year == 1959


# ---------------------------------------------------------------------------
# TestTitleExtraction
# ---------------------------------------------------------------------------


class TestTitleExtraction:
    def test_title_from_title_tag_fallback(self, tmp_path: Path):
        """Falls back to <title> tag (stripping 'India Code:' prefix)."""
        html = """\
<html>
<head><title>India Code: The Wildlife Protection Act, 1972</title></head>
<body><p>Empty body.</p></body>
</html>
"""
        path = _write_html(tmp_path, html)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.title == "The Wildlife Protection Act, 1972"


# ---------------------------------------------------------------------------
# TestSectionOutput
# ---------------------------------------------------------------------------


class TestSectionOutput:
    def test_creates_preamble_section(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert len(result.sections) == 1
        preamble = result.sections[0]
        assert preamble.level == SectionLevel.PREAMBLE
        assert preamble.id == "ic_preamble"
        assert preamble.token_count > 0

    def test_no_sections_for_empty_body(self, tmp_path: Path):
        html = "<html><head><title>Empty</title></head><body></body></html>"
        path = _write_html(tmp_path, html)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.sections == []


# ---------------------------------------------------------------------------
# TestOutputContract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_parsed_document_fields(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.source_type == SourceType.INDIA_CODE
        assert result.document_type == DocumentType.STATUTE
        assert result.content_format == ContentFormat.HTML
        assert result.parser_used == ParserType.HTML_INDIA_CODE
        assert result.parsing_duration_seconds >= 0
        assert result.raw_content_path == str(path)
        assert result.quality.passed is False  # placeholder

    def test_defaults_to_statute_when_no_doc_type(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc(doc_type=None)
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert result.document_type == DocumentType.STATUTE

    def test_raw_text_contains_metadata(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        assert "Academy of Scientific" in result.raw_text
        assert len(result.raw_text) > 0

    def test_round_trips_through_json(self, tmp_path: Path):
        path = _write_html(tmp_path, SAMPLE_INDIA_CODE_HTML)
        raw = _make_raw_doc()
        parser = IndiaCodeHtmlParser()

        result = parser.parse(path, raw)

        json_str = result.model_dump_json()
        assert '"html_india_code"' in json_str
        assert '"india_code"' in json_str
