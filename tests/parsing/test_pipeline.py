"""Integration tests for ParsingPipeline, CLI, and end-to-end flows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    PreliminaryMetadata,
    RawDocument,
    SourceType,
)
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    ParsingConfig,
    ParsingResult,
    ParsingSettings,
    QualityReport,
    SectionLevel,
)
from src.parsing.pipeline import ParsingPipeline, _merge_metadata, _resolve_source_filter

if TYPE_CHECKING:
    from pathlib import Path

# Re-use HTML samples from conftest
from tests.parsing.conftest import (
    SAMPLE_INDIA_CODE_HTML,
    SAMPLE_JUDGMENT_HTML,
    SAMPLE_STATUTE_HTML,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_ik_document(
    input_dir: Path,
    doc_id: str,
    html: str,
    *,
    doc_type: DocumentType | None = DocumentType.JUDGMENT,
) -> RawDocument:
    """Write an IK HTML file + meta.json, return the RawDocument."""
    source_dir = input_dir / "indian_kanoon"
    source_dir.mkdir(parents=True, exist_ok=True)

    html_path = source_dir / f"{doc_id}.html"
    html_path.write_text(html, encoding="utf-8")

    raw_doc = RawDocument(
        url=f"https://indiankanoon.org/doc/{doc_id}/",
        source_type=SourceType.INDIAN_KANOON,
        content_format=ContentFormat.HTML,
        raw_content_path=str(html_path),
        document_type=doc_type,
        preliminary_metadata=PreliminaryMetadata(title=f"IK Doc {doc_id}"),
    )
    meta_path = source_dir / f"{doc_id}.meta.json"
    meta_path.write_text(raw_doc.model_dump_json(indent=2), encoding="utf-8")
    return raw_doc


def _create_ic_document(
    input_dir: Path,
    doc_id: str,
    html: str,
    *,
    download_url: str = "https://www.indiacode.nic.in/bitstream/123456789/2110/1/a2012-13.pdf",
) -> RawDocument:
    """Write an IC HTML file + meta.json with download_url."""
    source_dir = input_dir / "india_code"
    source_dir.mkdir(parents=True, exist_ok=True)

    html_path = source_dir / f"{doc_id}.html"
    html_path.write_text(html, encoding="utf-8")

    raw_doc = RawDocument(
        url=f"https://www.indiacode.nic.in/handle/123456789/{doc_id}",
        source_type=SourceType.INDIA_CODE,
        content_format=ContentFormat.HTML,
        raw_content_path=str(html_path),
        document_type=DocumentType.STATUTE,
        preliminary_metadata=PreliminaryMetadata(download_url=download_url),
    )
    meta_path = source_dir / f"{doc_id}.meta.json"
    meta_path.write_text(raw_doc.model_dump_json(indent=2), encoding="utf-8")
    return raw_doc


def _make_pipeline(tmp_path: Path) -> ParsingPipeline:
    """Create a ParsingPipeline with temp directories."""
    config = ParsingConfig(
        settings=ParsingSettings(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "parsed",
            pdf_cache_dir=tmp_path / "cache" / "pdf",
        )
    )
    return ParsingPipeline(config=config)


def _fake_pdf_parsed_doc(
    raw_doc: RawDocument,
    pdf_path: Path,
) -> ParsedDocument:
    """Build a mock ParsedDocument as if Docling parsed a PDF."""
    return ParsedDocument(
        document_id=raw_doc.document_id,
        source_type=SourceType.INDIA_CODE,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.PDF,
        raw_text="Section 1. Short title.\nThis Act may be called the Test Act.",
        sections=[
            ParsedSection(
                id="sec_1",
                level=SectionLevel.SECTION,
                number="1",
                title="Short title",
                text="This Act may be called the Test Act.",
            ),
        ],
        parser_used=ParserType.DOCLING_PDF,
        quality=QualityReport(overall_score=0.0, passed=False),
        raw_content_path=str(pdf_path),
    )


# ---------------------------------------------------------------------------
# TestSourceResolution
# ---------------------------------------------------------------------------


class TestSourceResolution:
    def test_resolve_indian_kanoon(self) -> None:
        assert _resolve_source_filter("Indian Kanoon") == SourceType.INDIAN_KANOON

    def test_resolve_india_code(self) -> None:
        assert _resolve_source_filter("India Code") == SourceType.INDIA_CODE

    def test_resolve_case_insensitive(self) -> None:
        assert _resolve_source_filter("india code") == SourceType.INDIA_CODE
        assert _resolve_source_filter("INDIAN KANOON") == SourceType.INDIAN_KANOON

    def test_resolve_unknown_returns_none(self) -> None:
        assert _resolve_source_filter("Nonexistent") is None

    def test_resolve_none_returns_none(self) -> None:
        assert _resolve_source_filter(None) is None


# ---------------------------------------------------------------------------
# TestDiscovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discover_all_sources(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)
        _create_ic_document(input_dir, "2110", SAMPLE_INDIA_CODE_HTML)

        pipeline = _make_pipeline(tmp_path)
        meta_files = pipeline._discover_meta_files(source_filter=None)
        assert len(meta_files) == 2

    def test_discover_filtered_source(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)
        _create_ic_document(input_dir, "2110", SAMPLE_INDIA_CODE_HTML)

        pipeline = _make_pipeline(tmp_path)
        meta_files = pipeline._discover_meta_files(
            source_filter=SourceType.INDIAN_KANOON,
        )
        assert len(meta_files) == 1

    def test_discover_missing_dir(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        # input_dir doesn't exist (no raw/ created)
        meta_files = pipeline._discover_meta_files(source_filter=None)
        assert meta_files == []

    def test_discover_empty_source_dir(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw" / "indian_kanoon"
        input_dir.mkdir(parents=True)

        pipeline = _make_pipeline(tmp_path)
        meta_files = pipeline._discover_meta_files(
            source_filter=SourceType.INDIAN_KANOON,
        )
        assert meta_files == []


# ---------------------------------------------------------------------------
# TestIndianKanoonFlow
# ---------------------------------------------------------------------------


class TestIndianKanoonFlow:
    async def test_parse_judgment(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.documents_found == 1
        assert result.documents_parsed == 1
        assert result.documents_failed == 0

        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        assert len(parsed_files) == 1

        doc = ParsedDocument.model_validate_json(
            parsed_files[0].read_text(encoding="utf-8"),
        )
        assert doc.parser_used == ParserType.HTML_INDIAN_KANOON
        assert len(doc.sections) > 0

    async def test_parse_statute(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(
            input_dir,
            "201",
            SAMPLE_STATUTE_HTML,
            doc_type=DocumentType.STATUTE,
        )

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.documents_parsed == 1
        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        doc = ParsedDocument.model_validate_json(
            parsed_files[0].read_text(encoding="utf-8"),
        )
        assert doc.parser_used == ParserType.HTML_INDIAN_KANOON
        assert doc.document_type == DocumentType.STATUTE

    async def test_quality_validated(self, tmp_path: Path) -> None:
        """Output has real QualityReport, not the parser's placeholder."""
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)

        pipeline = _make_pipeline(tmp_path)
        await pipeline.run()

        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        doc = ParsedDocument.model_validate_json(
            parsed_files[0].read_text(encoding="utf-8"),
        )
        # Real validator runs multiple checks
        assert len(doc.quality.checks) > 0
        assert doc.quality.overall_score > 0.0


# ---------------------------------------------------------------------------
# TestIndiaCodeFlow
# ---------------------------------------------------------------------------


class TestIndiaCodeFlow:
    async def test_pdf_download_and_parse(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        raw_doc = _create_ic_document(input_dir, "2110", SAMPLE_INDIA_CODE_HTML)

        fake_pdf_path = tmp_path / "cache" / "pdf" / f"{raw_doc.document_id}.pdf"
        fake_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fake_pdf_path.write_bytes(b"%PDF-fake-content")

        pipeline = _make_pipeline(tmp_path)

        with (
            patch.object(
                pipeline._downloader,
                "download",
                new_callable=AsyncMock,
                return_value=fake_pdf_path,
            ),
            patch(
                "src.parsing.pipeline.DoclingPdfParser.parse",
                return_value=_fake_pdf_parsed_doc(raw_doc, fake_pdf_path),
            ),
        ):
            result = await pipeline.run()

        assert result.documents_parsed == 1
        assert result.documents_failed == 0

        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        assert len(parsed_files) == 1

    async def test_metadata_merged(self, tmp_path: Path) -> None:
        """HTML metadata (act_name, year) is merged onto PDF-parsed doc."""
        input_dir = tmp_path / "raw"
        raw_doc = _create_ic_document(input_dir, "2110", SAMPLE_INDIA_CODE_HTML)

        fake_pdf_path = tmp_path / "cache" / "pdf" / f"{raw_doc.document_id}.pdf"
        fake_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fake_pdf_path.write_bytes(b"%PDF-fake-content")

        pipeline = _make_pipeline(tmp_path)

        with (
            patch.object(
                pipeline._downloader,
                "download",
                new_callable=AsyncMock,
                return_value=fake_pdf_path,
            ),
            patch(
                "src.parsing.pipeline.DoclingPdfParser.parse",
                return_value=_fake_pdf_parsed_doc(raw_doc, fake_pdf_path),
            ),
        ):
            await pipeline.run()

        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        doc = ParsedDocument.model_validate_json(
            parsed_files[0].read_text(encoding="utf-8"),
        )
        # From IC HTML parser metadata
        assert doc.act_number == "13"
        assert doc.year == 2011
        # From Docling (content preserved)
        assert len(doc.sections) > 0

    async def test_no_download_url_uses_html_parser(self, tmp_path: Path) -> None:
        """IC doc without download_url falls through to HTML-only parse."""
        input_dir = tmp_path / "raw"
        source_dir = input_dir / "india_code"
        source_dir.mkdir(parents=True, exist_ok=True)

        html_path = source_dir / "9999.html"
        html_path.write_text(SAMPLE_INDIA_CODE_HTML, encoding="utf-8")

        raw_doc = RawDocument(
            url="https://www.indiacode.nic.in/handle/123456789/9999",
            source_type=SourceType.INDIA_CODE,
            content_format=ContentFormat.HTML,
            raw_content_path=str(html_path),
            document_type=DocumentType.STATUTE,
            preliminary_metadata=PreliminaryMetadata(download_url=None),
        )
        meta_path = source_dir / "9999.meta.json"
        meta_path.write_text(raw_doc.model_dump_json(indent=2), encoding="utf-8")

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.documents_parsed == 1
        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        doc = ParsedDocument.model_validate_json(
            parsed_files[0].read_text(encoding="utf-8"),
        )
        assert doc.parser_used == ParserType.HTML_INDIA_CODE


# ---------------------------------------------------------------------------
# TestIdempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_skips_existing_output(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)

        pipeline = _make_pipeline(tmp_path)

        result1 = await pipeline.run()
        assert result1.documents_parsed == 1

        result2 = await pipeline.run()
        assert result2.documents_parsed == 0
        assert result2.documents_skipped == 1

    async def test_skip_count_matches_found(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)
        _create_ik_document(input_dir, "102", SAMPLE_STATUTE_HTML, doc_type=DocumentType.STATUTE)

        pipeline = _make_pipeline(tmp_path)

        await pipeline.run()
        result2 = await pipeline.run()

        assert result2.documents_found == 2
        assert result2.documents_skipped == 2
        assert result2.documents_parsed == 0


# ---------------------------------------------------------------------------
# TestDryRun
# ---------------------------------------------------------------------------


class TestDryRun:
    async def test_no_output_files(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(dry_run=True)

        assert result.documents_found == 1
        assert result.documents_parsed == 0

        parsed_dir = tmp_path / "parsed"
        if parsed_dir.exists():
            assert len(list(parsed_dir.rglob("*.json"))) == 0

    async def test_returns_found_count(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)
        _create_ik_document(input_dir, "102", SAMPLE_STATUTE_HTML, doc_type=DocumentType.STATUTE)

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(dry_run=True)

        assert result.documents_found == 2
        assert result.documents_parsed == 0
        assert result.documents_skipped == 0


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_malformed_meta_json(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "raw" / "indian_kanoon"
        source_dir.mkdir(parents=True)
        bad_meta = source_dir / "bad.meta.json"
        bad_meta.write_text("{invalid json", encoding="utf-8")

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.documents_found == 1
        assert result.documents_failed == 1
        assert len(result.errors) == 1

    async def test_parser_error_isolated(self, tmp_path: Path) -> None:
        """One doc failing doesn't stop others from being parsed."""
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "good", SAMPLE_JUDGMENT_HTML)
        # Create a doc with empty HTML — parser may raise or produce minimal output
        _create_ik_document(input_dir, "empty", "<html><body></body></html>")

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.documents_found == 2
        # Pipeline did not crash — both were attempted
        assert result.documents_parsed + result.documents_failed == 2

    async def test_unknown_source_name(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run(source_name="Nonexistent")

        assert len(result.errors) == 1
        assert "Unknown source" in result.errors[0]

    async def test_empty_input_dir(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir(parents=True)

        pipeline = _make_pipeline(tmp_path)
        result = await pipeline.run()

        assert result.documents_found == 0
        assert result.documents_parsed == 0
        assert not result.errors


# ---------------------------------------------------------------------------
# TestOutputValidation
# ---------------------------------------------------------------------------


class TestOutputValidation:
    async def test_output_valid_parsed_document(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)

        pipeline = _make_pipeline(tmp_path)
        await pipeline.run()

        parsed_files = list((tmp_path / "parsed").rglob("*.json"))
        assert len(parsed_files) == 1

        doc = ParsedDocument.model_validate_json(
            parsed_files[0].read_text(encoding="utf-8"),
        )
        assert doc.document_id is not None
        assert doc.source_type == SourceType.INDIAN_KANOON
        assert doc.parser_used == ParserType.HTML_INDIAN_KANOON
        assert doc.raw_text
        assert doc.parsed_at is not None

    async def test_output_directory_structure(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "raw"
        raw_doc = _create_ik_document(input_dir, "101", SAMPLE_JUDGMENT_HTML)

        pipeline = _make_pipeline(tmp_path)
        await pipeline.run()

        expected_path = tmp_path / "parsed" / "indian_kanoon" / f"{raw_doc.document_id}.json"
        assert expected_path.exists()


# ---------------------------------------------------------------------------
# TestMergeMetadata
# ---------------------------------------------------------------------------


class TestMergeMetadata:
    def test_html_metadata_takes_precedence(self) -> None:
        pdf_doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.PDF,
            raw_text="Full content from PDF",
            title="PDF Title",
            year=None,
            parser_used=ParserType.DOCLING_PDF,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path="/fake/path.pdf",
        )
        html_doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.HTML,
            raw_text="Minimal HTML content",
            title="HTML Title",
            act_name="Test Act",
            year=2011,
            parser_used=ParserType.HTML_INDIA_CODE,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path="/fake/page.html",
        )

        merged = _merge_metadata(pdf_doc, html_doc)

        # HTML values take precedence
        assert merged.title == "HTML Title"
        assert merged.act_name == "Test Act"
        assert merged.year == 2011
        # PDF content preserved
        assert merged.raw_text == "Full content from PDF"

    def test_pdf_metadata_kept_when_html_is_none(self) -> None:
        pdf_doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.PDF,
            raw_text="Content",
            title="PDF Title",
            act_number="42",
            parser_used=ParserType.DOCLING_PDF,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path="/fake/path.pdf",
        )
        html_doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.HTML,
            raw_text="Minimal",
            title=None,
            act_number=None,
            parser_used=ParserType.HTML_INDIA_CODE,
            quality=QualityReport(overall_score=0.0, passed=False),
            raw_content_path="/fake/page.html",
        )

        merged = _merge_metadata(pdf_doc, html_doc)

        # PDF values preserved when HTML has None
        assert merged.title == "PDF Title"
        assert merged.act_number == "42"


# ---------------------------------------------------------------------------
# TestParsingResult
# ---------------------------------------------------------------------------


class TestParsingResult:
    def test_default_values(self) -> None:
        result = ParsingResult()
        assert result.documents_found == 0
        assert result.documents_parsed == 0
        assert result.documents_skipped == 0
        assert result.documents_failed == 0
        assert result.errors == []
        assert result.source_type is None
        assert result.started_at is not None
        assert result.finished_at is None

    def test_serialization_roundtrip(self) -> None:
        result = ParsingResult(
            source_type=SourceType.INDIAN_KANOON,
            documents_found=10,
            documents_parsed=8,
            documents_skipped=1,
            documents_failed=1,
            errors=["Some error"],
        )
        json_str = result.model_dump_json()
        restored = ParsingResult.model_validate_json(json_str)
        assert restored.documents_parsed == 8
        assert restored.errors == ["Some error"]
