from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParsedTable,
    ParserType,
    ParsingConfig,
    ParsingSettings,
    QualityCheckResult,
    QualityReport,
    SectionLevel,
)


class TestSectionLevel:
    """SectionLevel enum tests."""

    def test_statute_levels_exist(self) -> None:
        assert SectionLevel.PREAMBLE == "preamble"
        assert SectionLevel.PART == "part"
        assert SectionLevel.CHAPTER == "chapter"
        assert SectionLevel.SECTION == "section"
        assert SectionLevel.SUB_SECTION == "sub_section"
        assert SectionLevel.CLAUSE == "clause"
        assert SectionLevel.PROVISO == "proviso"
        assert SectionLevel.EXPLANATION == "explanation"
        assert SectionLevel.DEFINITION == "definition"
        assert SectionLevel.SCHEDULE == "schedule"

    def test_judgment_levels_exist(self) -> None:
        assert SectionLevel.HEADER == "header"
        assert SectionLevel.FACTS == "facts"
        assert SectionLevel.ISSUES == "issues"
        assert SectionLevel.REASONING == "reasoning"
        assert SectionLevel.HOLDING == "holding"
        assert SectionLevel.ORDER == "order"
        assert SectionLevel.DISSENT == "dissent"
        assert SectionLevel.OBITER == "obiter"

    def test_is_str_enum(self) -> None:
        assert isinstance(SectionLevel.SECTION, str)
        assert SectionLevel.SECTION == "section"


class TestParserType:
    """ParserType enum tests."""

    def test_all_parser_types(self) -> None:
        assert ParserType.DOCLING_PDF == "docling_pdf"
        assert ParserType.PYMUPDF_PDF == "pymupdf_pdf"
        assert ParserType.HTML_INDIAN_KANOON == "html_indian_kanoon"
        assert ParserType.HTML_INDIA_CODE == "html_india_code"

    def test_is_str_enum(self) -> None:
        assert isinstance(ParserType.DOCLING_PDF, str)


class TestParsedSection:
    """ParsedSection model tests."""

    def test_minimal_section(self) -> None:
        section = ParsedSection(
            id="s1",
            level=SectionLevel.SECTION,
            text="Short title and commencement.",
        )
        assert section.id == "s1"
        assert section.level == SectionLevel.SECTION
        assert section.number is None
        assert section.title is None
        assert section.children == []
        assert section.parent_id is None
        assert section.token_count == 0

    def test_section_with_children(self, sample_parsed_section: ParsedSection) -> None:
        assert sample_parsed_section.id == "s10"
        assert len(sample_parsed_section.children) == 2
        assert sample_parsed_section.children[0].level == SectionLevel.EXPLANATION
        assert sample_parsed_section.children[1].level == SectionLevel.PROVISO
        assert sample_parsed_section.children[0].parent_id == "s10"

    def test_section_json_roundtrip(self, sample_parsed_section: ParsedSection) -> None:
        json_str = sample_parsed_section.model_dump_json()
        restored = ParsedSection.model_validate_json(json_str)
        assert restored.id == sample_parsed_section.id
        assert len(restored.children) == len(sample_parsed_section.children)
        assert restored.children[0].level == SectionLevel.EXPLANATION

    def test_deeply_nested_sections(self) -> None:
        """Part > Chapter > Section > Sub-section > Clause."""
        clause = ParsedSection(
            id="p1_ch1_s2_ss1_a",
            level=SectionLevel.CLAUSE,
            number="a",
            text="clause (a) text",
            parent_id="p1_ch1_s2_ss1",
        )
        subsection = ParsedSection(
            id="p1_ch1_s2_ss1",
            level=SectionLevel.SUB_SECTION,
            number="1",
            text="(1) sub-section text",
            parent_id="p1_ch1_s2",
            children=[clause],
        )
        section = ParsedSection(
            id="p1_ch1_s2",
            level=SectionLevel.SECTION,
            number="2",
            title="Definitions",
            text="Section 2 text",
            parent_id="p1_ch1",
            children=[subsection],
        )
        chapter = ParsedSection(
            id="p1_ch1",
            level=SectionLevel.CHAPTER,
            number="I",
            title="Preliminary",
            text="",
            parent_id="p1",
            children=[section],
        )
        part = ParsedSection(
            id="p1",
            level=SectionLevel.PART,
            number="I",
            title="General",
            text="",
            children=[chapter],
        )

        # Verify full tree
        assert part.children[0].children[0].children[0].children[0].level == SectionLevel.CLAUSE
        json_str = part.model_dump_json()
        restored = ParsedSection.model_validate_json(json_str)
        leaf = restored.children[0].children[0].children[0].children[0]
        assert leaf.id == "p1_ch1_s2_ss1_a"

    def test_page_numbers(self) -> None:
        section = ParsedSection(
            id="s1",
            level=SectionLevel.SECTION,
            text="text",
            page_numbers=[1, 2],
        )
        assert section.page_numbers == [1, 2]


class TestParsedTable:
    """ParsedTable model tests."""

    def test_minimal_table(self) -> None:
        table = ParsedTable(
            id="t1",
            headers=["Column A", "Column B"],
            rows=[["val1", "val2"], ["val3", "val4"]],
            row_count=2,
            col_count=2,
        )
        assert table.id == "t1"
        assert len(table.rows) == 2
        assert table.row_count == 2
        assert table.col_count == 2

    def test_table_json_roundtrip(self) -> None:
        table = ParsedTable(
            id="t1",
            caption="Schedule I",
            headers=["Name", "Age"],
            rows=[["Alice", "30"]],
            row_count=1,
            col_count=2,
            section_id="sched1",
            page_number=5,
        )
        json_str = table.model_dump_json()
        restored = ParsedTable.model_validate_json(json_str)
        assert restored.caption == "Schedule I"
        assert restored.section_id == "sched1"


class TestQualityReport:
    """QualityReport and QualityCheckResult tests."""

    def test_passing_report(self, sample_quality_report: QualityReport) -> None:
        assert sample_quality_report.passed is True
        assert sample_quality_report.overall_score == 0.95
        assert sample_quality_report.flagged_for_review is False
        assert len(sample_quality_report.checks) == 1

    def test_failing_report(self) -> None:
        report = QualityReport(
            overall_score=0.3,
            passed=False,
            checks=[
                QualityCheckResult(
                    check_name="text_completeness",
                    passed=False,
                    score=0.3,
                    details="Insufficient text extracted",
                ),
            ],
            flagged_for_review=True,
        )
        assert report.passed is False
        assert report.flagged_for_review is True

    def test_quality_check_result(self) -> None:
        check = QualityCheckResult(
            check_name="ocr_confidence",
            passed=True,
            score=0.92,
            details="Mean OCR confidence: 92%",
        )
        assert check.check_name == "ocr_confidence"
        assert check.passed is True

    def test_report_json_roundtrip(self, sample_quality_report: QualityReport) -> None:
        json_str = sample_quality_report.model_dump_json()
        restored = QualityReport.model_validate_json(json_str)
        assert restored.overall_score == sample_quality_report.overall_score
        assert len(restored.checks) == len(sample_quality_report.checks)


class TestParsedDocument:
    """ParsedDocument model tests."""

    def test_minimal_document(self, sample_quality_report: QualityReport) -> None:
        doc = ParsedDocument(
            document_id=uuid4(),
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.PDF,
            raw_text="The Indian Contract Act, 1872...",
            parser_used=ParserType.DOCLING_PDF,
            quality=sample_quality_report,
            raw_content_path="data/raw/india_code/ic_2160.html",
        )
        assert doc.source_type == SourceType.INDIA_CODE
        assert doc.document_type == DocumentType.STATUTE
        assert doc.parser_used == ParserType.DOCLING_PDF
        assert doc.quality.passed is True

    def test_document_with_sections(
        self,
        sample_parsed_section: ParsedSection,
        sample_quality_report: QualityReport,
    ) -> None:
        doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.PDF,
            raw_text="Full text here...",
            sections=[sample_parsed_section],
            parser_used=ParserType.DOCLING_PDF,
            quality=sample_quality_report,
            raw_content_path="data/raw/india_code/ic_2160.html",
        )
        assert len(doc.sections) == 1
        assert doc.sections[0].id == "s10"
        assert len(doc.sections[0].children) == 2

    def test_document_with_tables(self, sample_quality_report: QualityReport) -> None:
        table = ParsedTable(
            id="t1",
            headers=["Col"],
            rows=[["val"]],
            row_count=1,
            col_count=1,
        )
        doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.SCHEDULE,
            content_format=ContentFormat.PDF,
            raw_text="Schedule I...",
            tables=[table],
            parser_used=ParserType.DOCLING_PDF,
            quality=sample_quality_report,
            raw_content_path="data/raw/india_code/ic_2160.html",
        )
        assert len(doc.tables) == 1

    def test_document_with_metadata(self, sample_quality_report: QualityReport) -> None:
        doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.PDF,
            raw_text="text",
            title="The Indian Contract Act, 1872",
            act_name="Indian Contract Act",
            act_number="9",
            year=1872,
            date="25 April 1872",
            page_count=42,
            parser_used=ParserType.DOCLING_PDF,
            quality=sample_quality_report,
            raw_content_path="path",
        )
        assert doc.title == "The Indian Contract Act, 1872"
        assert doc.year == 1872
        assert doc.page_count == 42

    def test_document_judgment_metadata(self, sample_quality_report: QualityReport) -> None:
        doc = ParsedDocument(
            source_type=SourceType.INDIAN_KANOON,
            document_type=DocumentType.JUDGMENT,
            content_format=ContentFormat.HTML,
            raw_text="judgment text",
            court="Supreme Court of India",
            case_citation="AIR 2024 SC 1500",
            parties="State of Maharashtra vs Rajesh Kumar",
            parser_used=ParserType.HTML_INDIAN_KANOON,
            quality=sample_quality_report,
            raw_content_path="path",
        )
        assert doc.court == "Supreme Court of India"
        assert doc.case_citation == "AIR 2024 SC 1500"

    def test_document_json_roundtrip(self, sample_quality_report: QualityReport) -> None:
        doc_id = uuid4()
        doc = ParsedDocument(
            document_id=doc_id,
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.PDF,
            raw_text="Full text",
            title="Test Act",
            parser_used=ParserType.DOCLING_PDF,
            ocr_applied=True,
            ocr_confidence=0.92,
            parsing_duration_seconds=3.5,
            quality=sample_quality_report,
            raw_content_path="data/raw/india_code/ic_test.html",
        )
        json_str = doc.model_dump_json(indent=2)
        restored = ParsedDocument.model_validate_json(json_str)
        assert restored.document_id == doc_id
        assert restored.ocr_applied is True
        assert restored.ocr_confidence == 0.92
        assert restored.parsing_duration_seconds == 3.5

    def test_document_defaults(self, sample_quality_report: QualityReport) -> None:
        doc = ParsedDocument(
            source_type=SourceType.INDIA_CODE,
            document_type=DocumentType.STATUTE,
            content_format=ContentFormat.HTML,
            raw_text="text",
            parser_used=ParserType.HTML_INDIA_CODE,
            quality=sample_quality_report,
            raw_content_path="path",
        )
        assert doc.ocr_applied is False
        assert doc.ocr_confidence is None
        assert doc.sections == []
        assert doc.tables == []
        assert doc.title is None
        assert doc.parsed_at is not None


class TestParsingSettings:
    """ParsingSettings model tests."""

    def test_defaults(self) -> None:
        settings = ParsingSettings()
        assert settings.input_dir == Path("data/raw")
        assert settings.output_dir == Path("data/parsed")
        assert settings.pdf_cache_dir == Path("data/cache/pdf")
        assert settings.prefer_docling is True
        assert settings.ocr_languages == ["eng", "hin"]
        assert settings.ocr_confidence_threshold == 0.85
        assert settings.min_text_completeness == 0.5
        assert settings.download_timeout_seconds == 120
        assert settings.max_pdf_size_mb == 100

    def test_custom_settings(self, tmp_path) -> None:
        settings = ParsingSettings(
            input_dir=tmp_path / "in",
            output_dir=tmp_path / "out",
            prefer_docling=False,
            ocr_confidence_threshold=0.90,
        )
        assert settings.prefer_docling is False
        assert settings.ocr_confidence_threshold == 0.90


class TestParsingConfig:
    """ParsingConfig root model tests."""

    def test_default_config(self) -> None:
        config = ParsingConfig()
        assert config.settings.prefer_docling is True

    def test_config_with_custom_settings(self) -> None:
        config = ParsingConfig(settings=ParsingSettings(prefer_docling=False))
        assert config.settings.prefer_docling is False

    def test_config_json_roundtrip(self) -> None:
        config = ParsingConfig(
            settings=ParsingSettings(
                min_text_completeness=0.7,
                max_pdf_size_mb=50,
            )
        )
        json_str = config.model_dump_json()
        restored = ParsingConfig.model_validate_json(json_str)
        assert restored.settings.min_text_completeness == 0.7
        assert restored.settings.max_pdf_size_mb == 50
