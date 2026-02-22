from __future__ import annotations

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParsedTable,
    ParserType,
    ParsingSettings,
    QualityReport,
    SectionLevel,
)
from src.parsing._validation import QualityValidator

# --- Helpers ---


def _make_doc(
    raw_text: str = "x" * 5000,
    content_format: ContentFormat = ContentFormat.PDF,
    page_count: int | None = 5,
    sections: list[ParsedSection] | None = None,
    tables: list[ParsedTable] | None = None,
    ocr_applied: bool = False,
    ocr_confidence: float | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        source_type=SourceType.INDIA_CODE,
        document_type=DocumentType.STATUTE,
        content_format=content_format,
        raw_text=raw_text,
        sections=sections or [],
        tables=tables or [],
        page_count=page_count,
        ocr_applied=ocr_applied,
        ocr_confidence=ocr_confidence,
        parser_used=ParserType.DOCLING_PDF,
        quality=QualityReport(overall_score=1.0, passed=True),
        raw_content_path="data/raw/india_code/ic_100.html",
    )


def _make_section(number: str, level: SectionLevel = SectionLevel.SECTION) -> ParsedSection:
    return ParsedSection(
        id=f"s{number}",
        level=level,
        number=number,
        text=f"Section {number} text.",
    )


def _make_table(
    table_id: str,
    rows: list[list[str]],
    row_count: int,
    col_count: int,
) -> ParsedTable:
    return ParsedTable(
        id=table_id,
        rows=rows,
        row_count=row_count,
        col_count=col_count,
    )


# --- Text Completeness Tests ---


class TestTextCompleteness:
    def test_html_always_passes(self, parsing_settings: ParsingSettings):
        doc = _make_doc(raw_text="short", content_format=ContentFormat.HTML, page_count=10)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "text_completeness")
        assert check.passed is True
        assert check.score == 1.0

    def test_pdf_adequate_text_passes(self, parsing_settings: ParsingSettings):
        # 5 pages * 2000 = 10000 expected. 6000/10000 = 0.6 > 0.5 threshold
        doc = _make_doc(raw_text="x" * 6000, page_count=5)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "text_completeness")
        assert check.passed is True

    def test_pdf_insufficient_text_fails(self, parsing_settings: ParsingSettings):
        # 10 pages * 2000 = 20000 expected. 2000/20000 = 0.1 < 0.5
        doc = _make_doc(raw_text="x" * 2000, page_count=10)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "text_completeness")
        assert check.passed is False

    def test_pdf_none_page_count_passes(self, parsing_settings: ParsingSettings):
        doc = _make_doc(raw_text="short", page_count=None)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "text_completeness")
        assert check.passed is True

    def test_pdf_zero_pages_passes(self, parsing_settings: ParsingSettings):
        doc = _make_doc(raw_text="short", page_count=0)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "text_completeness")
        assert check.passed is True

    def test_score_capped_at_one(self, parsing_settings: ParsingSettings):
        # 1 page * 2000 = 2000 expected. 10000/2000 = 5.0 → capped to 1.0
        doc = _make_doc(raw_text="x" * 10000, page_count=1)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "text_completeness")
        assert check.score == 1.0


# --- Section Sequence Tests ---


class TestSectionSequence:
    def test_consecutive_sections_pass(self, parsing_settings: ParsingSettings):
        sections = [_make_section(str(i)) for i in range(1, 6)]
        doc = _make_doc(sections=sections)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "section_sequence")
        assert check.passed is True
        assert check.score == 1.0

    def test_gap_in_sections_fails(self, parsing_settings: ParsingSettings):
        sections = [_make_section("1"), _make_section("2"), _make_section("5")]
        doc = _make_doc(sections=sections)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "section_sequence")
        assert check.passed is False
        assert "2->5" in check.details

    def test_fewer_than_two_sections_skipped(self, parsing_settings: ParsingSettings):
        sections = [_make_section("1")]
        doc = _make_doc(sections=sections)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "section_sequence")
        assert check.passed is True
        assert "skipped" in check.details.lower()

    def test_no_sections_skipped(self, parsing_settings: ParsingSettings):
        doc = _make_doc(sections=[])
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "section_sequence")
        assert check.passed is True

    def test_non_section_levels_ignored(self, parsing_settings: ParsingSettings):
        """Chapters and provisos don't participate in section numbering checks."""
        sections = [
            _make_section("I", level=SectionLevel.CHAPTER),
            _make_section("1"),
            _make_section("2"),
            _make_section("3"),
        ]
        doc = _make_doc(sections=sections)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "section_sequence")
        assert check.passed is True

    def test_alpha_suffix_sections_parsed(self, parsing_settings: ParsingSettings):
        """Section numbers like 302A are parsed for their numeric part."""
        sections = [_make_section("302"), _make_section("302A"), _make_section("303")]
        doc = _make_doc(sections=sections)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "section_sequence")
        # 302, 302, 303 — sorted: [302, 302, 303] — no gaps
        assert check.passed is True


# --- Table Integrity Tests ---


class TestTableIntegrity:
    def test_matching_cells_passes(self, parsing_settings: ParsingSettings):
        table = _make_table(
            "t1",
            rows=[["a", "b"], ["c", "d"]],
            row_count=2,
            col_count=2,
        )
        doc = _make_doc(tables=[table])
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "table_integrity_t1")
        assert check.passed is True

    def test_mismatched_cells_fails(self, parsing_settings: ParsingSettings):
        table = _make_table(
            "t1",
            rows=[["a", "b"], ["c"]],  # only 3 cells
            row_count=2,
            col_count=2,   # expects 4
        )
        doc = _make_doc(tables=[table])
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "table_integrity_t1")
        assert check.passed is False
        assert "Expected 4" in check.details

    def test_empty_table_passes(self, parsing_settings: ParsingSettings):
        table = _make_table("t1", rows=[], row_count=0, col_count=0)
        doc = _make_doc(tables=[table])
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "table_integrity_t1")
        assert check.passed is True

    def test_multiple_tables_checked_independently(self, parsing_settings: ParsingSettings):
        good_table = _make_table("t1", rows=[["a"]], row_count=1, col_count=1)
        bad_table = _make_table("t2", rows=[["a"]], row_count=2, col_count=1)
        doc = _make_doc(tables=[good_table, bad_table])
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        t1 = next(c for c in report.checks if c.check_name == "table_integrity_t1")
        t2 = next(c for c in report.checks if c.check_name == "table_integrity_t2")
        assert t1.passed is True
        assert t2.passed is False


# --- OCR Confidence Tests ---


class TestOCRConfidence:
    def test_high_confidence_passes(self, parsing_settings: ParsingSettings):
        doc = _make_doc(ocr_applied=True, ocr_confidence=0.92)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "ocr_confidence")
        assert check.passed is True

    def test_low_confidence_fails(self, parsing_settings: ParsingSettings):
        doc = _make_doc(ocr_applied=True, ocr_confidence=0.70)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        check = next(c for c in report.checks if c.check_name == "ocr_confidence")
        assert check.passed is False

    def test_not_checked_when_ocr_not_applied(self, parsing_settings: ParsingSettings):
        doc = _make_doc(ocr_applied=False, ocr_confidence=None)
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        ocr_checks = [c for c in report.checks if c.check_name == "ocr_confidence"]
        assert len(ocr_checks) == 0


# --- Aggregate Report Tests ---


class TestValidateAggregate:
    def test_all_pass_report_passes(self, parsing_settings: ParsingSettings):
        doc = _make_doc(
            raw_text="x" * 6000,
            page_count=5,
            sections=[_make_section("1"), _make_section("2"), _make_section("3")],
        )
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        assert report.passed is True
        assert report.flagged_for_review is False
        assert report.overall_score > 0.0

    def test_one_failure_flags_report(self, parsing_settings: ParsingSettings):
        doc = _make_doc(
            raw_text="x" * 100,   # too little text for 10 pages
            page_count=10,
            sections=[_make_section("1"), _make_section("2")],
        )
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        assert report.passed is False
        assert report.flagged_for_review is True

    def test_overall_score_is_average(self, parsing_settings: ParsingSettings):
        doc = _make_doc(
            raw_text="x" * 6000,
            page_count=5,
            sections=[_make_section("1"), _make_section("2")],
        )
        validator = QualityValidator(parsing_settings)
        report = validator.validate(doc)
        scores = [c.score for c in report.checks]
        expected = round(sum(scores) / len(scores), 4)
        assert report.overall_score == expected
