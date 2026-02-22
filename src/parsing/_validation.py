from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.parsing._models import QualityCheckResult, QualityReport
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument, ParsingSettings

_log = get_logger(__name__)

# Section number pattern for sequence checking
_SECTION_NUM_RE = re.compile(r"^(\d+)([A-Z]?)$")


class QualityValidator:
    """Runs post-parse quality checks on a ParsedDocument."""

    def __init__(self, settings: ParsingSettings) -> None:
        self._settings = settings

    def validate(self, doc: ParsedDocument) -> QualityReport:
        """Run all applicable quality checks and return a report."""
        checks: list[QualityCheckResult] = []

        checks.append(self._check_text_completeness(doc))
        checks.append(self._check_section_sequence(doc))

        for table in doc.tables:
            checks.append(self._check_table_integrity(table))

        if doc.ocr_applied and doc.ocr_confidence is not None:
            checks.append(self._check_ocr_confidence(doc))

        # Aggregate
        scores = [c.score for c in checks]
        overall = sum(scores) / len(scores) if scores else 0.0
        all_passed = all(c.passed for c in checks)

        return QualityReport(
            overall_score=round(overall, 4),
            passed=all_passed,
            checks=checks,
            flagged_for_review=not all_passed,
        )

    def _check_text_completeness(self, doc: ParsedDocument) -> QualityCheckResult:
        """Check if enough text was extracted relative to page count."""
        from src.acquisition._models import ContentFormat

        # HTML documents always pass (no page count concept)
        if doc.content_format == ContentFormat.HTML or doc.page_count is None:
            return QualityCheckResult(
                check_name="text_completeness",
                passed=True,
                score=1.0,
                details="HTML document — text completeness check skipped",
            )

        expected_chars = doc.page_count * 2000
        if expected_chars == 0:
            return QualityCheckResult(
                check_name="text_completeness",
                passed=True,
                score=1.0,
                details="Zero pages reported",
            )

        ratio = len(doc.raw_text) / expected_chars
        passed = ratio >= self._settings.min_text_completeness

        return QualityCheckResult(
            check_name="text_completeness",
            passed=passed,
            score=min(ratio, 1.0),
            details=f"Ratio: {ratio:.2f} (threshold: {self._settings.min_text_completeness})",
        )

    def _check_section_sequence(self, doc: ParsedDocument) -> QualityCheckResult:
        """Check for gaps in section numbering."""
        from src.parsing._models import SectionLevel

        section_numbers: list[int] = []
        for section in doc.sections:
            if section.level == SectionLevel.SECTION and section.number:
                match = _SECTION_NUM_RE.match(section.number)
                if match:
                    section_numbers.append(int(match.group(1)))

        if len(section_numbers) < 2:
            return QualityCheckResult(
                check_name="section_sequence",
                passed=True,
                score=1.0,
                details=f"Only {len(section_numbers)} sections — sequence check skipped",
            )

        section_numbers.sort()
        gaps = []
        for i in range(1, len(section_numbers)):
            expected = section_numbers[i - 1] + 1
            actual = section_numbers[i]
            if actual > expected:
                gaps.append(f"{section_numbers[i-1]}->{actual}")

        if not gaps:
            return QualityCheckResult(
                check_name="section_sequence",
                passed=True,
                score=1.0,
                details=f"{len(section_numbers)} sections, no gaps",
            )

        gap_ratio = len(gaps) / (len(section_numbers) - 1)
        return QualityCheckResult(
            check_name="section_sequence",
            passed=False,
            score=max(0.0, 1.0 - gap_ratio),
            details=f"Gaps found: {', '.join(gaps[:5])}{'...' if len(gaps) > 5 else ''}",
        )

    def _check_table_integrity(self, table) -> QualityCheckResult:
        """Check that table dimensions match cell data."""
        if table.row_count == 0 or table.col_count == 0:
            return QualityCheckResult(
                check_name=f"table_integrity_{table.id}",
                passed=True,
                score=1.0,
                details="Empty table — integrity check skipped",
            )

        expected_cells = table.row_count * table.col_count
        actual_cells = sum(len(row) for row in table.rows)

        if expected_cells == actual_cells:
            return QualityCheckResult(
                check_name=f"table_integrity_{table.id}",
                passed=True,
                score=1.0,
                details=f"{expected_cells} cells match",
            )

        ratio = min(actual_cells, expected_cells) / max(actual_cells, expected_cells)
        return QualityCheckResult(
            check_name=f"table_integrity_{table.id}",
            passed=False,
            score=ratio,
            details=f"Expected {expected_cells} cells, got {actual_cells}",
        )

    def _check_ocr_confidence(self, doc: ParsedDocument) -> QualityCheckResult:
        """Check OCR confidence against threshold."""
        confidence = doc.ocr_confidence or 0.0
        passed = confidence >= self._settings.ocr_confidence_threshold

        return QualityCheckResult(
            check_name="ocr_confidence",
            passed=passed,
            score=confidence,
            details=f"OCR confidence: {confidence:.2f} (threshold: {self._settings.ocr_confidence_threshold})",
        )
