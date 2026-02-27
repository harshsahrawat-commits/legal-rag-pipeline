"""Tests for Layer 2: Temporal Checker."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from src.hallucination._models import HallucinationSettings
from src.hallucination._temporal_checker import TemporalChecker


@pytest.fixture()
def settings() -> HallucinationSettings:
    return HallucinationSettings()


@pytest.fixture()
def mock_qb() -> AsyncMock:
    """Mock QueryBuilder."""
    qb = AsyncMock()
    qb.temporal_status = AsyncMock(return_value={"found": True, "is_in_force": True})
    qb.find_replacement = AsyncMock(return_value=None)
    return qb


class TestHardcodedRepeals:
    """Test IPC→BNS / CrPC→BNSS / Evidence Act→BSA transition."""

    async def test_ipc_repealed_after_july_2024(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 420 of the Indian Penal Code provides for punishment.",
            reference_date=date(2025, 1, 15),
        )
        assert len(warnings) == 1
        assert warnings[0].act == "Indian Penal Code"
        assert warnings[0].replacement_act == "Bharatiya Nyaya Sanhita"
        assert "repealed" in warnings[0].warning_text.lower()

    async def test_ipc_valid_before_july_2024(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 420 of the Indian Penal Code provides for punishment.",
            reference_date=date(2024, 6, 30),
        )
        assert len(warnings) == 0

    async def test_crpc_repealed(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 154 of the Code of Criminal Procedure requires FIR registration.",
            reference_date=date(2025, 1, 1),
        )
        assert len(warnings) == 1
        assert warnings[0].replacement_act == "Bharatiya Nagarik Suraksha Sanhita"

    async def test_evidence_act_repealed(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 45 of the Indian Evidence Act deals with expert opinion.",
            reference_date=date(2025, 1, 1),
        )
        assert len(warnings) == 1
        assert warnings[0].replacement_act == "Bharatiya Sakshya Adhiniyam"

    async def test_bns_no_warning(self, settings: HallucinationSettings) -> None:
        """BNS (new code) should NOT trigger a warning."""
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 420 of the Bharatiya Nyaya Sanhita provides for punishment.",
            reference_date=date(2025, 1, 1),
        )
        assert len(warnings) == 0

    async def test_ipc_on_repeal_date(self, settings: HallucinationSettings) -> None:
        """Exactly July 1, 2024 — should trigger warning."""
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 302 of the Indian Penal Code is about murder.",
            reference_date=date(2024, 7, 1),
        )
        assert len(warnings) == 1

    async def test_multiple_sections_same_act_deduped(
        self, settings: HallucinationSettings
    ) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 420 of the Indian Penal Code and Section 302 of the Indian Penal Code.",
            reference_date=date(2025, 1, 1),
        )
        # Both sections should produce warnings
        assert len(warnings) == 2

    async def test_repealed_by_field(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 420 of the Indian Penal Code.",
            reference_date=date(2025, 1, 1),
        )
        assert warnings[0].repealed_by == "Bharatiya Nyaya Sanhita, 2023"


class TestKGTemporalCheck:
    """Test KG-backed temporal checks."""

    async def test_in_force_no_warning(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.temporal_status.return_value = {
            "found": True,
            "is_in_force": True,
            "act_status": "in_force",
        }
        checker = TemporalChecker(settings, query_builder=mock_qb)
        warnings = await checker.check_response(
            "Section 10 of the Contract Act.",
            reference_date=date(2025, 1, 1),
        )
        # Contract Act is not in _REPEALED_ACTS, so it goes to KG
        assert len(warnings) == 0

    async def test_section_not_in_force(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.temporal_status.return_value = {
            "found": True,
            "is_in_force": False,
            "act_status": "in_force",
        }
        mock_qb.find_replacement.return_value = None
        checker = TemporalChecker(settings, query_builder=mock_qb)
        warnings = await checker.check_response(
            "Section 10 of the Contract Act.",
            reference_date=date(2025, 1, 1),
        )
        assert len(warnings) == 1
        assert "no longer in force" in warnings[0].warning_text

    async def test_act_repealed_in_kg(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.temporal_status.return_value = {
            "found": True,
            "is_in_force": True,
            "act_status": "repealed",
        }
        mock_qb.find_replacement.return_value = {
            "replacement_act": "New Act",
            "replacement_section": "10A",
        }
        checker = TemporalChecker(settings, query_builder=mock_qb)
        warnings = await checker.check_response(
            "Section 10 of the Repealed Act.",
            reference_date=date(2025, 1, 1),
        )
        assert len(warnings) == 1
        assert warnings[0].replacement_act == "New Act"

    async def test_section_not_found_in_kg(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.temporal_status.return_value = {"found": False}
        checker = TemporalChecker(settings, query_builder=mock_qb)
        warnings = await checker.check_response(
            "Section 999 of the Random Act.",
            reference_date=date(2025, 1, 1),
        )
        assert len(warnings) == 0

    async def test_kg_exception_graceful(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.temporal_status.side_effect = Exception("connection refused")
        checker = TemporalChecker(settings, query_builder=mock_qb)
        warnings = await checker.check_response(
            "Section 10 of the Contract Act.",
            reference_date=date(2025, 1, 1),
        )
        # Should gracefully return no warnings, not crash
        assert len(warnings) == 0


class TestTemporalCheckerEdgeCases:
    async def test_empty_response(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response("", reference_date=date(2025, 1, 1))
        assert warnings == []

    async def test_no_section_refs(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Article 21 guarantees the right to life.",
            reference_date=date(2025, 1, 1),
        )
        # Articles are not section refs — should produce no warnings
        assert warnings == []

    async def test_no_kg_no_crash(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings, query_builder=None)
        warnings = await checker.check_response(
            "Section 10 of the Contract Act.",
            reference_date=date(2025, 1, 1),
        )
        assert warnings == []

    async def test_default_ref_date(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        # No reference_date → uses today() → after July 2024
        warnings = await checker.check_response("Section 420 of the Indian Penal Code.")
        assert len(warnings) == 1

    async def test_reference_date_from_input(self, settings: HallucinationSettings) -> None:
        checker = TemporalChecker(settings)
        warnings = await checker.check_response(
            "Section 420 of the Indian Penal Code.",
            reference_date=date(2025, 6, 1),
        )
        assert len(warnings) == 1
        assert warnings[0].reference_date == date(2025, 6, 1)
