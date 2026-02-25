"""Tests for IntegrityChecker."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.knowledge_graph._exceptions import KGIntegrityError
from src.knowledge_graph._integrity import IntegrityChecker


@pytest.fixture()
def mock_client():
    client = AsyncMock()
    client.run_query = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def checker(mock_client) -> IntegrityChecker:
    return IntegrityChecker(mock_client)


class TestCheckSectionVersions:
    @pytest.mark.asyncio
    async def test_passes_when_all_have_versions(self, checker) -> None:
        result = await checker.check_section_versions()
        assert result.passed is True
        assert result.violations == []

    @pytest.mark.asyncio
    async def test_fails_when_missing_versions(self, checker, mock_client) -> None:
        mock_client.run_query.return_value = [
            {"act": "IPC", "section": "302"},
            {"act": "IPC", "section": "303"},
        ]
        result = await checker.check_section_versions()
        assert result.passed is False
        assert len(result.violations) == 2
        assert "IPC:s.302" in result.violations[0]

    @pytest.mark.asyncio
    async def test_wraps_errors(self, checker, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGIntegrityError, match="section_versions"):
            await checker.check_section_versions()


class TestCheckRepealedConsistency:
    @pytest.mark.asyncio
    async def test_passes_when_consistent(self, checker) -> None:
        result = await checker.check_repealed_consistency()
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_inconsistent(self, checker, mock_client) -> None:
        mock_client.run_query.return_value = [
            {"act": "IPC", "section": "302"},
        ]
        result = await checker.check_repealed_consistency()
        assert result.passed is False
        assert "is_in_force=true" in result.violations[0]

    @pytest.mark.asyncio
    async def test_wraps_errors(self, checker, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGIntegrityError, match="repealed_consistency"):
            await checker.check_repealed_consistency()


class TestCheckOverruleHierarchy:
    @pytest.mark.asyncio
    async def test_passes_when_valid(self, checker) -> None:
        result = await checker.check_overrule_hierarchy()
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_lower_court_overrules(self, checker, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "overruler": "(2024) HC 100",
                "overruled": "(2020) SC 50",
                "overruler_level": 2,
                "overruled_level": 1,
            },
        ]
        result = await checker.check_overrule_hierarchy()
        assert result.passed is False
        assert "level 2" in result.violations[0]

    @pytest.mark.asyncio
    async def test_wraps_errors(self, checker, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGIntegrityError, match="overrule_hierarchy"):
            await checker.check_overrule_hierarchy()


class TestCheckVersionDateOverlap:
    @pytest.mark.asyncio
    async def test_passes_when_no_overlap(self, checker) -> None:
        result = await checker.check_version_date_overlap()
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_overlapping(self, checker, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "act": "IPC",
                "section": "302",
                "v1_id": "IPC:302:v1",
                "v2_id": "IPC:302:v2",
                "v1_from": "1860-10-06",
                "v1_until": None,
                "v2_from": "2013-04-02",
            },
        ]
        result = await checker.check_version_date_overlap()
        assert result.passed is False
        assert "overlap" in result.violations[0]

    @pytest.mark.asyncio
    async def test_wraps_errors(self, checker, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGIntegrityError, match="version_date_overlap"):
            await checker.check_version_date_overlap()


class TestCheckAll:
    @pytest.mark.asyncio
    async def test_all_passing(self, checker) -> None:
        report = await checker.check_all()
        assert report.passed is True
        assert len(report.checks) == 4
        assert all(c.passed for c in report.checks)

    @pytest.mark.asyncio
    async def test_one_failing(self, checker, mock_client) -> None:
        # Make section_versions fail (first query returns results)
        call_count = 0
        original_return = []

        async def _mock_query(cypher, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"act": "IPC", "section": "302"}]
            return original_return

        mock_client.run_query = AsyncMock(side_effect=_mock_query)
        report = await checker.check_all()
        assert report.passed is False
        # First check should fail, rest pass
        assert report.checks[0].passed is False
        assert all(c.passed for c in report.checks[1:])
