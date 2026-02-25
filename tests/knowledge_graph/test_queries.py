"""Tests for QueryBuilder with mocked Neo4j client."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from src.knowledge_graph._exceptions import KGQueryError
from src.knowledge_graph._queries import QueryBuilder


@pytest.fixture()
def mock_client():
    client = AsyncMock()
    client.run_query = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def qb(mock_client) -> QueryBuilder:
    return QueryBuilder(mock_client)


class TestPointInTime:
    @pytest.mark.asyncio
    async def test_returns_version(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "version_id": "IPC:302:v1",
                "text_hash": "abc123",
                "effective_from": "1860-10-06",
                "effective_until": None,
                "amending_act": None,
            }
        ]
        result = await qb.point_in_time("Indian Penal Code", "302", date(2024, 1, 1))
        assert result is not None
        assert result["version_id"] == "IPC:302:v1"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, qb) -> None:
        result = await qb.point_in_time("Unknown Act", "999", date(2024, 1, 1))
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_correct_params(self, qb, mock_client) -> None:
        await qb.point_in_time("IPC", "302", date(2024, 1, 1))
        call_args = mock_client.run_query.call_args
        assert call_args[0][1]["act"] == "IPC"
        assert call_args[0][1]["section"] == "302"
        assert call_args[0][1]["query_date"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Point-in-time"):
            await qb.point_in_time("IPC", "302", date(2024, 1, 1))


class TestAmendmentCascade:
    @pytest.mark.asyncio
    async def test_returns_affected_sections(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "parent_act": "IPC",
                "section_number": "376",
                "change_type": "AMENDS",
                "amendment_date": "2013-04-02",
            },
            {
                "parent_act": "IPC",
                "section_number": "376A",
                "change_type": "INSERTS",
                "amendment_date": "2013-04-02",
            },
        ]
        results = await qb.amendment_cascade("Criminal Law Amendment Act, 2013")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_when_no_match(self, qb) -> None:
        results = await qb.amendment_cascade("Nonexistent Act")
        assert results == []

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Amendment cascade"):
            await qb.amendment_cascade("CLA 2013")


class TestCitationTraversal:
    @pytest.mark.asyncio
    async def test_returns_judgments(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "citation": "(2024) 1 SCC 100",
                "court": "SC",
                "date_decided": "2024-01-15",
                "status": "good_law",
            },
        ]
        results = await qb.citation_traversal("302", "Indian Penal Code")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_filters_by_court(self, qb, mock_client) -> None:
        await qb.citation_traversal("302", "IPC", court="Supreme Court of India")
        call_args = mock_client.run_query.call_args
        assert call_args[0][1]["court"] == "Supreme Court of India"

    @pytest.mark.asyncio
    async def test_no_court_filter(self, qb, mock_client) -> None:
        await qb.citation_traversal("302", "IPC")
        call_args = mock_client.run_query.call_args
        assert "court" not in call_args[0][1]

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Citation traversal"):
            await qb.citation_traversal("302", "IPC")


class TestHierarchyNavigation:
    @pytest.mark.asyncio
    async def test_returns_sections(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {"number": "1", "chapter": "I", "part": None, "is_in_force": True},
            {"number": "2", "chapter": "I", "part": None, "is_in_force": True},
        ]
        results = await qb.hierarchy_navigation("Indian Penal Code")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filters_by_chapter(self, qb, mock_client) -> None:
        await qb.hierarchy_navigation("IPC", chapter="XVI")
        call_args = mock_client.run_query.call_args
        assert call_args[0][1]["chapter"] == "XVI"

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Hierarchy navigation"):
            await qb.hierarchy_navigation("IPC")


class TestTemporalStatus:
    @pytest.mark.asyncio
    async def test_returns_status(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "is_in_force": True,
                "act_status": "in_force",
                "act_date_repealed": None,
                "amendments": [],
            }
        ]
        result = await qb.temporal_status("302", "IPC")
        assert result["found"] is True
        assert result["is_in_force"] is True

    @pytest.mark.asyncio
    async def test_not_found(self, qb) -> None:
        result = await qb.temporal_status("999", "Unknown Act")
        assert result["found"] is False

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Temporal status"):
            await qb.temporal_status("302", "IPC")


class TestJudgmentRelationships:
    @pytest.mark.asyncio
    async def test_returns_relationships(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {
                "citation": "(2024) 1 SCC 100",
                "status": "good_law",
                "overrules": [],
                "overruled_by": [],
                "follows": [],
                "followed_by": ["(2025) 1 SCC 50"],
                "distinguishes": [],
                "distinguished_by": [],
                "cites": ["(2020) 3 SCC 200"],
            }
        ]
        result = await qb.judgment_relationships("(2024) 1 SCC 100")
        assert result["found"] is True
        assert len(result["followed_by"]) == 1

    @pytest.mark.asyncio
    async def test_not_found(self, qb) -> None:
        result = await qb.judgment_relationships("(9999) 1 SCC 1")
        assert result["found"] is False

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Judgment relationships"):
            await qb.judgment_relationships("X")


class TestFindReplacement:
    @pytest.mark.asyncio
    async def test_returns_replacement(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [
            {"replacement_act": "BNS", "replacement_section": "103"}
        ]
        result = await qb.find_replacement("IPC", "302")
        assert result is not None
        assert result["replacement_act"] == "BNS"

    @pytest.mark.asyncio
    async def test_none_when_no_replacement(self, qb) -> None:
        result = await qb.find_replacement("Unknown", "1")
        assert result is None

    @pytest.mark.asyncio
    async def test_wraps_errors(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        with pytest.raises(KGQueryError, match="Find replacement"):
            await qb.find_replacement("IPC", "302")


class TestNodeExists:
    @pytest.mark.asyncio
    async def test_returns_true(self, qb, mock_client) -> None:
        mock_client.run_query.return_value = [{"cnt": 1}]
        assert await qb.node_exists("Act", {"name": "IPC"}) is True

    @pytest.mark.asyncio
    async def test_returns_false(self, qb) -> None:
        assert await qb.node_exists("Act", {"name": "Unknown"}) is False

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, qb, mock_client) -> None:
        mock_client.run_query.side_effect = RuntimeError("db error")
        assert await qb.node_exists("Act", {"name": "IPC"}) is False
