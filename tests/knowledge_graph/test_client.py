"""Tests for Neo4jClient with mocked async driver."""

from __future__ import annotations

import sys
from datetime import date
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.knowledge_graph._client import CONSTRAINTS, INDEXES, Neo4jClient
from src.knowledge_graph._exceptions import (
    KGConnectionError,
    KGIngestionError,
    KGNotAvailableError,
    KGSchemaError,
)
from src.knowledge_graph._models import (
    ActNode,
    AmendmentNode,
    CourtNode,
    JudgeNode,
    JudgmentNode,
    KGSettings,
    LegalConceptNode,
    Relationship,
    SectionNode,
    SectionVersionNode,
)


def _build_mock_neo4j_module():
    """Build a fake neo4j module."""
    mod = ModuleType("neo4j")
    driver = AsyncMock()
    session = AsyncMock()
    session.run = AsyncMock(return_value=AsyncMock(data=AsyncMock(return_value=[])))
    session.execute_write = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()

    mock_async_gdb = MagicMock()
    mock_async_gdb.driver = MagicMock(return_value=driver)
    mod.AsyncGraphDatabase = mock_async_gdb
    return mod, driver, session


@pytest.fixture(autouse=True)
def _mock_neo4j_module():
    """Inject a fake neo4j module for all tests."""
    mod, _driver, _session = _build_mock_neo4j_module()
    with patch.dict(sys.modules, {"neo4j": mod}):
        yield


@pytest.fixture()
def mock_session():
    session = AsyncMock()
    session.run = AsyncMock(return_value=AsyncMock(data=AsyncMock(return_value=[])))
    session.execute_write = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture()
def mock_driver(mock_session):
    driver = AsyncMock()
    driver.session = MagicMock(return_value=mock_session)
    driver.close = AsyncMock()
    return driver


@pytest.fixture()
def client(mock_driver):
    settings = KGSettings()
    c = Neo4jClient(settings)
    c._driver = mock_driver
    return c


class TestInit:
    def test_creates_with_settings(self) -> None:
        settings = KGSettings()
        c = Neo4jClient(settings)
        assert c._settings is settings
        assert c._driver is None


class TestMissingDependency:
    def test_raises_without_neo4j(self) -> None:
        c = Neo4jClient(KGSettings())
        with (
            patch.dict("sys.modules", {"neo4j": None}),
            pytest.raises(KGNotAvailableError, match="neo4j"),
        ):
            c._ensure_driver()


class TestEnsureDriver:
    def test_lazy_init(self) -> None:
        settings = KGSettings()
        c = Neo4jClient(settings)
        c._ensure_driver()
        assert c._driver is not None

    def test_idempotent(self, client, mock_driver) -> None:
        client._ensure_driver()
        # Driver should still be the same mock
        assert client._driver is mock_driver

    def test_connection_error(self) -> None:
        mod = ModuleType("neo4j")
        mock_async_gdb = MagicMock()
        mock_async_gdb.driver = MagicMock(side_effect=RuntimeError("connection refused"))
        mod.AsyncGraphDatabase = mock_async_gdb

        c = Neo4jClient(KGSettings())
        with (
            patch.dict(sys.modules, {"neo4j": mod}),
            pytest.raises(KGConnectionError, match="connection refused"),
        ):
            c._ensure_driver()


class TestSetupSchema:
    @pytest.mark.asyncio
    async def test_runs_all_statements(self, client, mock_session) -> None:
        await client.setup_schema()
        expected_calls = len(CONSTRAINTS) + len(INDEXES)
        assert mock_session.run.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_schema_error_wraps(self, client, mock_session) -> None:
        mock_session.run.side_effect = RuntimeError("constraint failed")
        with pytest.raises(KGSchemaError, match="Failed to set up schema"):
            await client.setup_schema()


class TestMergeAct:
    @pytest.mark.asyncio
    async def test_merge_act(self, client, mock_session) -> None:
        act = ActNode(name="Indian Penal Code", number="Act No. 45 of 1860", year=1860)
        await client.merge_act(act)
        mock_session.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_act_minimal(self, client, mock_session) -> None:
        act = ActNode(name="Test Act")
        await client.merge_act(act)
        mock_session.execute_write.assert_called_once()


class TestMergeSection:
    @pytest.mark.asyncio
    async def test_merge_section(self, client, mock_session) -> None:
        section = SectionNode(number="302", parent_act="IPC", chapter="XVI")
        await client.merge_section(section)
        mock_session.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_section_with_chunk_id(self, client, mock_session) -> None:
        section = SectionNode(number="302", parent_act="IPC", chunk_id=uuid4())
        await client.merge_section(section)
        mock_session.execute_write.assert_called_once()


class TestMergeSectionVersion:
    @pytest.mark.asyncio
    async def test_merge_section_version(self, client, mock_session) -> None:
        sv = SectionVersionNode(
            version_id="IPC:302:v1",
            text_hash="abc123",
            effective_from=date(1860, 10, 6),
        )
        await client.merge_section_version(sv)
        mock_session.execute_write.assert_called_once()


class TestMergeJudgment:
    @pytest.mark.asyncio
    async def test_merge_judgment(self, client, mock_session) -> None:
        j = JudgmentNode(
            citation="(2024) 1 SCC 100",
            court="Supreme Court of India",
            court_level=1,
            date_decided=date(2024, 1, 15),
        )
        await client.merge_judgment(j)
        mock_session.execute_write.assert_called_once()


class TestMergeAmendment:
    @pytest.mark.asyncio
    async def test_merge_amendment(self, client, mock_session) -> None:
        a = AmendmentNode(
            amending_act="CLA 2013",
            date=date(2013, 4, 2),
            nature="substitution",
        )
        await client.merge_amendment(a)
        mock_session.execute_write.assert_called_once()


class TestMergeLegalConcept:
    @pytest.mark.asyncio
    async def test_merge_legal_concept(self, client, mock_session) -> None:
        lc = LegalConceptNode(name="mens rea", definition_source="Section 2, IPC")
        await client.merge_legal_concept(lc)
        mock_session.execute_write.assert_called_once()


class TestMergeCourt:
    @pytest.mark.asyncio
    async def test_merge_court(self, client, mock_session) -> None:
        court = CourtNode(name="Supreme Court of India", hierarchy_level=1)
        await client.merge_court(court)
        mock_session.execute_write.assert_called_once()


class TestMergeJudge:
    @pytest.mark.asyncio
    async def test_merge_judge(self, client, mock_session) -> None:
        judge = JudgeNode(name="Justice D.Y. Chandrachud", courts_served=["SC"])
        await client.merge_judge(judge)
        mock_session.execute_write.assert_called_once()


class TestCreateRelationship:
    @pytest.mark.asyncio
    async def test_basic_relationship(self, client, mock_session) -> None:
        rel = Relationship(
            from_label="Act",
            from_key={"name": "IPC"},
            to_label="Section",
            to_key={"parent_act": "IPC", "number": "302"},
            rel_type="CONTAINS",
        )
        await client.create_relationship(rel)
        mock_session.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_relationship_with_properties(self, client, mock_session) -> None:
        rel = Relationship(
            from_label="Amendment",
            from_key={"amending_act": "CLA 2013"},
            to_label="Section",
            to_key={"parent_act": "IPC", "number": "376"},
            rel_type="AMENDS",
            properties={"before_text": "old", "after_text": "new"},
        )
        await client.create_relationship(rel)
        mock_session.execute_write.assert_called_once()


class TestExecuteBatch:
    @pytest.mark.asyncio
    async def test_batch_execution(self, client, mock_session) -> None:
        ops = [
            ("MERGE (a:Act {name: $name})", {"name": "IPC"}),
            ("MERGE (a:Act {name: $name})", {"name": "CrPC"}),
        ]
        await client.execute_batch(ops)
        mock_session.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_batch(self, client, mock_session) -> None:
        await client.execute_batch([])
        mock_session.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_error_wraps(self, client, mock_session) -> None:
        mock_session.execute_write.side_effect = RuntimeError("tx failed")
        with pytest.raises(KGIngestionError, match="Batch execution failed"):
            await client.execute_batch([("MERGE (a:Act {name: $n})", {"n": "X"})])


class TestRunQuery:
    @pytest.mark.asyncio
    async def test_returns_results(self, client, mock_session) -> None:
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"name": "IPC"}])
        mock_session.run.return_value = mock_result

        results = await client.run_query("MATCH (a:Act) RETURN a.name as name")
        assert results == [{"name": "IPC"}]

    @pytest.mark.asyncio
    async def test_with_params(self, client, mock_session) -> None:
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[])
        mock_session.run.return_value = mock_result

        await client.run_query("MATCH (a:Act {name: $name}) RETURN a", {"name": "IPC"})
        mock_session.run.assert_called_with("MATCH (a:Act {name: $name}) RETURN a", {"name": "IPC"})


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_driver(self, client, mock_driver) -> None:
        await client.close()
        mock_driver.close.assert_called_once()
        assert client._driver is None

    @pytest.mark.asyncio
    async def test_close_when_no_driver(self) -> None:
        c = Neo4jClient(KGSettings())
        await c.close()  # Should not raise


class TestWriteErrors:
    @pytest.mark.asyncio
    async def test_execute_write_wraps_errors(self, client, mock_session) -> None:
        mock_session.execute_write.side_effect = RuntimeError("write error")
        act = ActNode(name="Test")
        with pytest.raises(KGIngestionError, match="Write failed"):
            await client.merge_act(act)
