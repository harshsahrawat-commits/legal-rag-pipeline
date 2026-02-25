from __future__ import annotations

import pytest

from src.knowledge_graph._exceptions import (
    KGConnectionError,
    KGIngestionError,
    KGIntegrityError,
    KGNotAvailableError,
    KGQueryError,
    KGSchemaError,
    KnowledgeGraphError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    """All KG exceptions inherit from KnowledgeGraphError -> LegalRAGError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            KnowledgeGraphError,
            KGConnectionError,
            KGSchemaError,
            KGIngestionError,
            KGQueryError,
            KGIntegrityError,
            KGNotAvailableError,
        ],
    )
    def test_inherits_legal_rag_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, LegalRAGError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            KGConnectionError,
            KGSchemaError,
            KGIngestionError,
            KGQueryError,
            KGIntegrityError,
            KGNotAvailableError,
        ],
    )
    def test_inherits_knowledge_graph_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, KnowledgeGraphError)

    def test_knowledge_graph_error_is_base(self) -> None:
        assert issubclass(KnowledgeGraphError, LegalRAGError)
        assert not issubclass(LegalRAGError, KnowledgeGraphError)


class TestExceptionMessages:
    def test_connection_error_message(self) -> None:
        exc = KGConnectionError("bolt://localhost:7687 refused")
        assert "bolt://" in str(exc)

    def test_schema_error_message(self) -> None:
        exc = KGSchemaError("Failed to create constraint act_name")
        assert "constraint" in str(exc)

    def test_ingestion_error_message(self) -> None:
        exc = KGIngestionError("MERGE failed for Act node")
        assert "MERGE" in str(exc)

    def test_query_error_message(self) -> None:
        exc = KGQueryError("SyntaxError in Cypher query")
        assert "Cypher" in str(exc)

    def test_integrity_error_message(self) -> None:
        exc = KGIntegrityError("Section without SectionVersion found")
        assert "SectionVersion" in str(exc)

    def test_not_available_error_message(self) -> None:
        exc = KGNotAvailableError("neo4j package not installed")
        assert "neo4j" in str(exc)


class TestExceptionRaising:
    def test_catch_as_knowledge_graph_error(self) -> None:
        with pytest.raises(KnowledgeGraphError):
            raise KGConnectionError("test")

    def test_catch_as_legal_rag_error(self) -> None:
        with pytest.raises(LegalRAGError):
            raise KGIngestionError("test")

    def test_exception_chaining(self) -> None:
        original = RuntimeError("driver error")
        try:
            raise KGConnectionError("connection failed") from original
        except KGConnectionError as exc:
            assert exc.__cause__ is original
