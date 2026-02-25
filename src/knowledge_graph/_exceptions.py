from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class KnowledgeGraphError(LegalRAGError):
    """Base exception for the knowledge graph module."""


class KGConnectionError(KnowledgeGraphError):
    """Failed to connect to the Neo4j database."""


class KGSchemaError(KnowledgeGraphError):
    """Failed to create or validate the graph schema."""


class KGIngestionError(KnowledgeGraphError):
    """Failed to ingest entities or relationships into the graph."""


class KGQueryError(KnowledgeGraphError):
    """A Cypher query failed or returned unexpected results."""


class KGIntegrityError(KnowledgeGraphError):
    """Post-ingestion integrity check found violations."""


class KGNotAvailableError(KnowledgeGraphError):
    """The neo4j driver package is not installed."""
