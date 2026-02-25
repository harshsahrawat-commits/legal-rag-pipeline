"""Neo4j async client for the knowledge graph.

Handles connection, schema setup, MERGE operations, and batch execution.
All operations are idempotent via MERGE with ON CREATE SET / ON MATCH SET.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.knowledge_graph._exceptions import (
    KGConnectionError,
    KGIngestionError,
    KGNotAvailableError,
    KGSchemaError,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
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

_log = get_logger(__name__)

# --- Schema statements ---

CONSTRAINTS = [
    "CREATE CONSTRAINT act_name IF NOT EXISTS FOR (a:Act) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT section_unique IF NOT EXISTS FOR (s:Section) REQUIRE (s.parent_act, s.number) IS UNIQUE",
    "CREATE CONSTRAINT judgment_citation IF NOT EXISTS FOR (j:Judgment) REQUIRE j.citation IS UNIQUE",
    "CREATE CONSTRAINT section_version_id IF NOT EXISTS FOR (sv:SectionVersion) REQUIRE sv.version_id IS UNIQUE",
    "CREATE CONSTRAINT amendment_unique IF NOT EXISTS FOR (am:Amendment) REQUIRE (am.amending_act, am.date, am.nature) IS UNIQUE",
    "CREATE CONSTRAINT court_name IF NOT EXISTS FOR (c:Court) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT judge_name IF NOT EXISTS FOR (j:Judge) REQUIRE j.name IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (lc:LegalConcept) REQUIRE lc.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX section_in_force IF NOT EXISTS FOR (s:Section) ON (s.is_in_force)",
    "CREATE INDEX judgment_date IF NOT EXISTS FOR (j:Judgment) ON (j.date_decided)",
    "CREATE INDEX judgment_court IF NOT EXISTS FOR (j:Judgment) ON (j.court)",
]


class Neo4jClient:
    """Async Neo4j client with lazy driver initialization."""

    def __init__(self, settings: KGSettings) -> None:
        self._settings = settings
        self._driver = None

    async def setup_schema(self) -> None:
        """Create all constraints and indexes. Idempotent via IF NOT EXISTS."""
        self._ensure_driver()
        try:
            async with self._driver.session(database=self._settings.neo4j_database) as session:
                for stmt in CONSTRAINTS:
                    await session.run(stmt)
                for stmt in INDEXES:
                    await session.run(stmt)
            _log.info("schema_setup_complete", constraints=len(CONSTRAINTS), indexes=len(INDEXES))
        except Exception as exc:
            if isinstance(exc, KGSchemaError):
                raise
            msg = f"Failed to set up schema: {exc}"
            raise KGSchemaError(msg) from exc

    async def merge_act(self, node: ActNode) -> None:
        """MERGE an Act node."""
        cypher = """
        MERGE (a:Act {name: $name})
        ON CREATE SET a.number = $number, a.year = $year,
                      a.date_enacted = $date_enacted, a.date_effective = $date_effective,
                      a.date_repealed = $date_repealed, a.jurisdiction = $jurisdiction,
                      a.status = $status
        ON MATCH SET  a.number = coalesce($number, a.number),
                      a.year = coalesce($year, a.year),
                      a.date_enacted = coalesce($date_enacted, a.date_enacted),
                      a.date_effective = coalesce($date_effective, a.date_effective),
                      a.date_repealed = coalesce($date_repealed, a.date_repealed),
                      a.status = $status
        """
        await self._execute_write(cypher, node.model_dump(mode="json"))

    async def merge_section(self, node: SectionNode) -> None:
        """MERGE a Section node."""
        cypher = """
        MERGE (s:Section {parent_act: $parent_act, number: $number})
        ON CREATE SET s.chapter = $chapter, s.part = $part,
                      s.is_in_force = $is_in_force, s.chunk_id = $chunk_id
        ON MATCH SET  s.chapter = coalesce($chapter, s.chapter),
                      s.part = coalesce($part, s.part),
                      s.is_in_force = $is_in_force,
                      s.chunk_id = coalesce($chunk_id, s.chunk_id)
        """
        params = node.model_dump(mode="json")
        if params.get("chunk_id"):
            params["chunk_id"] = str(params["chunk_id"])
        await self._execute_write(cypher, params)

    async def merge_section_version(self, node: SectionVersionNode) -> None:
        """MERGE a SectionVersion node."""
        cypher = """
        MERGE (sv:SectionVersion {version_id: $version_id})
        ON CREATE SET sv.text_hash = $text_hash,
                      sv.effective_from = $effective_from,
                      sv.effective_until = $effective_until,
                      sv.amending_act = $amending_act
        ON MATCH SET  sv.text_hash = $text_hash,
                      sv.effective_from = coalesce($effective_from, sv.effective_from),
                      sv.effective_until = coalesce($effective_until, sv.effective_until),
                      sv.amending_act = coalesce($amending_act, sv.amending_act)
        """
        await self._execute_write(cypher, node.model_dump(mode="json"))

    async def merge_judgment(self, node: JudgmentNode) -> None:
        """MERGE a Judgment node."""
        cypher = """
        MERGE (j:Judgment {citation: $citation})
        ON CREATE SET j.alt_citations = $alt_citations, j.court = $court,
                      j.court_level = $court_level, j.bench_type = $bench_type,
                      j.bench_strength = $bench_strength, j.date_decided = $date_decided,
                      j.case_type = $case_type,
                      j.parties_petitioner = $parties_petitioner,
                      j.parties_respondent = $parties_respondent,
                      j.status = $status, j.chunk_id = $chunk_id
        ON MATCH SET  j.alt_citations = coalesce($alt_citations, j.alt_citations),
                      j.court = $court, j.court_level = $court_level,
                      j.bench_type = coalesce($bench_type, j.bench_type),
                      j.bench_strength = coalesce($bench_strength, j.bench_strength),
                      j.date_decided = coalesce($date_decided, j.date_decided),
                      j.case_type = coalesce($case_type, j.case_type),
                      j.parties_petitioner = coalesce($parties_petitioner, j.parties_petitioner),
                      j.parties_respondent = coalesce($parties_respondent, j.parties_respondent),
                      j.status = $status,
                      j.chunk_id = coalesce($chunk_id, j.chunk_id)
        """
        params = node.model_dump(mode="json")
        if params.get("chunk_id"):
            params["chunk_id"] = str(params["chunk_id"])
        await self._execute_write(cypher, params)

    async def merge_amendment(self, node: AmendmentNode) -> None:
        """MERGE an Amendment node."""
        cypher = """
        MERGE (am:Amendment {amending_act: $amending_act, date: $date, nature: $nature})
        ON CREATE SET am.gazette_ref = $gazette_ref
        ON MATCH SET  am.gazette_ref = coalesce($gazette_ref, am.gazette_ref)
        """
        await self._execute_write(cypher, node.model_dump(mode="json"))

    async def merge_legal_concept(self, node: LegalConceptNode) -> None:
        """MERGE a LegalConcept node."""
        cypher = """
        MERGE (lc:LegalConcept {name: $name})
        ON CREATE SET lc.definition_source = $definition_source, lc.category = $category
        ON MATCH SET  lc.definition_source = coalesce($definition_source, lc.definition_source),
                      lc.category = coalesce($category, lc.category)
        """
        await self._execute_write(cypher, node.model_dump(mode="json"))

    async def merge_court(self, node: CourtNode) -> None:
        """MERGE a Court node."""
        cypher = """
        MERGE (c:Court {name: $name})
        ON CREATE SET c.hierarchy_level = $hierarchy_level, c.state = $state,
                      c.jurisdiction_type = $jurisdiction_type
        ON MATCH SET  c.hierarchy_level = $hierarchy_level,
                      c.state = coalesce($state, c.state),
                      c.jurisdiction_type = coalesce($jurisdiction_type, c.jurisdiction_type)
        """
        await self._execute_write(cypher, node.model_dump(mode="json"))

    async def merge_judge(self, node: JudgeNode) -> None:
        """MERGE a Judge node."""
        cypher = """
        MERGE (j:Judge {name: $name})
        ON CREATE SET j.courts_served = $courts_served
        ON MATCH SET  j.courts_served = $courts_served
        """
        await self._execute_write(cypher, node.model_dump(mode="json"))

    async def create_relationship(self, rel: Relationship) -> None:
        """MERGE a relationship between two nodes."""
        from_key_clause = " AND ".join(f"a.{k} = $from_{k}" for k in rel.from_key)
        to_key_clause = " AND ".join(f"b.{k} = $to_{k}" for k in rel.to_key)

        prop_clause = ""
        if rel.properties:
            prop_sets = ", ".join(f"r.{k} = $prop_{k}" for k in rel.properties)
            prop_clause = f" ON CREATE SET {prop_sets} ON MATCH SET {prop_sets}"

        cypher = (
            f"MATCH (a:{rel.from_label} WHERE {from_key_clause}), "
            f"(b:{rel.to_label} WHERE {to_key_clause}) "
            f"MERGE (a)-[r:{rel.rel_type}]->(b){prop_clause}"
        )

        params: dict[str, Any] = {}
        for k, v in rel.from_key.items():
            params[f"from_{k}"] = v
        for k, v in rel.to_key.items():
            params[f"to_{k}"] = v
        for k, v in rel.properties.items():
            params[f"prop_{k}"] = v

        await self._execute_write(cypher, params)

    async def execute_batch(self, operations: list[tuple[str, dict[str, Any]]]) -> None:
        """Execute multiple Cypher statements in a single transaction."""
        self._ensure_driver()
        try:
            async with self._driver.session(database=self._settings.neo4j_database) as session:

                async def _tx(tx: Any) -> None:
                    for cypher, params in operations:
                        await tx.run(cypher, params)

                await session.execute_write(_tx)
            _log.debug("batch_executed", count=len(operations))
        except Exception as exc:
            if isinstance(exc, KGIngestionError):
                raise
            msg = f"Batch execution failed ({len(operations)} ops): {exc}"
            raise KGIngestionError(msg) from exc

    async def run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read query and return results as list of dicts."""
        self._ensure_driver()
        async with self._driver.session(database=self._settings.neo4j_database) as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def close(self) -> None:
        """Clean shutdown of the driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            _log.info("neo4j_driver_closed")

    def _ensure_driver(self) -> None:
        """Lazy-initialize the async Neo4j driver."""
        if self._driver is not None:
            return

        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            msg = "neo4j is required. Install with: pip install neo4j"
            raise KGNotAvailableError(msg) from exc

        try:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            )
        except Exception as exc:
            msg = f"Failed to create Neo4j driver: {exc}"
            raise KGConnectionError(msg) from exc

    async def _execute_write(self, cypher: str, params: dict[str, Any]) -> None:
        """Execute a single write Cypher statement."""
        self._ensure_driver()
        try:
            async with self._driver.session(database=self._settings.neo4j_database) as session:

                async def _tx(tx: Any) -> None:
                    await tx.run(cypher, params)

                await session.execute_write(_tx)
        except Exception as exc:
            if isinstance(exc, (KGIngestionError, KGNotAvailableError, KGConnectionError)):
                raise
            msg = f"Write failed: {exc}"
            raise KGIngestionError(msg) from exc
