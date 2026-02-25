"""Reusable Cypher query builders for the knowledge graph.

These queries are consumed by Phase 7 (Retrieval) and Phase 8 (Hallucination Mitigation).
All use parameterized Cypher — no string interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.knowledge_graph._exceptions import KGQueryError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from datetime import date

    from src.knowledge_graph._client import Neo4jClient

_log = get_logger(__name__)


class QueryBuilder:
    """Reusable Cypher query functions for the knowledge graph."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    async def point_in_time(
        self, act: str, section: str, query_date: date
    ) -> dict[str, Any] | None:
        """What was Section X of Act Y as on a given date?"""
        cypher = """
        MATCH (s:Section {number: $section, parent_act: $act})-[:HAS_VERSION]->(v:SectionVersion)
        WHERE v.effective_from <= date($query_date)
          AND (v.effective_until IS NULL OR v.effective_until > date($query_date))
        RETURN v.version_id AS version_id, v.text_hash AS text_hash,
               v.effective_from AS effective_from, v.effective_until AS effective_until,
               v.amending_act AS amending_act
        ORDER BY v.effective_from DESC
        LIMIT 1
        """
        try:
            results = await self._client.run_query(
                cypher,
                {
                    "act": act,
                    "section": section,
                    "query_date": str(query_date),
                },
            )
            return results[0] if results else None
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Point-in-time query failed for {act} s.{section} @ {query_date}: {exc}"
            raise KGQueryError(msg) from exc

    async def amendment_cascade(self, amending_act: str) -> list[dict[str, Any]]:
        """Find all sections affected by an amendment."""
        cypher = """
        MATCH (a:Amendment {amending_act: $amending_act})-[r:AMENDS|INSERTS|OMITS]->(s:Section)
        RETURN s.parent_act AS parent_act, s.number AS section_number,
               type(r) AS change_type, a.date AS amendment_date
        ORDER BY s.parent_act, s.number
        """
        try:
            return await self._client.run_query(cypher, {"amending_act": amending_act})
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Amendment cascade query failed for {amending_act}: {exc}"
            raise KGQueryError(msg) from exc

    async def citation_traversal(
        self,
        section: str,
        act: str,
        court: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find all judgments interpreting/citing a section."""
        if court:
            cypher = """
            MATCH (j:Judgment)-[:INTERPRETS|CITES_SECTION]->(s:Section {number: $section, parent_act: $act})
            WHERE j.court = $court
            RETURN j.citation AS citation, j.court AS court, j.date_decided AS date_decided,
                   j.status AS status
            ORDER BY j.date_decided DESC
            """
            params: dict[str, Any] = {"section": section, "act": act, "court": court}
        else:
            cypher = """
            MATCH (j:Judgment)-[:INTERPRETS|CITES_SECTION]->(s:Section {number: $section, parent_act: $act})
            RETURN j.citation AS citation, j.court AS court, j.date_decided AS date_decided,
                   j.status AS status
            ORDER BY j.date_decided DESC
            """
            params = {"section": section, "act": act}

        try:
            return await self._client.run_query(cypher, params)
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Citation traversal failed for s.{section} of {act}: {exc}"
            raise KGQueryError(msg) from exc

    async def hierarchy_navigation(
        self, act: str, chapter: str | None = None
    ) -> list[dict[str, Any]]:
        """List all sections under an Act, optionally filtered by chapter."""
        if chapter:
            cypher = """
            MATCH (a:Act {name: $act})-[:CONTAINS]->(s:Section)
            WHERE s.chapter = $chapter
            RETURN s.number AS number, s.chapter AS chapter, s.part AS part,
                   s.is_in_force AS is_in_force
            ORDER BY s.number
            """
            params: dict[str, Any] = {"act": act, "chapter": chapter}
        else:
            cypher = """
            MATCH (a:Act {name: $act})-[:CONTAINS]->(s:Section)
            RETURN s.number AS number, s.chapter AS chapter, s.part AS part,
                   s.is_in_force AS is_in_force
            ORDER BY s.number
            """
            params = {"act": act}

        try:
            return await self._client.run_query(cypher, params)
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Hierarchy navigation failed for {act}: {exc}"
            raise KGQueryError(msg) from exc

    async def temporal_status(
        self, section: str, act: str, ref_date: date | None = None
    ) -> dict[str, Any]:
        """Check if a section is currently in force, repealed, or superseded."""
        cypher = """
        MATCH (s:Section {number: $section, parent_act: $act})
        OPTIONAL MATCH (s)<-[:AMENDS|INSERTS|OMITS]-(a:Amendment)
        OPTIONAL MATCH (parent:Act {name: $act})
        RETURN s.is_in_force AS is_in_force,
               parent.status AS act_status,
               parent.date_repealed AS act_date_repealed,
               collect(DISTINCT {amending_act: a.amending_act, date: a.date, nature: a.nature}) AS amendments
        """
        try:
            results = await self._client.run_query(
                cypher,
                {
                    "section": section,
                    "act": act,
                },
            )
            if not results:
                return {"found": False, "section": section, "act": act}
            row = results[0]
            row["found"] = True
            row["section"] = section
            row["act"] = act
            return row
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Temporal status query failed for s.{section} of {act}: {exc}"
            raise KGQueryError(msg) from exc

    async def judgment_relationships(self, citation: str) -> dict[str, Any]:
        """Get all relationships for a judgment."""
        cypher = """
        MATCH (j:Judgment {citation: $citation})
        OPTIONAL MATCH (j)-[:OVERRULES]->(overruled:Judgment)
        OPTIONAL MATCH (overruler:Judgment)-[:OVERRULES]->(j)
        OPTIONAL MATCH (j)-[:FOLLOWS]->(followed:Judgment)
        OPTIONAL MATCH (follower:Judgment)-[:FOLLOWS]->(j)
        OPTIONAL MATCH (j)-[:DISTINGUISHES]->(distinguished:Judgment)
        OPTIONAL MATCH (distinguisher:Judgment)-[:DISTINGUISHES]->(j)
        OPTIONAL MATCH (j)-[:CITES_CASE]->(cited:Judgment)
        RETURN j.citation AS citation, j.status AS status,
               collect(DISTINCT overruled.citation) AS overrules,
               collect(DISTINCT overruler.citation) AS overruled_by,
               collect(DISTINCT followed.citation) AS follows,
               collect(DISTINCT follower.citation) AS followed_by,
               collect(DISTINCT distinguished.citation) AS distinguishes,
               collect(DISTINCT distinguisher.citation) AS distinguished_by,
               collect(DISTINCT cited.citation) AS cites
        """
        try:
            results = await self._client.run_query(cypher, {"citation": citation})
            if not results:
                return {"found": False, "citation": citation}
            row = results[0]
            row["found"] = True
            return row
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Judgment relationships query failed for {citation}: {exc}"
            raise KGQueryError(msg) from exc

    async def find_replacement(self, old_act: str, section: str) -> dict[str, Any] | None:
        """Find the replacement section/act for a repealed provision."""
        cypher = """
        MATCH (new_act:Act)-[:REPLACES]->(old:Act {name: $old_act})
        OPTIONAL MATCH (old)-[:CONTAINS]->(s:Section {number: $section})
        RETURN new_act.name AS replacement_act,
               s.replaced_by_section AS replacement_section
        LIMIT 1
        """
        try:
            results = await self._client.run_query(
                cypher,
                {
                    "old_act": old_act,
                    "section": section,
                },
            )
            return results[0] if results else None
        except Exception as exc:
            if isinstance(exc, KGQueryError):
                raise
            msg = f"Find replacement failed for {old_act} s.{section}: {exc}"
            raise KGQueryError(msg) from exc

    async def node_exists(self, label: str, key: dict[str, Any]) -> bool:
        """Check if a node exists in the graph."""
        where_clause = " AND ".join(f"n.{k} = ${k}" for k in key)
        cypher = f"MATCH (n:{label} WHERE {where_clause}) RETURN count(n) AS cnt"
        try:
            results = await self._client.run_query(cypher, key)
            return bool(results and results[0].get("cnt", 0) > 0)
        except Exception:
            return False
