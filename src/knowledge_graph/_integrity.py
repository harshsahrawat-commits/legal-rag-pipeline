"""Post-ingestion integrity checker for the knowledge graph.

Validates data consistency rules from docs/knowledge_graph_schema.md:
1. Every Section must have at least one SectionVersion.
2. If Act is "repealed", all sections must have is_in_force = false.
3. OVERRULES requires equal or higher court hierarchy.
4. SectionVersion effective_from ranges must not overlap for the same section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.knowledge_graph._exceptions import KGIntegrityError
from src.knowledge_graph._models import IntegrityCheck, IntegrityReport
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.knowledge_graph._client import Neo4jClient

_log = get_logger(__name__)


class IntegrityChecker:
    """Run post-ingestion integrity checks on the knowledge graph."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    async def check_all(self) -> IntegrityReport:
        """Run all integrity checks and return a combined report."""
        checks = [
            await self.check_section_versions(),
            await self.check_repealed_consistency(),
            await self.check_overrule_hierarchy(),
            await self.check_version_date_overlap(),
        ]
        passed = all(c.passed for c in checks)
        report = IntegrityReport(passed=passed, checks=checks)
        _log.info(
            "integrity_check_complete",
            passed=passed,
            total_checks=len(checks),
            violations=sum(len(c.violations) for c in checks),
        )
        return report

    async def check_section_versions(self) -> IntegrityCheck:
        """Every Section must have at least one SectionVersion."""
        cypher = """
        MATCH (s:Section)
        WHERE NOT (s)-[:HAS_VERSION]->(:SectionVersion)
        RETURN s.parent_act AS act, s.number AS section
        LIMIT 100
        """
        try:
            results = await self._client.run_query(cypher)
            violations = [f"{r['act']}:s.{r['section']} has no SectionVersion" for r in results]
            return IntegrityCheck(
                name="section_versions",
                passed=len(violations) == 0,
                violations=violations,
            )
        except Exception as exc:
            msg = f"section_versions check failed: {exc}"
            raise KGIntegrityError(msg) from exc

    async def check_repealed_consistency(self) -> IntegrityCheck:
        """If an Act is repealed, all its sections must have is_in_force=false."""
        cypher = """
        MATCH (a:Act {status: 'repealed'})-[:CONTAINS]->(s:Section)
        WHERE s.is_in_force = true
        RETURN a.name AS act, s.number AS section
        LIMIT 100
        """
        try:
            results = await self._client.run_query(cypher)
            violations = [
                f"{r['act']}:s.{r['section']} is_in_force=true but act is repealed" for r in results
            ]
            return IntegrityCheck(
                name="repealed_consistency",
                passed=len(violations) == 0,
                violations=violations,
            )
        except Exception as exc:
            msg = f"repealed_consistency check failed: {exc}"
            raise KGIntegrityError(msg) from exc

    async def check_overrule_hierarchy(self) -> IntegrityCheck:
        """OVERRULES requires the overruling judgment to be from equal or higher court."""
        cypher = """
        MATCH (overruler:Judgment)-[:OVERRULES]->(overruled:Judgment)
        WHERE overruler.court_level > overruled.court_level
        RETURN overruler.citation AS overruler, overruled.citation AS overruled,
               overruler.court_level AS overruler_level,
               overruled.court_level AS overruled_level
        LIMIT 100
        """
        try:
            results = await self._client.run_query(cypher)
            violations = [
                f"{r['overruler']} (level {r['overruler_level']}) overrules "
                f"{r['overruled']} (level {r['overruled_level']})"
                for r in results
            ]
            return IntegrityCheck(
                name="overrule_hierarchy",
                passed=len(violations) == 0,
                violations=violations,
            )
        except Exception as exc:
            msg = f"overrule_hierarchy check failed: {exc}"
            raise KGIntegrityError(msg) from exc

    async def check_version_date_overlap(self) -> IntegrityCheck:
        """SectionVersion date ranges must not overlap for the same section."""
        cypher = """
        MATCH (s:Section)-[:HAS_VERSION]->(v1:SectionVersion),
              (s)-[:HAS_VERSION]->(v2:SectionVersion)
        WHERE v1.version_id < v2.version_id
          AND v1.effective_from IS NOT NULL
          AND v2.effective_from IS NOT NULL
          AND v1.effective_from < v2.effective_from
          AND (v1.effective_until IS NULL OR v1.effective_until > v2.effective_from)
        RETURN s.parent_act AS act, s.number AS section,
               v1.version_id AS v1_id, v2.version_id AS v2_id,
               v1.effective_from AS v1_from, v1.effective_until AS v1_until,
               v2.effective_from AS v2_from
        LIMIT 100
        """
        try:
            results = await self._client.run_query(cypher)
            violations = [
                f"{r['act']}:s.{r['section']} versions overlap: "
                f"{r['v1_id']} ({r['v1_from']}-{r['v1_until']}) and "
                f"{r['v2_id']} (from {r['v2_from']})"
                for r in results
            ]
            return IntegrityCheck(
                name="version_date_overlap",
                passed=len(violations) == 0,
                violations=violations,
            )
        except Exception as exc:
            msg = f"version_date_overlap check failed: {exc}"
            raise KGIntegrityError(msg) from exc
