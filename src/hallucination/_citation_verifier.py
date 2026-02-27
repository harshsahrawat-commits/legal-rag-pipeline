"""Layer 1: Citation Verification against the Knowledge Graph.

Maps extracted citations to KG node lookups, verifying existence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.hallucination._citation_extractor import extract_citations
from src.hallucination._models import (
    CitationResult,
    CitationStatus,
    CitationType,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.hallucination._models import ExtractedCitation, HallucinationSettings
    from src.knowledge_graph._queries import QueryBuilder

_log = get_logger(__name__)


class CitationVerifier:
    """Verify extracted citations against the knowledge graph."""

    def __init__(
        self,
        settings: HallucinationSettings,
        query_builder: QueryBuilder | None = None,
    ) -> None:
        self._settings = settings
        self._qb = query_builder

    async def verify_response(self, response_text: str) -> list[CitationResult]:
        """Extract and verify all citations in the response text."""
        citations = extract_citations(response_text)
        if not citations:
            return []

        results: list[CitationResult] = []
        for citation in citations:
            try:
                result = await self._verify_one(citation)
            except Exception as exc:
                _log.warning(
                    "citation_verification_error",
                    citation=citation.text,
                    error=str(exc),
                )
                result = CitationResult(
                    citation=citation,
                    status=CitationStatus.KG_UNAVAILABLE,
                    error=str(exc),
                )
            results.append(result)
        return results

    async def _verify_one(self, citation: ExtractedCitation) -> CitationResult:
        """Verify a single citation against the KG."""
        if self._qb is None:
            return CitationResult(
                citation=citation,
                status=CitationStatus.KG_UNAVAILABLE,
                error="Knowledge graph not available",
            )

        if citation.citation_type == CitationType.SECTION_REF:
            return await self._verify_section(citation)
        if citation.citation_type == CitationType.ARTICLE_REF:
            return await self._verify_article(citation)
        if citation.citation_type == CitationType.CASE_CITATION:
            return await self._verify_case(citation)
        # Notifications and circulars are not in KG schema
        return CitationResult(
            citation=citation,
            status=CitationStatus.KG_UNAVAILABLE,
            error=f"Citation type {citation.citation_type} not in KG schema",
        )

    async def _verify_section(self, citation: ExtractedCitation) -> CitationResult:
        """Verify a section reference exists in the KG."""
        if not citation.section or not citation.act:
            return CitationResult(
                citation=citation,
                status=CitationStatus.NOT_FOUND,
                error="Section or act missing from citation",
            )

        try:
            exists = await self._qb.node_exists(
                "Section",
                {"number": citation.section, "parent_act": citation.act},
            )
        except Exception as exc:
            return CitationResult(
                citation=citation,
                status=CitationStatus.KG_UNAVAILABLE,
                error=str(exc),
            )

        if exists:
            return CitationResult(
                citation=citation,
                status=CitationStatus.VERIFIED,
                kg_node_label="Section",
            )
        return CitationResult(
            citation=citation,
            status=CitationStatus.NOT_FOUND,
            error=f"Section {citation.section} of {citation.act} not found in KG",
        )

    async def _verify_article(self, citation: ExtractedCitation) -> CitationResult:
        """Verify an article reference exists in the KG."""
        if not citation.article:
            return CitationResult(
                citation=citation,
                status=CitationStatus.NOT_FOUND,
                error="Article number missing",
            )

        try:
            exists = await self._qb.node_exists(
                "Section",
                {
                    "number": citation.article,
                    "parent_act": "Constitution of India",
                },
            )
        except Exception as exc:
            return CitationResult(
                citation=citation,
                status=CitationStatus.KG_UNAVAILABLE,
                error=str(exc),
            )

        if exists:
            return CitationResult(
                citation=citation,
                status=CitationStatus.VERIFIED,
                kg_node_label="Section",
            )
        return CitationResult(
            citation=citation,
            status=CitationStatus.NOT_FOUND,
            error=f"Article {citation.article} not found in KG",
        )

    async def _verify_case(self, citation: ExtractedCitation) -> CitationResult:
        """Verify a case citation exists in the KG."""
        if not citation.case_citation:
            return CitationResult(
                citation=citation,
                status=CitationStatus.NOT_FOUND,
                error="Case citation string missing",
            )

        try:
            exists = await self._qb.node_exists(
                "Judgment",
                {"citation": citation.case_citation},
            )
        except Exception as exc:
            return CitationResult(
                citation=citation,
                status=CitationStatus.KG_UNAVAILABLE,
                error=str(exc),
            )

        if exists:
            return CitationResult(
                citation=citation,
                status=CitationStatus.VERIFIED,
                kg_node_label="Judgment",
            )
        return CitationResult(
            citation=citation,
            status=CitationStatus.NOT_FOUND,
            error=f"Judgment {citation.case_citation} not found in KG",
        )
