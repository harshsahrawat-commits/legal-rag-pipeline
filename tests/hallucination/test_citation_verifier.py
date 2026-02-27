"""Tests for Layer 1: Citation Verifier."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.hallucination._citation_verifier import CitationVerifier
from src.hallucination._models import (
    CitationStatus,
    HallucinationSettings,
)


@pytest.fixture()
def settings() -> HallucinationSettings:
    return HallucinationSettings()


@pytest.fixture()
def mock_qb() -> AsyncMock:
    """A mock QueryBuilder."""
    qb = AsyncMock()
    qb.node_exists = AsyncMock(return_value=True)
    return qb


class TestCitationVerifierWithKG:
    async def test_section_verified(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.node_exists.return_value = True
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("Section 420 of the Indian Penal Code.")
        section_results = [r for r in results if r.citation.section == "420"]
        assert len(section_results) >= 1
        assert section_results[0].status == CitationStatus.VERIFIED
        assert section_results[0].kg_node_label == "Section"

    async def test_section_not_found(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.node_exists.return_value = False
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("Section 999 of the Indian Penal Code.")
        section_results = [r for r in results if r.citation.section == "999"]
        assert len(section_results) >= 1
        assert section_results[0].status == CitationStatus.NOT_FOUND

    async def test_case_verified(self, settings: HallucinationSettings, mock_qb: AsyncMock) -> None:
        mock_qb.node_exists.return_value = True
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("In AIR 2023 SC 1234, the court held...")
        case_results = [r for r in results if r.citation.case_citation is not None]
        assert len(case_results) == 1
        assert case_results[0].status == CitationStatus.VERIFIED
        assert case_results[0].kg_node_label == "Judgment"

    async def test_case_not_found(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.node_exists.return_value = False
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("See AIR 1999 SC 0000.")
        case_results = [r for r in results if r.citation.case_citation is not None]
        assert len(case_results) == 1
        assert case_results[0].status == CitationStatus.NOT_FOUND

    async def test_article_verified(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.node_exists.return_value = True
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("Article 21 of the Constitution.")
        article_results = [r for r in results if r.citation.article is not None]
        assert len(article_results) >= 1
        assert article_results[0].status == CitationStatus.VERIFIED

    async def test_notification_is_kg_unavailable(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("GSR 1234(E) was notified.")
        notif_results = [r for r in results if r.citation.notification_ref is not None]
        assert len(notif_results) == 1
        assert notif_results[0].status == CitationStatus.KG_UNAVAILABLE

    async def test_circular_is_kg_unavailable(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("RBI/2023-24/45 was issued.")
        circ_results = [r for r in results if r.citation.circular_ref is not None]
        assert len(circ_results) == 1
        assert circ_results[0].status == CitationStatus.KG_UNAVAILABLE

    async def test_multiple_citations(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        # First call (section) returns True, second (case) returns False
        mock_qb.node_exists.side_effect = [True, True, False]
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response(
            "Section 420 of the Indian Penal Code. Article 21. AIR 2023 SC 1234."
        )
        assert len(results) >= 3

    async def test_kg_exception_graceful(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.node_exists.side_effect = Exception("connection timeout")
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("Section 420 of the Indian Penal Code.")
        section_results = [r for r in results if r.citation.section == "420"]
        assert len(section_results) >= 1
        assert section_results[0].status == CitationStatus.KG_UNAVAILABLE

    async def test_duplicate_citations(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        mock_qb.node_exists.return_value = True
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        # Same citation repeated — extractor deduplicates by span
        results = await verifier.verify_response(
            "Section 420 of the Indian Penal Code is important. "
        )
        section_results = [r for r in results if r.citation.section == "420"]
        assert len(section_results) >= 1


class TestCitationVerifierWithoutKG:
    async def test_all_kg_unavailable(self, settings: HallucinationSettings) -> None:
        verifier = CitationVerifier(settings, query_builder=None)
        results = await verifier.verify_response(
            "Section 420 of the Indian Penal Code. AIR 2023 SC 1234."
        )
        assert all(r.status == CitationStatus.KG_UNAVAILABLE for r in results)

    async def test_empty_response(self, settings: HallucinationSettings) -> None:
        verifier = CitationVerifier(settings, query_builder=None)
        results = await verifier.verify_response("")
        assert results == []

    async def test_no_citations(self, settings: HallucinationSettings) -> None:
        verifier = CitationVerifier(settings, query_builder=None)
        results = await verifier.verify_response("The weather is nice today.")
        assert results == []


class TestCitationVerifierEdgeCases:
    async def test_section_missing_act(
        self, settings: HallucinationSettings, mock_qb: AsyncMock
    ) -> None:
        # S.302 abbreviation — act should be resolved
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        mock_qb.node_exists.return_value = True
        results = await verifier.verify_response("S. 302 IPC.")
        assert len(results) >= 1

    async def test_scc_citation(self, settings: HallucinationSettings, mock_qb: AsyncMock) -> None:
        mock_qb.node_exists.return_value = True
        verifier = CitationVerifier(settings, query_builder=mock_qb)
        results = await verifier.verify_response("(2023) 5 SCC 678.")
        case_results = [r for r in results if r.citation.case_citation is not None]
        assert len(case_results) == 1
        assert case_results[0].status == CitationStatus.VERIFIED
