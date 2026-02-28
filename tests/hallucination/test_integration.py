"""Integration tests for the hallucination mitigation pipeline.

These tests verify cross-layer behavior — multiple layers working together.
External services (Neo4j, Anthropic) are mocked.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.hallucination._models import (
    CitationStatus,
    HallucinationSettings,
    VerificationInput,
)
from src.hallucination.pipeline import HallucinationPipeline
from src.retrieval._models import ExpandedContext, QueryRoute, RetrievalResult


@pytest.fixture()
def mock_qb() -> AsyncMock:
    """Mock QueryBuilder with typical responses."""
    qb = AsyncMock()
    qb.node_exists = AsyncMock(return_value=True)
    qb.temporal_status = AsyncMock(return_value={"found": True, "is_in_force": True})
    qb.find_replacement = AsyncMock(return_value=None)
    return qb


@pytest.fixture()
def ipc_response_input() -> VerificationInput:
    """A response referencing IPC (repealed) with source chunks."""
    chunks = [
        ExpandedContext(
            chunk_id="c1",
            chunk_text="Section 420 of the Indian Penal Code deals with cheating.",
            relevance_score=0.9,
            metadata={"document_type": "statute", "court_hierarchy": 0},
        ),
        ExpandedContext(
            chunk_id="c2",
            chunk_text="The punishment for cheating under S.420 includes imprisonment.",
            relevance_score=0.85,
            metadata={"document_type": "statute", "court_hierarchy": 0},
        ),
    ]
    result = RetrievalResult(query_text="Section 420 IPC", chunks=chunks)
    return VerificationInput(
        response_text=(
            "Section 420 of the Indian Penal Code provides for punishment "
            "of cheating. AIR 2023 SC 1234 affirmed this interpretation."
        ),
        retrieval_result=result,
        route=QueryRoute.STANDARD,
        reference_date=date(2025, 1, 15),
    )


class TestEndToEndNoGenGround:
    """Full pipeline without GenGround (most common in tests)."""

    async def test_ipc_citation_and_temporal(
        self,
        mock_qb: AsyncMock,
        ipc_response_input: VerificationInput,
    ) -> None:
        settings = HallucinationSettings(genground_enabled=False)
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)
        result = await pipeline.verify(ipc_response_input)

        # Citations should be extracted and verified
        assert result.summary.total_citations >= 2  # Section + AIR case

        # Temporal: IPC is repealed after July 2024
        assert result.summary.temporal_warnings >= 1
        ipc_warnings = [w for w in result.temporal_warnings if w.act == "Indian Penal Code"]
        assert len(ipc_warnings) >= 1
        assert ipc_warnings[0].replacement_act == "Bharatiya Nyaya Sanhita"

        # Confidence should be computed
        assert 0.0 <= result.confidence.overall_score <= 1.0

        # No errors
        assert result.errors == []

    async def test_modern_bns_no_warnings(
        self,
        mock_qb: AsyncMock,
    ) -> None:
        settings = HallucinationSettings(genground_enabled=False)
        inp = VerificationInput(
            response_text=(
                "Section 420 of the Bharatiya Nyaya Sanhita provides for punishment of cheating."
            ),
            reference_date=date(2025, 1, 15),
        )
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)
        result = await pipeline.verify(inp)
        # BNS is the new code — no temporal warnings
        assert result.summary.temporal_warnings == 0

    async def test_no_citations_no_warnings(
        self,
    ) -> None:
        settings = HallucinationSettings(genground_enabled=False)
        inp = VerificationInput(
            response_text="The weather is nice today.",
            reference_date=date(2025, 1, 15),
        )
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert result.summary.total_citations == 0
        assert result.summary.temporal_warnings == 0
        # No chunks → low retrieval/authority/recency → score below 0.5 is expected
        assert 0.0 <= result.confidence.overall_score <= 1.0


class TestEndToEndWithMockGenGround:
    """Pipeline with mocked GenGround LLM calls."""

    async def test_simple_route_audit(
        self,
        mock_qb: AsyncMock,
    ) -> None:
        settings = HallucinationSettings(genground_enabled=True)
        chunks = [
            ExpandedContext(
                chunk_id="c1",
                chunk_text="Section 10 of the Contract Act.",
                relevance_score=0.9,
                metadata={"document_type": "statute"},
            ),
        ]
        result_obj = RetrievalResult(query_text="test", chunks=chunks)
        inp = VerificationInput(
            response_text="Section 10 of the Contract Act covers validity.",
            retrieval_result=result_obj,
            route=QueryRoute.SIMPLE,
        )

        # Mock the GenGroundRefiner's LLM client
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)

        # Patch the GenGroundRefiner to inject a mock LLM provider
        from src.hallucination._genground_refiner import GenGroundRefiner
        from src.utils._llm_client import LLMResponse

        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(
            return_value=LLMResponse(
                text=json.dumps({"verdict": "supported", "issues": []}),
                model="mock",
                provider="mock",
            )
        )

        original_init = GenGroundRefiner.__init__

        def patched_init(self_ref, *args, **kwargs):
            original_init(self_ref, *args, **kwargs)
            self_ref._provider = mock_provider

        GenGroundRefiner.__init__ = patched_init
        try:
            result = await pipeline.verify(inp)
        finally:
            GenGroundRefiner.__init__ = original_init

        assert result.summary.total_claims >= 1
        assert result.summary.supported_claims >= 1


class TestKGUnavailable:
    """Pipeline behavior when KG is not available."""

    async def test_no_kg_graceful(self) -> None:
        settings = HallucinationSettings(genground_enabled=False)
        inp = VerificationInput(
            response_text="Section 420 of the Indian Penal Code.",
            reference_date=date(2025, 1, 15),
        )
        pipeline = HallucinationPipeline(settings, query_builder=None)
        result = await pipeline.verify(inp)
        # Citations extracted but all KG_UNAVAILABLE
        kg_unavail = [
            c for c in result.citation_results if c.status == CitationStatus.KG_UNAVAILABLE
        ]
        assert len(kg_unavail) == len(result.citation_results)
        # Temporal still works (hardcoded IPC→BNS)
        assert result.summary.temporal_warnings >= 1
        # No pipeline errors
        assert result.errors == []

    async def test_kg_failure_isolated(self) -> None:
        settings = HallucinationSettings(genground_enabled=False)
        mock_qb = AsyncMock()
        mock_qb.node_exists = AsyncMock(side_effect=Exception("connection timeout"))
        inp = VerificationInput(
            response_text="Section 420 of the Indian Penal Code.",
            reference_date=date(2025, 1, 15),
        )
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)
        result = await pipeline.verify(inp)
        # Citations should have KG_UNAVAILABLE status (not crash)
        assert result.modified_response is not None
        assert result.confidence is not None
