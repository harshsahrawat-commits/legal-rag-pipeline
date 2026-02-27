"""Tests for HallucinationPipeline orchestrator."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from src.hallucination._models import (
    HallucinationSettings,
    VerificationInput,
)
from src.hallucination.pipeline import HallucinationPipeline
from src.retrieval._models import ExpandedContext, QueryRoute, RetrievalResult


@pytest.fixture()
def settings() -> HallucinationSettings:
    return HallucinationSettings(genground_enabled=False)


@pytest.fixture()
def genground_settings() -> HallucinationSettings:
    return HallucinationSettings(genground_enabled=True)


@pytest.fixture()
def sample_chunks() -> list[ExpandedContext]:
    return [
        ExpandedContext(
            chunk_id="c1",
            chunk_text="Section 420 deals with cheating.",
            relevance_score=0.9,
            metadata={"document_type": "statute", "court_hierarchy": 0},
        ),
    ]


@pytest.fixture()
def sample_input(sample_chunks: list[ExpandedContext]) -> VerificationInput:
    result = RetrievalResult(
        query_text="test",
        chunks=sample_chunks,
    )
    return VerificationInput(
        response_text="Section 420 of the Indian Penal Code provides for punishment.",
        retrieval_result=result,
        route=QueryRoute.STANDARD,
        reference_date=date(2025, 1, 15),
    )


class TestPipelineBasic:
    async def test_returns_verified_response(
        self,
        settings: HallucinationSettings,
        sample_input: VerificationInput,
    ) -> None:
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(sample_input)
        assert result.original_response == sample_input.response_text
        assert result.modified_response is not None
        assert result.elapsed_ms >= 0.0

    async def test_no_retrieval_result(
        self,
        settings: HallucinationSettings,
    ) -> None:
        inp = VerificationInput(response_text="Some legal response.")
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert result.modified_response == "Some legal response."
        assert result.errors == []

    async def test_empty_response(
        self,
        settings: HallucinationSettings,
    ) -> None:
        inp = VerificationInput(response_text="")
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert result.modified_response == ""
        assert result.summary.total_citations == 0

    async def test_default_route(
        self,
        settings: HallucinationSettings,
    ) -> None:
        """No route provided → defaults to STANDARD."""
        inp = VerificationInput(response_text="test")
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert result.confidence is not None


class TestPipelineCitationLayer:
    async def test_citation_results_populated(
        self,
        settings: HallucinationSettings,
        sample_input: VerificationInput,
    ) -> None:
        mock_qb = AsyncMock()
        mock_qb.node_exists = AsyncMock(return_value=True)
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)
        result = await pipeline.verify(sample_input)
        assert len(result.citation_results) >= 1
        assert result.summary.total_citations >= 1

    async def test_citation_layer_failure_isolated(
        self,
        settings: HallucinationSettings,
        sample_input: VerificationInput,
    ) -> None:
        mock_qb = AsyncMock()
        mock_qb.node_exists = AsyncMock(side_effect=Exception("KG down"))
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)
        result = await pipeline.verify(sample_input)
        # Pipeline doesn't crash; citation results have KG_UNAVAILABLE
        assert result.modified_response is not None


class TestPipelineTemporalLayer:
    async def test_temporal_warnings_populated(
        self,
        settings: HallucinationSettings,
    ) -> None:
        inp = VerificationInput(
            response_text="Section 420 of the Indian Penal Code is about cheating.",
            reference_date=date(2025, 1, 15),
        )
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert len(result.temporal_warnings) >= 1
        assert result.summary.temporal_warnings >= 1

    async def test_no_temporal_warnings_before_repeal(
        self,
        settings: HallucinationSettings,
    ) -> None:
        inp = VerificationInput(
            response_text="Section 420 of the Indian Penal Code is about cheating.",
            reference_date=date(2024, 6, 1),
        )
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert len(result.temporal_warnings) == 0


class TestPipelineGenGroundLayer:
    async def test_genground_disabled(
        self,
        settings: HallucinationSettings,
        sample_input: VerificationInput,
    ) -> None:
        """GenGround disabled → no claim verdicts, no LLM calls."""
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(sample_input)
        assert result.claim_verdicts == []
        assert result.summary.total_claims == 0

    async def test_genground_enabled_mock(
        self,
        genground_settings: HallucinationSettings,
        sample_input: VerificationInput,
    ) -> None:
        """GenGround enabled but anthropic not installed → error isolated."""
        pipeline = HallucinationPipeline(genground_settings)
        with patch.dict("sys.modules", {"anthropic": None}):
            result = await pipeline.verify(sample_input)
        # Should record error but not crash
        assert len(result.errors) >= 1
        assert "GenGround" in result.errors[0]


class TestPipelineConfidenceLayer:
    async def test_confidence_always_present(
        self,
        settings: HallucinationSettings,
        sample_input: VerificationInput,
    ) -> None:
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(sample_input)
        assert 0.0 <= result.confidence.overall_score <= 1.0
        assert result.confidence.explanation is not None

    async def test_high_quality_chunks_boost_confidence(
        self,
        settings: HallucinationSettings,
    ) -> None:
        chunks = [
            ExpandedContext(
                chunk_id="c1",
                chunk_text="Definitive legal text.",
                relevance_score=0.95,
                metadata={"document_type": "statute"},
            ),
        ]
        result_obj = RetrievalResult(
            query_text="test",
            chunks=chunks,
        )
        inp = VerificationInput(
            response_text="test response",
            retrieval_result=result_obj,
            route=QueryRoute.SIMPLE,
        )
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert result.confidence.overall_score >= 0.5


class TestPipelineSummary:
    async def test_summary_counts(
        self,
        settings: HallucinationSettings,
    ) -> None:
        mock_qb = AsyncMock()
        mock_qb.node_exists = AsyncMock(side_effect=[True, False])
        inp = VerificationInput(
            response_text="Section 420 of the Indian Penal Code. AIR 2023 SC 1234.",
            reference_date=date(2025, 1, 15),
        )
        pipeline = HallucinationPipeline(settings, query_builder=mock_qb)
        result = await pipeline.verify(inp)
        assert result.summary.total_citations >= 2

    async def test_errors_list_empty_on_success(
        self,
        settings: HallucinationSettings,
    ) -> None:
        inp = VerificationInput(response_text="No citations here.")
        pipeline = HallucinationPipeline(settings)
        result = await pipeline.verify(inp)
        assert result.errors == []
