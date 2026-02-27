"""Tests for the EvaluationPipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from src.evaluation._models import (
    EvaluationResult,
    EvaluationSettings,
    LatencyAggregate,
    LegalMetricsAggregate,
    QIMetrics,
    RagasAggregateResult,
)
from src.evaluation.pipeline import EvaluationPipeline

if TYPE_CHECKING:
    from src.evaluation._models import EvaluationInput


@pytest.fixture()
def settings() -> EvaluationSettings:
    return EvaluationSettings(ragas_enabled=False, human_eval_enabled=False)


@pytest.fixture()
def pipeline(settings: EvaluationSettings) -> EvaluationPipeline:
    return EvaluationPipeline(settings)


class TestEvaluationPipelineInit:
    def test_stores_settings(self, settings: EvaluationSettings) -> None:
        pipe = EvaluationPipeline(settings)
        assert pipe._settings is settings


class TestEvaluationPipelineEvaluate:
    @pytest.mark.asyncio
    async def test_returns_evaluation_result(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_queries_evaluated_count(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert result.queries_evaluated == len(sample_evaluation_inputs)

    @pytest.mark.asyncio
    async def test_empty_inputs(self, pipeline: EvaluationPipeline) -> None:
        result = await pipeline.evaluate([])
        assert result.queries_evaluated == 0

    @pytest.mark.asyncio
    async def test_has_timings(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert "legal_metrics_ms" in result.timings
        assert "latency_ms" in result.timings
        assert "qi_metrics_ms" in result.timings

    @pytest.mark.asyncio
    async def test_has_timestamps(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_elapsed_ms_positive(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert result.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_legal_metrics_populated(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert isinstance(result.legal_metrics, LegalMetricsAggregate)

    @pytest.mark.asyncio
    async def test_latency_populated(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert isinstance(result.latency, LatencyAggregate)

    @pytest.mark.asyncio
    async def test_qi_populated(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert isinstance(result.qi_metrics, QIMetrics)

    @pytest.mark.asyncio
    async def test_ragas_skipped_when_disabled(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        # RAGAS disabled → default empty aggregate
        assert result.ragas.queries_evaluated == 0
        assert "ragas_ms" not in result.timings

    @pytest.mark.asyncio
    async def test_human_eval_skipped_when_disabled(
        self, pipeline: EvaluationPipeline, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        result = await pipeline.evaluate(sample_evaluation_inputs)
        assert result.human_eval.total_evaluations == 0


class TestPipelineErrorIsolation:
    @pytest.mark.asyncio
    async def test_legal_metrics_failure_isolated(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)

        with patch(
            "src.evaluation.pipeline.EvaluationPipeline._run_legal_metrics",
            side_effect=RuntimeError("legal boom"),
        ):
            # The error is raised inside the patched method, but the pipeline
            # catches exceptions inside each _run_* method. Let's test the
            # internal error isolation instead.
            pass

        # Test actual internal isolation by patching the evaluator
        with patch(
            "src.evaluation._legal_metrics.LegalMetricsEvaluator.evaluate",
            side_effect=RuntimeError("legal boom"),
        ):
            result = await pipe.evaluate(sample_evaluation_inputs)
            assert any("Legal metrics failed" in e for e in result.errors)
            # Other layers still ran
            assert isinstance(result.latency, LatencyAggregate)
            assert isinstance(result.qi_metrics, QIMetrics)

    @pytest.mark.asyncio
    async def test_latency_failure_isolated(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)

        with patch(
            "src.evaluation._latency_metrics.LatencyEvaluator.evaluate",
            side_effect=RuntimeError("latency boom"),
        ):
            result = await pipe.evaluate(sample_evaluation_inputs)
            assert any("Latency metrics failed" in e for e in result.errors)
            assert isinstance(result.legal_metrics, LegalMetricsAggregate)

    @pytest.mark.asyncio
    async def test_qi_failure_isolated(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)

        with patch(
            "src.evaluation._qi_metrics.QIMetricsEvaluator.evaluate",
            side_effect=RuntimeError("qi boom"),
        ):
            result = await pipe.evaluate(sample_evaluation_inputs)
            assert any("QI metrics failed" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_ragas_failure_isolated(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(ragas_enabled=True)
        pipe = EvaluationPipeline(settings)

        with patch(
            "src.evaluation._ragas_evaluator.RagasEvaluator.evaluate",
            new_callable=AsyncMock,
            side_effect=RuntimeError("ragas boom"),
        ):
            result = await pipe.evaluate(sample_evaluation_inputs)
            assert any("RAGAS evaluation failed" in e for e in result.errors)
            # Other layers still work
            assert isinstance(result.legal_metrics, LegalMetricsAggregate)

    @pytest.mark.asyncio
    async def test_all_layers_fail_still_returns_result(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)

        with (
            patch(
                "src.evaluation._legal_metrics.LegalMetricsEvaluator.evaluate",
                side_effect=RuntimeError("boom1"),
            ),
            patch(
                "src.evaluation._latency_metrics.LatencyEvaluator.evaluate",
                side_effect=RuntimeError("boom2"),
            ),
            patch(
                "src.evaluation._qi_metrics.QIMetricsEvaluator.evaluate",
                side_effect=RuntimeError("boom3"),
            ),
        ):
            result = await pipe.evaluate(sample_evaluation_inputs)
            assert len(result.errors) == 3
            assert result.queries_evaluated == len(sample_evaluation_inputs)


class TestPipelineRagasEnabled:
    @pytest.mark.asyncio
    async def test_ragas_runs_when_enabled(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(ragas_enabled=True)
        pipe = EvaluationPipeline(settings)

        mock_result = RagasAggregateResult(
            context_recall=0.95,
            faithfulness=0.98,
            queries_evaluated=3,
        )

        with patch(
            "src.evaluation._ragas_evaluator.RagasEvaluator.evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await pipe.evaluate(sample_evaluation_inputs)
            assert result.ragas.context_recall == 0.95
            assert result.ragas.faithfulness == 0.98
            assert "ragas_ms" in result.timings


class TestPipelineHumanEvalEnabled:
    @pytest.mark.asyncio
    async def test_human_eval_runs_when_enabled_with_scoresheets(
        self, sample_evaluation_inputs: list[EvaluationInput], tmp_path
    ) -> None:
        import json

        # Create a scoresheet
        scoresheet = tmp_path / "scores.json"
        scoresheet.write_text(
            json.dumps(
                [
                    {
                        "query_id": "q1",
                        "evaluator_id": "eval1",
                        "accuracy": 5,
                        "completeness": 4,
                        "recency": 3,
                        "usefulness": 4,
                    }
                ]
            ),
            encoding="utf-8",
        )

        settings = EvaluationSettings(
            ragas_enabled=False,
            human_eval_enabled=True,
            scoresheets_dir=str(tmp_path),
        )
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate(sample_evaluation_inputs)
        assert result.human_eval.total_evaluations == 1
        assert result.human_eval.avg_accuracy == 5.0

    @pytest.mark.asyncio
    async def test_human_eval_no_scoresheets_dir(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(
            ragas_enabled=False,
            human_eval_enabled=True,
            scoresheets_dir="/nonexistent/path",
        )
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate(sample_evaluation_inputs)
        assert result.human_eval.total_evaluations == 0

    @pytest.mark.asyncio
    async def test_legal_metrics_disabled(
        self, sample_evaluation_inputs: list[EvaluationInput]
    ) -> None:
        settings = EvaluationSettings(
            ragas_enabled=False,
            legal_metrics_enabled=False,
        )
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate(sample_evaluation_inputs)
        # Should return defaults when disabled
        assert result.legal_metrics.citation_accuracy == 0.0
