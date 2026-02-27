"""EvaluationPipeline: orchestrates all 5 evaluation layers.

Order: Legal Metrics → Latency → QI → RAGAS (optional) → Human Eval (optional).
Each layer is error-isolated — one failure does not crash the pipeline.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.evaluation._models import EvaluationResult
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.evaluation._models import (
        EvaluationInput,
        EvaluationSettings,
        HumanEvalAggregate,
        LatencyAggregate,
        LegalMetricsAggregate,
        QIMetrics,
        RagasAggregateResult,
    )

_log = get_logger(__name__)


class EvaluationPipeline:
    """Orchestrate 5-layer evaluation.

    Layers:
        1. Legal Metrics — citation, temporal, completeness, cross-ref
        2. Latency Metrics — per-route TTFT compliance
        3. QI Metrics — cache, routing, GenGround, parent, FLARE
        4. RAGAS Metrics — context recall/precision, faithfulness, relevancy
        5. Human Evaluation — import scoresheets (if enabled + available)
    """

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    async def evaluate(self, inputs: list[EvaluationInput]) -> EvaluationResult:
        """Run all evaluation layers and return an EvaluationResult."""
        started_at = datetime.now(UTC)
        errors: list[str] = []
        timings: dict[str, float] = {}

        # Layer 1: Legal Metrics
        legal = self._run_legal_metrics(inputs, errors, timings)

        # Layer 2: Latency Metrics
        latency = self._run_latency_metrics(inputs, errors, timings)

        # Layer 3: QI Metrics
        qi = self._run_qi_metrics(inputs, errors, timings)

        # Layer 4: RAGAS Metrics (optional, async)
        ragas = await self._run_ragas(inputs, errors, timings)

        # Layer 5: Human Evaluation (optional)
        human = self._run_human_eval(errors, timings)

        finished_at = datetime.now(UTC)

        result = EvaluationResult(
            ragas=ragas,
            legal_metrics=legal,
            latency=latency,
            qi_metrics=qi,
            human_eval=human,
            queries_evaluated=len(inputs),
            timings=timings,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

        _log.info(
            "evaluation_pipeline_complete",
            queries=len(inputs),
            all_targets_met=result.all_targets_met,
            elapsed_ms=result.elapsed_ms,
            errors=len(errors),
        )

        return result

    def _run_legal_metrics(
        self,
        inputs: list[EvaluationInput],
        errors: list[str],
        timings: dict[str, float],
    ) -> LegalMetricsAggregate:
        """Layer 1: Custom legal metrics."""
        from src.evaluation._models import LegalMetricsAggregate

        if not self._settings.legal_metrics_enabled:
            return LegalMetricsAggregate()

        t0 = time.perf_counter()
        try:
            from src.evaluation._legal_metrics import LegalMetricsEvaluator

            evaluator = LegalMetricsEvaluator(self._settings)
            result = evaluator.evaluate(inputs)
            timings["legal_metrics_ms"] = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:
            _log.warning("layer1_legal_metrics_failed", error=str(exc))
            errors.append(f"Legal metrics failed: {exc}")
            timings["legal_metrics_ms"] = (time.perf_counter() - t0) * 1000
            return LegalMetricsAggregate()

    def _run_latency_metrics(
        self,
        inputs: list[EvaluationInput],
        errors: list[str],
        timings: dict[str, float],
    ) -> LatencyAggregate:
        """Layer 2: Latency compliance."""
        from src.evaluation._models import LatencyAggregate

        t0 = time.perf_counter()
        try:
            from src.evaluation._latency_metrics import LatencyEvaluator

            evaluator = LatencyEvaluator(self._settings)
            result = evaluator.evaluate(inputs)
            timings["latency_ms"] = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:
            _log.warning("layer2_latency_metrics_failed", error=str(exc))
            errors.append(f"Latency metrics failed: {exc}")
            timings["latency_ms"] = (time.perf_counter() - t0) * 1000
            return LatencyAggregate()

    def _run_qi_metrics(
        self,
        inputs: list[EvaluationInput],
        errors: list[str],
        timings: dict[str, float],
    ) -> QIMetrics:
        """Layer 3: Query Intelligence analytics."""
        from src.evaluation._models import QIMetrics

        t0 = time.perf_counter()
        try:
            from src.evaluation._qi_metrics import QIMetricsEvaluator

            evaluator = QIMetricsEvaluator(self._settings)
            result = evaluator.evaluate(inputs)
            timings["qi_metrics_ms"] = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:
            _log.warning("layer3_qi_metrics_failed", error=str(exc))
            errors.append(f"QI metrics failed: {exc}")
            timings["qi_metrics_ms"] = (time.perf_counter() - t0) * 1000
            return QIMetrics()

    async def _run_ragas(
        self,
        inputs: list[EvaluationInput],
        errors: list[str],
        timings: dict[str, float],
    ) -> RagasAggregateResult:
        """Layer 4: RAGAS metrics (optional, async)."""
        from src.evaluation._models import RagasAggregateResult

        if not self._settings.ragas_enabled:
            return RagasAggregateResult()

        t0 = time.perf_counter()
        try:
            from src.evaluation._ragas_evaluator import RagasEvaluator

            evaluator = RagasEvaluator(self._settings)
            result = await evaluator.evaluate(inputs)
            timings["ragas_ms"] = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:
            _log.warning("layer4_ragas_failed", error=str(exc))
            errors.append(f"RAGAS evaluation failed: {exc}")
            timings["ragas_ms"] = (time.perf_counter() - t0) * 1000
            return RagasAggregateResult()

    def _run_human_eval(
        self,
        errors: list[str],
        timings: dict[str, float],
    ) -> HumanEvalAggregate:
        """Layer 5: Human evaluation (optional)."""
        from pathlib import Path

        from src.evaluation._models import HumanEvalAggregate

        if not self._settings.human_eval_enabled:
            return HumanEvalAggregate()

        t0 = time.perf_counter()
        try:
            from src.evaluation._human_harness import HumanEvalHarness

            harness = HumanEvalHarness(self._settings)
            scoresheets_dir = Path(self._settings.scoresheets_dir)
            if not scoresheets_dir.exists():
                _log.info("human_eval_no_scoresheets", dir=str(scoresheets_dir))
                timings["human_eval_ms"] = (time.perf_counter() - t0) * 1000
                return HumanEvalAggregate()

            result = harness.import_scoresheets(scoresheets_dir)
            timings["human_eval_ms"] = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:
            _log.warning("layer5_human_eval_failed", error=str(exc))
            errors.append(f"Human evaluation failed: {exc}")
            timings["human_eval_ms"] = (time.perf_counter() - t0) * 1000
            return HumanEvalAggregate()
