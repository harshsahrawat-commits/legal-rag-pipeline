"""Query Intelligence metrics evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.evaluation._models import (
        EvaluationInput,
        EvaluationSettings,
        QIMetrics,
    )

_log = get_logger(__name__)


class QIMetricsEvaluator:
    """Compute Query Intelligence metrics from upstream result fields."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, inputs: list[EvaluationInput]) -> QIMetrics:
        """Compute all QI metrics."""
        from src.evaluation._models import QIMetrics

        errors: list[str] = []

        cache_hit = self._safe(self._cache_hit_rate, inputs, errors, "cache_hit_rate")
        routing = self._safe(self._routing_accuracy, inputs, errors, "routing_accuracy")
        genground = self._safe(
            self._genground_verification_rate,
            inputs,
            errors,
            "genground_verification_rate",
        )
        parent = self._safe(
            self._parent_context_utilization,
            inputs,
            errors,
            "parent_context_utilization",
        )
        flare = self._safe(self._flare_frequency, inputs, errors, "flare_frequency")

        details: dict[str, float] = {
            "cache_hit_rate": cache_hit,
            "routing_accuracy": routing,
            "genground_verification_rate": genground,
            "parent_context_utilization": parent,
            "flare_frequency": flare,
        }

        _log.info("qi_metrics_evaluate_complete", total=len(inputs), details=details)

        return QIMetrics(
            cache_hit_rate=cache_hit,
            routing_accuracy=routing,
            genground_verification_rate=genground,
            parent_context_utilization=parent,
            flare_frequency=flare,
            details=details,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def _cache_hit_rate(self, inputs: list[EvaluationInput]) -> float:
        """Fraction of queries where qi_result.get('cache_hit', False) is True."""
        if not inputs:
            return 0.0
        hits = sum(1 for inp in inputs if (inp.qi_result or {}).get("cache_hit", False) is True)
        return hits / len(inputs)

    def _routing_accuracy(self, inputs: list[EvaluationInput]) -> float:
        """Fraction of queries where qi_result route matches expected_route.

        Skip queries where expected_route is empty.
        """
        eligible = [inp for inp in inputs if inp.expected_route]
        if not eligible:
            return 0.0
        correct = sum(
            1 for inp in eligible if (inp.qi_result or {}).get("route", "") == inp.expected_route
        )
        return correct / len(eligible)

    def _genground_verification_rate(self, inputs: list[EvaluationInput]) -> float:
        """supported_claims / total_claims from verification_result['summary'].

        If total_claims is 0, count as 1.0 (all supported).
        Default to 0.0 if summary not present.
        """
        if not inputs:
            return 0.0

        total_supported = 0
        total_claims = 0
        has_data = False

        for inp in inputs:
            vr = inp.verification_result or {}
            summary = vr.get("summary")
            if summary is None:
                continue
            has_data = True
            tc = summary.get("total_claims", 0)
            sc = summary.get("supported_claims", 0)
            if tc == 0:
                # No claims = fully supported
                total_supported += 1
                total_claims += 1
            else:
                total_supported += sc
                total_claims += tc

        if not has_data:
            return 0.0
        if total_claims == 0:
            return 0.0
        return total_supported / total_claims

    def _parent_context_utilization(self, inputs: list[EvaluationInput]) -> float:
        """Fraction of queries where at least one chunk has parent_text != None."""
        if not inputs:
            return 0.0
        count = 0
        for inp in inputs:
            rr = inp.retrieval_result or {}
            chunks = rr.get("chunks", [])
            has_parent = any(
                c.get("parent_text") is not None for c in chunks if isinstance(c, dict)
            )
            if has_parent:
                count += 1
        return count / len(inputs)

    def _flare_frequency(self, inputs: list[EvaluationInput]) -> float:
        """Fraction of queries where retrieval_result.get('flare_retrievals', 0) > 0."""
        if not inputs:
            return 0.0
        count = sum(
            1 for inp in inputs if (inp.retrieval_result or {}).get("flare_retrievals", 0) > 0
        )
        return count / len(inputs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe(
        fn: object,
        inputs: list[EvaluationInput],
        errors: list[str],
        metric_name: str,
    ) -> float:
        """Call *fn* and return the result, catching exceptions."""
        try:
            # fn is a bound method
            return fn(inputs)  # type: ignore[operator]
        except Exception as exc:
            errors.append(f"{metric_name}: {exc}")
            _log.warning("qi_metric_error", metric=metric_name, error=str(exc))
            return 0.0
