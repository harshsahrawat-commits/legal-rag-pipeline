"""Latency evaluator — checks query latency against per-route targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.evaluation._models import (
        EvaluationInput,
        EvaluationSettings,
        LatencyAggregate,
        LatencyResult,
    )

_log = get_logger(__name__)


class LatencyEvaluator:
    """Check query latency against per-route targets."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, inputs: list[EvaluationInput]) -> LatencyAggregate:
        """Check all queries against route-specific latency targets."""
        from src.evaluation._models import LatencyAggregate

        if not inputs:
            _log.info("latency_evaluate_no_inputs")
            return LatencyAggregate(pass_rate=0.0)

        results: list[LatencyResult] = []
        errors: list[str] = []

        for inp in inputs:
            try:
                results.append(self._check_one(inp))
            except Exception as exc:
                errors.append(f"query={inp.query_id}: {exc}")
                _log.warning("latency_check_error", query_id=inp.query_id, error=str(exc))

        # Aggregate pass rate
        if results:
            passed = sum(1 for r in results if r.status.value == "pass")
            pass_rate = passed / len(results)
        else:
            pass_rate = 0.0

        # Per-route average elapsed ms
        per_route: dict[str, float] = {}
        route_totals: dict[str, list[float]] = {}
        for r in results:
            route_totals.setdefault(r.route, []).append(r.elapsed_ms)
        for route, elapsed_list in route_totals.items():
            per_route[route] = sum(elapsed_list) / len(elapsed_list)

        _log.info(
            "latency_evaluate_complete",
            total=len(inputs),
            passed=sum(1 for r in results if r.status.value == "pass"),
            failed=sum(1 for r in results if r.status.value == "fail"),
            errors=len(errors),
        )

        return LatencyAggregate(
            pass_rate=pass_rate,
            per_route=per_route,
            per_query=results,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _target_for_route(self, route: str) -> float:
        """Map route string to target ms.

        simple     -> settings.latency_simple_ms      (200)
        standard   -> settings.latency_standard_ms     (800)
        complex    -> settings.latency_complex_ms      (2000)
        analytical -> settings.latency_analytical_ms   (5000)
        Unknown    -> settings.latency_standard_ms     (default)
        """
        route_lower = route.lower()
        targets = {
            "simple": self._settings.latency_simple_ms,
            "standard": self._settings.latency_standard_ms,
            "complex": self._settings.latency_complex_ms,
            "analytical": self._settings.latency_analytical_ms,
        }
        return targets.get(route_lower, self._settings.latency_standard_ms)

    def _check_one(self, inp: EvaluationInput) -> LatencyResult:
        """Check one query's latency against its route target.

        Route comes from inp.qi_result.get("route") falling back to
        inp.expected_route.  elapsed_ms from inp.total_elapsed_ms.
        Status is PASS if elapsed <= target, FAIL otherwise.
        """
        from src.evaluation._models import LatencyResult, MetricStatus

        qi = inp.qi_result or {}
        route = qi.get("route", "") or inp.expected_route or "standard"
        elapsed_ms = inp.total_elapsed_ms
        target_ms = self._target_for_route(route)

        status = MetricStatus.PASS if elapsed_ms <= target_ms else MetricStatus.FAIL

        return LatencyResult(
            query_id=inp.query_id,
            route=route,
            elapsed_ms=elapsed_ms,
            target_ms=target_ms,
            status=status,
        )
