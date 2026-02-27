"""Tests for latency evaluator."""

from __future__ import annotations

import pytest

from src.evaluation._latency_metrics import LatencyEvaluator
from src.evaluation._models import (
    EvaluationInput,
    EvaluationSettings,
    MetricStatus,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _inp(
    query_id: str = "q1",
    elapsed_ms: float = 100.0,
    route: str = "simple",
    expected_route: str = "",
    qi_result: dict | None = None,
) -> EvaluationInput:
    """Build a minimal EvaluationInput for latency tests."""
    return EvaluationInput(
        query_id=query_id,
        query_text="test query",
        total_elapsed_ms=elapsed_ms,
        expected_route=expected_route,
        qi_result=qi_result if qi_result is not None else {"route": route},
    )


@pytest.fixture()
def settings() -> EvaluationSettings:
    return EvaluationSettings()


@pytest.fixture()
def evaluator(settings: EvaluationSettings) -> LatencyEvaluator:
    return LatencyEvaluator(settings)


# ── target_for_route ─────────────────────────────────────────────────


class TestTargetForRoute:
    def test_simple(self, evaluator: LatencyEvaluator) -> None:
        assert evaluator._target_for_route("simple") == 200.0

    def test_standard(self, evaluator: LatencyEvaluator) -> None:
        assert evaluator._target_for_route("standard") == 800.0

    def test_complex(self, evaluator: LatencyEvaluator) -> None:
        assert evaluator._target_for_route("complex") == 2000.0

    def test_analytical(self, evaluator: LatencyEvaluator) -> None:
        assert evaluator._target_for_route("analytical") == 5000.0

    def test_unknown_defaults_to_standard(self, evaluator: LatencyEvaluator) -> None:
        assert evaluator._target_for_route("mystery") == 800.0

    def test_case_insensitive(self, evaluator: LatencyEvaluator) -> None:
        assert evaluator._target_for_route("SIMPLE") == 200.0
        assert evaluator._target_for_route("Complex") == 2000.0


# ── _check_one ───────────────────────────────────────────────────────


class TestCheckOne:
    def test_simple_under_target_pass(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(elapsed_ms=150.0, route="simple")
        result = evaluator._check_one(inp)
        assert result.status == MetricStatus.PASS
        assert result.route == "simple"
        assert result.target_ms == 200.0
        assert result.elapsed_ms == 150.0

    def test_simple_over_target_fail(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(elapsed_ms=250.0, route="simple")
        result = evaluator._check_one(inp)
        assert result.status == MetricStatus.FAIL

    def test_exact_boundary_passes(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(elapsed_ms=200.0, route="simple")
        result = evaluator._check_one(inp)
        assert result.status == MetricStatus.PASS

    def test_analytical_under_target(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(elapsed_ms=4500.0, route="analytical")
        result = evaluator._check_one(inp)
        assert result.status == MetricStatus.PASS
        assert result.target_ms == 5000.0

    def test_complex_over_target(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(elapsed_ms=2500.0, route="complex")
        result = evaluator._check_one(inp)
        assert result.status == MetricStatus.FAIL

    def test_uses_expected_route_when_qi_missing(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(
            elapsed_ms=100.0,
            expected_route="simple",
            qi_result={},
        )
        result = evaluator._check_one(inp)
        assert result.route == "simple"
        assert result.status == MetricStatus.PASS

    def test_defaults_to_standard_when_no_route(self, evaluator: LatencyEvaluator) -> None:
        inp = _inp(elapsed_ms=500.0, expected_route="", qi_result={})
        result = evaluator._check_one(inp)
        assert result.route == "standard"
        assert result.target_ms == 800.0


# ── evaluate (aggregate) ────────────────────────────────────────────


class TestEvaluateAggregate:
    def test_empty_inputs(self, evaluator: LatencyEvaluator) -> None:
        agg = evaluator.evaluate([])
        assert agg.pass_rate == 0.0
        assert agg.per_route == {}
        assert agg.per_query == []

    def test_all_pass(self, evaluator: LatencyEvaluator) -> None:
        inputs = [
            _inp("q1", 100.0, "simple"),
            _inp("q2", 500.0, "standard"),
            _inp("q3", 1500.0, "complex"),
        ]
        agg = evaluator.evaluate(inputs)
        assert agg.pass_rate == 1.0
        assert len(agg.per_query) == 3
        assert all(r.status == MetricStatus.PASS for r in agg.per_query)

    def test_half_fail(self, evaluator: LatencyEvaluator) -> None:
        inputs = [
            _inp("q1", 100.0, "simple"),  # pass (target 200)
            _inp("q2", 300.0, "simple"),  # fail (target 200)
        ]
        agg = evaluator.evaluate(inputs)
        assert agg.pass_rate == 0.5

    def test_per_route_averages(self, evaluator: LatencyEvaluator) -> None:
        inputs = [
            _inp("q1", 100.0, "simple"),
            _inp("q2", 200.0, "simple"),
            _inp("q3", 600.0, "standard"),
        ]
        agg = evaluator.evaluate(inputs)
        assert agg.per_route["simple"] == pytest.approx(150.0)
        assert agg.per_route["standard"] == pytest.approx(600.0)

    def test_aggregate_has_all_fields(self, evaluator: LatencyEvaluator) -> None:
        inputs = [_inp("q1", 100.0, "simple")]
        agg = evaluator.evaluate(inputs)
        assert hasattr(agg, "pass_rate")
        assert hasattr(agg, "per_route")
        assert hasattr(agg, "per_query")
        assert hasattr(agg, "errors")
        assert isinstance(agg.errors, list)

    def test_error_isolation(self, evaluator: LatencyEvaluator) -> None:
        """If _check_one raises, the error is captured and other queries proceed."""
        good = _inp("q1", 100.0, "simple")
        bad = _inp("q2", 100.0, "simple")

        # Patch _check_one to blow up for q2 specifically
        original = evaluator._check_one

        def broken_check(inp: EvaluationInput) -> object:
            if inp.query_id == "q2":
                msg = "deliberate failure"
                raise RuntimeError(msg)
            return original(inp)

        evaluator._check_one = broken_check  # type: ignore[assignment]

        agg = evaluator.evaluate([good, bad])
        assert len(agg.per_query) == 1
        assert agg.per_query[0].query_id == "q1"
        assert len(agg.errors) == 1
        assert "deliberate failure" in agg.errors[0]
        # pass_rate computed on successful results only
        assert agg.pass_rate == 1.0
