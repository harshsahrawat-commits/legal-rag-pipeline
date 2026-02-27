"""Tests for QI metrics evaluator."""

from __future__ import annotations

import pytest

from src.evaluation._models import EvaluationInput, EvaluationSettings
from src.evaluation._qi_metrics import QIMetricsEvaluator

# ── Helpers ──────────────────────────────────────────────────────────


def _inp(
    query_id: str = "q1",
    *,
    cache_hit: bool = False,
    route: str = "standard",
    expected_route: str = "standard",
    supported_claims: int = 1,
    total_claims: int = 1,
    has_verification: bool = True,
    parent_texts: list[str | None] | None = None,
    flare_retrievals: int = 0,
    qi_result: dict | None = None,
    retrieval_result: dict | None = None,
    verification_result: dict | None = None,
) -> EvaluationInput:
    """Build a minimal EvaluationInput for QI tests."""
    if qi_result is None:
        qi_result = {"cache_hit": cache_hit, "route": route}

    if retrieval_result is None:
        chunks = []
        if parent_texts is not None:
            chunks = [{"parent_text": pt} for pt in parent_texts]
        retrieval_result = {"flare_retrievals": flare_retrievals, "chunks": chunks}

    if verification_result is None:
        if has_verification:
            verification_result = {
                "summary": {
                    "supported_claims": supported_claims,
                    "total_claims": total_claims,
                }
            }
        else:
            verification_result = {}

    return EvaluationInput(
        query_id=query_id,
        query_text="test",
        expected_route=expected_route,
        qi_result=qi_result,
        retrieval_result=retrieval_result,
        verification_result=verification_result,
    )


@pytest.fixture()
def settings() -> EvaluationSettings:
    return EvaluationSettings()


@pytest.fixture()
def evaluator(settings: EvaluationSettings) -> QIMetricsEvaluator:
    return QIMetricsEvaluator(settings)


# ── cache_hit_rate ───────────────────────────────────────────────────


class TestCacheHitRate:
    def test_all_hits(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [_inp("q1", cache_hit=True), _inp("q2", cache_hit=True)]
        assert evaluator._cache_hit_rate(inputs) == 1.0

    def test_no_hits(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [_inp("q1", cache_hit=False), _inp("q2", cache_hit=False)]
        assert evaluator._cache_hit_rate(inputs) == 0.0

    def test_partial_hits(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", cache_hit=True),
            _inp("q2", cache_hit=False),
            _inp("q3", cache_hit=True),
            _inp("q4", cache_hit=False),
        ]
        assert evaluator._cache_hit_rate(inputs) == 0.5

    def test_empty_inputs(self, evaluator: QIMetricsEvaluator) -> None:
        assert evaluator._cache_hit_rate([]) == 0.0

    def test_missing_qi_result(self, evaluator: QIMetricsEvaluator) -> None:
        inp = _inp("q1", qi_result={})
        assert evaluator._cache_hit_rate([inp]) == 0.0


# ── routing_accuracy ─────────────────────────────────────────────────


class TestRoutingAccuracy:
    def test_all_correct(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", route="simple", expected_route="simple"),
            _inp("q2", route="complex", expected_route="complex"),
        ]
        assert evaluator._routing_accuracy(inputs) == 1.0

    def test_some_wrong(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", route="simple", expected_route="simple"),
            _inp("q2", route="standard", expected_route="complex"),
        ]
        assert evaluator._routing_accuracy(inputs) == 0.5

    def test_missing_expected_route_skipped(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", route="simple", expected_route="simple"),
            _inp("q2", route="complex", expected_route=""),  # skipped
        ]
        assert evaluator._routing_accuracy(inputs) == 1.0

    def test_all_skipped(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [_inp("q1", expected_route="")]
        assert evaluator._routing_accuracy(inputs) == 0.0

    def test_empty_inputs(self, evaluator: QIMetricsEvaluator) -> None:
        assert evaluator._routing_accuracy([]) == 0.0


# ── genground_verification_rate ──────────────────────────────────────


class TestGengroundRate:
    def test_all_supported(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", supported_claims=3, total_claims=3),
            _inp("q2", supported_claims=2, total_claims=2),
        ]
        assert evaluator._genground_verification_rate(inputs) == pytest.approx(1.0)

    def test_mixed(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", supported_claims=1, total_claims=2),
            _inp("q2", supported_claims=2, total_claims=2),
        ]
        # 3 supported / 4 total = 0.75
        assert evaluator._genground_verification_rate(inputs) == pytest.approx(0.75)

    def test_zero_claims_counts_as_fully_supported(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [_inp("q1", supported_claims=0, total_claims=0)]
        assert evaluator._genground_verification_rate(inputs) == pytest.approx(1.0)

    def test_no_summary_defaults_zero(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [_inp("q1", has_verification=False)]
        assert evaluator._genground_verification_rate(inputs) == 0.0

    def test_empty_inputs(self, evaluator: QIMetricsEvaluator) -> None:
        assert evaluator._genground_verification_rate([]) == 0.0


# ── parent_context_utilization ───────────────────────────────────────


class TestParentUtilization:
    def test_all_expanded(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", parent_texts=["text"]),
            _inp("q2", parent_texts=["text"]),
        ]
        assert evaluator._parent_context_utilization(inputs) == 1.0

    def test_none_expanded(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", parent_texts=[None]),
            _inp("q2", parent_texts=[None]),
        ]
        assert evaluator._parent_context_utilization(inputs) == 0.0

    def test_partial(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", parent_texts=["text"]),
            _inp("q2", parent_texts=[None]),
        ]
        assert evaluator._parent_context_utilization(inputs) == 0.5

    def test_empty_chunks(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [_inp("q1", parent_texts=[])]
        assert evaluator._parent_context_utilization(inputs) == 0.0

    def test_empty_inputs(self, evaluator: QIMetricsEvaluator) -> None:
        assert evaluator._parent_context_utilization([]) == 0.0


# ── flare_frequency ──────────────────────────────────────────────────


class TestFlareFrequency:
    def test_none_triggered(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", flare_retrievals=0),
            _inp("q2", flare_retrievals=0),
        ]
        assert evaluator._flare_frequency(inputs) == 0.0

    def test_some_triggered(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", flare_retrievals=0),
            _inp("q2", flare_retrievals=2),
        ]
        assert evaluator._flare_frequency(inputs) == 0.5

    def test_all_triggered(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp("q1", flare_retrievals=1),
            _inp("q2", flare_retrievals=3),
        ]
        assert evaluator._flare_frequency(inputs) == 1.0

    def test_empty_inputs(self, evaluator: QIMetricsEvaluator) -> None:
        assert evaluator._flare_frequency([]) == 0.0


# ── evaluate (aggregate) ────────────────────────────────────────────


class TestEvaluateAggregate:
    def test_returns_all_fields(self, evaluator: QIMetricsEvaluator) -> None:
        inputs = [
            _inp(
                "q1",
                cache_hit=True,
                route="simple",
                expected_route="simple",
                supported_claims=2,
                total_claims=2,
                parent_texts=["text"],
                flare_retrievals=1,
            )
        ]
        result = evaluator.evaluate(inputs)
        assert result.cache_hit_rate == 1.0
        assert result.routing_accuracy == 1.0
        assert result.genground_verification_rate == pytest.approx(1.0)
        assert result.parent_context_utilization == 1.0
        assert result.flare_frequency == 1.0
        assert "cache_hit_rate" in result.details
        assert isinstance(result.errors, list)

    def test_empty_inputs_all_zeros(self, evaluator: QIMetricsEvaluator) -> None:
        result = evaluator.evaluate([])
        assert result.cache_hit_rate == 0.0
        assert result.routing_accuracy == 0.0
        assert result.genground_verification_rate == 0.0
        assert result.parent_context_utilization == 0.0
        assert result.flare_frequency == 0.0

    def test_missing_qi_result_graceful(self, evaluator: QIMetricsEvaluator) -> None:
        inp = EvaluationInput(
            query_id="q1",
            query_text="test",
            qi_result={},
            retrieval_result={},
            verification_result={},
        )
        result = evaluator.evaluate([inp])
        assert result.cache_hit_rate == 0.0
        assert result.routing_accuracy == 0.0

    def test_missing_retrieval_result_graceful(self, evaluator: QIMetricsEvaluator) -> None:
        inp = EvaluationInput(
            query_id="q1",
            query_text="test",
            retrieval_result={},
        )
        result = evaluator.evaluate([inp])
        assert result.parent_context_utilization == 0.0
        assert result.flare_frequency == 0.0

    def test_missing_verification_result_graceful(self, evaluator: QIMetricsEvaluator) -> None:
        inp = EvaluationInput(
            query_id="q1",
            query_text="test",
            verification_result={},
        )
        result = evaluator.evaluate([inp])
        assert result.genground_verification_rate == 0.0

    def test_error_isolation(self, evaluator: QIMetricsEvaluator) -> None:
        """If one metric raises, the rest still compute."""
        inputs = [_inp("q1", cache_hit=True)]

        original = evaluator._cache_hit_rate

        def broken(_inputs: list[EvaluationInput]) -> float:
            msg = "deliberate"
            raise RuntimeError(msg)

        evaluator._cache_hit_rate = broken  # type: ignore[assignment]

        result = evaluator.evaluate(inputs)
        # cache_hit_rate errored → 0.0
        assert result.cache_hit_rate == 0.0
        # other metrics still computed
        assert len(result.errors) >= 1
        assert "cache_hit_rate" in result.errors[0]

        evaluator._cache_hit_rate = original  # type: ignore[assignment]
