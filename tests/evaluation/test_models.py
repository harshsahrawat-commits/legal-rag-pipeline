"""Tests for evaluation Pydantic models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.evaluation._models import (
    CitationAccuracyResult,
    CrossReferenceResult,
    EvaluationConfig,
    EvaluationInput,
    EvaluationResult,
    EvaluationSettings,
    HumanScore,
    LatencyResult,
    LegalMetricsAggregate,
    MetricStatus,
    PracticeArea,
    QIMetrics,
    QueryType,
    RagasAggregateResult,
    RagasMetricResult,
    SectionCompletenessResult,
    TemporalAccuracyResult,
    TestQuery,
    TestQueryDataset,
)


class TestEnums:
    """Test enum values."""

    def test_practice_area_values(self) -> None:
        assert len(PracticeArea) == 5
        assert PracticeArea.CRIMINAL == "criminal"
        assert PracticeArea.CIVIL_CONTRACT == "civil_contract"
        assert PracticeArea.CORPORATE_COMMERCIAL == "corporate_commercial"
        assert PracticeArea.TAX == "tax"
        assert PracticeArea.CONSTITUTIONAL == "constitutional"

    def test_query_type_values(self) -> None:
        assert len(QueryType) == 4
        assert QueryType.FACTUAL == "factual"
        assert QueryType.ANALYTICAL == "analytical"
        assert QueryType.CROSS_REFERENCE == "cross_reference"
        assert QueryType.TEMPORAL == "temporal"

    def test_metric_status_values(self) -> None:
        assert len(MetricStatus) == 4
        assert MetricStatus.PASS == "pass"
        assert MetricStatus.FAIL == "fail"
        assert MetricStatus.SKIP == "skip"
        assert MetricStatus.ERROR == "error"


class TestTestQuery:
    """Test TestQuery model."""

    def test_valid_query(self) -> None:
        q = TestQuery(
            query_id="q1",
            query_text="What is Section 420 IPC?",
            practice_area="criminal",
            query_type="factual",
        )
        assert q.query_id == "q1"
        assert q.practice_area == PracticeArea.CRIMINAL

    def test_defaults(self) -> None:
        q = TestQuery(
            query_id="q1",
            query_text="test",
            practice_area="criminal",
            query_type="factual",
        )
        assert q.expected_route == "standard"
        assert q.expected_citations == []
        assert q.expected_answer_contains == []
        assert q.reference_answer == ""
        assert q.expected_sections == []
        assert q.temporal_test is False
        assert q.cross_reference_test is False

    def test_invalid_practice_area_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestQuery(
                query_id="q1",
                query_text="test",
                practice_area="invalid_area",
                query_type="factual",
            )

    def test_invalid_query_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestQuery(
                query_id="q1",
                query_text="test",
                practice_area="criminal",
                query_type="invalid_type",
            )


class TestTestQueryDataset:
    """Test TestQueryDataset model."""

    def test_empty_dataset(self) -> None:
        ds = TestQueryDataset()
        assert ds.queries == []
        assert ds.version == "1.0"

    def test_with_queries(self, sample_test_query: TestQuery) -> None:
        ds = TestQueryDataset(queries=[sample_test_query])
        assert len(ds.queries) == 1


class TestRagasMetricResult:
    """Test RAGAS metric result model."""

    def test_creation(self) -> None:
        r = RagasMetricResult(query_id="q1", metric_name="faithfulness", score=0.95)
        assert r.score == 0.95
        assert r.status == MetricStatus.PASS
        assert r.error is None


class TestRagasAggregateResult:
    """Test RAGAS aggregate model."""

    def test_defaults(self) -> None:
        r = RagasAggregateResult()
        assert r.context_recall == 0.0
        assert r.context_precision == 0.0
        assert r.faithfulness == 0.0
        assert r.answer_relevancy == 0.0
        assert r.queries_evaluated == 0


class TestCitationAccuracyResult:
    """Test citation accuracy model."""

    def test_creation(self) -> None:
        r = CitationAccuracyResult(
            query_id="q1",
            total_citations_in_response=5,
            citations_found_in_contexts=4,
            accuracy=0.8,
            missing_citations=["Section 420 IPC"],
        )
        assert r.accuracy == 0.8
        assert len(r.missing_citations) == 1


class TestTemporalAccuracyResult:
    """Test temporal accuracy model."""

    def test_creation(self) -> None:
        r = TemporalAccuracyResult(
            query_id="q1",
            total_temporal_references=3,
            correct_temporal_references=2,
            accuracy=0.667,
            violations=["IPC cited without repeal note"],
        )
        assert r.accuracy == pytest.approx(0.667)


class TestSectionCompletenessResult:
    """Test section completeness model."""

    def test_creation(self) -> None:
        r = SectionCompletenessResult(
            query_id="q1",
            expected_sections=["302", "304"],
            retrieved_sections=["302"],
            completeness=0.5,
        )
        assert r.completeness == 0.5


class TestCrossReferenceResult:
    """Test cross-reference resolution model."""

    def test_creation(self) -> None:
        r = CrossReferenceResult(
            query_id="q1",
            expected_sections=["7", "9", "10"],
            retrieved_sections=["7", "9"],
            resolution_rate=0.667,
        )
        assert r.resolution_rate == pytest.approx(0.667)


class TestLatencyResult:
    """Test latency result model."""

    def test_pass(self) -> None:
        r = LatencyResult(
            query_id="q1", route="simple", elapsed_ms=150.0, target_ms=200.0
        )
        assert r.status == MetricStatus.PASS

    def test_fail(self) -> None:
        r = LatencyResult(
            query_id="q1",
            route="simple",
            elapsed_ms=250.0,
            target_ms=200.0,
            status=MetricStatus.FAIL,
        )
        assert r.status == MetricStatus.FAIL


class TestQIMetrics:
    """Test QI metrics model."""

    def test_defaults(self) -> None:
        qi = QIMetrics()
        assert qi.cache_hit_rate == 0.0
        assert qi.routing_accuracy == 0.0
        assert qi.flare_frequency == 0.0


class TestHumanScore:
    """Test human evaluation score model."""

    def test_valid_scores(self) -> None:
        s = HumanScore(
            query_id="q1",
            evaluator_id="lawyer1",
            accuracy=5,
            completeness=4,
            recency=3,
            usefulness=4,
        )
        assert s.accuracy == 5

    def test_rejects_score_below_1(self) -> None:
        with pytest.raises(ValidationError):
            HumanScore(
                query_id="q1",
                evaluator_id="e1",
                accuracy=0,
                completeness=3,
                recency=3,
                usefulness=3,
            )

    def test_rejects_score_above_5(self) -> None:
        with pytest.raises(ValidationError):
            HumanScore(
                query_id="q1",
                evaluator_id="e1",
                accuracy=6,
                completeness=3,
                recency=3,
                usefulness=3,
            )

    def test_notes_default_empty(self) -> None:
        s = HumanScore(
            query_id="q1",
            evaluator_id="e1",
            accuracy=3,
            completeness=3,
            recency=3,
            usefulness=3,
        )
        assert s.notes == ""


class TestEvaluationInput:
    """Test EvaluationInput model."""

    def test_defaults_empty(self) -> None:
        inp = EvaluationInput(query_id="q1", query_text="test")
        assert inp.response_text == ""
        assert inp.retrieved_contexts == []
        assert inp.qi_result == {}
        assert inp.retrieval_result == {}
        assert inp.verification_result == {}
        assert inp.total_elapsed_ms == 0.0

    def test_with_upstream_results(
        self, sample_evaluation_input: EvaluationInput
    ) -> None:
        assert sample_evaluation_input.response_text != ""
        assert len(sample_evaluation_input.retrieved_contexts) == 2


class TestEvaluationResult:
    """Test EvaluationResult model."""

    def test_elapsed_ms_none_finished(self) -> None:
        r = EvaluationResult()
        assert r.elapsed_ms == 0.0

    def test_elapsed_ms_computed(self) -> None:
        start = datetime.now(UTC)
        r = EvaluationResult(
            started_at=start,
            finished_at=start + timedelta(seconds=1.5),
        )
        assert r.elapsed_ms == pytest.approx(1500.0)

    def test_all_targets_met_true(self) -> None:
        r = EvaluationResult(
            ragas=RagasAggregateResult(
                context_recall=0.95,
                context_precision=0.90,
                faithfulness=0.98,
                answer_relevancy=0.90,
            ),
            legal_metrics=LegalMetricsAggregate(
                citation_accuracy=0.99,
                temporal_accuracy=1.0,
                section_completeness=0.95,
                cross_reference_resolution=0.85,
            ),
        )
        assert r.all_targets_met is True

    def test_all_targets_met_false_ragas(self) -> None:
        r = EvaluationResult(
            ragas=RagasAggregateResult(
                context_recall=0.80,  # Below 0.90 target
                context_precision=0.90,
                faithfulness=0.98,
                answer_relevancy=0.90,
            ),
            legal_metrics=LegalMetricsAggregate(
                citation_accuracy=0.99,
                temporal_accuracy=1.0,
                section_completeness=0.95,
                cross_reference_resolution=0.85,
            ),
        )
        assert r.all_targets_met is False

    def test_all_targets_met_false_legal(self) -> None:
        r = EvaluationResult(
            ragas=RagasAggregateResult(
                context_recall=0.95,
                context_precision=0.90,
                faithfulness=0.98,
                answer_relevancy=0.90,
            ),
            legal_metrics=LegalMetricsAggregate(
                citation_accuracy=0.95,  # Below 0.98 target
                temporal_accuracy=1.0,
                section_completeness=0.95,
                cross_reference_resolution=0.85,
            ),
        )
        assert r.all_targets_met is False

    def test_defaults(self) -> None:
        r = EvaluationResult()
        assert r.queries_evaluated == 0
        assert r.errors == []
        assert r.timings == {}
        assert r.all_targets_met is False


class TestEvaluationSettings:
    """Test EvaluationSettings model."""

    def test_defaults(self) -> None:
        s = EvaluationSettings()
        assert s.ragas_enabled is True
        assert s.ragas_llm_model == "claude-haiku-4-5-20251001"
        assert s.latency_simple_ms == 200.0
        assert s.latency_standard_ms == 800.0
        assert s.latency_complex_ms == 2000.0
        assert s.latency_analytical_ms == 5000.0
        assert s.cache_hit_target == 0.30
        assert s.routing_accuracy_target == 0.90
        assert s.human_eval_enabled is False
        assert s.accuracy_pass_threshold == 4

    def test_custom_values(self) -> None:
        s = EvaluationSettings(latency_simple_ms=100.0, ragas_enabled=False)
        assert s.latency_simple_ms == 100.0
        assert s.ragas_enabled is False


class TestEvaluationConfig:
    """Test EvaluationConfig root model."""

    def test_default_settings(self) -> None:
        c = EvaluationConfig()
        assert isinstance(c.settings, EvaluationSettings)
        assert c.settings.ragas_enabled is True
