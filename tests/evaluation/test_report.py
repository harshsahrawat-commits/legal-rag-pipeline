"""Tests for evaluation report generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.evaluation._exceptions import ReportError
from src.evaluation._models import (
    EvaluationResult,
    EvaluationSettings,
    HumanEvalAggregate,
    LatencyAggregate,
    LegalMetricsAggregate,
    QIMetrics,
    RagasAggregateResult,
)
from src.evaluation._report import EvaluationReporter


@pytest.fixture()
def reporter() -> EvaluationReporter:
    return EvaluationReporter(EvaluationSettings())


@pytest.fixture()
def sample_result() -> EvaluationResult:
    return EvaluationResult(
        ragas=RagasAggregateResult(
            context_recall=0.92,
            context_precision=0.88,
            faithfulness=0.96,
            answer_relevancy=0.87,
            queries_evaluated=3,
        ),
        legal_metrics=LegalMetricsAggregate(
            citation_accuracy=0.99,
            temporal_accuracy=1.0,
            section_completeness=0.95,
            cross_reference_resolution=0.85,
        ),
        latency=LatencyAggregate(
            pass_rate=0.90,
            per_route={"standard": 500.0, "complex": 1500.0},
        ),
        qi_metrics=QIMetrics(
            cache_hit_rate=0.35,
            routing_accuracy=0.92,
            genground_verification_rate=0.97,
            parent_context_utilization=0.85,
            flare_frequency=0.07,
        ),
        queries_evaluated=3,
        timings={"legal_metrics_ms": 5.0, "latency_ms": 1.0},
    )


class TestEvaluationReporterInit:
    def test_stores_settings(self) -> None:
        s = EvaluationSettings()
        r = EvaluationReporter(s)
        assert r._settings is s


class TestGenerateJSON:
    def test_returns_valid_json(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult
    ) -> None:
        output = reporter.generate(sample_result)
        parsed = json.loads(output)
        assert parsed["queries_evaluated"] == 3

    def test_json_contains_ragas(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult
    ) -> None:
        parsed = json.loads(reporter.generate(sample_result))
        assert parsed["ragas"]["context_recall"] == 0.92

    def test_json_contains_legal(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult
    ) -> None:
        parsed = json.loads(reporter.generate(sample_result))
        assert parsed["legal_metrics"]["citation_accuracy"] == 0.99

    def test_json_contains_latency(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult
    ) -> None:
        parsed = json.loads(reporter.generate(sample_result))
        assert parsed["latency"]["pass_rate"] == 0.90

    def test_json_contains_qi(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult
    ) -> None:
        parsed = json.loads(reporter.generate(sample_result))
        assert parsed["qi_metrics"]["cache_hit_rate"] == 0.35


class TestGenerateText:
    def test_text_format(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(sample_result)
        assert "EVALUATION REPORT" in output

    def test_text_contains_ragas_section(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(sample_result)
        assert "RAGAS METRICS" in output
        assert "Context Recall" in output

    def test_text_contains_legal_section(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(sample_result)
        assert "LEGAL METRICS" in output
        assert "Citation Accuracy" in output

    def test_text_pass_fail_indicators(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(sample_result)
        assert "[PASS]" in output

    def test_text_contains_timings(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(sample_result)
        assert "TIMINGS" in output
        assert "legal_metrics_ms" in output

    def test_text_errors_section(self) -> None:
        result = EvaluationResult(errors=["something failed"])
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(result)
        assert "ERRORS" in output
        assert "something failed" in output

    def test_text_human_eval_when_enabled(self) -> None:
        result = EvaluationResult(
            human_eval=HumanEvalAggregate(
                avg_accuracy=4.5,
                total_evaluations=10,
            ),
        )
        settings = EvaluationSettings(report_format="text", human_eval_enabled=True)
        reporter = EvaluationReporter(settings)
        output = reporter.generate(result)
        assert "HUMAN EVALUATION" in output

    def test_text_no_human_eval_when_disabled(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="text"))
        output = reporter.generate(sample_result)
        assert "HUMAN EVALUATION" not in output


class TestUnknownFormat:
    def test_raises_report_error(self, sample_result: EvaluationResult) -> None:
        reporter = EvaluationReporter(EvaluationSettings(report_format="xml"))
        with pytest.raises(ReportError, match="Unknown report format"):
            reporter.generate(sample_result)


class TestSave:
    def test_saves_json_to_file(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.json"
        path = reporter.save(sample_result, out)
        assert path == out
        assert out.exists()
        parsed = json.loads(out.read_text(encoding="utf-8"))
        assert parsed["queries_evaluated"] == 3

    def test_saves_to_default_path(self, sample_result: EvaluationResult, tmp_path: Path) -> None:
        default_path = tmp_path / "eval" / "report.json"
        settings = EvaluationSettings(report_output_path=str(default_path))
        reporter = EvaluationReporter(settings)
        path = reporter.save(sample_result)
        assert path == default_path
        assert default_path.exists()

    def test_creates_parent_dirs(
        self, reporter: EvaluationReporter, sample_result: EvaluationResult, tmp_path: Path
    ) -> None:
        out = tmp_path / "deep" / "nested" / "report.json"
        reporter.save(sample_result, out)
        assert out.exists()


class TestMetricLine:
    def test_pass_line(self) -> None:
        line = EvaluationReporter._metric_line("Test Metric", 0.95, 0.90)
        assert "[PASS]" in line
        assert "Test Metric" in line

    def test_fail_line(self) -> None:
        line = EvaluationReporter._metric_line("Test Metric", 0.80, 0.90)
        assert "[FAIL]" in line
