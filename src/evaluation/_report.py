"""Evaluation report generation — JSON and text formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation._exceptions import ReportError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from src.evaluation._models import EvaluationResult, EvaluationSettings

_log = get_logger(__name__)


class EvaluationReporter:
    """Generate evaluation reports in JSON or text format."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    def generate(self, result: EvaluationResult) -> str:
        """Generate a report string from an EvaluationResult.

        Returns JSON or text based on settings.report_format.
        """
        fmt = self._settings.report_format.lower()
        if fmt == "json":
            return self._generate_json(result)
        if fmt == "text":
            return self._generate_text(result)
        raise ReportError(f"Unknown report format: {fmt}")

    def save(self, result: EvaluationResult, output_path: Path | None = None) -> Path:
        """Generate and save a report to disk.

        Returns the path where the report was saved.
        """
        from pathlib import Path as _Path

        path = _Path(output_path or self._settings.report_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate(result)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ReportError(f"Failed to write report to {path}: {exc}") from exc

        _log.info("report_saved", path=str(path), format=self._settings.report_format)
        return path

    def _generate_json(self, result: EvaluationResult) -> str:
        """Generate a JSON report."""
        return result.model_dump_json(indent=2)

    def _generate_text(self, result: EvaluationResult) -> str:
        """Generate a human-readable text report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"Queries Evaluated: {result.queries_evaluated}")
        lines.append(f"All Targets Met:   {result.all_targets_met}")
        lines.append(f"Elapsed:           {result.elapsed_ms:.0f} ms")
        lines.append("")

        # RAGAS Metrics
        lines.append("-" * 40)
        lines.append("RAGAS METRICS")
        lines.append("-" * 40)
        r = result.ragas
        lines.append(self._metric_line("Context Recall", r.context_recall, 0.90))
        lines.append(self._metric_line("Context Precision", r.context_precision, 0.85))
        lines.append(self._metric_line("Faithfulness", r.faithfulness, 0.95))
        lines.append(self._metric_line("Answer Relevancy", r.answer_relevancy, 0.85))
        lines.append("")

        # Legal Metrics
        lines.append("-" * 40)
        lines.append("LEGAL METRICS")
        lines.append("-" * 40)
        lm = result.legal_metrics
        lines.append(self._metric_line("Citation Accuracy", lm.citation_accuracy, 0.98))
        lines.append(self._metric_line("Temporal Accuracy", lm.temporal_accuracy, 0.99))
        lines.append(self._metric_line("Section Completeness", lm.section_completeness, 0.90))
        lines.append(self._metric_line("Cross-Ref Resolution", lm.cross_reference_resolution, 0.80))
        lines.append("")

        # Latency
        lines.append("-" * 40)
        lines.append("LATENCY")
        lines.append("-" * 40)
        lines.append(f"  Pass Rate: {result.latency.pass_rate:.1%}")
        for route, avg_ms in sorted(result.latency.per_route.items()):
            lines.append(f"  {route:>12s}: {avg_ms:.0f} ms avg")
        lines.append("")

        # QI Metrics
        lines.append("-" * 40)
        lines.append("QUERY INTELLIGENCE")
        lines.append("-" * 40)
        qi = result.qi_metrics
        lines.append(f"  Cache Hit Rate:       {qi.cache_hit_rate:.1%}")
        lines.append(f"  Routing Accuracy:     {qi.routing_accuracy:.1%}")
        lines.append(f"  GenGround Rate:       {qi.genground_verification_rate:.1%}")
        lines.append(f"  Parent Utilization:   {qi.parent_context_utilization:.1%}")
        lines.append(f"  FLARE Frequency:      {qi.flare_frequency:.1%}")
        lines.append("")

        # Human Eval
        if self._settings.human_eval_enabled and result.human_eval.total_evaluations > 0:
            lines.append("-" * 40)
            lines.append("HUMAN EVALUATION")
            lines.append("-" * 40)
            he = result.human_eval
            lines.append(f"  Evaluations:     {he.total_evaluations}")
            lines.append(f"  Avg Accuracy:    {he.avg_accuracy:.2f}")
            lines.append(f"  Avg Completeness:{he.avg_completeness:.2f}")
            lines.append(f"  Avg Recency:     {he.avg_recency:.2f}")
            lines.append(f"  Avg Usefulness:  {he.avg_usefulness:.2f}")
            lines.append(f"  Pass Rate:       {he.accuracy_pass_rate:.1%}")
            lines.append("")

        # Errors
        if result.errors:
            lines.append("-" * 40)
            lines.append("ERRORS")
            lines.append("-" * 40)
            for err in result.errors:
                lines.append(f"  - {err}")
            lines.append("")

        # Timings
        if result.timings:
            lines.append("-" * 40)
            lines.append("TIMINGS")
            lines.append("-" * 40)
            for name, ms in sorted(result.timings.items()):
                lines.append(f"  {name}: {ms:.1f} ms")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _metric_line(name: str, score: float, target: float) -> str:
        """Format a single metric line: name, score, target, PASS/FAIL."""
        status = "PASS" if score >= target else "FAIL"
        return f"  {name:<25s} {score:.4f}  (target: {target:.2f})  [{status}]"
