"""Custom legal metrics evaluator — citation, temporal, section, cross-ref."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.evaluation._models import (
        CitationAccuracyResult,
        CrossReferenceResult,
        EvaluationInput,
        EvaluationSettings,
        LegalMetricsAggregate,
        SectionCompletenessResult,
        TemporalAccuracyResult,
    )

_log = get_logger(__name__)


class LegalMetricsEvaluator:
    """Compute 4 custom legal metrics for RAG evaluation.

    1. Citation accuracy — citations in response that appear in contexts
    2. Temporal accuracy — repealed-act references acknowledged correctly
    3. Section completeness — expected sections found in contexts
    4. Cross-reference resolution — cross-ref sections retrieved
    """

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, inputs: list[EvaluationInput]) -> LegalMetricsAggregate:
        """Compute all 4 custom legal metrics across *inputs*."""
        from src.evaluation._models import LegalMetricsAggregate

        if not inputs:
            _log.info("legal_metrics_no_inputs")
            return LegalMetricsAggregate()

        citation_details: list[CitationAccuracyResult] = []
        temporal_details: list[TemporalAccuracyResult] = []
        completeness_details: list[SectionCompletenessResult] = []
        cross_ref_details: list[CrossReferenceResult] = []
        errors: list[str] = []

        for inp in inputs:
            # Citation accuracy
            try:
                citation_details.append(self._citation_accuracy(inp))
            except Exception as exc:
                errors.append(f"citation query={inp.query_id}: {exc}")
                _log.warning(
                    "legal_metric_citation_error",
                    query_id=inp.query_id,
                    error=str(exc),
                )

            # Temporal accuracy
            try:
                temporal_details.append(self._temporal_accuracy(inp))
            except Exception as exc:
                errors.append(f"temporal query={inp.query_id}: {exc}")
                _log.warning(
                    "legal_metric_temporal_error",
                    query_id=inp.query_id,
                    error=str(exc),
                )

            # Section completeness
            try:
                completeness_details.append(self._section_completeness(inp))
            except Exception as exc:
                errors.append(f"completeness query={inp.query_id}: {exc}")
                _log.warning(
                    "legal_metric_completeness_error",
                    query_id=inp.query_id,
                    error=str(exc),
                )

            # Cross-reference resolution
            try:
                cross_ref_details.append(self._cross_reference_resolution(inp))
            except Exception as exc:
                errors.append(f"cross_ref query={inp.query_id}: {exc}")
                _log.warning(
                    "legal_metric_cross_ref_error",
                    query_id=inp.query_id,
                    error=str(exc),
                )

        # Aggregate averages
        citation_avg = (
            sum(d.accuracy for d in citation_details) / len(citation_details)
            if citation_details
            else 0.0
        )
        temporal_avg = (
            sum(d.accuracy for d in temporal_details) / len(temporal_details)
            if temporal_details
            else 0.0
        )
        completeness_avg = (
            sum(d.completeness for d in completeness_details) / len(completeness_details)
            if completeness_details
            else 0.0
        )
        cross_ref_avg = (
            sum(d.resolution_rate for d in cross_ref_details) / len(cross_ref_details)
            if cross_ref_details
            else 0.0
        )

        _log.info(
            "legal_metrics_evaluate_complete",
            total=len(inputs),
            citation_avg=round(citation_avg, 4),
            temporal_avg=round(temporal_avg, 4),
            completeness_avg=round(completeness_avg, 4),
            cross_ref_avg=round(cross_ref_avg, 4),
            errors=len(errors),
        )

        return LegalMetricsAggregate(
            citation_accuracy=citation_avg,
            temporal_accuracy=temporal_avg,
            section_completeness=completeness_avg,
            cross_reference_resolution=cross_ref_avg,
            citation_details=citation_details,
            temporal_details=temporal_details,
            completeness_details=completeness_details,
            cross_reference_details=cross_ref_details,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Citation accuracy
    # ------------------------------------------------------------------

    def _citation_accuracy(self, inp: EvaluationInput) -> CitationAccuracyResult:
        """Extract citations from response, verify each appears in contexts."""
        from src.evaluation._models import CitationAccuracyResult
        from src.hallucination._citation_extractor import extract_citations

        citations = extract_citations(inp.response_text)

        if not citations:
            return CitationAccuracyResult(
                query_id=inp.query_id,
                total_citations_in_response=0,
                citations_found_in_contexts=0,
                accuracy=1.0,
            )

        joined_contexts = " ".join(inp.retrieved_contexts).lower()
        found = 0
        missing: list[str] = []

        for citation in citations:
            # Check if citation text OR section appears in any context
            text_lower = citation.text.lower()
            section_lower = (citation.section or "").lower()

            in_context = text_lower in joined_contexts
            if not in_context and section_lower:
                in_context = section_lower in joined_contexts

            if in_context:
                found += 1
            else:
                missing.append(citation.text)

        accuracy = found / len(citations)

        return CitationAccuracyResult(
            query_id=inp.query_id,
            total_citations_in_response=len(citations),
            citations_found_in_contexts=found,
            accuracy=accuracy,
            missing_citations=missing,
        )

    # ------------------------------------------------------------------
    # Temporal accuracy
    # ------------------------------------------------------------------

    def _temporal_accuracy(self, inp: EvaluationInput) -> TemporalAccuracyResult:
        """Check IPC/CrPC/Evidence Act references are acknowledged as repealed."""
        from src.evaluation._models import TemporalAccuracyResult
        from src.hallucination._citation_extractor import (
            extract_citations,
            resolve_act_alias,
        )
        from src.hallucination._temporal_checker import get_repealed_acts

        citations = extract_citations(inp.response_text)
        repealed_acts = get_repealed_acts()
        response_lower = inp.response_text.lower()

        total_temporal = 0
        correct = 0
        violations: list[str] = []

        for citation in citations:
            # Resolve the act name via alias mapping
            act_name = resolve_act_alias(citation.act) if citation.act else None
            if not act_name:
                continue

            repeal_info = repealed_acts.get(act_name)
            if repeal_info is None:
                continue

            total_temporal += 1

            # If query_type == "temporal", skip violation flagging
            if inp.query_type == "temporal":
                correct += 1
                continue

            replacement = repeal_info["replacement_act"].lower()
            # Check if response acknowledges the repeal
            acknowledged = (
                replacement in response_lower
                or "repealed" in response_lower
                or "replaced" in response_lower
                or "no longer in force" in response_lower
            )

            if acknowledged:
                correct += 1
            else:
                violations.append(
                    f"{citation.text} — {act_name} repealed, "
                    f"replacement: {repeal_info['replacement_act']}"
                )

        accuracy = correct / total_temporal if total_temporal > 0 else 1.0

        return TemporalAccuracyResult(
            query_id=inp.query_id,
            total_temporal_references=total_temporal,
            correct_temporal_references=correct,
            accuracy=accuracy,
            violations=violations,
        )

    # ------------------------------------------------------------------
    # Section completeness
    # ------------------------------------------------------------------

    def _section_completeness(self, inp: EvaluationInput) -> SectionCompletenessResult:
        """Check expected sections found in retrieved contexts."""
        from src.evaluation._models import SectionCompletenessResult

        if not inp.expected_sections:
            return SectionCompletenessResult(
                query_id=inp.query_id,
                expected_sections=[],
                retrieved_sections=[],
                completeness=1.0,
            )

        joined_contexts = " ".join(inp.retrieved_contexts)
        retrieved: list[str] = []

        for section in inp.expected_sections:
            pattern = rf"\b{re.escape(section)}\b"
            if re.search(pattern, joined_contexts):
                retrieved.append(section)

        completeness = len(retrieved) / len(inp.expected_sections)

        return SectionCompletenessResult(
            query_id=inp.query_id,
            expected_sections=list(inp.expected_sections),
            retrieved_sections=retrieved,
            completeness=completeness,
        )

    # ------------------------------------------------------------------
    # Cross-reference resolution
    # ------------------------------------------------------------------

    def _cross_reference_resolution(self, inp: EvaluationInput) -> CrossReferenceResult:
        """Check all expected cross-ref sections are retrieved."""
        from src.evaluation._models import CrossReferenceResult

        if not inp.cross_reference_test:
            return CrossReferenceResult(
                query_id=inp.query_id,
                expected_sections=list(inp.expected_sections),
                retrieved_sections=[],
                resolution_rate=1.0,
            )

        joined_contexts = " ".join(inp.retrieved_contexts)
        retrieved: list[str] = []

        for section in inp.expected_sections:
            pattern = rf"\b{re.escape(section)}\b"
            if re.search(pattern, joined_contexts):
                retrieved.append(section)

        rate = len(retrieved) / len(inp.expected_sections) if inp.expected_sections else 1.0

        return CrossReferenceResult(
            query_id=inp.query_id,
            expected_sections=list(inp.expected_sections),
            retrieved_sections=retrieved,
            resolution_rate=rate,
        )
