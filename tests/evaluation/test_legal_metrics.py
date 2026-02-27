"""Tests for LegalMetricsEvaluator — 4 custom legal metrics."""

from __future__ import annotations

import pytest

from src.evaluation._legal_metrics import LegalMetricsEvaluator
from src.evaluation._models import (
    CitationAccuracyResult,
    CrossReferenceResult,
    EvaluationInput,
    EvaluationSettings,
    LegalMetricsAggregate,
    SectionCompletenessResult,
)

# --- Helpers ---


def _make_input(
    *,
    query_id: str = "q1",
    query_text: str = "test query",
    query_type: str = "factual",
    response_text: str = "",
    retrieved_contexts: list[str] | None = None,
    expected_sections: list[str] | None = None,
    cross_reference_test: bool = False,
    temporal_test: bool = False,
) -> EvaluationInput:
    return EvaluationInput(
        query_id=query_id,
        query_text=query_text,
        query_type=query_type,
        response_text=response_text,
        retrieved_contexts=retrieved_contexts or [],
        expected_sections=expected_sections or [],
        cross_reference_test=cross_reference_test,
        temporal_test=temporal_test,
    )


@pytest.fixture()
def evaluator() -> LegalMetricsEvaluator:
    return LegalMetricsEvaluator(EvaluationSettings())


# ===================================================================
# Citation accuracy (~10 tests)
# ===================================================================


class TestCitationAccuracy:
    """Tests for _citation_accuracy."""

    def test_all_citations_found(self, evaluator: LegalMetricsEvaluator) -> None:
        """All citations present in contexts -> accuracy 1.0."""
        inp = _make_input(
            response_text="Section 302 of the Indian Penal Code prescribes murder.",
            retrieved_contexts=["Section 302. Punishment for murder under Indian Penal Code."],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.missing_citations == []
        assert result.citations_found_in_contexts > 0

    def test_no_citations_in_response(self, evaluator: LegalMetricsEvaluator) -> None:
        """No citations extracted -> accuracy 1.0 (vacuously true)."""
        inp = _make_input(
            response_text="This is a general statement with no citations.",
            retrieved_contexts=["Some context text."],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.total_citations_in_response == 0

    def test_missing_citation(self, evaluator: LegalMetricsEvaluator) -> None:
        """Citation NOT in contexts -> accuracy < 1.0, missing populated."""
        inp = _make_input(
            response_text="Section 420 of the Indian Penal Code deals with cheating.",
            retrieved_contexts=["Section 302. Punishment for murder."],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy < 1.0
        assert len(result.missing_citations) > 0

    def test_case_citation_air_found(self, evaluator: LegalMetricsEvaluator) -> None:
        """AIR case citation present in contexts."""
        inp = _make_input(
            response_text="The court held in AIR 2023 SC 1234 that...",
            retrieved_contexts=["As per AIR 2023 SC 1234, the ratio was..."],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.total_citations_in_response == 1

    def test_case_citation_scc_found(self, evaluator: LegalMetricsEvaluator) -> None:
        """SCC citation present in contexts."""
        inp = _make_input(
            response_text="The ruling in (2023) 5 SCC 678 established...",
            retrieved_contexts=["Reported at (2023) 5 SCC 678."],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy == 1.0

    def test_section_ref_partial_match(self, evaluator: LegalMetricsEvaluator) -> None:
        """Section number appears in context even if full citation text doesn't."""
        inp = _make_input(
            response_text="S. 302 IPC is the key provision.",
            retrieved_contexts=["302. Punishment for murder..."],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy == 1.0

    def test_multiple_mixed_citations(self, evaluator: LegalMetricsEvaluator) -> None:
        """Some found, some missing -> accuracy between 0 and 1."""
        inp = _make_input(
            response_text=(
                "Section 302 of the Indian Penal Code for murder. "
                "Section 420 of the Indian Penal Code for cheating."
            ),
            retrieved_contexts=["Section 302. Punishment for murder. Indian Penal Code."],
        )
        result = evaluator._citation_accuracy(inp)
        assert 0.0 < result.accuracy < 1.0
        assert len(result.missing_citations) >= 1

    def test_empty_contexts_all_missing(self, evaluator: LegalMetricsEvaluator) -> None:
        """Empty contexts -> all citations missing."""
        inp = _make_input(
            response_text="Section 302 of the Indian Penal Code.",
            retrieved_contexts=[],
        )
        result = evaluator._citation_accuracy(inp)
        assert result.accuracy == 0.0
        assert len(result.missing_citations) > 0

    def test_aggregate_citation_accuracy(self, evaluator: LegalMetricsEvaluator) -> None:
        """Aggregate citation accuracy across multiple queries."""
        inputs = [
            _make_input(
                query_id="q1",
                response_text="Section 302 of the Indian Penal Code.",
                retrieved_contexts=["Section 302. Murder. Indian Penal Code."],
            ),
            _make_input(
                query_id="q2",
                response_text="No citations here.",
                retrieved_contexts=["Some text."],
            ),
        ]
        result = evaluator.evaluate(inputs)
        assert result.citation_accuracy == 1.0
        assert len(result.citation_details) == 2

    def test_citation_accuracy_result_model(self, evaluator: LegalMetricsEvaluator) -> None:
        """CitationAccuracyResult fields are correctly populated."""
        inp = _make_input(
            response_text="Section 302 of the Indian Penal Code.",
            retrieved_contexts=["302 punishment for murder."],
        )
        result = evaluator._citation_accuracy(inp)
        assert isinstance(result, CitationAccuracyResult)
        assert result.query_id == "q1"
        assert result.total_citations_in_response >= 1


# ===================================================================
# Temporal accuracy (~10 tests)
# ===================================================================


class TestTemporalAccuracy:
    """Tests for _temporal_accuracy."""

    def test_ipc_cited_no_repeal_note(self, evaluator: LegalMetricsEvaluator) -> None:
        """IPC cited without any acknowledgement of repeal -> violation."""
        inp = _make_input(
            response_text="Section 302 of the Indian Penal Code provides punishment for murder.",
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 0.0
        assert len(result.violations) > 0

    def test_ipc_cited_with_bns_mentioned(self, evaluator: LegalMetricsEvaluator) -> None:
        """IPC cited but BNS mentioned -> correct."""
        inp = _make_input(
            response_text=(
                "Section 302 of the Indian Penal Code was the provision for murder. "
                "It has been replaced by the Bharatiya Nyaya Sanhita."
            ),
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.violations == []

    def test_crpc_cited_with_bnss_mentioned(self, evaluator: LegalMetricsEvaluator) -> None:
        """CrPC cited with BNSS mentioned -> correct."""
        inp = _make_input(
            response_text=(
                "Section 437 of the Code of Criminal Procedure deals with bail. "
                "Now covered under Bharatiya Nagarik Suraksha Sanhita."
            ),
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0

    def test_evidence_act_with_bsa_mentioned(self, evaluator: LegalMetricsEvaluator) -> None:
        """Evidence Act cited with BSA mentioned -> correct."""
        inp = _make_input(
            response_text=(
                "Section 65B of the Indian Evidence Act governs electronic evidence. "
                "Bharatiya Sakshya Adhiniyam is the new act."
            ),
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0

    def test_no_repealed_acts_referenced(self, evaluator: LegalMetricsEvaluator) -> None:
        """No repealed act referenced -> accuracy 1.0."""
        inp = _make_input(
            response_text="Section 7 of the Insolvency and Bankruptcy Code is key.",
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.total_temporal_references == 0

    def test_mixed_correct_and_violations(self, evaluator: LegalMetricsEvaluator) -> None:
        """One correct, one violation -> accuracy 0.5."""
        inp = _make_input(
            response_text=(
                "Section 302 of the Indian Penal Code prescribes murder. "
                "Section 437 of the Code of Criminal Procedure was about bail. "
                "The Bharatiya Nagarik Suraksha Sanhita replaces CrPC."
            ),
        )
        result = evaluator._temporal_accuracy(inp)
        # IPC 302 -> violation (BNS not mentioned for IPC context, but BNSS is mentioned)
        # Actually BNSS replacement covers CrPC, not IPC. BNS not mentioned.
        # CrPC 437 -> correct (BNSS mentioned)
        # IPC 302 -> violation (BNS not mentioned)
        assert result.accuracy == pytest.approx(0.5)
        assert len(result.violations) == 1

    def test_response_says_repealed(self, evaluator: LegalMetricsEvaluator) -> None:
        """Response says 'repealed' -> counts as correct."""
        inp = _make_input(
            response_text=("Section 302 of the Indian Penal Code has been repealed."),
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0

    def test_bns_cited_no_issue(self, evaluator: LegalMetricsEvaluator) -> None:
        """BNS (current law) cited -> no temporal issue."""
        inp = _make_input(
            response_text="Section 103 of the Bharatiya Nyaya Sanhita deals with murder.",
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.total_temporal_references == 0

    def test_temporal_query_type_skips_violations(self, evaluator: LegalMetricsEvaluator) -> None:
        """query_type=='temporal' -> skip violation flagging."""
        inp = _make_input(
            query_type="temporal",
            response_text="Section 302 of the Indian Penal Code was the old provision.",
        )
        result = evaluator._temporal_accuracy(inp)
        assert result.accuracy == 1.0
        assert result.violations == []
        assert result.total_temporal_references > 0

    def test_aggregate_temporal_accuracy(self, evaluator: LegalMetricsEvaluator) -> None:
        """Aggregate temporal accuracy across queries."""
        inputs = [
            _make_input(
                query_id="q1",
                response_text="Section 302 of the Indian Penal Code. Bharatiya Nyaya Sanhita replaces it.",
            ),
            _make_input(
                query_id="q2",
                response_text="Section 7 of the Companies Act.",
            ),
        ]
        result = evaluator.evaluate(inputs)
        assert result.temporal_accuracy == 1.0


# ===================================================================
# Section completeness (~6 tests)
# ===================================================================


class TestSectionCompleteness:
    """Tests for _section_completeness."""

    def test_all_sections_retrieved(self, evaluator: LegalMetricsEvaluator) -> None:
        """All expected sections found -> 1.0."""
        inp = _make_input(
            expected_sections=["302", "304"],
            retrieved_contexts=[
                "Section 302. Punishment for murder.",
                "Section 304. Punishment for culpable homicide.",
            ],
        )
        result = evaluator._section_completeness(inp)
        assert result.completeness == 1.0
        assert result.retrieved_sections == ["302", "304"]

    def test_no_expected_sections(self, evaluator: LegalMetricsEvaluator) -> None:
        """No expected sections -> 1.0 (vacuously true)."""
        inp = _make_input(
            expected_sections=[],
            retrieved_contexts=["Some text."],
        )
        result = evaluator._section_completeness(inp)
        assert result.completeness == 1.0

    def test_partial_sections_retrieved(self, evaluator: LegalMetricsEvaluator) -> None:
        """Only some sections found -> between 0 and 1."""
        inp = _make_input(
            expected_sections=["302", "304", "307"],
            retrieved_contexts=[
                "Section 302. Punishment for murder.",
                "Section 304. Punishment for culpable homicide.",
            ],
        )
        result = evaluator._section_completeness(inp)
        assert result.completeness == pytest.approx(2.0 / 3.0)
        assert "302" in result.retrieved_sections
        assert "304" in result.retrieved_sections
        assert "307" not in result.retrieved_sections

    def test_zero_sections_found(self, evaluator: LegalMetricsEvaluator) -> None:
        """Expected sections not in any context -> 0.0."""
        inp = _make_input(
            expected_sections=["420", "467"],
            retrieved_contexts=["Section 302. Murder."],
        )
        result = evaluator._section_completeness(inp)
        assert result.completeness == 0.0
        assert result.retrieved_sections == []

    def test_section_word_boundary(self, evaluator: LegalMetricsEvaluator) -> None:
        """Section number must match on word boundary (e.g. 80 vs 80C)."""
        inp = _make_input(
            expected_sections=["80"],
            retrieved_contexts=["Section 80C allows deductions."],
        )
        result = evaluator._section_completeness(inp)
        # "80" appears as part of "80C" — but \b80\b should not match "80C"
        assert result.completeness == 0.0

    def test_section_completeness_result_model(self, evaluator: LegalMetricsEvaluator) -> None:
        """SectionCompletenessResult fields populated correctly."""
        inp = _make_input(
            expected_sections=["302"],
            retrieved_contexts=["Section 302."],
        )
        result = evaluator._section_completeness(inp)
        assert isinstance(result, SectionCompletenessResult)
        assert result.query_id == "q1"
        assert result.expected_sections == ["302"]


# ===================================================================
# Cross-reference resolution (~6 tests)
# ===================================================================


class TestCrossReferenceResolution:
    """Tests for _cross_reference_resolution."""

    def test_all_resolved(self, evaluator: LegalMetricsEvaluator) -> None:
        """All cross-ref sections found -> 1.0."""
        inp = _make_input(
            cross_reference_test=True,
            expected_sections=["73", "74"],
            retrieved_contexts=[
                "Section 73. Compensation for loss.",
                "Section 74. Compensation for breach where penalty stipulated.",
            ],
        )
        result = evaluator._cross_reference_resolution(inp)
        assert result.resolution_rate == 1.0
        assert result.retrieved_sections == ["73", "74"]

    def test_partial_resolved(self, evaluator: LegalMetricsEvaluator) -> None:
        """Only some sections found -> between 0 and 1."""
        inp = _make_input(
            cross_reference_test=True,
            expected_sections=["73", "74"],
            retrieved_contexts=["Section 73. Compensation for loss."],
        )
        result = evaluator._cross_reference_resolution(inp)
        assert result.resolution_rate == pytest.approx(0.5)

    def test_not_cross_reference_test(self, evaluator: LegalMetricsEvaluator) -> None:
        """cross_reference_test=False -> resolution_rate=1.0."""
        inp = _make_input(
            cross_reference_test=False,
            expected_sections=["73", "74"],
            retrieved_contexts=[],
        )
        result = evaluator._cross_reference_resolution(inp)
        assert result.resolution_rate == 1.0

    def test_zero_resolved(self, evaluator: LegalMetricsEvaluator) -> None:
        """No sections found -> 0.0."""
        inp = _make_input(
            cross_reference_test=True,
            expected_sections=["73", "74"],
            retrieved_contexts=["Unrelated text."],
        )
        result = evaluator._cross_reference_resolution(inp)
        assert result.resolution_rate == 0.0

    def test_cross_ref_result_model(self, evaluator: LegalMetricsEvaluator) -> None:
        """CrossReferenceResult fields populated correctly."""
        inp = _make_input(
            cross_reference_test=True,
            expected_sections=["73"],
            retrieved_contexts=["Section 73."],
        )
        result = evaluator._cross_reference_resolution(inp)
        assert isinstance(result, CrossReferenceResult)
        assert result.query_id == "q1"

    def test_empty_expected_sections_with_cross_ref(self, evaluator: LegalMetricsEvaluator) -> None:
        """cross_reference_test=True but no expected sections -> 1.0."""
        inp = _make_input(
            cross_reference_test=True,
            expected_sections=[],
            retrieved_contexts=["Some text."],
        )
        result = evaluator._cross_reference_resolution(inp)
        assert result.resolution_rate == 1.0


# ===================================================================
# Aggregate / evaluate (~4+ tests)
# ===================================================================


class TestAggregate:
    """Tests for evaluate() orchestration and aggregation."""

    def test_evaluate_returns_all_4_metrics(self, evaluator: LegalMetricsEvaluator) -> None:
        """evaluate() populates all 4 metric fields."""
        inputs = [
            _make_input(
                response_text="Section 302 of the Indian Penal Code. This act was repealed.",
                retrieved_contexts=["Section 302. Murder."],
                expected_sections=["302"],
            ),
        ]
        result = evaluator.evaluate(inputs)
        assert isinstance(result, LegalMetricsAggregate)
        assert result.citation_accuracy >= 0.0
        assert result.temporal_accuracy >= 0.0
        assert result.section_completeness >= 0.0
        assert result.cross_reference_resolution >= 0.0
        assert len(result.citation_details) == 1
        assert len(result.temporal_details) == 1
        assert len(result.completeness_details) == 1
        assert len(result.cross_reference_details) == 1

    def test_error_isolation(self, evaluator: LegalMetricsEvaluator) -> None:
        """If one metric fails for a query, others still run."""
        # We'll use a valid input — error isolation is about per-query exceptions.
        # Force an error by monkeypatching one method.
        original = evaluator._citation_accuracy

        def _boom(inp: EvaluationInput) -> CitationAccuracyResult:
            msg = "test error"
            raise RuntimeError(msg)

        evaluator._citation_accuracy = _boom  # type: ignore[assignment]

        inputs = [
            _make_input(
                response_text="Section 302 of the Indian Penal Code. Repealed.",
                retrieved_contexts=["Section 302."],
                expected_sections=["302"],
            ),
        ]
        result = evaluator.evaluate(inputs)

        # Citation failed, but temporal/completeness/cross-ref still ran
        assert len(result.citation_details) == 0
        assert len(result.temporal_details) == 1
        assert len(result.completeness_details) == 1
        assert len(result.cross_reference_details) == 1
        assert len(result.errors) >= 1
        assert "citation" in result.errors[0]

        evaluator._citation_accuracy = original  # type: ignore[assignment]

    def test_empty_inputs(self, evaluator: LegalMetricsEvaluator) -> None:
        """Empty input list -> default aggregate."""
        result = evaluator.evaluate([])
        assert isinstance(result, LegalMetricsAggregate)
        assert result.citation_accuracy == 0.0
        assert result.temporal_accuracy == 0.0
        assert result.section_completeness == 0.0
        assert result.cross_reference_resolution == 0.0

    def test_mixed_query_types(self, evaluator: LegalMetricsEvaluator) -> None:
        """Mix of temporal, cross-ref, and factual queries."""
        inputs = [
            _make_input(
                query_id="q1",
                query_type="temporal",
                response_text="Section 302 of the Indian Penal Code was the old law.",
                retrieved_contexts=["Section 302. Murder."],
                expected_sections=["302"],
            ),
            _make_input(
                query_id="q2",
                query_type="cross_reference",
                cross_reference_test=True,
                response_text="Section 73 of the Indian Contract Act and Section 74 of the Indian Contract Act.",
                retrieved_contexts=[
                    "Section 73. Compensation.",
                    "Section 74. Liquidated damages.",
                ],
                expected_sections=["73", "74"],
            ),
            _make_input(
                query_id="q3",
                query_type="factual",
                response_text="Section 80C allows tax deductions.",
                retrieved_contexts=["Section 80C. Deductions."],
                expected_sections=["80C"],
            ),
        ]
        result = evaluator.evaluate(inputs)
        assert len(result.citation_details) == 3
        assert len(result.temporal_details) == 3
        assert len(result.completeness_details) == 3
        assert len(result.cross_reference_details) == 3
        assert result.errors == []

    def test_aggregate_averages_correct(self, evaluator: LegalMetricsEvaluator) -> None:
        """Verify aggregate averages are arithmetic means."""
        inputs = [
            _make_input(
                query_id="q1",
                response_text="No citations.",
                retrieved_contexts=[],
                expected_sections=["302"],
            ),
            _make_input(
                query_id="q2",
                response_text="No citations.",
                retrieved_contexts=["Section 302. Murder."],
                expected_sections=["302"],
            ),
        ]
        result = evaluator.evaluate(inputs)
        # q1: completeness=0.0 (302 not in empty contexts)
        # q2: completeness=1.0 (302 found)
        assert result.section_completeness == pytest.approx(0.5)
