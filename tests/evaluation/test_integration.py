"""Integration tests for the evaluation module.

End-to-end flows with mock upstream results — temporal violation detection,
citation accuracy, latency pass/fail, worksheet roundtrip, full pipeline
error isolation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.evaluation._models import (
    EvaluationInput,
    EvaluationResult,
    EvaluationSettings,
)
from src.evaluation.pipeline import EvaluationPipeline

if TYPE_CHECKING:
    from pathlib import Path


class TestTemporalViolationDetection:
    """Verify that temporal accuracy catches IPC references without repeal acknowledgment."""

    @pytest.mark.asyncio
    async def test_ipc_reference_without_acknowledgment(self) -> None:
        inp = EvaluationInput(
            query_id="temporal_001",
            query_text="What is the punishment for theft?",
            practice_area="criminal",
            query_type="factual",
            response_text="Section 379 of the Indian Penal Code prescribes punishment for theft.",
            retrieved_contexts=[
                "Section 379. Punishment for theft.—Whoever commits theft shall be punished "
                "with imprisonment of either description for a term which may extend to three years.",
            ],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])

        # Should detect temporal violation — IPC referenced without noting repeal
        td = result.legal_metrics.temporal_details
        assert len(td) == 1
        assert td[0].accuracy < 1.0
        assert len(td[0].violations) > 0

    @pytest.mark.asyncio
    async def test_ipc_reference_with_bns_acknowledgment(self) -> None:
        inp = EvaluationInput(
            query_id="temporal_002",
            query_text="What is the punishment for theft?",
            practice_area="criminal",
            query_type="factual",
            response_text=(
                "Section 379 of the Indian Penal Code prescribed punishment for theft. "
                "Note: IPC has been replaced by the Bharatiya Nyaya Sanhita."
            ),
            retrieved_contexts=["Section 379. Punishment for theft..."],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])

        td = result.legal_metrics.temporal_details
        assert len(td) == 1
        assert td[0].accuracy == 1.0
        assert len(td[0].violations) == 0

    @pytest.mark.asyncio
    async def test_temporal_query_type_skips_violation(self) -> None:
        """Historical queries (query_type=temporal) should not flag violations."""
        inp = EvaluationInput(
            query_id="temporal_003",
            query_text="What was the bail provision under CrPC before BNSS?",
            practice_area="criminal",
            query_type="temporal",
            response_text="Section 437 of the Code of Criminal Procedure provided bail in non-bailable cases.",
            retrieved_contexts=["Section 437. When bail may be taken..."],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])

        td = result.legal_metrics.temporal_details
        assert len(td) == 1
        assert td[0].accuracy == 1.0


class TestCitationAccuracy:
    """Verify citation accuracy across different scenarios."""

    @pytest.mark.asyncio
    async def test_all_citations_found(self) -> None:
        inp = EvaluationInput(
            query_id="cite_001",
            query_text="What is Section 302 IPC?",
            response_text="Section 302 of IPC prescribes punishment for murder.",
            retrieved_contexts=[
                "Section 302. Punishment for murder.—Whoever commits murder shall be punished with death.",
            ],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])

        cd = result.legal_metrics.citation_details
        assert len(cd) == 1
        assert cd[0].accuracy == 1.0

    @pytest.mark.asyncio
    async def test_missing_citation(self) -> None:
        inp = EvaluationInput(
            query_id="cite_002",
            query_text="What about murder and attempt to murder?",
            response_text=(
                "Section 302 of IPC covers murder. "
                "Section 307 of IPC covers attempt to murder."
            ),
            retrieved_contexts=[
                "Section 302. Punishment for murder.—Whoever commits murder...",
            ],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])

        cd = result.legal_metrics.citation_details
        assert len(cd) == 1
        assert cd[0].accuracy < 1.0
        assert len(cd[0].missing_citations) > 0


class TestLatencyCompliance:
    """Verify latency pass/fail for different route types."""

    @pytest.mark.asyncio
    async def test_fast_query_passes(self) -> None:
        inp = EvaluationInput(
            query_id="latency_001",
            query_text="Quick factual query",
            response_text="Answer.",
            qi_result={"route": "simple"},
            total_elapsed_ms=100.0,
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])
        assert result.latency.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_slow_simple_fails(self) -> None:
        inp = EvaluationInput(
            query_id="latency_002",
            query_text="Simple but slow query",
            response_text="Answer.",
            qi_result={"route": "simple"},
            total_elapsed_ms=500.0,
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])
        assert result.latency.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_complex_within_target_passes(self) -> None:
        inp = EvaluationInput(
            query_id="latency_003",
            query_text="Complex analysis query",
            response_text="Detailed answer.",
            qi_result={"route": "complex"},
            total_elapsed_ms=1800.0,
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])
        assert result.latency.pass_rate == 1.0


class TestSectionCompleteness:
    """Verify section completeness metric."""

    @pytest.mark.asyncio
    async def test_all_sections_found(self) -> None:
        inp = EvaluationInput(
            query_id="comp_001",
            query_text="What are Sections 73 and 74 Contract Act?",
            expected_sections=["73", "74"],
            response_text="Section 73 and 74 of Contract Act...",
            retrieved_contexts=[
                "Section 73. Compensation for loss or damage...",
                "Section 74. Compensation for breach of contract...",
            ],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])
        assert result.legal_metrics.section_completeness == 1.0

    @pytest.mark.asyncio
    async def test_missing_section(self) -> None:
        inp = EvaluationInput(
            query_id="comp_002",
            query_text="What are Sections 73 and 74 Contract Act?",
            expected_sections=["73", "74"],
            response_text="Section 73 and 74...",
            retrieved_contexts=["Section 73. Compensation for loss..."],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])
        assert result.legal_metrics.section_completeness == 0.5


class TestCrossReferenceResolution:
    """Verify cross-reference resolution metric."""

    @pytest.mark.asyncio
    async def test_cross_ref_all_resolved(self) -> None:
        inp = EvaluationInput(
            query_id="xref_001",
            query_text="How do Sections 7 and 9 IBC differ?",
            cross_reference_test=True,
            expected_sections=["7", "9"],
            response_text="Section 7 IBC allows financial creditors. Section 9 IBC allows operational creditors.",
            retrieved_contexts=[
                "Section 7. Initiation by financial creditor.",
                "Section 9. Application by operational creditor.",
            ],
        )
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate([inp])
        assert result.legal_metrics.cross_reference_resolution == 1.0


class TestWorksheetRoundTrip:
    """Generate worksheets, fill scores, import back."""

    @pytest.mark.asyncio
    async def test_generate_fill_import(self, tmp_path: Path) -> None:
        from src.evaluation._human_harness import HumanEvalHarness

        settings = EvaluationSettings(
            ragas_enabled=False,
            human_eval_enabled=True,
            scoresheets_dir=str(tmp_path / "scoresheets"),
        )

        inputs = [
            EvaluationInput(
                query_id="ws_001",
                query_text="What is Section 302?",
                response_text="Section 302 covers murder.",
                retrieved_contexts=["Section 302. Punishment for murder..."],
            ),
        ]

        harness = HumanEvalHarness(settings)
        worksheet_dir = tmp_path / "worksheets"
        worksheets = harness.generate_worksheets(inputs, worksheet_dir)
        assert len(worksheets) == 1

        # Simulate evaluator filling in the worksheet
        ws_data = json.loads(worksheets[0].read_text(encoding="utf-8"))
        ws_data["scores"]["evaluator_id"] = "evaluator_1"
        ws_data["scores"]["accuracy"] = 5
        ws_data["scores"]["completeness"] = 4
        ws_data["scores"]["recency"] = 4
        ws_data["scores"]["usefulness"] = 5

        # Save as scoresheet
        scoresheets_dir = tmp_path / "scoresheets"
        scoresheets_dir.mkdir(parents=True, exist_ok=True)
        scoresheet_path = scoresheets_dir / "ws_001_worksheet.json"
        scoresheet_path.write_text(
            json.dumps(ws_data, indent=2),
            encoding="utf-8",
        )

        # Import
        aggregate = harness.import_scoresheets(scoresheets_dir)
        assert aggregate.total_evaluations == 1
        assert aggregate.avg_accuracy == 5.0
        assert aggregate.avg_completeness == 4.0
        assert aggregate.accuracy_pass_rate == 1.0


class TestFullPipelineErrorIsolation:
    """Verify that failures in one layer don't crash the pipeline."""

    @pytest.mark.asyncio
    async def test_mixed_valid_and_problematic_inputs(self) -> None:
        inputs = [
            EvaluationInput(
                query_id="mix_001",
                query_text="Valid factual query",
                response_text="Section 302 IPC covers murder.",
                retrieved_contexts=["Section 302. Punishment for murder..."],
                qi_result={"cache_hit": True, "route": "simple"},
                total_elapsed_ms=100.0,
                verification_result={"summary": {"supported_claims": 1, "total_claims": 1}},
            ),
            EvaluationInput(
                query_id="mix_002",
                query_text="Empty response query",
                response_text="",
                retrieved_contexts=[],
                qi_result={},
                total_elapsed_ms=0.0,
            ),
            EvaluationInput(
                query_id="mix_003",
                query_text="Query with all upstream data",
                response_text="Section 80C allows deductions.",
                retrieved_contexts=["Section 80C. Deduction..."],
                expected_route="standard",
                expected_sections=["80C"],
                qi_result={"cache_hit": False, "route": "standard"},
                retrieval_result={
                    "flare_retrievals": 0,
                    "chunks": [{"parent_text": "Full text..."}],
                },
                verification_result={"summary": {"supported_claims": 1, "total_claims": 1}},
                total_elapsed_ms=500.0,
            ),
        ]
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate(inputs)

        assert isinstance(result, EvaluationResult)
        assert result.queries_evaluated == 3
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_all_targets_met_property(self) -> None:
        """EvaluationResult.all_targets_met works correctly."""
        result = EvaluationResult()
        # Default 0.0 values — targets not met
        assert result.all_targets_met is False


class TestQIMetricsIntegration:
    """Verify QI metrics aggregate correctly from upstream data."""

    @pytest.mark.asyncio
    async def test_cache_hit_and_routing(self) -> None:
        inputs = [
            EvaluationInput(
                query_id="qi_001",
                query_text="Q1",
                response_text="A1",
                expected_route="simple",
                qi_result={"cache_hit": True, "route": "simple"},
                total_elapsed_ms=50.0,
            ),
            EvaluationInput(
                query_id="qi_002",
                query_text="Q2",
                response_text="A2",
                expected_route="standard",
                qi_result={"cache_hit": False, "route": "standard"},
                total_elapsed_ms=400.0,
            ),
            EvaluationInput(
                query_id="qi_003",
                query_text="Q3",
                response_text="A3",
                expected_route="complex",
                qi_result={"cache_hit": False, "route": "standard"},
                total_elapsed_ms=1000.0,
            ),
        ]
        settings = EvaluationSettings(ragas_enabled=False)
        pipe = EvaluationPipeline(settings)
        result = await pipe.evaluate(inputs)

        qi = result.qi_metrics
        # 1 of 3 cache hits
        assert abs(qi.cache_hit_rate - 1 / 3) < 0.01
        # 2 of 3 routing correct (qi_003 expected complex but got standard)
        assert abs(qi.routing_accuracy - 2 / 3) < 0.01
