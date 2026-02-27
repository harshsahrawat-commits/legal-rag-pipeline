"""Tests for the human evaluation harness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.evaluation._exceptions import HumanEvalError
from src.evaluation._human_harness import HumanEvalHarness
from src.evaluation._models import EvaluationInput, EvaluationSettings, HumanScore

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def harness(evaluation_settings: EvaluationSettings) -> HumanEvalHarness:
    """Harness with default settings."""
    return HumanEvalHarness(evaluation_settings)


@pytest.fixture()
def _single_input() -> EvaluationInput:
    """A single EvaluationInput for basic tests."""
    return EvaluationInput(
        query_id="q1",
        query_text="What is Section 302 IPC?",
        practice_area="criminal",
        query_type="factual",
        response_text="Section 302 deals with murder.",
        retrieved_contexts=["Section 302. Punishment for murder."],
    )


# ---------------------------------------------------------------------------
# TestGenerateWorksheets
# ---------------------------------------------------------------------------


class TestGenerateWorksheets:
    """Tests for HumanEvalHarness.generate_worksheets."""

    def test_generates_one_file_per_input(
        self,
        harness: HumanEvalHarness,
        sample_evaluation_inputs: list[EvaluationInput],
        tmp_path: Path,
    ) -> None:
        """3 inputs with response text produce 3 worksheet files."""
        paths = harness.generate_worksheets(sample_evaluation_inputs, tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_skips_empty_responses(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Inputs with blank response_text are skipped."""
        inputs = [
            EvaluationInput(
                query_id="e1",
                query_text="Q1?",
                response_text="   ",
            ),
            EvaluationInput(
                query_id="e2",
                query_text="Q2?",
                response_text="",
            ),
            EvaluationInput(
                query_id="e3",
                query_text="Q3?",
                response_text="Has content.",
            ),
        ]
        paths = harness.generate_worksheets(inputs, tmp_path)
        assert len(paths) == 1
        assert paths[0].name == "e3_worksheet.json"

    def test_worksheet_structure(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
        _single_input: EvaluationInput,
    ) -> None:
        """Worksheet JSON has all expected top-level keys."""
        paths = harness.generate_worksheets([_single_input], tmp_path)
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        expected_keys = {
            "query_id",
            "query_text",
            "response_text",
            "retrieved_contexts",
            "practice_area",
            "query_type",
            "scores",
        }
        assert set(data.keys()) == expected_keys

    def test_worksheet_maps_fields(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
        _single_input: EvaluationInput,
    ) -> None:
        """Worksheet field values match the input."""
        paths = harness.generate_worksheets([_single_input], tmp_path)
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["query_id"] == "q1"
        assert data["query_text"] == "What is Section 302 IPC?"
        assert data["response_text"] == "Section 302 deals with murder."
        assert data["practice_area"] == "criminal"
        assert data["query_type"] == "factual"
        assert data["retrieved_contexts"] == ["Section 302. Punishment for murder."]

    def test_creates_output_dir(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
        _single_input: EvaluationInput,
    ) -> None:
        """Output directory is created if it does not exist."""
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        harness.generate_worksheets([_single_input], nested)
        assert nested.exists()
        assert nested.is_dir()

    def test_returns_paths(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
        sample_evaluation_inputs: list[EvaluationInput],
    ) -> None:
        """Returned paths correspond to actual files on disk."""
        paths = harness.generate_worksheets(sample_evaluation_inputs, tmp_path)
        on_disk = sorted(tmp_path.glob("*.json"))
        assert sorted(paths) == on_disk

    def test_empty_inputs(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Empty input list produces empty output list."""
        paths = harness.generate_worksheets([], tmp_path)
        assert paths == []

    def test_worksheet_scores_blank(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
        _single_input: EvaluationInput,
    ) -> None:
        """Generated worksheet has blank score fields for evaluator."""
        paths = harness.generate_worksheets([_single_input], tmp_path)
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        scores = data["scores"]
        assert scores["evaluator_id"] == ""
        assert scores["accuracy"] is None
        assert scores["completeness"] is None
        assert scores["recency"] is None
        assert scores["usefulness"] is None
        assert scores["notes"] == ""


# ---------------------------------------------------------------------------
# TestImportScoresheets
# ---------------------------------------------------------------------------


class TestImportScoresheets:
    """Tests for HumanEvalHarness.import_scoresheets."""

    def test_missing_dir_raises(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Nonexistent directory raises HumanEvalError."""
        with pytest.raises(HumanEvalError, match="not found"):
            harness.import_scoresheets(tmp_path / "does_not_exist")

    def test_empty_dir_returns_default(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Empty directory returns a default HumanEvalAggregate."""
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 0
        assert result.scores == []
        assert result.avg_accuracy == 0.0

    def test_imports_array_format(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """JSON array of score entries is parsed correctly."""
        scores = [
            {
                "query_id": "q1",
                "evaluator_id": "eval_a",
                "accuracy": 5,
                "completeness": 4,
                "recency": 3,
                "usefulness": 4,
            },
            {
                "query_id": "q2",
                "evaluator_id": "eval_a",
                "accuracy": 4,
                "completeness": 3,
                "recency": 4,
                "usefulness": 5,
            },
        ]
        (tmp_path / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 2
        assert result.scores[0].query_id == "q1"
        assert result.scores[1].query_id == "q2"

    def test_imports_worksheet_format(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Filled worksheet JSON with 'scores' key is imported."""
        worksheet = {
            "query_id": "q1",
            "query_text": "What is murder?",
            "response_text": "Section 302.",
            "retrieved_contexts": [],
            "practice_area": "criminal",
            "query_type": "factual",
            "scores": {
                "evaluator_id": "eval_b",
                "accuracy": 5,
                "completeness": 5,
                "recency": 4,
                "usefulness": 5,
                "notes": "Excellent",
            },
        }
        (tmp_path / "q1_worksheet.json").write_text(json.dumps(worksheet), encoding="utf-8")
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 1
        assert result.scores[0].evaluator_id == "eval_b"
        assert result.scores[0].notes == "Excellent"

    def test_imports_direct_format(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Single HumanScore dict format is imported."""
        direct = {
            "query_id": "q1",
            "evaluator_id": "eval_c",
            "accuracy": 3,
            "completeness": 3,
            "recency": 3,
            "usefulness": 3,
        }
        (tmp_path / "direct.json").write_text(json.dumps(direct), encoding="utf-8")
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 1
        assert result.scores[0].evaluator_id == "eval_c"

    def test_invalid_scores_raise(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Score values outside 1-5 raise HumanEvalError."""
        bad = {
            "query_id": "q1",
            "evaluator_id": "eval_d",
            "accuracy": 6,
            "completeness": 3,
            "recency": 3,
            "usefulness": 3,
        }
        (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(HumanEvalError, match="Invalid score entry"):
            harness.import_scoresheets(tmp_path)

    def test_invalid_json_records_error(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Malformed JSON file adds an entry to errors list."""
        (tmp_path / "broken.json").write_text("{bad json", encoding="utf-8")
        # Also add a valid file so we get a result with errors
        valid = [
            {
                "query_id": "q1",
                "evaluator_id": "eval_e",
                "accuracy": 4,
                "completeness": 4,
                "recency": 4,
                "usefulness": 4,
            }
        ]
        (tmp_path / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 1
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]
        assert "broken.json" in result.errors[0]

    def test_multiple_files_combined(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Scores from multiple files are aggregated together."""
        file1 = [
            {
                "query_id": "q1",
                "evaluator_id": "eval_f",
                "accuracy": 5,
                "completeness": 5,
                "recency": 5,
                "usefulness": 5,
            }
        ]
        file2 = [
            {
                "query_id": "q2",
                "evaluator_id": "eval_f",
                "accuracy": 3,
                "completeness": 3,
                "recency": 3,
                "usefulness": 3,
            }
        ]
        (tmp_path / "a_scores.json").write_text(json.dumps(file1), encoding="utf-8")
        (tmp_path / "b_scores.json").write_text(json.dumps(file2), encoding="utf-8")
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 2
        assert result.avg_accuracy == pytest.approx(4.0)

    def test_unrecognized_format_raises(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Dict without query_id or scores raises HumanEvalError."""
        bad = {"random_key": "value"}
        (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(HumanEvalError, match="Unrecognized scoresheet format"):
            harness.import_scoresheets(tmp_path)

    def test_worksheet_with_unfilled_scores_skipped(
        self,
        harness: HumanEvalHarness,
        tmp_path: Path,
    ) -> None:
        """Worksheet with accuracy=None (unfilled) produces no score."""
        worksheet = {
            "query_id": "q1",
            "query_text": "Q?",
            "response_text": "A.",
            "retrieved_contexts": [],
            "practice_area": "criminal",
            "query_type": "factual",
            "scores": {
                "evaluator_id": "",
                "accuracy": None,
                "completeness": None,
                "recency": None,
                "usefulness": None,
                "notes": "",
            },
        }
        (tmp_path / "unfilled.json").write_text(json.dumps(worksheet), encoding="utf-8")
        result = harness.import_scoresheets(tmp_path)
        assert result.total_evaluations == 0
        assert result.scores == []


# ---------------------------------------------------------------------------
# TestAggregateScores
# ---------------------------------------------------------------------------


class TestAggregateScores:
    """Tests for HumanEvalHarness._aggregate_scores."""

    def test_computes_averages(
        self,
        harness: HumanEvalHarness,
    ) -> None:
        """Average scores are computed correctly across 3 entries."""
        scores = [
            HumanScore(
                query_id="q1",
                evaluator_id="e1",
                accuracy=5,
                completeness=4,
                recency=3,
                usefulness=5,
            ),
            HumanScore(
                query_id="q2",
                evaluator_id="e1",
                accuracy=3,
                completeness=2,
                recency=5,
                usefulness=3,
            ),
            HumanScore(
                query_id="q3",
                evaluator_id="e1",
                accuracy=4,
                completeness=3,
                recency=4,
                usefulness=4,
            ),
        ]
        result = harness._aggregate_scores(scores)
        assert result.avg_accuracy == pytest.approx(4.0)
        assert result.avg_completeness == pytest.approx(3.0)
        assert result.avg_recency == pytest.approx(4.0)
        assert result.avg_usefulness == pytest.approx(4.0)

    def test_accuracy_pass_rate(
        self,
        harness: HumanEvalHarness,
    ) -> None:
        """Pass rate: threshold=4, 2 of 3 pass -> 0.667."""
        # Default threshold is 4
        scores = [
            HumanScore(
                query_id="q1",
                evaluator_id="e1",
                accuracy=5,
                completeness=3,
                recency=3,
                usefulness=3,
            ),
            HumanScore(
                query_id="q2",
                evaluator_id="e1",
                accuracy=4,
                completeness=3,
                recency=3,
                usefulness=3,
            ),
            HumanScore(
                query_id="q3",
                evaluator_id="e1",
                accuracy=2,
                completeness=3,
                recency=3,
                usefulness=3,
            ),
        ]
        result = harness._aggregate_scores(scores)
        assert result.accuracy_pass_rate == pytest.approx(2 / 3)

    def test_empty_scores(
        self,
        harness: HumanEvalHarness,
    ) -> None:
        """Empty score list returns default aggregate."""
        result = harness._aggregate_scores([])
        assert result.total_evaluations == 0
        assert result.avg_accuracy == 0.0
        assert result.accuracy_pass_rate == 0.0
        assert result.scores == []

    def test_total_evaluations(
        self,
        harness: HumanEvalHarness,
    ) -> None:
        """Total evaluations equals length of scores list."""
        scores = [
            HumanScore(
                query_id=f"q{i}",
                evaluator_id="e1",
                accuracy=3,
                completeness=3,
                recency=3,
                usefulness=3,
            )
            for i in range(7)
        ]
        result = harness._aggregate_scores(scores)
        assert result.total_evaluations == 7

    def test_custom_threshold(self) -> None:
        """Custom accuracy_pass_threshold alters pass rate calculation."""
        settings = EvaluationSettings(accuracy_pass_threshold=3)
        harness = HumanEvalHarness(settings)
        scores = [
            HumanScore(
                query_id="q1",
                evaluator_id="e1",
                accuracy=3,
                completeness=3,
                recency=3,
                usefulness=3,
            ),
            HumanScore(
                query_id="q2",
                evaluator_id="e1",
                accuracy=2,
                completeness=3,
                recency=3,
                usefulness=3,
            ),
        ]
        result = harness._aggregate_scores(scores)
        # Only first score passes (accuracy=3 >= threshold=3)
        assert result.accuracy_pass_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end: generate worksheets, fill them, import."""

    def test_generate_fill_import(
        self,
        harness: HumanEvalHarness,
        sample_evaluation_inputs: list[EvaluationInput],
        tmp_path: Path,
    ) -> None:
        """Generate worksheets, fill in scores, import back."""
        worksheets_dir = tmp_path / "worksheets"
        paths = harness.generate_worksheets(sample_evaluation_inputs, worksheets_dir)
        assert len(paths) == 3

        # Simulate evaluator filling in scores
        for p in paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["scores"] = {
                "evaluator_id": "lawyer_1",
                "accuracy": 4,
                "completeness": 5,
                "recency": 4,
                "usefulness": 4,
                "notes": "Good answer",
            }
            p.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Import filled worksheets
        result = harness.import_scoresheets(worksheets_dir)
        assert result.total_evaluations == 3
        assert result.avg_accuracy == pytest.approx(4.0)
        assert result.avg_completeness == pytest.approx(5.0)
        assert result.accuracy_pass_rate == pytest.approx(1.0)
        assert result.errors == []
