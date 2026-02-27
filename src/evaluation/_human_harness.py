"""Human evaluation harness: worksheet generation and scoresheet import."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.evaluation._exceptions import HumanEvalError
from src.evaluation._models import HumanEvalAggregate, HumanScore
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from src.evaluation._models import EvaluationInput, EvaluationSettings

_log = get_logger(__name__)


class HumanEvalHarness:
    """Generate worksheets for human evaluators and import completed scoresheets."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    def generate_worksheets(
        self,
        inputs: list[EvaluationInput],
        output_dir: Path,
    ) -> list[Path]:
        """Generate JSON worksheet files for human evaluation.

        Each worksheet contains the query, response, and retrieved contexts,
        plus blank score fields for the evaluator to fill in.
        One file per query with response text.

        Returns list of created file paths.
        """
        from pathlib import Path as _Path

        output_dir = _Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created: list[Path] = []
        for inp in inputs:
            if not inp.response_text.strip():
                continue

            worksheet = {
                "query_id": inp.query_id,
                "query_text": inp.query_text,
                "response_text": inp.response_text,
                "retrieved_contexts": inp.retrieved_contexts,
                "practice_area": inp.practice_area,
                "query_type": inp.query_type,
                "scores": {
                    "evaluator_id": "",
                    "accuracy": None,
                    "completeness": None,
                    "recency": None,
                    "usefulness": None,
                    "notes": "",
                },
            }

            path = output_dir / f"{inp.query_id}_worksheet.json"
            try:
                path.write_text(
                    json.dumps(worksheet, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                created.append(path)
            except OSError as exc:
                raise HumanEvalError(
                    f"Failed to write worksheet for {inp.query_id}: {exc}"
                ) from exc

        _log.info("worksheets_generated", count=len(created), output_dir=str(output_dir))
        return created

    def import_scoresheets(self, scoresheets_dir: Path) -> HumanEvalAggregate:
        """Import completed scoresheets from a directory.

        Each scoresheet is a JSON file with a list of HumanScore entries,
        or a single worksheet JSON with filled-in scores.

        Returns aggregated human evaluation results.
        """
        from pathlib import Path as _Path

        scoresheets_dir = _Path(scoresheets_dir)
        if not scoresheets_dir.exists():
            raise HumanEvalError(f"Scoresheets directory not found: {scoresheets_dir}")

        all_scores: list[HumanScore] = []
        errors: list[str] = []

        json_files = sorted(scoresheets_dir.glob("*.json"))
        if not json_files:
            _log.warning("no_scoresheets_found", dir=str(scoresheets_dir))
            return HumanEvalAggregate()

        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                scores = self._parse_scoresheet(data, f.name)
                all_scores.extend(scores)
            except HumanEvalError:
                raise
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON in {f.name}: {exc}")
            except Exception as exc:
                errors.append(f"Error reading {f.name}: {exc}")

        result = self._aggregate_scores(all_scores)
        result.errors = errors
        return result

    def _parse_scoresheet(self, data: dict | list, filename: str) -> list[HumanScore]:
        """Parse a single scoresheet file into HumanScore objects."""
        scores: list[HumanScore] = []

        if isinstance(data, list):
            # Array of score entries
            for entry in data:
                scores.append(self._validate_score_entry(entry, filename))
        elif isinstance(data, dict):
            # Single worksheet format with "scores" key
            if "scores" in data and "query_id" in data:
                score_data = data["scores"]
                if score_data.get("accuracy") is not None:
                    entry = {
                        "query_id": data["query_id"],
                        **score_data,
                    }
                    scores.append(self._validate_score_entry(entry, filename))
            elif "query_id" in data and "evaluator_id" in data:
                # Direct HumanScore format
                scores.append(self._validate_score_entry(data, filename))
            else:
                raise HumanEvalError(f"Unrecognized scoresheet format in {filename}")
        else:
            raise HumanEvalError(f"Invalid scoresheet data type in {filename}")

        return scores

    def _validate_score_entry(self, entry: dict, filename: str) -> HumanScore:
        """Validate and create a HumanScore from a dict entry."""
        try:
            return HumanScore(**entry)
        except Exception as exc:
            raise HumanEvalError(f"Invalid score entry in {filename}: {exc}") from exc

    def _aggregate_scores(self, scores: list[HumanScore]) -> HumanEvalAggregate:
        """Compute aggregate statistics from individual scores."""
        if not scores:
            return HumanEvalAggregate()

        n = len(scores)
        avg_accuracy = sum(s.accuracy for s in scores) / n
        avg_completeness = sum(s.completeness for s in scores) / n
        avg_recency = sum(s.recency for s in scores) / n
        avg_usefulness = sum(s.usefulness for s in scores) / n

        # Pass rate: % of scores with accuracy >= threshold
        threshold = self._settings.accuracy_pass_threshold
        passes = sum(1 for s in scores if s.accuracy >= threshold)
        accuracy_pass_rate = passes / n

        return HumanEvalAggregate(
            avg_accuracy=avg_accuracy,
            avg_completeness=avg_completeness,
            avg_recency=avg_recency,
            avg_usefulness=avg_usefulness,
            accuracy_pass_rate=accuracy_pass_rate,
            total_evaluations=n,
            scores=scores,
        )
