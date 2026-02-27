"""Load and validate test query datasets for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.evaluation._exceptions import TestDatasetError
from src.evaluation._models import EvaluationInput, TestQueryDataset
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.evaluation._models import EvaluationSettings

_log = get_logger(__name__)


class TestDatasetLoader:
    """Load and validate test_queries.json, convert to evaluation inputs."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    def load(self, path: Path | None = None) -> TestQueryDataset:
        """Load test queries from JSON file.

        Args:
            path: Path to test_queries.json. Defaults to settings.test_queries_path.

        Returns:
            Validated TestQueryDataset.

        Raises:
            TestDatasetError: If file missing, invalid JSON, or validation fails.
        """
        file_path = path or Path(self._settings.test_queries_path)

        if not file_path.exists():
            msg = f"Test queries file not found: {file_path}"
            raise TestDatasetError(msg)

        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in test queries file {file_path}: {exc}"
            raise TestDatasetError(msg) from exc

        try:
            dataset = TestQueryDataset.model_validate(raw)
        except Exception as exc:
            msg = f"Test query validation failed: {exc}"
            raise TestDatasetError(msg) from exc

        _log.info(
            "test_dataset_loaded",
            path=str(file_path),
            query_count=len(dataset.queries),
            version=dataset.version,
        )
        return dataset

    def to_evaluation_inputs(
        self,
        dataset: TestQueryDataset,
    ) -> list[EvaluationInput]:
        """Convert TestQueryDataset to list of EvaluationInput.

        Populates query and expected fields. Upstream result fields remain
        empty until the full pipeline fills them.
        """
        inputs = []
        for q in dataset.queries:
            inputs.append(
                EvaluationInput(
                    query_id=q.query_id,
                    query_text=q.query_text,
                    practice_area=q.practice_area,
                    query_type=q.query_type,
                    expected_route=q.expected_route,
                    expected_citations=q.expected_citations,
                    expected_answer_contains=q.expected_answer_contains,
                    expected_sections=q.expected_sections,
                    reference_answer=q.reference_answer,
                    temporal_test=q.temporal_test,
                    cross_reference_test=q.cross_reference_test,
                )
            )
        return inputs

    @staticmethod
    def to_ragas_samples(
        inputs: list[EvaluationInput],
    ) -> list[dict]:
        """Convert completed EvaluationInputs to RAGAS-compatible dicts.

        Maps:
            query_text → user_input
            response_text → response
            retrieved_contexts → retrieved_contexts
            reference_answer → reference

        Skips inputs with empty response_text.
        """
        samples = []
        for inp in inputs:
            if not inp.response_text:
                continue
            samples.append({
                "user_input": inp.query_text,
                "response": inp.response_text,
                "retrieved_contexts": inp.retrieved_contexts or [],
                "reference": inp.reference_answer or None,
            })
        return samples
