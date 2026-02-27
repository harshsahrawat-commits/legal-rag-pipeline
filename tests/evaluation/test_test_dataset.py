"""Tests for test dataset loading and conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation._exceptions import TestDatasetError
from src.evaluation._models import (
    EvaluationInput,
    EvaluationSettings,
    PracticeArea,
    QueryType,
    TestQueryDataset,
)
from src.evaluation._test_dataset import TestDatasetLoader


@pytest.fixture()
def loader(evaluation_settings: EvaluationSettings) -> TestDatasetLoader:
    return TestDatasetLoader(evaluation_settings)


class TestTestDatasetLoaderLoad:
    """Test loading test_queries.json."""

    def test_load_valid_dataset(
        self, loader: TestDatasetLoader, test_queries_path: Path
    ) -> None:
        dataset = loader.load(test_queries_path)
        assert isinstance(dataset, TestQueryDataset)
        assert len(dataset.queries) == 3
        assert dataset.version == "1.0"

    def test_load_missing_file_raises(self, loader: TestDatasetLoader) -> None:
        with pytest.raises(TestDatasetError, match="not found"):
            loader.load(Path("/nonexistent/path.json"))

    def test_load_invalid_json_raises(
        self, loader: TestDatasetLoader, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(TestDatasetError, match="Invalid JSON"):
            loader.load(bad_file)

    def test_load_invalid_schema_raises(
        self, loader: TestDatasetLoader, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad_schema.json"
        bad_file.write_text(
            json.dumps({
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "test",
                        "practice_area": "invalid_area",
                        "query_type": "factual",
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(TestDatasetError, match="validation failed"):
            loader.load(bad_file)

    def test_load_empty_queries_accepted(
        self, loader: TestDatasetLoader, tmp_path: Path
    ) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(
            json.dumps({"version": "1.0", "queries": []}), encoding="utf-8"
        )
        dataset = loader.load(empty_file)
        assert len(dataset.queries) == 0

    def test_load_validates_practice_area(
        self, loader: TestDatasetLoader, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.json"
        f.write_text(
            json.dumps({
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "test",
                        "practice_area": "maritime",
                        "query_type": "factual",
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(TestDatasetError):
            loader.load(f)

    def test_load_validates_query_type(
        self, loader: TestDatasetLoader, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.json"
        f.write_text(
            json.dumps({
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "test",
                        "practice_area": "criminal",
                        "query_type": "hypothetical",
                    }
                ]
            }),
            encoding="utf-8",
        )
        with pytest.raises(TestDatasetError):
            loader.load(f)

    def test_load_preserves_all_fields(
        self, loader: TestDatasetLoader, test_queries_path: Path
    ) -> None:
        dataset = loader.load(test_queries_path)
        q = dataset.queries[0]
        assert q.query_id == "test_001"
        assert q.expected_citations == ["Section 318 BNS"]
        assert q.expected_sections == ["318"]


class TestTestDatasetLoaderConvert:
    """Test converting dataset to EvaluationInputs."""

    def test_to_evaluation_inputs_count(
        self,
        loader: TestDatasetLoader,
        sample_test_dataset: TestQueryDataset,
    ) -> None:
        inputs = loader.to_evaluation_inputs(sample_test_dataset)
        assert len(inputs) == len(sample_test_dataset.queries)

    def test_to_evaluation_inputs_maps_fields(
        self,
        loader: TestDatasetLoader,
        sample_test_dataset: TestQueryDataset,
    ) -> None:
        inputs = loader.to_evaluation_inputs(sample_test_dataset)
        inp = inputs[0]
        assert inp.query_id == "test_001"
        assert inp.query_text == "What is the punishment for cheating under BNS?"
        assert inp.expected_citations == ["Section 318 BNS"]
        assert inp.expected_sections == ["318"]
        assert inp.practice_area == "criminal"
        assert inp.query_type == "factual"

    def test_to_evaluation_inputs_upstream_empty(
        self,
        loader: TestDatasetLoader,
        sample_test_dataset: TestQueryDataset,
    ) -> None:
        inputs = loader.to_evaluation_inputs(sample_test_dataset)
        inp = inputs[0]
        assert inp.response_text == ""
        assert inp.retrieved_contexts == []
        assert inp.qi_result == {}
        assert inp.total_elapsed_ms == 0.0

    def test_to_evaluation_inputs_preserves_flags(
        self,
        loader: TestDatasetLoader,
        sample_test_dataset: TestQueryDataset,
    ) -> None:
        inputs = loader.to_evaluation_inputs(sample_test_dataset)
        # Second query has temporal_test=True
        assert inputs[1].temporal_test is True
        # Third query has cross_reference_test=True
        assert inputs[2].cross_reference_test is True


class TestToRagasSamples:
    """Test RAGAS format conversion."""

    def test_maps_fields(self) -> None:
        inputs = [
            EvaluationInput(
                query_id="q1",
                query_text="What is Section 420?",
                response_text="Section 420 defines cheating.",
                retrieved_contexts=["Section 420. Cheating..."],
                reference_answer="Section 420 IPC defines cheating.",
            )
        ]
        samples = TestDatasetLoader.to_ragas_samples(inputs)
        assert len(samples) == 1
        s = samples[0]
        assert s["user_input"] == "What is Section 420?"
        assert s["response"] == "Section 420 defines cheating."
        assert s["retrieved_contexts"] == ["Section 420. Cheating..."]
        assert s["reference"] == "Section 420 IPC defines cheating."

    def test_skips_empty_response(self) -> None:
        inputs = [
            EvaluationInput(query_id="q1", query_text="test", response_text=""),
            EvaluationInput(
                query_id="q2", query_text="test2", response_text="has response"
            ),
        ]
        samples = TestDatasetLoader.to_ragas_samples(inputs)
        assert len(samples) == 1
        assert samples[0]["user_input"] == "test2"

    def test_none_reference_when_empty(self) -> None:
        inputs = [
            EvaluationInput(
                query_id="q1",
                query_text="test",
                response_text="answer",
                reference_answer="",
            )
        ]
        samples = TestDatasetLoader.to_ragas_samples(inputs)
        assert samples[0]["reference"] is None

    def test_empty_inputs(self) -> None:
        samples = TestDatasetLoader.to_ragas_samples([])
        assert samples == []

    def test_empty_contexts_default(self) -> None:
        inputs = [
            EvaluationInput(
                query_id="q1",
                query_text="test",
                response_text="answer",
            )
        ]
        samples = TestDatasetLoader.to_ragas_samples(inputs)
        assert samples[0]["retrieved_contexts"] == []


class TestRealTestQueriesFile:
    """Test loading the actual data/eval/test_queries.json file."""

    def test_load_real_dataset(self) -> None:
        path = Path("data/eval/test_queries.json")
        if not path.exists():
            pytest.skip("test_queries.json not found")
        settings = EvaluationSettings()
        loader = TestDatasetLoader(settings)
        dataset = loader.load(path)
        assert len(dataset.queries) == 50
        assert dataset.version == "1.0"

    def test_real_dataset_query_ids_unique(self) -> None:
        path = Path("data/eval/test_queries.json")
        if not path.exists():
            pytest.skip("test_queries.json not found")
        settings = EvaluationSettings()
        loader = TestDatasetLoader(settings)
        dataset = loader.load(path)
        ids = [q.query_id for q in dataset.queries]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"

    def test_real_dataset_all_practice_areas_covered(self) -> None:
        path = Path("data/eval/test_queries.json")
        if not path.exists():
            pytest.skip("test_queries.json not found")
        settings = EvaluationSettings()
        loader = TestDatasetLoader(settings)
        dataset = loader.load(path)
        areas = {q.practice_area for q in dataset.queries}
        for area in PracticeArea:
            assert area in areas, f"Missing practice area: {area}"

    def test_real_dataset_all_query_types_covered(self) -> None:
        path = Path("data/eval/test_queries.json")
        if not path.exists():
            pytest.skip("test_queries.json not found")
        settings = EvaluationSettings()
        loader = TestDatasetLoader(settings)
        dataset = loader.load(path)
        types = {q.query_type for q in dataset.queries}
        for qt in QueryType:
            assert qt in types, f"Missing query type: {qt}"
