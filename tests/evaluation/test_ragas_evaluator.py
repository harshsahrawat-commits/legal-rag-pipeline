"""Tests for RAGAS evaluator."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation._exceptions import RagasEvaluationError, RagasNotAvailableError
from src.evaluation._models import (
    EvaluationInput,
    EvaluationSettings,
    MetricStatus,
    RagasAggregateResult,
)
from src.evaluation._ragas_evaluator import RagasEvaluator

# --- Helpers ---


def _make_input(
    *,
    query_id: str = "q1",
    query_text: str = "test query",
    response_text: str = "test response",
    retrieved_contexts: list[str] | None = None,
    reference_answer: str = "",
) -> EvaluationInput:
    return EvaluationInput(
        query_id=query_id,
        query_text=query_text,
        response_text=response_text,
        retrieved_contexts=retrieved_contexts or ["context text"],
        reference_answer=reference_answer,
    )


def _make_mock_df(rows: list[dict[str, float]]) -> MagicMock:
    """Create a mock DataFrame with iloc and len support."""
    mock_df = MagicMock()
    mock_df.__len__ = lambda self: len(rows)

    class _MockRow:
        def __init__(self, data: dict[str, float]) -> None:
            self._data = data

        def get(self, key: str, default: float = 0.0) -> float:
            return self._data.get(key, default)

    mock_df.iloc.__getitem__ = lambda self, idx: _MockRow(rows[idx])
    return mock_df


def _make_mock_result(rows: list[dict[str, float]]) -> MagicMock:
    """Create a mock RAGAS evaluation result with to_pandas()."""
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = _make_mock_df(rows)
    return mock_result


def _make_fake_ragas(
    evaluate_fn: MagicMock | None = None,
) -> tuple[types.ModuleType, MagicMock]:
    """Create a fake ragas module with evaluate function for sys.modules patching.

    Returns:
        Tuple of (fake_ragas_module, mock_evaluate_fn).
    """
    mock_eval = evaluate_fn or MagicMock()
    fake_ragas = types.ModuleType("ragas")
    fake_ragas.evaluate = mock_eval  # type: ignore[attr-defined]
    return fake_ragas, mock_eval


@pytest.fixture()
def settings() -> EvaluationSettings:
    return EvaluationSettings()


@pytest.fixture()
def evaluator(settings: EvaluationSettings) -> RagasEvaluator:
    return RagasEvaluator(settings)


# ===================================================================
# TestRagasEvaluatorInit
# ===================================================================


class TestRagasEvaluatorInit:
    """Tests for RagasEvaluator initialization."""

    def test_init_stores_settings(self, settings: EvaluationSettings) -> None:
        """Settings are stored on the evaluator instance."""
        ev = RagasEvaluator(settings)
        assert ev._settings is settings


# ===================================================================
# TestRagasEvaluatorEvaluate
# ===================================================================


class TestRagasEvaluatorEvaluate:
    """Tests for the evaluate() async method."""

    @pytest.mark.asyncio
    async def test_empty_inputs_returns_default(self, evaluator: RagasEvaluator) -> None:
        """Empty list returns default RagasAggregateResult."""
        result = await evaluator.evaluate([])
        assert isinstance(result, RagasAggregateResult)
        assert result.queries_evaluated == 0
        assert result.per_query == []

    @pytest.mark.asyncio
    async def test_all_empty_responses_returns_default(self, evaluator: RagasEvaluator) -> None:
        """Inputs with only empty response_text return default result."""
        inputs = [
            _make_input(query_id="q1", response_text=""),
            _make_input(query_id="q2", response_text="   "),
        ]
        result = await evaluator.evaluate(inputs)
        assert isinstance(result, RagasAggregateResult)
        assert result.queries_evaluated == 0

    @pytest.mark.asyncio
    async def test_raises_ragas_not_available_when_ragas_missing(
        self, evaluator: RagasEvaluator
    ) -> None:
        """ImportError on _import_ragas raises RagasNotAvailableError."""
        with (
            patch.object(evaluator, "_import_ragas", side_effect=ImportError("no ragas")),
            pytest.raises(RagasNotAvailableError, match="RAGAS dependencies"),
        ):
            await evaluator.evaluate([_make_input()])

    @pytest.mark.asyncio
    async def test_raises_ragas_not_available_when_langchain_missing(
        self, evaluator: RagasEvaluator
    ) -> None:
        """RagasNotAvailableError from _init_llm propagates."""
        mock_mod = MagicMock(spec=types.ModuleType)
        with (
            patch.object(evaluator, "_import_ragas", return_value=mock_mod),
            patch.object(
                evaluator,
                "_init_llm",
                side_effect=RagasNotAvailableError("langchain missing"),
            ),
            pytest.raises(RagasNotAvailableError, match="langchain missing"),
        ):
            await evaluator.evaluate([_make_input()])

    @pytest.mark.asyncio
    async def test_successful_evaluation_returns_aggregate(self, evaluator: RagasEvaluator) -> None:
        """Full successful chain returns populated RagasAggregateResult."""
        mock_mod = MagicMock(spec=types.ModuleType)
        mock_llm = MagicMock()
        mock_metrics = [MagicMock() for _ in range(4)]
        mock_dataset = MagicMock()
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 0.9,
                    "context_precision": 0.85,
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.88,
                }
            ]
        )

        fake_ragas, _ = _make_fake_ragas(MagicMock(return_value=mock_result))

        with (
            patch.object(evaluator, "_import_ragas", return_value=mock_mod),
            patch.object(evaluator, "_init_llm", return_value=mock_llm),
            patch.object(evaluator, "_build_metrics", return_value=mock_metrics),
            patch.object(evaluator, "_build_dataset", return_value=mock_dataset),
            patch.dict("sys.modules", {"ragas": fake_ragas}),
        ):
            result = await evaluator.evaluate([_make_input()])

        assert isinstance(result, RagasAggregateResult)
        assert result.queries_evaluated == 1
        assert result.context_recall == pytest.approx(0.9)
        assert result.context_precision == pytest.approx(0.85)
        assert result.faithfulness == pytest.approx(0.95)
        assert result.answer_relevancy == pytest.approx(0.88)
        assert len(result.per_query) == 4

    @pytest.mark.asyncio
    async def test_ragas_evaluation_error_on_unexpected_exception(
        self, evaluator: RagasEvaluator
    ) -> None:
        """Unexpected exception during evaluation raises RagasEvaluationError."""
        mock_mod = MagicMock(spec=types.ModuleType)
        mock_llm = MagicMock()

        with (
            patch.object(evaluator, "_import_ragas", return_value=mock_mod),
            patch.object(evaluator, "_init_llm", return_value=mock_llm),
            patch.object(
                evaluator,
                "_build_metrics",
                side_effect=RuntimeError("metrics boom"),
            ),
            pytest.raises(RagasEvaluationError, match="RAGAS evaluation failed"),
        ):
            await evaluator.evaluate([_make_input()])

    @pytest.mark.asyncio
    async def test_filters_empty_responses(self, evaluator: RagasEvaluator) -> None:
        """Only inputs with non-empty response_text are evaluated."""
        inputs = [
            _make_input(query_id="q1", response_text="valid response"),
            _make_input(query_id="q2", response_text=""),
            _make_input(query_id="q3", response_text="another valid"),
        ]

        mock_mod = MagicMock(spec=types.ModuleType)
        mock_llm = MagicMock()
        mock_metrics = [MagicMock() for _ in range(4)]
        mock_dataset = MagicMock()
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 0.8,
                    "context_precision": 0.8,
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.8,
                },
                {
                    "context_recall": 0.9,
                    "context_precision": 0.9,
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.9,
                },
            ]
        )

        build_dataset_calls: list[list[EvaluationInput]] = []

        def _capture_dataset(_mod: object, valid_inputs: list[EvaluationInput]) -> MagicMock:
            build_dataset_calls.append(valid_inputs)
            return mock_dataset

        fake_ragas, _ = _make_fake_ragas(MagicMock(return_value=mock_result))

        with (
            patch.object(evaluator, "_import_ragas", return_value=mock_mod),
            patch.object(evaluator, "_init_llm", return_value=mock_llm),
            patch.object(evaluator, "_build_metrics", return_value=mock_metrics),
            patch.object(evaluator, "_build_dataset", side_effect=_capture_dataset),
            patch.dict("sys.modules", {"ragas": fake_ragas}),
        ):
            result = await evaluator.evaluate(inputs)

        # Only 2 valid inputs passed to _build_dataset
        assert len(build_dataset_calls) == 1
        assert len(build_dataset_calls[0]) == 2
        assert result.queries_evaluated == 2

    @pytest.mark.asyncio
    async def test_nan_scores_default_to_zero(self, evaluator: RagasEvaluator) -> None:
        """NaN scores in result are replaced with 0.0 and logged as errors."""
        mock_mod = MagicMock(spec=types.ModuleType)
        mock_llm = MagicMock()
        mock_metrics = [MagicMock() for _ in range(4)]
        mock_dataset = MagicMock()
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": float("nan"),
                    "context_precision": 0.85,
                    "faithfulness": float("nan"),
                    "answer_relevancy": 0.88,
                }
            ]
        )

        fake_ragas, _ = _make_fake_ragas(MagicMock(return_value=mock_result))

        with (
            patch.object(evaluator, "_import_ragas", return_value=mock_mod),
            patch.object(evaluator, "_init_llm", return_value=mock_llm),
            patch.object(evaluator, "_build_metrics", return_value=mock_metrics),
            patch.object(evaluator, "_build_dataset", return_value=mock_dataset),
            patch.dict("sys.modules", {"ragas": fake_ragas}),
        ):
            result = await evaluator.evaluate([_make_input()])

        # NaN scores become 0.0
        recall_scores = [r.score for r in result.per_query if r.metric_name == "context_recall"]
        assert recall_scores == [0.0]
        faith_scores = [r.score for r in result.per_query if r.metric_name == "faithfulness"]
        assert faith_scores == [0.0]
        # Errors recorded
        assert any("NaN" in e and "context_recall" in e for e in result.errors)
        assert any("NaN" in e and "faithfulness" in e for e in result.errors)
        # Non-NaN scores preserved
        assert result.context_precision == pytest.approx(0.85)
        assert result.answer_relevancy == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_batch_size_from_settings(self) -> None:
        """Settings ragas_batch_size is passed to ragas.evaluate."""
        custom_settings = EvaluationSettings(ragas_batch_size=42)
        ev = RagasEvaluator(custom_settings)

        mock_mod = MagicMock(spec=types.ModuleType)
        mock_llm = MagicMock()
        mock_metrics = [MagicMock() for _ in range(4)]
        mock_dataset = MagicMock()
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 0.5,
                    "context_precision": 0.5,
                    "faithfulness": 0.5,
                    "answer_relevancy": 0.5,
                }
            ]
        )

        mock_ragas_evaluate = MagicMock(return_value=mock_result)
        fake_ragas, _ = _make_fake_ragas(mock_ragas_evaluate)

        with (
            patch.object(ev, "_import_ragas", return_value=mock_mod),
            patch.object(ev, "_init_llm", return_value=mock_llm),
            patch.object(ev, "_build_metrics", return_value=mock_metrics),
            patch.object(ev, "_build_dataset", return_value=mock_dataset),
            patch.dict("sys.modules", {"ragas": fake_ragas}),
        ):
            await ev.evaluate([_make_input()])

        mock_ragas_evaluate.assert_called_once()
        call_kwargs = mock_ragas_evaluate.call_args
        assert call_kwargs.kwargs["batch_size"] == 42


# ===================================================================
# TestRagasEvaluatorImportRagas
# ===================================================================


class TestRagasEvaluatorImportRagas:
    """Tests for _import_ragas."""

    def test_import_ragas_success(self, evaluator: RagasEvaluator) -> None:
        """If ragas is installed, returns the module."""
        fake_ragas = types.ModuleType("ragas")
        with patch.dict("sys.modules", {"ragas": fake_ragas}):
            result = evaluator._import_ragas()
        assert result is fake_ragas

    def test_import_ragas_missing_raises(self, evaluator: RagasEvaluator) -> None:
        """If ragas is not installed, raises ImportError."""
        with patch.dict("sys.modules", {"ragas": None}), pytest.raises(ImportError):
            evaluator._import_ragas()


# ===================================================================
# TestRagasEvaluatorInitLLM
# ===================================================================


class TestRagasEvaluatorInitLLM:
    """Tests for _init_llm."""

    def test_init_llm_returns_langchain_model(self, evaluator: RagasEvaluator) -> None:
        """_init_llm returns a LangChain chat model via get_langchain_llm."""
        mock_llm = MagicMock()
        with patch(
            "src.utils._llm_client.get_langchain_llm", return_value=mock_llm
        ):
            result = evaluator._init_llm()

        assert result is mock_llm

    def test_init_llm_raises_when_provider_unavailable(self, evaluator: RagasEvaluator) -> None:
        """LLMNotAvailableError from get_langchain_llm raises RagasNotAvailableError."""
        from src.utils._exceptions import LLMNotAvailableError

        with (
            patch(
                "src.utils._llm_client.get_langchain_llm",
                side_effect=LLMNotAvailableError("no provider configured"),
            ),
            pytest.raises(RagasNotAvailableError, match="no provider configured"),
        ):
            evaluator._init_llm()

    def test_init_llm_calls_with_ragas_component(self, evaluator: RagasEvaluator) -> None:
        """get_langchain_llm is called with 'ragas' component name."""
        mock_llm = MagicMock()
        with patch(
            "src.utils._llm_client.get_langchain_llm", return_value=mock_llm
        ) as mock_get:
            evaluator._init_llm()

        mock_get.assert_called_once_with("ragas")


# ===================================================================
# TestRagasEvaluatorBuildMetrics
# ===================================================================


class TestRagasEvaluatorBuildMetrics:
    """Tests for _build_metrics."""

    def _setup_ragas_metrics_mocks(self) -> dict[str, MagicMock]:
        """Create mock ragas.metrics module with 4 metric classes."""
        mocks: dict[str, MagicMock] = {}
        for name in [
            "ContextRecall",
            "ContextPrecision",
            "Faithfulness",
            "AnswerRelevancy",
        ]:
            mocks[name] = MagicMock()

        fake_metrics = types.ModuleType("ragas.metrics")
        fake_ragas = types.ModuleType("ragas")
        for name, mock in mocks.items():
            setattr(fake_metrics, name, mock)

        return {
            "mocks": mocks,
            "fake_metrics": fake_metrics,
            "fake_ragas": fake_ragas,
        }

    def test_build_metrics_returns_four(self, evaluator: RagasEvaluator) -> None:
        """_build_metrics returns exactly 4 metric instances."""
        setup = self._setup_ragas_metrics_mocks()

        with patch.dict(
            "sys.modules",
            {
                "ragas": setup["fake_ragas"],
                "ragas.metrics": setup["fake_metrics"],
            },
        ):
            result = evaluator._build_metrics(setup["fake_ragas"], MagicMock())

        assert len(result) == 4

    def test_build_metrics_types(self, evaluator: RagasEvaluator) -> None:
        """_build_metrics creates ContextRecall, ContextPrecision, Faithfulness, AnswerRelevancy."""
        setup = self._setup_ragas_metrics_mocks()
        mocks = setup["mocks"]

        mock_llm = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "ragas": setup["fake_ragas"],
                "ragas.metrics": setup["fake_metrics"],
            },
        ):
            evaluator._build_metrics(setup["fake_ragas"], mock_llm)

        # Each metric class was called with llm=mock_llm
        for name in ["ContextRecall", "ContextPrecision", "Faithfulness", "AnswerRelevancy"]:
            mocks[name].assert_called_once_with(llm=mock_llm)


# ===================================================================
# TestRagasEvaluatorBuildDataset
# ===================================================================


class TestRagasEvaluatorBuildDataset:
    """Tests for _build_dataset."""

    def _setup_ragas_dataset_mocks(self) -> dict[str, MagicMock]:
        """Create mock ragas module with EvaluationDataset and SingleTurnSample."""
        mock_sample_cls = MagicMock()
        mock_dataset_cls = MagicMock()

        fake_ragas = types.ModuleType("ragas")
        fake_ragas.EvaluationDataset = mock_dataset_cls  # type: ignore[attr-defined]
        fake_ragas.SingleTurnSample = mock_sample_cls  # type: ignore[attr-defined]

        return {
            "mock_sample_cls": mock_sample_cls,
            "mock_dataset_cls": mock_dataset_cls,
            "fake_ragas": fake_ragas,
        }

    def test_build_dataset_maps_fields(self, evaluator: RagasEvaluator) -> None:
        """EvaluationInput fields are correctly mapped to SingleTurnSample."""
        setup = self._setup_ragas_dataset_mocks()
        mock_sample_cls = setup["mock_sample_cls"]

        inp = _make_input(
            query_text="What is Section 302?",
            response_text="Section 302 provides...",
            retrieved_contexts=["Section 302. Murder."],
            reference_answer="Section 302 of IPC.",
        )

        with patch.dict("sys.modules", {"ragas": setup["fake_ragas"]}):
            evaluator._build_dataset(setup["fake_ragas"], [inp])

        mock_sample_cls.assert_called_once_with(
            user_input="What is Section 302?",
            response="Section 302 provides...",
            retrieved_contexts=["Section 302. Murder."],
            reference="Section 302 of IPC.",
        )

    def test_build_dataset_none_reference_for_empty(self, evaluator: RagasEvaluator) -> None:
        """Empty reference_answer becomes None in the sample."""
        setup = self._setup_ragas_dataset_mocks()
        mock_sample_cls = setup["mock_sample_cls"]

        inp = _make_input(reference_answer="")

        with patch.dict("sys.modules", {"ragas": setup["fake_ragas"]}):
            evaluator._build_dataset(setup["fake_ragas"], [inp])

        call_kwargs = mock_sample_cls.call_args.kwargs
        assert call_kwargs["reference"] is None

    def test_build_dataset_empty_contexts(self, evaluator: RagasEvaluator) -> None:
        """Empty retrieved_contexts results in [] in sample."""
        setup = self._setup_ragas_dataset_mocks()
        mock_sample_cls = setup["mock_sample_cls"]

        inp = EvaluationInput(
            query_id="q1",
            query_text="query",
            response_text="response",
            retrieved_contexts=[],
        )

        with patch.dict("sys.modules", {"ragas": setup["fake_ragas"]}):
            evaluator._build_dataset(setup["fake_ragas"], [inp])

        call_kwargs = mock_sample_cls.call_args.kwargs
        assert call_kwargs["retrieved_contexts"] == []

    def test_build_dataset_multiple_inputs(self, evaluator: RagasEvaluator) -> None:
        """Multiple inputs create multiple samples."""
        setup = self._setup_ragas_dataset_mocks()
        mock_sample_cls = setup["mock_sample_cls"]
        mock_dataset_cls = setup["mock_dataset_cls"]

        inputs = [
            _make_input(query_id="q1", query_text="first query"),
            _make_input(query_id="q2", query_text="second query"),
        ]

        with patch.dict("sys.modules", {"ragas": setup["fake_ragas"]}):
            evaluator._build_dataset(setup["fake_ragas"], inputs)

        assert mock_sample_cls.call_count == 2
        mock_dataset_cls.assert_called_once()
        # The dataset receives a list of 2 samples
        samples_arg = mock_dataset_cls.call_args.kwargs.get(
            "samples",
            mock_dataset_cls.call_args.args[0] if mock_dataset_cls.call_args.args else None,
        )
        assert len(samples_arg) == 2


# ===================================================================
# TestRagasEvaluatorParseResult
# ===================================================================


class TestRagasEvaluatorParseResult:
    """Tests for _parse_result."""

    def test_parse_result_computes_aggregates(self, evaluator: RagasEvaluator) -> None:
        """Parse result computes correct averages across queries."""
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 0.8,
                    "context_precision": 0.7,
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.85,
                },
                {
                    "context_recall": 0.6,
                    "context_precision": 0.5,
                    "faithfulness": 0.7,
                    "answer_relevancy": 0.75,
                },
            ]
        )

        inputs = [
            _make_input(query_id="q1"),
            _make_input(query_id="q2"),
        ]

        result = evaluator._parse_result(mock_result, inputs)

        assert result.context_recall == pytest.approx(0.7)
        assert result.context_precision == pytest.approx(0.6)
        assert result.faithfulness == pytest.approx(0.8)
        assert result.answer_relevancy == pytest.approx(0.8)
        assert result.queries_evaluated == 2
        assert len(result.per_query) == 8  # 2 queries * 4 metrics

    def test_parse_result_handles_missing_metrics(self, evaluator: RagasEvaluator) -> None:
        """Missing metric column in dataframe defaults to NaN -> 0.0."""
        # Only context_recall present, others missing -> get(name, nan) -> NaN -> 0.0
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 0.9,
                    # context_precision, faithfulness, answer_relevancy missing
                }
            ]
        )

        inputs = [_make_input(query_id="q1")]
        result = evaluator._parse_result(mock_result, inputs)

        assert result.context_recall == pytest.approx(0.9)
        # Missing metrics should produce NaN -> 0.0
        assert result.context_precision == pytest.approx(0.0)
        assert result.faithfulness == pytest.approx(0.0)
        assert result.answer_relevancy == pytest.approx(0.0)
        # Errors for each NaN
        assert len(result.errors) == 3
        assert all("NaN" in e for e in result.errors)

    def test_parse_result_records_errors_on_exception(self, evaluator: RagasEvaluator) -> None:
        """Exception during parsing records error string."""
        mock_result = MagicMock()
        mock_result.to_pandas.side_effect = RuntimeError("pandas boom")

        inputs = [_make_input(query_id="q1")]
        result = evaluator._parse_result(mock_result, inputs)

        assert len(result.errors) == 1
        assert "Failed to parse RAGAS results" in result.errors[0]
        assert "pandas boom" in result.errors[0]
        assert result.queries_evaluated == 1
        assert result.per_query == []

    def test_parse_result_single_query_all_metrics(self, evaluator: RagasEvaluator) -> None:
        """Single query produces 4 per_query entries."""
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 1.0,
                    "context_precision": 0.95,
                    "faithfulness": 0.98,
                    "answer_relevancy": 0.92,
                }
            ]
        )

        inputs = [_make_input(query_id="single")]
        result = evaluator._parse_result(mock_result, inputs)

        assert len(result.per_query) == 4
        query_ids = {r.query_id for r in result.per_query}
        assert query_ids == {"single"}
        metric_names = {r.metric_name for r in result.per_query}
        assert metric_names == {
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_relevancy",
        }
        # All should be PASS status
        assert all(r.status == MetricStatus.PASS for r in result.per_query)

    def test_parse_result_more_inputs_than_rows(self, evaluator: RagasEvaluator) -> None:
        """If DataFrame has fewer rows than inputs, only available rows are parsed."""
        mock_result = _make_mock_result(
            [
                {
                    "context_recall": 0.8,
                    "context_precision": 0.8,
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.8,
                }
            ]
        )

        inputs = [
            _make_input(query_id="q1"),
            _make_input(query_id="q2"),
            _make_input(query_id="q3"),
        ]
        result = evaluator._parse_result(mock_result, inputs)

        # Only 1 row in df, so only 4 per-query results (1 query * 4 metrics)
        assert len(result.per_query) == 4
        assert all(r.query_id == "q1" for r in result.per_query)
        assert result.queries_evaluated == 3
