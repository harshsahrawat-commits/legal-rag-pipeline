"""RAGAS v0.4 evaluation wrapper.

Lazy-imports ragas, langchain-anthropic, and datasets.
Raises RagasNotAvailableError if dependencies are missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation._exceptions import RagasEvaluationError, RagasNotAvailableError
from src.evaluation._models import MetricStatus, RagasAggregateResult, RagasMetricResult
from src.utils._logging import get_logger

if TYPE_CHECKING:
    import types

    from src.evaluation._models import EvaluationInput, EvaluationSettings

_log = get_logger(__name__)


class RagasEvaluator:
    """Wrapper around RAGAS v0.4 for RAG evaluation metrics."""

    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    async def evaluate(self, inputs: list[EvaluationInput]) -> RagasAggregateResult:
        """Run RAGAS evaluation on the provided inputs.

        Metrics: ContextRecall, ContextPrecision, Faithfulness, AnswerRelevancy.

        Args:
            inputs: List of EvaluationInput with response_text and retrieved_contexts.

        Returns:
            RagasAggregateResult with per-query and aggregate scores.

        Raises:
            RagasNotAvailableError: If ragas or langchain-anthropic not installed.
            RagasEvaluationError: If RAGAS evaluation fails during execution.
        """
        if not inputs:
            return RagasAggregateResult()

        # Filter to inputs with non-empty response_text
        valid = [inp for inp in inputs if inp.response_text.strip()]
        if not valid:
            return RagasAggregateResult()

        try:
            ragas_mod = self._import_ragas()
        except ImportError as exc:
            raise RagasNotAvailableError(
                "RAGAS dependencies not installed. "
                "Install with: pip install 'legal-rag-pipeline[evaluation]'"
            ) from exc

        try:
            llm = self._init_llm()
            metrics = self._build_metrics(ragas_mod, llm)
            dataset = self._build_dataset(ragas_mod, valid)

            from ragas import evaluate as ragas_evaluate

            result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                batch_size=self._settings.ragas_batch_size,
            )

            return self._parse_result(result, valid)
        except RagasNotAvailableError:
            raise
        except Exception as exc:
            raise RagasEvaluationError(f"RAGAS evaluation failed: {exc}") from exc

    def _import_ragas(self) -> types.ModuleType:
        """Lazy-import ragas module.

        Returns:
            The ragas module.

        Raises:
            ImportError: If ragas is not installed.
        """
        import ragas

        return ragas

    def _init_llm(self) -> object:
        """Initialize the LLM wrapper for RAGAS.

        Returns:
            A ChatAnthropic instance configured with the model from settings.

        Raises:
            RagasNotAvailableError: If langchain-anthropic is not installed.
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RagasNotAvailableError(
                "langchain-anthropic not installed. Install with: pip install langchain-anthropic"
            ) from exc

        return ChatAnthropic(model=self._settings.ragas_llm_model)

    def _build_metrics(self, ragas_mod: types.ModuleType, llm: object) -> list[object]:
        """Build the 4 core RAGAS metrics.

        Args:
            ragas_mod: The ragas module (unused but kept for API consistency).
            llm: The LLM wrapper to pass to each metric.

        Returns:
            List of 4 RAGAS metric instances.
        """
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        return [
            ContextRecall(llm=llm),
            ContextPrecision(llm=llm),
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm),
        ]

    def _build_dataset(self, ragas_mod: types.ModuleType, inputs: list[EvaluationInput]) -> object:
        """Convert EvaluationInputs to a RAGAS EvaluationDataset.

        Args:
            ragas_mod: The ragas module (unused but kept for API consistency).
            inputs: Filtered list of EvaluationInput with non-empty responses.

        Returns:
            A RAGAS EvaluationDataset.
        """
        from ragas import EvaluationDataset, SingleTurnSample

        samples = []
        for inp in inputs:
            sample = SingleTurnSample(
                user_input=inp.query_text,
                response=inp.response_text,
                retrieved_contexts=inp.retrieved_contexts or [],
                reference=inp.reference_answer if inp.reference_answer else None,
            )
            samples.append(sample)

        return EvaluationDataset(samples=samples)

    def _parse_result(self, result: object, inputs: list[EvaluationInput]) -> RagasAggregateResult:
        """Parse RAGAS result into our RagasAggregateResult model.

        Args:
            result: The raw RAGAS evaluation result (has to_pandas()).
            inputs: The EvaluationInputs that were evaluated.

        Returns:
            RagasAggregateResult with per-query scores and aggregates.
        """
        per_query: list[RagasMetricResult] = []
        errors: list[str] = []

        metric_names = [
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_relevancy",
        ]

        # Extract per-query scores from the result dataframe
        try:
            df = result.to_pandas()  # type: ignore[union-attr]
            for i, inp in enumerate(inputs):
                if i >= len(df):
                    break
                row = df.iloc[i]
                for metric_name in metric_names:
                    score = row.get(metric_name, float("nan"))
                    # Handle NaN -- NaN != NaN
                    if score != score:
                        score = 0.0
                        errors.append(f"NaN score for {metric_name} on {inp.query_id}")
                    per_query.append(
                        RagasMetricResult(
                            query_id=inp.query_id,
                            metric_name=metric_name,
                            score=float(score),
                            status=MetricStatus.PASS,
                        )
                    )
        except Exception as exc:
            errors.append(f"Failed to parse RAGAS results: {exc}")

        # Compute aggregate scores
        def _mean_for(name: str) -> float:
            scores = [r.score for r in per_query if r.metric_name == name]
            return sum(scores) / len(scores) if scores else 0.0

        return RagasAggregateResult(
            context_recall=_mean_for("context_recall"),
            context_precision=_mean_for("context_precision"),
            faithfulness=_mean_for("faithfulness"),
            answer_relevancy=_mean_for("answer_relevancy"),
            per_query=per_query,
            queries_evaluated=len(inputs),
            errors=errors,
        )
