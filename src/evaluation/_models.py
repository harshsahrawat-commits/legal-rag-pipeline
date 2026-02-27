"""Pydantic models for the evaluation module."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# --- Enums ---


class PracticeArea(StrEnum):
    """Legal practice area for evaluation queries."""

    CRIMINAL = "criminal"
    CIVIL_CONTRACT = "civil_contract"
    CORPORATE_COMMERCIAL = "corporate_commercial"
    TAX = "tax"
    CONSTITUTIONAL = "constitutional"


class QueryType(StrEnum):
    """Type of evaluation query."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CROSS_REFERENCE = "cross_reference"
    TEMPORAL = "temporal"


class MetricStatus(StrEnum):
    """Result status for a single metric check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


# --- Test Query Dataset ---


class TestQuery(BaseModel):
    """A single test query with expected outcomes."""

    query_id: str
    query_text: str
    practice_area: PracticeArea
    query_type: QueryType
    expected_route: str = "standard"
    expected_citations: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    expected_sections: list[str] = Field(default_factory=list)
    temporal_test: bool = False
    cross_reference_test: bool = False


class TestQueryDataset(BaseModel):
    """Collection of test queries for evaluation."""

    version: str = "1.0"
    description: str = ""
    queries: list[TestQuery] = Field(default_factory=list)


# --- RAGAS Metric Results ---


class RagasMetricResult(BaseModel):
    """Result of a single RAGAS metric on one query."""

    query_id: str
    metric_name: str
    score: float
    status: MetricStatus = MetricStatus.PASS
    error: str | None = None


class RagasAggregateResult(BaseModel):
    """Aggregate RAGAS results across all queries."""

    context_recall: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    per_query: list[RagasMetricResult] = Field(default_factory=list)
    queries_evaluated: int = 0
    errors: list[str] = Field(default_factory=list)


# --- Custom Legal Metric Results ---


class CitationAccuracyResult(BaseModel):
    """Citation accuracy metric result for one query."""

    query_id: str
    total_citations_in_response: int = 0
    citations_found_in_contexts: int = 0
    accuracy: float = 0.0
    missing_citations: list[str] = Field(default_factory=list)


class TemporalAccuracyResult(BaseModel):
    """Temporal accuracy metric result for one query."""

    query_id: str
    total_temporal_references: int = 0
    correct_temporal_references: int = 0
    accuracy: float = 0.0
    violations: list[str] = Field(default_factory=list)


class SectionCompletenessResult(BaseModel):
    """Section completeness for 'what does Section X say' queries."""

    query_id: str
    expected_sections: list[str] = Field(default_factory=list)
    retrieved_sections: list[str] = Field(default_factory=list)
    completeness: float = 0.0


class CrossReferenceResult(BaseModel):
    """Cross-reference resolution for multi-section queries."""

    query_id: str
    expected_sections: list[str] = Field(default_factory=list)
    retrieved_sections: list[str] = Field(default_factory=list)
    resolution_rate: float = 0.0


class LegalMetricsAggregate(BaseModel):
    """Aggregate custom legal metrics across all queries."""

    citation_accuracy: float = 0.0
    temporal_accuracy: float = 0.0
    section_completeness: float = 0.0
    cross_reference_resolution: float = 0.0
    citation_details: list[CitationAccuracyResult] = Field(default_factory=list)
    temporal_details: list[TemporalAccuracyResult] = Field(default_factory=list)
    completeness_details: list[SectionCompletenessResult] = Field(default_factory=list)
    cross_reference_details: list[CrossReferenceResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# --- Latency Metric Results ---


class LatencyResult(BaseModel):
    """Latency check for one query."""

    query_id: str
    route: str
    elapsed_ms: float
    target_ms: float
    status: MetricStatus = MetricStatus.PASS


class LatencyAggregate(BaseModel):
    """Aggregate latency metrics."""

    pass_rate: float = 0.0
    per_route: dict[str, float] = Field(default_factory=dict)
    per_query: list[LatencyResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# --- Query Intelligence Metrics ---


class QIMetrics(BaseModel):
    """Query Intelligence layer metrics."""

    cache_hit_rate: float = 0.0
    routing_accuracy: float = 0.0
    genground_verification_rate: float = 0.0
    parent_context_utilization: float = 0.0
    flare_frequency: float = 0.0
    details: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# --- Human Evaluation ---


class HumanScore(BaseModel):
    """One evaluator's score for one query."""

    query_id: str
    evaluator_id: str
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    recency: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    notes: str = ""


class HumanEvalAggregate(BaseModel):
    """Aggregate human evaluation scores."""

    avg_accuracy: float = 0.0
    avg_completeness: float = 0.0
    avg_recency: float = 0.0
    avg_usefulness: float = 0.0
    accuracy_pass_rate: float = 0.0
    total_evaluations: int = 0
    scores: list[HumanScore] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# --- Pipeline Input ---


class EvaluationInput(BaseModel):
    """Collected pipeline results for one query, fed to the evaluator."""

    query_id: str
    query_text: str
    practice_area: str = ""
    query_type: str = ""
    expected_route: str = ""
    expected_citations: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    temporal_test: bool = False
    cross_reference_test: bool = False
    # Upstream results (populated by pipeline run)
    response_text: str = ""
    retrieved_contexts: list[str] = Field(default_factory=list)
    qi_result: dict = Field(default_factory=dict)
    retrieval_result: dict = Field(default_factory=dict)
    verification_result: dict = Field(default_factory=dict)
    total_elapsed_ms: float = 0.0


# --- Pipeline Output ---


class EvaluationResult(BaseModel):
    """Full evaluation output."""

    ragas: RagasAggregateResult = Field(default_factory=RagasAggregateResult)
    legal_metrics: LegalMetricsAggregate = Field(default_factory=LegalMetricsAggregate)
    latency: LatencyAggregate = Field(default_factory=LatencyAggregate)
    qi_metrics: QIMetrics = Field(default_factory=QIMetrics)
    human_eval: HumanEvalAggregate = Field(default_factory=HumanEvalAggregate)
    queries_evaluated: int = 0
    timings: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed time in milliseconds."""
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds() * 1000

    @property
    def all_targets_met(self) -> bool:
        """Check if all primary metric targets are met."""
        return (
            self.ragas.context_recall >= 0.90
            and self.ragas.context_precision >= 0.85
            and self.ragas.faithfulness >= 0.95
            and self.ragas.answer_relevancy >= 0.85
            and self.legal_metrics.citation_accuracy >= 0.98
            and self.legal_metrics.temporal_accuracy >= 0.99
            and self.legal_metrics.section_completeness >= 0.90
            and self.legal_metrics.cross_reference_resolution >= 0.80
        )


# --- Settings ---


class EvaluationSettings(BaseModel):
    """Evaluation settings from configs/evaluation.yaml."""

    # RAGAS
    ragas_enabled: bool = True
    ragas_llm_model: str = "claude-haiku-4-5-20251001"
    ragas_batch_size: int = 10

    # Legal metrics
    legal_metrics_enabled: bool = True

    # Latency targets (ms)
    latency_simple_ms: float = 200.0
    latency_standard_ms: float = 800.0
    latency_complex_ms: float = 2000.0
    latency_analytical_ms: float = 5000.0

    # QI metrics targets
    cache_hit_target: float = 0.30
    routing_accuracy_target: float = 0.90
    genground_target: float = 0.95
    parent_utilization_target: float = 0.80
    flare_min: float = 0.05
    flare_max: float = 0.10

    # Human evaluation
    human_eval_enabled: bool = False
    accuracy_pass_threshold: int = 4
    accuracy_pass_rate_target: float = 0.85

    # Test dataset path
    test_queries_path: str = "data/eval/test_queries.json"
    worksheets_dir: str = "data/eval/worksheets"
    scoresheets_dir: str = "data/eval/scoresheets"

    # Report
    report_format: str = "json"
    report_output_path: str = "data/eval/report.json"


class EvaluationConfig(BaseModel):
    """Root model for configs/evaluation.yaml."""

    settings: EvaluationSettings = Field(default_factory=EvaluationSettings)
