"""Pydantic models for the hallucination mitigation module."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from src.retrieval._models import QueryRoute, RetrievalResult

# --- Enums ---


class CitationStatus(StrEnum):
    """Result of verifying a citation against the knowledge graph."""

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    MISATTRIBUTED = "misattributed"
    KG_UNAVAILABLE = "kg_unavailable"


class ClaimVerdictType(StrEnum):
    """GenGround verdict for an atomic claim."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIALLY_SUPPORTED = "partially_supported"


class CitationType(StrEnum):
    """Type of legal citation extracted from a response."""

    SECTION_REF = "section_ref"
    ARTICLE_REF = "article_ref"
    CASE_CITATION = "case_citation"
    NOTIFICATION_REF = "notification_ref"
    CIRCULAR_REF = "circular_ref"


# --- Citation extraction models ---


class ExtractedCitation(BaseModel):
    """A citation extracted from the LLM response text."""

    text: str
    citation_type: CitationType
    section: str | None = None
    act: str | None = None
    article: str | None = None
    case_citation: str | None = None
    notification_ref: str | None = None
    circular_ref: str | None = None
    span_start: int = 0
    span_end: int = 0


# --- Layer 1: Citation verification ---


class CitationResult(BaseModel):
    """Result of verifying one citation against the knowledge graph."""

    citation: ExtractedCitation
    status: CitationStatus
    kg_node_label: str | None = None
    error: str | None = None


# --- Layer 2: Temporal checking ---


class TemporalWarning(BaseModel):
    """Warning about a section that may not be in force."""

    section: str
    act: str
    warning_text: str
    repealed_by: str | None = None
    replacement_act: str | None = None
    replacement_section: str | None = None
    reference_date: date | None = None


# --- Layer 4: GenGround claim models ---


class ExtractedClaim(BaseModel):
    """An atomic factual claim decomposed from the LLM response."""

    claim_id: int = 0
    text: str
    source_span: str | None = None


class ClaimVerdict(BaseModel):
    """GenGround verdict for a single claim."""

    claim: ExtractedClaim
    verdict: ClaimVerdictType
    confidence: float = 0.0
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    reasoning: str = ""


# --- Layer 3: Confidence scoring ---


class ConfidenceBreakdown(BaseModel):
    """Weighted confidence score with per-factor breakdown."""

    overall_score: float = 0.0
    retrieval_relevance: float = 0.0
    citation_verification_rate: float = 0.0
    source_authority: float = 0.0
    chunk_agreement: float = 0.0
    source_recency: float = 0.0
    query_specificity: float = 0.0
    explanation: str = ""


# --- Pipeline input/output ---


class VerificationInput(BaseModel):
    """Input to the hallucination mitigation pipeline."""

    response_text: str
    retrieval_result: RetrievalResult = Field(default_factory=RetrievalResult)
    reference_date: date | None = None
    route: QueryRoute = QueryRoute.STANDARD


class VerificationSummary(BaseModel):
    """Aggregate stats for a verification run."""

    total_citations: int = 0
    verified_citations: int = 0
    not_found_citations: int = 0
    misattributed_citations: int = 0
    kg_unavailable_citations: int = 0
    temporal_warnings: int = 0
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    partially_supported_claims: int = 0
    confidence_score: float = 0.0
    llm_calls: int = 0


class VerifiedResponse(BaseModel):
    """Output of the hallucination mitigation pipeline."""

    original_response: str
    modified_response: str
    citation_results: list[CitationResult] = Field(default_factory=list)
    temporal_warnings: list[TemporalWarning] = Field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    summary: VerificationSummary = Field(default_factory=VerificationSummary)
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


# --- Config models ---


class HallucinationSettings(BaseModel):
    """Hallucination mitigation settings from configs/hallucination.yaml."""

    # Neo4j (for citation verification & temporal checks)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # LLM (for GenGround)
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # GenGround
    genground_enabled: bool = True
    genground_simple_route_only_audit: bool = True
    genground_max_claims: int = 50
    genground_re_retrieval_top_k: int = 5

    # Confidence scoring weights
    weight_retrieval_relevance: float = 0.25
    weight_citation_verification: float = 0.20
    weight_source_authority: float = 0.20
    weight_chunk_agreement: float = 0.15
    weight_source_recency: float = 0.10
    weight_query_specificity: float = 0.10

    # Thresholds
    confidence_warning_threshold: float = 0.6
    confidence_critical_threshold: float = 0.3

    # Temporal
    ipc_repeal_date: str = "2024-07-01"


class HallucinationConfig(BaseModel):
    """Root model for configs/hallucination.yaml."""

    settings: HallucinationSettings = Field(default_factory=HallucinationSettings)
