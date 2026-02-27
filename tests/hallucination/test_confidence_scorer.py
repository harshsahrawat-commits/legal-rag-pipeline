"""Tests for Layer 3: Confidence Scorer."""

from __future__ import annotations

import pytest

from src.hallucination._confidence_scorer import ConfidenceScorer
from src.hallucination._models import (
    CitationResult,
    CitationStatus,
    CitationType,
    ClaimVerdict,
    ClaimVerdictType,
    ExtractedCitation,
    ExtractedClaim,
    HallucinationSettings,
)
from src.retrieval._models import ExpandedContext, QueryRoute


@pytest.fixture()
def settings() -> HallucinationSettings:
    return HallucinationSettings()


@pytest.fixture()
def scorer(settings: HallucinationSettings) -> ConfidenceScorer:
    return ConfidenceScorer(settings)


def _make_chunk(
    chunk_id: str = "c1",
    relevance: float = 0.9,
    doc_type: str = "statute",
    court_level: int = 0,
    date_str: str = "2023-01-15",
) -> ExpandedContext:
    return ExpandedContext(
        chunk_id=chunk_id,
        chunk_text="text",
        relevance_score=relevance,
        metadata={
            "document_type": doc_type,
            "court_hierarchy": court_level,
            "date_decided": date_str,
        },
    )


def _make_citation_result(status: CitationStatus) -> CitationResult:
    return CitationResult(
        citation=ExtractedCitation(text="S.420 IPC", citation_type=CitationType.SECTION_REF),
        status=status,
    )


def _make_claim_verdict(verdict: ClaimVerdictType) -> ClaimVerdict:
    return ClaimVerdict(
        claim=ExtractedClaim(claim_id=1, text="claim"),
        verdict=verdict,
        confidence=0.9,
    )


class TestConfidenceScoreRange:
    def test_all_high_inputs(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(relevance=0.95, doc_type="statute")]
        citations = [_make_citation_result(CitationStatus.VERIFIED)]
        claims = [_make_claim_verdict(ClaimVerdictType.SUPPORTED)]
        result = scorer.score(chunks, citations, claims, QueryRoute.SIMPLE)
        assert 0.7 <= result.overall_score <= 1.0

    def test_all_low_inputs(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(relevance=0.1, doc_type="unknown", court_level=5)]
        citations = [_make_citation_result(CitationStatus.NOT_FOUND)]
        claims = [_make_claim_verdict(ClaimVerdictType.UNSUPPORTED)]
        result = scorer.score(chunks, citations, claims, QueryRoute.ANALYTICAL)
        assert result.overall_score < 0.5

    def test_mixed_inputs(self, scorer: ConfidenceScorer) -> None:
        chunks = [
            _make_chunk(relevance=0.9),
            _make_chunk(chunk_id="c2", relevance=0.3),
        ]
        citations = [
            _make_citation_result(CitationStatus.VERIFIED),
            _make_citation_result(CitationStatus.NOT_FOUND),
        ]
        claims = [
            _make_claim_verdict(ClaimVerdictType.SUPPORTED),
            _make_claim_verdict(ClaimVerdictType.UNSUPPORTED),
        ]
        result = scorer.score(chunks, citations, claims, QueryRoute.STANDARD)
        assert 0.3 <= result.overall_score <= 0.8

    def test_score_clamped_to_0_1(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(relevance=0.95)]
        result = scorer.score(chunks, [], [], QueryRoute.SIMPLE)
        assert 0.0 <= result.overall_score <= 1.0


class TestRetrievalRelevance:
    def test_empty_chunks(self, scorer: ConfidenceScorer) -> None:
        result = scorer.score([], [], [], QueryRoute.STANDARD)
        assert result.retrieval_relevance == 0.0

    def test_high_relevance(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(relevance=0.95)]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert result.retrieval_relevance >= 0.9


class TestCitationVerificationRate:
    def test_all_verified(self, scorer: ConfidenceScorer) -> None:
        citations = [_make_citation_result(CitationStatus.VERIFIED)] * 3
        result = scorer.score([], citations, [], QueryRoute.STANDARD)
        assert result.citation_verification_rate == 1.0

    def test_none_verified(self, scorer: ConfidenceScorer) -> None:
        citations = [_make_citation_result(CitationStatus.NOT_FOUND)] * 3
        result = scorer.score([], citations, [], QueryRoute.STANDARD)
        assert result.citation_verification_rate == 0.0

    def test_no_citations(self, scorer: ConfidenceScorer) -> None:
        result = scorer.score([], [], [], QueryRoute.STANDARD)
        assert result.citation_verification_rate == 1.0  # No penalty


class TestSourceAuthority:
    def test_supreme_court(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(doc_type="judgment", court_level=1)]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert result.source_authority == 1.0

    def test_statute(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(doc_type="statute")]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert result.source_authority == 0.8

    def test_high_court(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(doc_type="judgment", court_level=2)]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert result.source_authority == 0.7

    def test_mixed_authority(self, scorer: ConfidenceScorer) -> None:
        chunks = [
            _make_chunk(doc_type="statute"),
            _make_chunk(chunk_id="c2", doc_type="judgment", court_level=3),
        ]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert 0.5 <= result.source_authority <= 0.7


class TestChunkAgreement:
    def test_all_supported(self, scorer: ConfidenceScorer) -> None:
        claims = [_make_claim_verdict(ClaimVerdictType.SUPPORTED)] * 3
        result = scorer.score([], [], claims, QueryRoute.STANDARD)
        assert result.chunk_agreement == 1.0

    def test_none_supported(self, scorer: ConfidenceScorer) -> None:
        claims = [_make_claim_verdict(ClaimVerdictType.UNSUPPORTED)] * 2
        result = scorer.score([], [], claims, QueryRoute.STANDARD)
        assert result.chunk_agreement == 0.0

    def test_no_claims(self, scorer: ConfidenceScorer) -> None:
        result = scorer.score([], [], [], QueryRoute.STANDARD)
        assert result.chunk_agreement == 1.0


class TestSourceRecency:
    def test_recent_source(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(date_str="2026-01-01")]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert result.source_recency >= 0.8

    def test_old_source(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(date_str="1950-01-26")]
        result = scorer.score(chunks, [], [], QueryRoute.STANDARD)
        assert result.source_recency <= 0.4


class TestQuerySpecificity:
    def test_simple_few_chunks(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk()]
        result = scorer.score(chunks, [], [], QueryRoute.SIMPLE)
        assert result.query_specificity >= 0.9

    def test_analytical_many_chunks(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(20)]
        result = scorer.score(chunks, [], [], QueryRoute.ANALYTICAL)
        assert result.query_specificity <= 0.3


class TestConfidenceExplanation:
    def test_warning_below_threshold(self, scorer: ConfidenceScorer) -> None:
        chunks = [_make_chunk(relevance=0.1, doc_type="unknown", court_level=5)]
        citations = [_make_citation_result(CitationStatus.NOT_FOUND)]
        claims = [_make_claim_verdict(ClaimVerdictType.UNSUPPORTED)]
        result = scorer.score(chunks, citations, claims, QueryRoute.ANALYTICAL)
        assert "WARNING" in result.explanation or "CRITICAL" in result.explanation

    def test_explanation_contains_factors(self, scorer: ConfidenceScorer) -> None:
        result = scorer.score([], [], [], QueryRoute.STANDARD)
        assert "Factors:" in result.explanation


class TestCustomWeights:
    def test_custom_weights(self) -> None:
        settings = HallucinationSettings(
            weight_retrieval_relevance=0.5,
            weight_citation_verification=0.5,
            weight_source_authority=0.0,
            weight_chunk_agreement=0.0,
            weight_source_recency=0.0,
            weight_query_specificity=0.0,
        )
        scorer = ConfidenceScorer(settings)
        chunks = [_make_chunk(relevance=1.0)]
        citations = [_make_citation_result(CitationStatus.VERIFIED)]
        result = scorer.score(chunks, citations, [], QueryRoute.STANDARD)
        assert result.overall_score == 1.0
