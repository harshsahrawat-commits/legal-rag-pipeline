"""Shared fixtures for hallucination mitigation tests."""

from __future__ import annotations

from datetime import date

import pytest

from src.hallucination._models import (
    CitationResult,
    CitationStatus,
    CitationType,
    ClaimVerdict,
    ClaimVerdictType,
    ExtractedCitation,
    ExtractedClaim,
    HallucinationSettings,
    VerificationInput,
)
from src.retrieval._models import ExpandedContext, QueryRoute, RetrievalResult


@pytest.fixture()
def hallucination_settings() -> HallucinationSettings:
    """Default settings for tests."""
    return HallucinationSettings()


@pytest.fixture()
def sample_response_text() -> str:
    """A response citing Section 420 IPC and a case."""
    return (
        "Section 420 of the Indian Penal Code provides for punishment of cheating. "
        "In AIR 2023 SC 1234, the Supreme Court held that mens rea is essential. "
        "Article 21 of the Constitution guarantees the right to life."
    )


@pytest.fixture()
def sample_retrieval_result() -> RetrievalResult:
    """A RetrievalResult with 3 expanded chunks."""
    return RetrievalResult(
        query_text="What is Section 420 IPC?",
        route=QueryRoute.STANDARD,
        chunks=[
            ExpandedContext(
                chunk_id="chunk-1",
                chunk_text="Section 420 deals with cheating and dishonestly inducing delivery of property.",
                relevance_score=0.95,
                total_tokens=50,
                metadata={
                    "document_type": "statute",
                    "court_hierarchy": 0,
                    "date_decided": "2023-01-15",
                },
            ),
            ExpandedContext(
                chunk_id="chunk-2",
                chunk_text="The Supreme Court in AIR 2023 SC 1234 held that mens rea is essential.",
                relevance_score=0.88,
                total_tokens=40,
                metadata={
                    "document_type": "judgment",
                    "court_hierarchy": 1,
                    "date_decided": "2023-06-10",
                },
            ),
            ExpandedContext(
                chunk_id="chunk-3",
                chunk_text="Article 21 protects the fundamental right to life and personal liberty.",
                relevance_score=0.75,
                total_tokens=35,
                metadata={
                    "document_type": "statute",
                    "court_hierarchy": 0,
                    "date_decided": "1950-01-26",
                },
            ),
        ],
        search_channels_used=["dense", "bm25"],
        total_context_tokens=125,
    )


@pytest.fixture()
def sample_verification_input(
    sample_response_text: str,
    sample_retrieval_result: RetrievalResult,
) -> VerificationInput:
    """Complete verification input."""
    return VerificationInput(
        response_text=sample_response_text,
        retrieval_result=sample_retrieval_result,
        reference_date=date(2025, 1, 15),
        route=QueryRoute.STANDARD,
    )


@pytest.fixture()
def sample_citation_results() -> list[CitationResult]:
    """Sample citation results for confidence scoring."""
    return [
        CitationResult(
            citation=ExtractedCitation(
                text="Section 420 of the Indian Penal Code",
                citation_type=CitationType.SECTION_REF,
                section="420",
                act="Indian Penal Code",
            ),
            status=CitationStatus.VERIFIED,
            kg_node_label="Section",
        ),
        CitationResult(
            citation=ExtractedCitation(
                text="AIR 2023 SC 1234",
                citation_type=CitationType.CASE_CITATION,
                case_citation="AIR 2023 SC 1234",
            ),
            status=CitationStatus.NOT_FOUND,
        ),
    ]


@pytest.fixture()
def sample_claim_verdicts() -> list[ClaimVerdict]:
    """Sample claim verdicts for confidence scoring."""
    return [
        ClaimVerdict(
            claim=ExtractedClaim(claim_id=1, text="Section 420 punishes cheating"),
            verdict=ClaimVerdictType.SUPPORTED,
            confidence=0.95,
            evidence_chunk_ids=["chunk-1"],
        ),
        ClaimVerdict(
            claim=ExtractedClaim(claim_id=2, text="Mens rea is essential"),
            verdict=ClaimVerdictType.SUPPORTED,
            confidence=0.88,
            evidence_chunk_ids=["chunk-2"],
        ),
        ClaimVerdict(
            claim=ExtractedClaim(claim_id=3, text="Maximum sentence is 10 years"),
            verdict=ClaimVerdictType.UNSUPPORTED,
            confidence=0.2,
        ),
    ]
