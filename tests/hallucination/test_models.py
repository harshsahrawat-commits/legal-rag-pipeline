"""Tests for hallucination Pydantic models."""

from __future__ import annotations

from datetime import UTC, date, datetime

from src.hallucination._models import (
    CitationResult,
    CitationStatus,
    CitationType,
    ClaimVerdict,
    ClaimVerdictType,
    ConfidenceBreakdown,
    ExtractedCitation,
    ExtractedClaim,
    HallucinationConfig,
    HallucinationSettings,
    TemporalWarning,
    VerificationInput,
    VerificationSummary,
    VerifiedResponse,
)
from src.retrieval._models import QueryRoute, RetrievalResult


class TestEnums:
    def test_citation_status_values(self) -> None:
        assert CitationStatus.VERIFIED == "verified"
        assert CitationStatus.NOT_FOUND == "not_found"
        assert CitationStatus.MISATTRIBUTED == "misattributed"
        assert CitationStatus.KG_UNAVAILABLE == "kg_unavailable"

    def test_claim_verdict_type_values(self) -> None:
        assert ClaimVerdictType.SUPPORTED == "supported"
        assert ClaimVerdictType.UNSUPPORTED == "unsupported"
        assert ClaimVerdictType.PARTIALLY_SUPPORTED == "partially_supported"

    def test_citation_type_values(self) -> None:
        assert CitationType.SECTION_REF == "section_ref"
        assert CitationType.ARTICLE_REF == "article_ref"
        assert CitationType.CASE_CITATION == "case_citation"
        assert CitationType.NOTIFICATION_REF == "notification_ref"
        assert CitationType.CIRCULAR_REF == "circular_ref"


class TestExtractedCitation:
    def test_section_ref(self) -> None:
        c = ExtractedCitation(
            text="Section 420 IPC",
            citation_type=CitationType.SECTION_REF,
            section="420",
            act="Indian Penal Code",
        )
        assert c.section == "420"
        assert c.act == "Indian Penal Code"
        assert c.case_citation is None

    def test_case_citation(self) -> None:
        c = ExtractedCitation(
            text="AIR 2023 SC 1234",
            citation_type=CitationType.CASE_CITATION,
            case_citation="AIR 2023 SC 1234",
        )
        assert c.case_citation == "AIR 2023 SC 1234"
        assert c.section is None


class TestCitationResult:
    def test_verified(self) -> None:
        r = CitationResult(
            citation=ExtractedCitation(text="S.420 IPC", citation_type=CitationType.SECTION_REF),
            status=CitationStatus.VERIFIED,
            kg_node_label="Section",
        )
        assert r.status == CitationStatus.VERIFIED

    def test_not_found_with_error(self) -> None:
        r = CitationResult(
            citation=ExtractedCitation(text="S.999 IPC", citation_type=CitationType.SECTION_REF),
            status=CitationStatus.NOT_FOUND,
            error="Section 999 not found in KG",
        )
        assert r.error is not None


class TestTemporalWarning:
    def test_fields(self) -> None:
        w = TemporalWarning(
            section="420",
            act="Indian Penal Code",
            warning_text="IPC repealed; use BNS",
            repealed_by="Bharatiya Nyaya Sanhita, 2023",
            replacement_act="Bharatiya Nyaya Sanhita",
            reference_date=date(2025, 1, 1),
        )
        assert w.repealed_by is not None
        assert w.reference_date == date(2025, 1, 1)


class TestExtractedClaim:
    def test_basic(self) -> None:
        c = ExtractedClaim(claim_id=1, text="Section 420 punishes cheating")
        assert c.claim_id == 1


class TestClaimVerdict:
    def test_supported(self) -> None:
        v = ClaimVerdict(
            claim=ExtractedClaim(claim_id=1, text="claim"),
            verdict=ClaimVerdictType.SUPPORTED,
            confidence=0.95,
            evidence_chunk_ids=["chunk-1"],
        )
        assert v.verdict == ClaimVerdictType.SUPPORTED
        assert v.confidence == 0.95


class TestConfidenceBreakdown:
    def test_defaults(self) -> None:
        cb = ConfidenceBreakdown()
        assert cb.overall_score == 0.0
        assert cb.explanation == ""


class TestVerificationInput:
    def test_minimal(self) -> None:
        vi = VerificationInput(response_text="test")
        assert vi.route == QueryRoute.STANDARD
        assert vi.reference_date is None

    def test_with_all_fields(self) -> None:
        vi = VerificationInput(
            response_text="Section 420 IPC...",
            retrieval_result=RetrievalResult(query_text="q"),
            reference_date=date(2025, 1, 1),
            route=QueryRoute.ANALYTICAL,
        )
        assert vi.route == QueryRoute.ANALYTICAL


class TestVerificationSummary:
    def test_defaults(self) -> None:
        s = VerificationSummary()
        assert s.total_citations == 0
        assert s.llm_calls == 0


class TestVerifiedResponse:
    def test_defaults(self) -> None:
        vr = VerifiedResponse(original_response="test", modified_response="test")
        assert vr.elapsed_ms == 0.0
        assert vr.errors == []

    def test_elapsed_ms(self) -> None:
        t1 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC)
        vr = VerifiedResponse(
            original_response="a",
            modified_response="b",
            started_at=t1,
            finished_at=t2,
        )
        assert vr.elapsed_ms == 1000.0


class TestHallucinationSettings:
    def test_defaults(self) -> None:
        s = HallucinationSettings()
        assert s.genground_enabled is True
        assert s.confidence_warning_threshold == 0.6
        assert s.ipc_repeal_date == "2024-07-01"

    def test_weights_sum_to_one(self) -> None:
        s = HallucinationSettings()
        total = (
            s.weight_retrieval_relevance
            + s.weight_citation_verification
            + s.weight_source_authority
            + s.weight_chunk_agreement
            + s.weight_source_recency
            + s.weight_query_specificity
        )
        assert abs(total - 1.0) < 1e-9


class TestHallucinationConfig:
    def test_default(self) -> None:
        c = HallucinationConfig()
        assert isinstance(c.settings, HallucinationSettings)
