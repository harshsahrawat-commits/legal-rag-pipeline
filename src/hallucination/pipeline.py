"""HallucinationPipeline: orchestrates all 4 verification layers.

Order: Citation → Temporal → GenGround → Confidence.
Confidence scoring is last because it uses Layer 1 + Layer 4 results.
Each layer is error-isolated — one failure does not crash the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.hallucination._citation_verifier import CitationVerifier
from src.hallucination._confidence_scorer import ConfidenceScorer
from src.hallucination._genground_refiner import GenGroundRefiner
from src.hallucination._models import (
    ConfidenceBreakdown,
    VerificationSummary,
    VerifiedResponse,
)
from src.hallucination._temporal_checker import TemporalChecker
from src.retrieval._models import QueryRoute
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.hallucination._models import (
        CitationResult,
        ClaimVerdict,
        HallucinationSettings,
        TemporalWarning,
        VerificationInput,
    )
    from src.knowledge_graph._queries import QueryBuilder
    from src.retrieval._engine import RetrievalEngine
    from src.retrieval._models import ExpandedContext

_log = get_logger(__name__)


class HallucinationPipeline:
    """Orchestrate 4-layer hallucination mitigation.

    Layers:
        1. Citation Verification — verify citations against KG
        2. Temporal Consistency — check laws are in force
        3. GenGround Verification — per-claim LLM re-retrieval
        4. Confidence Scoring — weighted composite score
    """

    def __init__(
        self,
        settings: HallucinationSettings,
        query_builder: QueryBuilder | None = None,
        retrieval_engine: RetrievalEngine | None = None,
    ) -> None:
        self._settings = settings
        self._qb = query_builder
        self._engine = retrieval_engine

    async def verify(self, inp: VerificationInput) -> VerifiedResponse:
        """Run all 4 verification layers and return a VerifiedResponse."""
        errors: list[str] = []

        # Resolve route
        route = inp.route or QueryRoute.STANDARD

        # Resolve chunks from retrieval result or empty
        chunks: list[ExpandedContext] = []
        if inp.retrieval_result is not None:
            chunks = inp.retrieval_result.chunks or []

        # Layer 1: Citation Verification
        citation_results = await self._run_citation_verification(inp.response_text, errors)

        # Layer 2: Temporal Consistency
        temporal_warnings = await self._run_temporal_check(
            inp.response_text, inp.reference_date, errors
        )

        # Layer 3: GenGround Verification
        is_simple = route == QueryRoute.SIMPLE
        modified_text, claim_verdicts = await self._run_genground(
            inp.response_text, chunks, is_simple, errors
        )

        # Layer 4: Confidence Scoring
        confidence = await self._run_confidence_scoring(
            chunks, citation_results, claim_verdicts, route, errors
        )

        # Build summary
        summary = self._build_summary(
            citation_results,
            temporal_warnings,
            claim_verdicts,
        )

        return VerifiedResponse(
            modified_response=modified_text,
            original_response=inp.response_text,
            citation_results=citation_results,
            temporal_warnings=temporal_warnings,
            claim_verdicts=claim_verdicts,
            confidence=confidence,
            summary=summary,
            errors=errors,
        )

    async def _run_citation_verification(
        self,
        response_text: str,
        errors: list[str],
    ) -> list[CitationResult]:
        """Layer 1: Citation Verification."""
        try:
            verifier = CitationVerifier(self._settings, query_builder=self._qb)
            return await verifier.verify_response(response_text)
        except Exception as exc:
            _log.warning("layer1_citation_verification_failed", error=str(exc))
            errors.append(f"Citation verification failed: {exc}")
            return []

    async def _run_temporal_check(
        self,
        response_text: str,
        reference_date: object,
        errors: list[str],
    ) -> list[TemporalWarning]:
        """Layer 2: Temporal Consistency."""
        try:
            checker = TemporalChecker(self._settings, query_builder=self._qb)
            kwargs: dict = {}
            if reference_date is not None:
                kwargs["reference_date"] = reference_date
            return await checker.check_response(response_text, **kwargs)
        except Exception as exc:
            _log.warning("layer2_temporal_check_failed", error=str(exc))
            errors.append(f"Temporal check failed: {exc}")
            return []

    async def _run_genground(
        self,
        response_text: str,
        chunks: list[ExpandedContext],
        is_simple: bool,
        errors: list[str],
    ) -> tuple[str, list[ClaimVerdict]]:
        """Layer 3: GenGround Verification."""
        try:
            refiner = GenGroundRefiner(
                self._settings,
                retrieval_engine=self._engine,
            )
            return await refiner.verify(response_text, chunks, is_simple=is_simple)
        except Exception as exc:
            _log.warning("layer3_genground_failed", error=str(exc))
            errors.append(f"GenGround verification failed: {exc}")
            return response_text, []

    async def _run_confidence_scoring(
        self,
        chunks: list[ExpandedContext],
        citation_results: list[CitationResult],
        claim_verdicts: list[ClaimVerdict],
        route: QueryRoute,
        errors: list[str],
    ) -> ConfidenceBreakdown:
        """Layer 4: Confidence Scoring."""
        try:
            scorer = ConfidenceScorer(self._settings)
            return scorer.score(chunks, citation_results, claim_verdicts, route)
        except Exception as exc:
            _log.warning("layer4_confidence_scoring_failed", error=str(exc))
            errors.append(f"Confidence scoring failed: {exc}")
            return ConfidenceBreakdown(
                overall_score=0.0,
                retrieval_relevance=0.0,
                citation_verification_rate=0.0,
                source_authority=0.0,
                chunk_agreement=0.0,
                source_recency=0.0,
                query_specificity=0.0,
                explanation="Confidence scoring failed",
            )

    def _build_summary(
        self,
        citations: list[CitationResult],
        warnings: list[TemporalWarning],
        verdicts: list[ClaimVerdict],
    ) -> VerificationSummary:
        """Build aggregate summary statistics."""
        from src.hallucination._models import CitationStatus, ClaimVerdictType

        total_citations = len(citations)
        verified = sum(1 for c in citations if c.status == CitationStatus.VERIFIED)
        total_claims = len(verdicts)
        supported = sum(1 for v in verdicts if v.verdict == ClaimVerdictType.SUPPORTED)
        unsupported = sum(1 for v in verdicts if v.verdict == ClaimVerdictType.UNSUPPORTED)
        llm_calls = 0  # Could be tracked via GenGround, but that's internal

        return VerificationSummary(
            total_citations=total_citations,
            verified_citations=verified,
            not_found_citations=total_citations - verified,
            temporal_warnings=len(warnings),
            total_claims=total_claims,
            supported_claims=supported,
            unsupported_claims=unsupported,
            llm_calls=llm_calls,
        )
