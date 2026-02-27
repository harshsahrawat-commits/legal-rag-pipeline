"""Layer 3: Confidence Scoring.

Computes a weighted composite confidence score from 6 factors.
Pure computation — no external dependencies.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from src.hallucination._models import (
    CitationStatus,
    ClaimVerdictType,
    ConfidenceBreakdown,
)
from src.retrieval._models import QueryRoute
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.hallucination._models import (
        CitationResult,
        ClaimVerdict,
        HallucinationSettings,
    )
    from src.retrieval._models import ExpandedContext

_log = get_logger(__name__)

# Court hierarchy → authority score mapping
_COURT_AUTHORITY: dict[int, float] = {
    1: 1.0,  # Supreme Court
    2: 0.7,  # High Court
    3: 0.4,  # District Court
    4: 0.2,  # Tribunal
    5: 0.1,  # Quasi-judicial
}
# Statute authority (not a court, but has high authority)
_STATUTE_AUTHORITY = 0.8


class ConfidenceScorer:
    """Compute weighted confidence score from verification results."""

    def __init__(self, settings: HallucinationSettings) -> None:
        self._settings = settings

    def score(
        self,
        chunks: list[ExpandedContext],
        citation_results: list[CitationResult],
        claim_verdicts: list[ClaimVerdict],
        route: QueryRoute,
    ) -> ConfidenceBreakdown:
        """Compute the composite confidence score."""
        retrieval_relevance = self._retrieval_relevance(chunks)
        citation_rate = self._citation_verification_rate(citation_results)
        source_auth = self._source_authority(chunks)
        chunk_agree = self._chunk_agreement(claim_verdicts)
        recency = self._source_recency(chunks)
        specificity = self._query_specificity(route, chunks)

        overall = (
            self._settings.weight_retrieval_relevance * retrieval_relevance
            + self._settings.weight_citation_verification * citation_rate
            + self._settings.weight_source_authority * source_auth
            + self._settings.weight_chunk_agreement * chunk_agree
            + self._settings.weight_source_recency * recency
            + self._settings.weight_query_specificity * specificity
        )

        # Clamp to [0, 1]
        overall = max(0.0, min(1.0, overall))

        explanation_parts: list[str] = []
        if overall < self._settings.confidence_critical_threshold:
            explanation_parts.append("CRITICAL: Very low confidence.")
        elif overall < self._settings.confidence_warning_threshold:
            explanation_parts.append("WARNING: Confidence below threshold.")

        explanation_parts.append(
            f"Factors: retrieval={retrieval_relevance:.2f}, "
            f"citations={citation_rate:.2f}, authority={source_auth:.2f}, "
            f"agreement={chunk_agree:.2f}, recency={recency:.2f}, "
            f"specificity={specificity:.2f}"
        )

        return ConfidenceBreakdown(
            overall_score=round(overall, 4),
            retrieval_relevance=round(retrieval_relevance, 4),
            citation_verification_rate=round(citation_rate, 4),
            source_authority=round(source_auth, 4),
            chunk_agreement=round(chunk_agree, 4),
            source_recency=round(recency, 4),
            query_specificity=round(specificity, 4),
            explanation=" ".join(explanation_parts),
        )

    def _retrieval_relevance(self, chunks: list[ExpandedContext]) -> float:
        """Average relevance score from chunks (0-1)."""
        if not chunks:
            return 0.0
        scores = [c.relevance_score for c in chunks if c.relevance_score > 0]
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return min(1.0, avg)

    def _citation_verification_rate(self, citation_results: list[CitationResult]) -> float:
        """Fraction of citations that were verified."""
        if not citation_results:
            return 1.0  # No citations to verify = no penalty
        verified = sum(1 for r in citation_results if r.status == CitationStatus.VERIFIED)
        return verified / len(citation_results)

    def _source_authority(self, chunks: list[ExpandedContext]) -> float:
        """Average authority score from chunk metadata."""
        if not chunks:
            return 0.0
        scores: list[float] = []
        for chunk in chunks:
            doc_type = chunk.metadata.get("document_type", "")
            court_level = chunk.metadata.get("court_hierarchy", 0)
            if doc_type == "statute":
                scores.append(_STATUTE_AUTHORITY)
            elif court_level and isinstance(court_level, int):
                scores.append(_COURT_AUTHORITY.get(court_level, 0.1))
            else:
                scores.append(0.1)
        return sum(scores) / len(scores)

    def _chunk_agreement(self, claim_verdicts: list[ClaimVerdict]) -> float:
        """Fraction of claims that are supported."""
        if not claim_verdicts:
            return 1.0  # No claims verified = no penalty
        supported = sum(1 for v in claim_verdicts if v.verdict == ClaimVerdictType.SUPPORTED)
        return supported / len(claim_verdicts)

    def _source_recency(self, chunks: list[ExpandedContext]) -> float:
        """Score based on how recent the source documents are."""
        if not chunks:
            return 0.0
        today = date.today()
        scores: list[float] = []
        for chunk in chunks:
            date_str = chunk.metadata.get("date_decided", "")
            if not date_str or not isinstance(date_str, str):
                scores.append(0.5)
                continue
            try:
                doc_date = date.fromisoformat(date_str)
                age_days = (today - doc_date).days
                # Recent = higher score; decay over 5 years
                if age_days <= 0:
                    scores.append(1.0)
                elif age_days < 365:
                    scores.append(0.9)
                elif age_days < 365 * 2:
                    scores.append(0.7)
                elif age_days < 365 * 5:
                    scores.append(0.5)
                else:
                    scores.append(0.3)
            except ValueError:
                scores.append(0.5)
        return sum(scores) / len(scores)

    def _query_specificity(self, route: QueryRoute, chunks: list[ExpandedContext]) -> float:
        """Heuristic: more specific queries → higher specificity score."""
        # Route-based base score
        route_scores: dict[QueryRoute, float] = {
            QueryRoute.SIMPLE: 0.9,
            QueryRoute.STANDARD: 0.7,
            QueryRoute.COMPLEX: 0.5,
            QueryRoute.ANALYTICAL: 0.3,
        }
        base = route_scores.get(route, 0.5)

        # More chunks = less specific
        if len(chunks) <= 3:
            return min(1.0, base + 0.2)
        if len(chunks) <= 10:
            return base
        return max(0.1, base - 0.2)
