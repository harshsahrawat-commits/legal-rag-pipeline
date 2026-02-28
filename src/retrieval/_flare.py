"""FLARE Active Retrieval for ANALYTICAL queries.

Forward-Looking Active REtrieval (FLARE) improves retrieval for
analytical legal queries by:

1. Segmenting retrieved context into token-bounded segments
2. Asking an LLM to assess which segments are insufficient
3. Generating follow-up queries for low-confidence segments
4. Re-retrieving and adding new chunks to the context

Uses the LLM provider abstraction (``get_llm_provider``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.retrieval._exceptions import FLAREError
from src.utils._exceptions import LLMNotAvailableError
from src.utils._llm_client import LLMMessage, get_llm_provider
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.retrieval._models import ExpandedContext, RetrievalSettings, ScoredChunk
    from src.utils._llm_client import BaseLLMProvider

_log = get_logger(__name__)


class FLAREActiveRetriever:
    """Runs FLARE active retrieval on ANALYTICAL queries.

    After standard retrieval produces initial results, FLARE:
    - Segments the retrieved text into manageable chunks
    - Asks an LLM to assess confidence per segment
    - For low-confidence segments, generates follow-up queries
    - Re-retrieves via ``hybrid_search`` to augment context

    Re-retrievals are capped at ``flare_max_retrievals`` (default 5).
    """

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._provider: BaseLLMProvider | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether FLARE is enabled and an LLM provider is available."""
        if not self._settings.flare_enabled:
            return False
        try:
            provider = get_llm_provider("flare")
            return provider.is_available
        except LLMNotAvailableError:
            return False

    # ------------------------------------------------------------------
    # Provider lifecycle
    # ------------------------------------------------------------------

    def _ensure_provider(self) -> None:
        """Lazy-initialize the LLM provider.

        Raises:
            FLAREError: If no LLM provider can be initialized.
        """
        if self._provider is not None:
            return
        try:
            self._provider = get_llm_provider("flare")
        except LLMNotAvailableError as exc:
            msg = f"LLM provider required for FLARE: {exc}"
            raise FLAREError(msg) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def active_retrieve(
        self,
        query: str,
        initial_chunks: list[ExpandedContext],
        search_fn: Callable[[str, int], Awaitable[list[ScoredChunk]]],
    ) -> tuple[list[ExpandedContext], int]:
        """Run FLARE on initial retrieval results.

        Args:
            query: The user's query text.
            initial_chunks: Chunks from standard retrieval.
            search_fn: Async callable ``(text, top_k) -> list[ScoredChunk]``.
                Typically ``engine.hybrid_search``.

        Returns:
            Tuple of ``(augmented_chunks, num_re_retrievals)``.
            If all segments are high-confidence, returns the original
            chunks with 0 re-retrievals.

        Raises:
            FLAREError: If the anthropic client cannot be initialized.
        """
        if not self._settings.flare_enabled:
            return initial_chunks, 0

        self._ensure_provider()

        segments = self._segment_chunks(initial_chunks)
        if not segments:
            return initial_chunks, 0

        retrieval_count = 0
        existing_ids: set[str] = {c.chunk_id for c in initial_chunks}
        new_chunks: list[ExpandedContext] = []

        try:
            # Assess confidence per segment
            confidences = await self._assess_segments(query, segments)

            # Find low-confidence segments
            threshold = self._settings.flare_confidence_threshold
            low_conf = [
                seg for seg, conf in zip(segments, confidences, strict=False) if conf < threshold
            ]

            if not low_conf:
                _log.info("flare_all_high_confidence", segments=len(segments))
                return initial_chunks, 0

            # Generate follow-up queries
            follow_ups = await self._generate_follow_ups(query, low_conf)

            # Re-retrieve for each follow-up (capped)
            for fq in follow_ups:
                if retrieval_count >= self._settings.flare_max_retrievals:
                    _log.info(
                        "flare_max_retrievals_reached", cap=self._settings.flare_max_retrievals
                    )
                    break

                re_results = await search_fn(fq, 5)
                retrieval_count += 1

                for sc in re_results:
                    if sc.chunk_id not in existing_ids:
                        existing_ids.add(sc.chunk_id)
                        new_chunks.append(
                            _scored_to_expanded(sc),
                        )

        except FLAREError:
            raise
        except Exception as exc:
            _log.warning("flare_llm_error", error=str(exc))
            # Graceful fallback — return original chunks
            return initial_chunks, retrieval_count

        _log.info(
            "flare_complete",
            re_retrievals=retrieval_count,
            new_chunks=len(new_chunks),
        )
        return initial_chunks + new_chunks, retrieval_count

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _segment_chunks(self, chunks: list[ExpandedContext]) -> list[str]:
        """Split chunks into segments of ~flare_segment_tokens words."""
        all_text = "\n\n".join(c.chunk_text for c in chunks if c.chunk_text)
        if not all_text.strip():
            return []

        words = all_text.split()
        segment_size = self._settings.flare_segment_tokens
        segments: list[str] = []

        for i in range(0, len(words), segment_size):
            segment = " ".join(words[i : i + segment_size])
            if segment.strip():
                segments.append(segment)

        return segments

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _assess_segments(
        self,
        query: str,
        segments: list[str],
    ) -> list[float]:
        """Ask LLM to rate retrieval confidence per segment.

        Returns a list of floats in [0.0, 1.0], one per segment.
        Falls back to threshold values if parsing fails.
        """
        segment_list = "\n".join(
            f"[Segment {i + 1}]: {seg[:500]}" for i, seg in enumerate(segments)
        )
        prompt = (
            "You are evaluating retrieval quality for a legal research query.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved segments:\n{segment_list}\n\n"
            "For each segment, rate how well it helps answer the query on a "
            "scale of 0.0 to 1.0. Return ONLY a JSON array of floats, one per "
            "segment. Example: [0.9, 0.3, 0.7]"
        )

        response = await self._provider.acomplete(
            [LLMMessage(role="user", content=prompt)],
            max_tokens=256,
        )

        return self._parse_confidence_scores(response.text.strip(), len(segments))

    async def _generate_follow_ups(
        self,
        query: str,
        low_segments: list[str],
    ) -> list[str]:
        """Generate follow-up search queries for low-confidence segments.

        Returns a list of query strings for re-retrieval.
        Falls back to the original query if parsing fails.
        """
        segment_text = "\n".join(f"[{i + 1}]: {seg[:300]}" for i, seg in enumerate(low_segments))
        prompt = (
            "You are a legal research assistant. The following segments were "
            "retrieved for the query but are insufficient.\n\n"
            f"Original query: {query}\n\n"
            f"Low-quality segments:\n{segment_text}\n\n"
            f"Generate {len(low_segments)} focused follow-up search queries "
            "to find better information. Return ONLY a JSON array of strings."
        )

        response = await self._provider.acomplete(
            [LLMMessage(role="user", content=prompt)],
            max_tokens=512,
        )

        return self._parse_follow_up_queries(response.text.strip(), query)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_confidence_scores(
        self,
        text: str,
        expected_count: int,
    ) -> list[float]:
        """Parse LLM confidence response into a list of floats."""
        try:
            scores = json.loads(text)
            if isinstance(scores, list) and all(isinstance(s, (int, float)) for s in scores):
                # Pad or truncate to match expected count
                while len(scores) < expected_count:
                    scores.append(self._settings.flare_confidence_threshold)
                return [float(s) for s in scores[:expected_count]]
        except (json.JSONDecodeError, TypeError):
            pass

        _log.warning("flare_confidence_parse_failed", raw=text[:200])
        return [self._settings.flare_confidence_threshold] * expected_count

    def _parse_follow_up_queries(
        self,
        text: str,
        fallback_query: str,
    ) -> list[str]:
        """Parse LLM follow-up response into a list of query strings."""
        try:
            queries = json.loads(text)
            if isinstance(queries, list) and queries:
                return [str(q) for q in queries]
        except (json.JSONDecodeError, TypeError):
            pass

        _log.warning("flare_followup_parse_failed", raw=text[:200])
        return [fallback_query]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored_to_expanded(sc: ScoredChunk) -> ExpandedContext:
    """Convert a ScoredChunk from re-retrieval into an ExpandedContext."""
    from src.retrieval._models import ExpandedContext

    return ExpandedContext(
        chunk_id=sc.chunk_id,
        chunk_text=sc.contextualized_text or sc.text,
        relevance_score=sc.score,
        total_tokens=max(1, len((sc.contextualized_text or sc.text).split())),
        metadata=sc.payload,
    )
