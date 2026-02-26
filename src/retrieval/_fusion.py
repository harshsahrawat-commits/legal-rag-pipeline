"""Reciprocal Rank Fusion for combining multi-channel retrieval results."""

from __future__ import annotations

from src.retrieval._models import FusedChunk, ScoredChunk
from src.utils._logging import get_logger

_log = get_logger(__name__)


class ReciprocalRankFusion:
    """Combines results from multiple search channels using RRF.

    RRF score for a chunk = sum over channels of 1 / (k + rank_in_channel)
    where rank starts at 1 for the highest-scored chunk.
    """

    def __init__(self, k: int = 60) -> None:
        """Initialize with the RRF constant k.

        Args:
            k: The RRF constant. Higher values reduce the impact of rank
               differences. Standard default is 60.
        """
        self._k = k

    @property
    def k(self) -> int:
        """The RRF constant."""
        return self._k

    def fuse(
        self,
        channel_results: dict[str, list[ScoredChunk]],
        top_k: int = 150,
    ) -> list[FusedChunk]:
        """Fuse results from multiple channels.

        Args:
            channel_results: Mapping of channel name to ranked results.
                Each list should be pre-sorted by score descending.
            top_k: Maximum number of fused results to return.

        Returns:
            List of FusedChunks sorted by RRF score descending, max top_k.
        """
        if top_k <= 0:
            return []

        if not channel_results:
            return []

        # Accumulate RRF scores, channel lists, and best-channel data per chunk.
        # rrf_scores[chunk_id] = cumulative RRF score
        rrf_scores: dict[str, float] = {}
        # channels[chunk_id] = list of channel names that returned this chunk
        channels: dict[str, list[str]] = {}
        # best_source[chunk_id] = (channel_score, ScoredChunk) — highest-scoring
        best_source: dict[str, tuple[float, ScoredChunk]] = {}

        for channel_name, results in channel_results.items():
            if not results:
                continue

            for rank_zero_idx, scored_chunk in enumerate(results):
                rank = rank_zero_idx + 1  # 1-based rank
                rrf_contribution = 1.0 / (self._k + rank)

                cid = scored_chunk.chunk_id

                # Accumulate RRF score.
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

                # Track contributing channels.
                if cid not in channels:
                    channels[cid] = []
                channels[cid].append(channel_name)

                # Keep the highest-scoring source for text/payload.
                if cid not in best_source or scored_chunk.score > best_source[cid][0]:
                    best_source[cid] = (scored_chunk.score, scored_chunk)

        if not rrf_scores:
            return []

        # Sort by RRF score descending, then by chunk_id ascending for
        # deterministic ordering when scores are tied.
        sorted_ids = sorted(
            rrf_scores,
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        fused: list[FusedChunk] = []
        for cid in sorted_ids[:top_k]:
            source = best_source[cid][1]
            fused.append(
                FusedChunk(
                    chunk_id=cid,
                    text=source.text,
                    contextualized_text=source.contextualized_text,
                    rrf_score=rrf_scores[cid],
                    channels=channels[cid],
                    payload=source.payload,
                ),
            )

        _log.info(
            "rrf_fusion_complete",
            num_channels=len(channel_results),
            input_chunks=sum(len(r) for r in channel_results.values()),
            unique_chunks=len(rrf_scores),
            output_chunks=len(fused),
            k=self._k,
        )

        return fused
