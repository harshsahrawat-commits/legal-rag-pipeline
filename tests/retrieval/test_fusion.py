"""Tests for ReciprocalRankFusion."""

from __future__ import annotations

import pytest

from src.retrieval._fusion import ReciprocalRankFusion
from src.retrieval._models import FusedChunk, ScoredChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sc(
    chunk_id: str,
    score: float,
    channel: str,
    text: str = "",
    contextualized_text: str | None = None,
    payload: dict | None = None,
) -> ScoredChunk:
    """Shorthand factory for ScoredChunk."""
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text or f"Text for {chunk_id}",
        contextualized_text=contextualized_text,
        score=score,
        channel=channel,
        payload=payload or {},
    )


# ---------------------------------------------------------------------------
# TestEmptyInput
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Edge cases where there are no results to fuse."""

    def test_no_channels(self) -> None:
        """Empty dict returns empty list."""
        rrf = ReciprocalRankFusion()
        assert rrf.fuse({}) == []

    def test_single_empty_channel(self) -> None:
        """One channel with no results returns empty list."""
        rrf = ReciprocalRankFusion()
        result = rrf.fuse({"dense": []})
        assert result == []

    def test_all_empty_channels(self) -> None:
        """Multiple channels, all empty, returns empty list."""
        rrf = ReciprocalRankFusion()
        result = rrf.fuse({"dense": [], "bm25": [], "quim": []})
        assert result == []


# ---------------------------------------------------------------------------
# TestSingleChannel
# ---------------------------------------------------------------------------


class TestSingleChannel:
    """Single channel pass-through with RRF scores."""

    def test_single_channel_preserves_order(self) -> None:
        """Results from one channel keep their ordering."""
        rrf = ReciprocalRankFusion(k=60)
        results = rrf.fuse(
            {
                "dense": [
                    _sc("c1", 0.95, "dense"),
                    _sc("c2", 0.88, "dense"),
                    _sc("c3", 0.80, "dense"),
                ],
            },
        )
        assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]

    def test_single_channel_rrf_scores(self) -> None:
        """RRF scores are correctly computed for a single channel."""
        k = 60
        rrf = ReciprocalRankFusion(k=k)
        results = rrf.fuse(
            {"dense": [_sc("c1", 0.95, "dense"), _sc("c2", 0.88, "dense")]},
        )

        assert len(results) == 2
        assert results[0].rrf_score == pytest.approx(1.0 / (k + 1))
        assert results[1].rrf_score == pytest.approx(1.0 / (k + 2))

    def test_single_channel_channels_list(self) -> None:
        """Each fused chunk records the contributing channel."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse({"bm25": [_sc("c1", 3.0, "bm25")]})
        assert results[0].channels == ["bm25"]

    def test_single_channel_text_from_source(self) -> None:
        """Fused chunk text comes from the source ScoredChunk."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {"dense": [_sc("c1", 0.95, "dense", text="Section 420...")]},
        )
        assert results[0].text == "Section 420..."

    def test_single_channel_contextualized_text(self) -> None:
        """Contextualized text is propagated."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {
                "dense": [
                    _sc(
                        "c1",
                        0.95,
                        "dense",
                        contextualized_text="Section 420 of IPC deals with cheating.",
                    ),
                ],
            },
        )
        assert results[0].contextualized_text == "Section 420 of IPC deals with cheating."


# ---------------------------------------------------------------------------
# TestTwoChannels
# ---------------------------------------------------------------------------


class TestTwoChannels:
    """Two channels — disjoint and overlapping results."""

    def test_disjoint_results(self) -> None:
        """Non-overlapping chunks from two channels are all returned."""
        rrf = ReciprocalRankFusion(k=60)
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.95, "dense")],
                "bm25": [_sc("c2", 3.0, "bm25")],
            },
        )
        ids = {r.chunk_id for r in results}
        assert ids == {"c1", "c2"}
        # Both have the same RRF score (rank 1 in their channel).
        assert results[0].rrf_score == pytest.approx(results[1].rrf_score)

    def test_overlapping_results_combined_score(self) -> None:
        """Chunk in both channels gets combined RRF score."""
        k = 60
        rrf = ReciprocalRankFusion(k=k)
        results = rrf.fuse(
            {
                "dense": [
                    _sc("c1", 0.95, "dense"),
                    _sc("c2", 0.88, "dense"),
                ],
                "bm25": [
                    _sc("c2", 3.2, "bm25"),
                    _sc("c3", 2.8, "bm25"),
                ],
            },
        )

        by_id = {r.chunk_id: r for r in results}

        # c1: rank 1 in dense only -> 1/(60+1)
        assert by_id["c1"].rrf_score == pytest.approx(1.0 / 61)
        # c2: rank 2 in dense + rank 1 in bm25 -> 1/(60+2) + 1/(60+1)
        assert by_id["c2"].rrf_score == pytest.approx(1.0 / 62 + 1.0 / 61)
        # c3: rank 2 in bm25 -> 1/(60+2)
        assert by_id["c3"].rrf_score == pytest.approx(1.0 / 62)

        # c2 should be ranked first (highest combined RRF score).
        assert results[0].chunk_id == "c2"

    def test_overlapping_results_channels_combined(self) -> None:
        """Chunk in both channels lists both channel names."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.95, "dense")],
                "bm25": [_sc("c1", 3.0, "bm25")],
            },
        )
        assert len(results) == 1
        assert set(results[0].channels) == {"dense", "bm25"}

    def test_overlapping_text_from_highest_score(self) -> None:
        """For overlapping chunk, text comes from the channel with higher score."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.95, "dense", text="Dense version")],
                "bm25": [_sc("c1", 3.2, "bm25", text="BM25 version")],
            },
        )
        # bm25 score (3.2) > dense score (0.95)
        assert results[0].text == "BM25 version"

    def test_overlapping_payload_from_highest_score(self) -> None:
        """For overlapping chunk, payload comes from the highest-scoring channel."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {
                "dense": [
                    _sc("c1", 0.95, "dense", payload={"src": "dense"}),
                ],
                "bm25": [
                    _sc("c1", 3.2, "bm25", payload={"src": "bm25"}),
                ],
            },
        )
        assert results[0].payload == {"src": "bm25"}


# ---------------------------------------------------------------------------
# TestMultipleChannels
# ---------------------------------------------------------------------------


class TestMultipleChannels:
    """Three or more channels."""

    def test_three_channels_all_contribute(self) -> None:
        """Chunk appearing in all three channels gets three RRF contributions."""
        k = 60
        rrf = ReciprocalRankFusion(k=k)
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.95, "dense")],
                "bm25": [_sc("c1", 3.0, "bm25")],
                "quim": [_sc("c1", 0.70, "quim")],
            },
        )

        expected = 3 * (1.0 / (k + 1))
        assert len(results) == 1
        assert results[0].rrf_score == pytest.approx(expected)
        assert set(results[0].channels) == {"dense", "bm25", "quim"}

    def test_four_channels_mixed_overlap(self) -> None:
        """Four channels with partial overlap produces correct ranking."""
        k = 60
        rrf = ReciprocalRankFusion(k=k)
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.9, "dense"), _sc("c2", 0.8, "dense")],
                "bm25": [_sc("c2", 3.0, "bm25"), _sc("c3", 2.5, "bm25")],
                "quim": [_sc("c3", 0.7, "quim"), _sc("c4", 0.6, "quim")],
                "graph": [_sc("c1", 0.5, "graph")],
            },
        )

        by_id = {r.chunk_id: r for r in results}

        # c1: dense rank 1 + graph rank 1 = 1/61 + 1/61
        assert by_id["c1"].rrf_score == pytest.approx(2.0 / 61)
        # c2: dense rank 2 + bm25 rank 1 = 1/62 + 1/61
        assert by_id["c2"].rrf_score == pytest.approx(1.0 / 62 + 1.0 / 61)
        # c3: bm25 rank 2 + quim rank 1 = 1/62 + 1/61
        assert by_id["c3"].rrf_score == pytest.approx(1.0 / 62 + 1.0 / 61)
        # c4: quim rank 2 = 1/62
        assert by_id["c4"].rrf_score == pytest.approx(1.0 / 62)

    def test_total_unique_chunks(self) -> None:
        """Deduplication across channels gives correct count."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.9, "dense"), _sc("c2", 0.8, "dense")],
                "bm25": [_sc("c2", 3.0, "bm25"), _sc("c3", 2.5, "bm25")],
            },
        )
        assert len(results) == 3  # c1, c2, c3


# ---------------------------------------------------------------------------
# TestTopK
# ---------------------------------------------------------------------------


class TestTopK:
    """Truncation and boundary behavior for top_k."""

    def test_truncation(self) -> None:
        """Only top_k results are returned."""
        rrf = ReciprocalRankFusion()
        chunks = [_sc(f"c{i}", 1.0 - i * 0.01, "dense") for i in range(10)]
        results = rrf.fuse({"dense": chunks}, top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_available(self) -> None:
        """When top_k > available chunks, all are returned."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {"dense": [_sc("c1", 0.9, "dense"), _sc("c2", 0.8, "dense")]},
            top_k=100,
        )
        assert len(results) == 2

    def test_top_k_zero(self) -> None:
        """top_k=0 returns empty list."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {"dense": [_sc("c1", 0.9, "dense")]},
            top_k=0,
        )
        assert results == []

    def test_top_k_one(self) -> None:
        """top_k=1 returns only the highest-scoring chunk."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse(
            {
                "dense": [_sc("c1", 0.95, "dense"), _sc("c2", 0.88, "dense")],
                "bm25": [_sc("c2", 3.2, "bm25")],
            },
            top_k=1,
        )
        assert len(results) == 1
        # c2 is in both channels, so highest RRF score
        assert results[0].chunk_id == "c2"


# ---------------------------------------------------------------------------
# TestRRFScoring
# ---------------------------------------------------------------------------


class TestRRFScoring:
    """Verify RRF formula: 1/(k + rank) with various k values."""

    def test_formula_k60(self) -> None:
        """Standard k=60 formula check."""
        rrf = ReciprocalRankFusion(k=60)
        results = rrf.fuse({"ch": [_sc("a", 1.0, "ch"), _sc("b", 0.5, "ch")]})
        assert results[0].rrf_score == pytest.approx(1.0 / 61)
        assert results[1].rrf_score == pytest.approx(1.0 / 62)

    def test_formula_k1(self) -> None:
        """Small k=1 gives more weight to rank differences."""
        rrf = ReciprocalRankFusion(k=1)
        results = rrf.fuse({"ch": [_sc("a", 1.0, "ch"), _sc("b", 0.5, "ch")]})
        # rank 1 -> 1/(1+1) = 0.5, rank 2 -> 1/(1+2) = 0.333...
        assert results[0].rrf_score == pytest.approx(0.5)
        assert results[1].rrf_score == pytest.approx(1.0 / 3)

    def test_formula_large_k(self) -> None:
        """Large k diminishes the impact of rank differences."""
        rrf = ReciprocalRankFusion(k=1000)
        results = rrf.fuse({"ch": [_sc("a", 1.0, "ch"), _sc("b", 0.5, "ch")]})
        # With k=1000, ranks 1 and 2 should be nearly equal.
        diff = abs(results[0].rrf_score - results[1].rrf_score)
        assert diff < 1e-6

    def test_k_property(self) -> None:
        """k property is accessible."""
        rrf = ReciprocalRankFusion(k=42)
        assert rrf.k == 42

    def test_multi_channel_sum(self) -> None:
        """RRF score is the sum of contributions from each channel."""
        k = 60
        rrf = ReciprocalRankFusion(k=k)

        # c1 appears at rank 2 in dense, rank 3 in bm25
        results = rrf.fuse(
            {
                "dense": [_sc("c0", 1.0, "dense"), _sc("c1", 0.9, "dense")],
                "bm25": [
                    _sc("c0", 3.0, "bm25"),
                    _sc("c9", 2.5, "bm25"),
                    _sc("c1", 2.0, "bm25"),
                ],
            },
        )

        by_id = {r.chunk_id: r for r in results}
        # c1: 1/(60+2) + 1/(60+3) = 1/62 + 1/63
        expected = 1.0 / 62 + 1.0 / 63
        assert by_id["c1"].rrf_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tied scores produce deterministic ordering by chunk_id."""

    def test_tied_scores_sorted_by_id(self) -> None:
        """Chunks with equal RRF scores are sorted by chunk_id."""
        rrf = ReciprocalRankFusion(k=60)
        # Each chunk is rank 1 in its own channel -> same RRF score.
        results = rrf.fuse(
            {
                "ch1": [_sc("zebra", 0.9, "ch1")],
                "ch2": [_sc("apple", 0.9, "ch2")],
                "ch3": [_sc("mango", 0.9, "ch3")],
            },
        )

        # All have same score 1/(60+1).
        assert results[0].rrf_score == results[1].rrf_score == results[2].rrf_score
        # Sorted alphabetically by chunk_id.
        assert [r.chunk_id for r in results] == ["apple", "mango", "zebra"]

    def test_repeated_calls_same_order(self) -> None:
        """Multiple calls with the same input produce the same order."""
        rrf = ReciprocalRankFusion()
        channel_results = {
            "dense": [
                _sc("c1", 0.9, "dense"),
                _sc("c2", 0.8, "dense"),
            ],
            "bm25": [
                _sc("c3", 3.0, "bm25"),
                _sc("c1", 2.5, "bm25"),
            ],
        }
        r1 = rrf.fuse(channel_results)
        r2 = rrf.fuse(channel_results)
        assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]
        assert [c.rrf_score for c in r1] == [c.rrf_score for c in r2]


# ---------------------------------------------------------------------------
# TestReturnType
# ---------------------------------------------------------------------------


class TestReturnType:
    """Ensure the returned objects are proper FusedChunk models."""

    def test_returns_fused_chunks(self) -> None:
        """Every result is a FusedChunk instance."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse({"dense": [_sc("c1", 0.9, "dense")]})
        assert len(results) == 1
        assert isinstance(results[0], FusedChunk)

    def test_rerank_score_is_none(self) -> None:
        """Fused chunks have no rerank_score until the reranker runs."""
        rrf = ReciprocalRankFusion()
        results = rrf.fuse({"dense": [_sc("c1", 0.9, "dense")]})
        assert results[0].rerank_score is None
