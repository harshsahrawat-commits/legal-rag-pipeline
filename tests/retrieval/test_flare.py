"""Tests for FLAREActiveRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval._exceptions import FLAREError
from src.retrieval._flare import FLAREActiveRetriever, _scored_to_expanded
from src.retrieval._models import (
    ExpandedContext,
    RetrievalSettings,
    ScoredChunk,
)

# --- Helpers ---


def _make_expanded(chunk_id: str, text: str = "Some legal text here.") -> ExpandedContext:
    return ExpandedContext(
        chunk_id=chunk_id,
        chunk_text=text,
        relevance_score=0.9,
        total_tokens=len(text.split()),
    )


def _make_scored(chunk_id: str, text: str = "Re-retrieved text.") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.85,
        channel="dense",
        payload={"id": chunk_id},
    )


def _mock_anthropic_client(response_text: str) -> MagicMock:
    """Create a mock AsyncAnthropic client that returns predefined text."""
    client = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    response = MagicMock()
    response.content = [content_block]
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.fixture()
def settings() -> RetrievalSettings:
    return RetrievalSettings(
        flare_enabled=True,
        flare_segment_tokens=10,  # small for testing
        flare_confidence_threshold=0.5,
        flare_max_retrievals=5,
    )


@pytest.fixture()
def flare(settings: RetrievalSettings) -> FLAREActiveRetriever:
    return FLAREActiveRetriever(settings)


# ===================================================================
# Availability
# ===================================================================


class TestIsAvailable:
    def test_available_when_enabled_and_anthropic_exists(self, flare: FLAREActiveRetriever) -> None:
        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            assert flare.is_available is True

    def test_not_available_when_disabled(self) -> None:
        settings = RetrievalSettings(flare_enabled=False)
        f = FLAREActiveRetriever(settings)
        assert f.is_available is False

    def test_not_available_when_anthropic_missing(self, flare: FLAREActiveRetriever) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            assert flare.is_available is False


# ===================================================================
# Client initialization
# ===================================================================


class TestEnsureClient:
    def test_raises_flare_error_when_no_anthropic(self, flare: FLAREActiveRetriever) -> None:
        with (
            patch.dict("sys.modules", {"anthropic": None}),
            pytest.raises(FLAREError, match="anthropic is required"),
        ):
            flare._ensure_client()

    def test_creates_client_when_available(self, flare: FLAREActiveRetriever) -> None:
        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            flare._ensure_client()
        assert flare._client is not None


# ===================================================================
# Segmentation
# ===================================================================


class TestSegmentChunks:
    def test_segments_by_word_count(self, flare: FLAREActiveRetriever) -> None:
        chunks = [_make_expanded("c1", " ".join(f"word{i}" for i in range(25)))]
        segments = flare._segment_chunks(chunks)
        # 25 words / 10 per segment = 3 segments
        assert len(segments) == 3

    def test_empty_chunks(self, flare: FLAREActiveRetriever) -> None:
        segments = flare._segment_chunks([])
        assert segments == []

    def test_chunks_with_no_text(self, flare: FLAREActiveRetriever) -> None:
        chunks = [_make_expanded("c1", "")]
        segments = flare._segment_chunks(chunks)
        assert segments == []

    def test_multiple_chunks_combined(self, flare: FLAREActiveRetriever) -> None:
        chunks = [
            _make_expanded("c1", " ".join(f"a{i}" for i in range(10))),
            _make_expanded("c2", " ".join(f"b{i}" for i in range(10))),
        ]
        segments = flare._segment_chunks(chunks)
        # Combined ~22 words (10 + separator + 10), segment_size=10 → 3 segments
        assert len(segments) >= 2


# ===================================================================
# Confidence parsing
# ===================================================================


class TestParseConfidenceScores:
    def test_valid_json_array(self, flare: FLAREActiveRetriever) -> None:
        scores = flare._parse_confidence_scores("[0.9, 0.3, 0.7]", 3)
        assert scores == [0.9, 0.3, 0.7]

    def test_pads_short_array(self, flare: FLAREActiveRetriever) -> None:
        scores = flare._parse_confidence_scores("[0.9]", 3)
        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.5  # threshold default

    def test_truncates_long_array(self, flare: FLAREActiveRetriever) -> None:
        scores = flare._parse_confidence_scores("[0.9, 0.8, 0.7, 0.6]", 2)
        assert len(scores) == 2

    def test_invalid_json_returns_defaults(self, flare: FLAREActiveRetriever) -> None:
        scores = flare._parse_confidence_scores("not json", 3)
        assert scores == [0.5, 0.5, 0.5]

    def test_non_array_json_returns_defaults(self, flare: FLAREActiveRetriever) -> None:
        scores = flare._parse_confidence_scores('{"score": 0.9}', 2)
        assert scores == [0.5, 0.5]


# ===================================================================
# Follow-up query parsing
# ===================================================================


class TestParseFollowUpQueries:
    def test_valid_json_array(self, flare: FLAREActiveRetriever) -> None:
        queries = flare._parse_follow_up_queries('["q1", "q2"]', "fallback")
        assert queries == ["q1", "q2"]

    def test_invalid_json_returns_fallback(self, flare: FLAREActiveRetriever) -> None:
        queries = flare._parse_follow_up_queries("not json", "fallback query")
        assert queries == ["fallback query"]


# ===================================================================
# Active retrieval (full flow)
# ===================================================================


class TestActiveRetrieve:
    @pytest.mark.asyncio
    async def test_flare_disabled_returns_original(self) -> None:
        settings = RetrievalSettings(flare_enabled=False)
        flare = FLAREActiveRetriever(settings)
        chunks = [_make_expanded("c1")]
        search_fn = AsyncMock(return_value=[])

        result, count = await flare.active_retrieve("query", chunks, search_fn)

        assert result is chunks
        assert count == 0
        search_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_high_confidence_no_re_retrieval(self, flare: FLAREActiveRetriever) -> None:
        """When all segments are high-confidence, no re-retrieval happens."""
        chunks = [_make_expanded("c1", " ".join(f"w{i}" for i in range(15)))]
        search_fn = AsyncMock(return_value=[])

        # Mock: all segments score above threshold
        flare._client = _mock_anthropic_client("[0.9, 0.8]")

        result, count = await flare.active_retrieve("query", chunks, search_fn)

        assert result is chunks
        assert count == 0
        search_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_re_retrieval(self, flare: FLAREActiveRetriever) -> None:
        """Low-confidence segments trigger follow-up queries and re-retrieval."""
        chunks = [_make_expanded("c1", " ".join(f"w{i}" for i in range(15)))]
        new_chunk = _make_scored("c_new", "New relevant info.")
        search_fn = AsyncMock(return_value=[new_chunk])

        # First LLM call: low confidence scores
        # Second LLM call: follow-up queries
        responses = [
            "[0.2, 0.1]",  # confidence assessment
            '["follow-up query 1", "follow-up query 2"]',  # follow-up queries
        ]
        call_count = 0

        async def multi_response(**kwargs):
            nonlocal call_count
            text = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            content_block = MagicMock()
            content_block.text = text
            resp = MagicMock()
            resp.content = [content_block]
            return resp

        flare._client = MagicMock()
        flare._client.messages.create = AsyncMock(side_effect=multi_response)

        result, count = await flare.active_retrieve("query", chunks, search_fn)

        assert count >= 1
        assert len(result) > len(chunks)
        # New chunk was added
        result_ids = {c.chunk_id for c in result}
        assert "c_new" in result_ids

    @pytest.mark.asyncio
    async def test_re_retrieval_deduplicates(self, flare: FLAREActiveRetriever) -> None:
        """Chunks already in initial results are not duplicated."""
        chunks = [_make_expanded("c1", " ".join(f"w{i}" for i in range(15)))]
        # Re-retrieve returns the same chunk ID
        same_chunk = _make_scored("c1", "Same chunk again.")
        search_fn = AsyncMock(return_value=[same_chunk])

        responses = ["[0.1]", '["follow-up"]']
        call_count = 0

        async def multi_response(**kwargs):
            nonlocal call_count
            text = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            block = MagicMock()
            block.text = text
            resp = MagicMock()
            resp.content = [block]
            return resp

        flare._client = MagicMock()
        flare._client.messages.create = AsyncMock(side_effect=multi_response)

        result, count = await flare.active_retrieve("query", chunks, search_fn)

        assert count >= 1
        # No new chunks added (same ID)
        assert len(result) == len(chunks)

    @pytest.mark.asyncio
    async def test_max_retrievals_cap(self, flare: FLAREActiveRetriever) -> None:
        """Re-retrieval is capped at flare_max_retrievals."""
        flare._settings.flare_max_retrievals = 2
        chunks = [_make_expanded("c1", " ".join(f"w{i}" for i in range(15)))]
        search_fn = AsyncMock(return_value=[_make_scored("cn")])

        # Return 5 follow-up queries, but cap is 2
        responses = ["[0.1]", '["q1","q2","q3","q4","q5"]']
        call_count = 0

        async def multi_response(**kwargs):
            nonlocal call_count
            text = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            block = MagicMock()
            block.text = text
            resp = MagicMock()
            resp.content = [block]
            return resp

        flare._client = MagicMock()
        flare._client.messages.create = AsyncMock(side_effect=multi_response)

        _, count = await flare.active_retrieve("query", chunks, search_fn)

        assert count == 2
        assert search_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_error_graceful_fallback(self, flare: FLAREActiveRetriever) -> None:
        """LLM API error → return original chunks gracefully."""
        chunks = [_make_expanded("c1", " ".join(f"w{i}" for i in range(15)))]
        search_fn = AsyncMock()

        flare._client = MagicMock()
        flare._client.messages.create = AsyncMock(side_effect=RuntimeError("API error"))

        result, count = await flare.active_retrieve("query", chunks, search_fn)

        # Original chunks returned unchanged
        assert result is chunks
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_immediately(self, flare: FLAREActiveRetriever) -> None:
        """No chunks → no segments → return immediately."""
        flare._client = _mock_anthropic_client("[]")
        search_fn = AsyncMock()

        result, count = await flare.active_retrieve("query", [], search_fn)

        assert result == []
        assert count == 0


# ===================================================================
# _scored_to_expanded helper
# ===================================================================


class TestScoredToExpanded:
    def test_converts_scored_chunk(self) -> None:
        sc = ScoredChunk(
            chunk_id="c1",
            text="Chunk text here.",
            score=0.9,
            channel="dense",
            payload={"id": "c1", "doc": "test"},
        )
        ec = _scored_to_expanded(sc)
        assert ec.chunk_id == "c1"
        assert ec.chunk_text == "Chunk text here."
        assert ec.relevance_score == 0.9
        assert ec.metadata == {"id": "c1", "doc": "test"}
        assert ec.total_tokens > 0

    def test_prefers_contextualized_text(self) -> None:
        sc = ScoredChunk(
            chunk_id="c1",
            text="Plain text.",
            contextualized_text="Contextual text here.",
            score=0.8,
            channel="bm25",
        )
        ec = _scored_to_expanded(sc)
        assert ec.chunk_text == "Contextual text here."
