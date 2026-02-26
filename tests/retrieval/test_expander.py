"""Tests for ParentDocumentExpander — Redis-backed context expansion."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.retrieval._exceptions import ContextExpansionError, SearchNotAvailableError
from src.retrieval._expander import ParentDocumentExpander
from src.retrieval._models import FusedChunk, RetrievalSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parent_json(
    chunk_id: str = "parent-1",
    text: str = "Full parent text for context expansion.",
    document_type: str = "statute",
    chunk_type: str = "statutory_text",
    **extra: Any,
) -> str:
    """Build a serialised parent record as stored in Redis."""
    data: dict[str, Any] = {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "text": text,
        "document_type": document_type,
        "chunk_type": chunk_type,
        **extra,
    }
    return json.dumps(data)


def _header_json(
    chunk_id: str = "header-1",
    text: str = "Case: State of Maharashtra v. Accused — Criminal Appeal No. 123/2024",
    **extra: Any,
) -> str:
    return _parent_json(
        chunk_id=chunk_id,
        text=text,
        document_type="judgment",
        chunk_type="judgment_header",
        **extra,
    )


def _fused_chunk(
    chunk_id: str = "c1",
    text: str = "Section 420 deals with cheating.",
    rrf_score: float = 0.032,
    rerank_score: float | None = 0.95,
    channels: list[str] | None = None,
    parent_chunk_id: str | None = None,
    judgment_header_chunk_id: str | None = None,
    contextualized_text: str | None = None,
) -> FusedChunk:
    """Build a FusedChunk with optional parent info."""
    payload: dict[str, Any] = {
        "parent_info": {
            "parent_chunk_id": parent_chunk_id,
            "judgment_header_chunk_id": judgment_header_chunk_id,
            "sibling_chunk_ids": [],
        },
    }
    return FusedChunk(
        chunk_id=chunk_id,
        text=text,
        contextualized_text=contextualized_text,
        rrf_score=rrf_score,
        rerank_score=rerank_score,
        channels=channels or ["dense", "bm25"],
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> RetrievalSettings:
    return RetrievalSettings(redis_key_prefix="parent:")


@pytest.fixture()
def mock_redis_client() -> AsyncMock:
    """Async Redis client mock with configurable key->value store."""
    client = AsyncMock()

    store: dict[str, str | None] = {
        "parent:parent-1": _parent_json("parent-1", "Full parent text for context expansion."),
        "parent:parent-2": _parent_json("parent-2", "Another parent text."),
        "parent:header-1": _header_json("header-1"),
        "parent:header-2": _header_json(
            "header-2",
            "Case: Union of India v. Respondent — Writ Petition (C) 456/2023",
        ),
    }

    async def mock_get(key: str) -> str | None:
        return store.get(key)

    client.get = AsyncMock(side_effect=mock_get)
    return client


@pytest.fixture()
def expander(settings: RetrievalSettings, mock_redis_client: AsyncMock) -> ParentDocumentExpander:
    """Expander with pre-injected mock Redis client."""
    exp = ParentDocumentExpander(settings)
    exp._client = mock_redis_client
    return exp


# ===========================================================================
# TestEnsureClient
# ===========================================================================


class TestEnsureClient:
    """Tests for lazy Redis client initialization."""

    def test_missing_redis_raises_not_available(self, settings: RetrievalSettings) -> None:
        """When redis package is not installed, raise SearchNotAvailableError."""
        exp = ParentDocumentExpander(settings)
        with (
            patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}),
            pytest.raises(SearchNotAvailableError, match="redis is required"),
        ):
            exp._ensure_client()

    def test_ensure_client_creates_client(self, settings: RetrievalSettings) -> None:
        """When redis is available, _ensure_client sets _client."""
        exp = ParentDocumentExpander(settings)
        mock_client = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
            exp._ensure_client()
            assert exp._client is mock_client
            mock_from_url.assert_called_once_with(settings.redis_url)

    def test_ensure_client_idempotent(
        self, settings: RetrievalSettings, mock_redis_client: AsyncMock
    ) -> None:
        """Calling _ensure_client twice does not recreate the client."""
        exp = ParentDocumentExpander(settings)
        exp._client = mock_redis_client
        exp._ensure_client()  # should be no-op
        assert exp._client is mock_redis_client


# ===========================================================================
# TestExpand
# ===========================================================================


class TestExpand:
    """Core expansion logic."""

    async def test_statute_chunk_with_parent(self, expander: ParentDocumentExpander) -> None:
        """Statute chunk with parent_chunk_id gets parent_text."""
        chunks = [_fused_chunk(parent_chunk_id="parent-1")]
        results = await expander.expand(chunks)

        assert len(results) == 1
        ctx = results[0]
        assert ctx.chunk_id == "c1"
        assert ctx.chunk_text == "Section 420 deals with cheating."
        assert ctx.parent_text == "Full parent text for context expansion."
        assert ctx.judgment_header_text is None

    async def test_judgment_chunk_with_header(self, expander: ParentDocumentExpander) -> None:
        """Judgment chunk with judgment_header_chunk_id gets header_text."""
        chunks = [
            _fused_chunk(
                chunk_id="j1",
                text="The court held mens rea is essential.",
                judgment_header_chunk_id="header-1",
            ),
        ]
        results = await expander.expand(chunks)

        assert len(results) == 1
        ctx = results[0]
        assert ctx.chunk_id == "j1"
        assert ctx.judgment_header_text is not None
        assert "State of Maharashtra" in ctx.judgment_header_text

    async def test_chunk_with_both_parent_and_header(
        self, expander: ParentDocumentExpander
    ) -> None:
        """Chunk referencing both parent and header gets both texts."""
        chunks = [
            _fused_chunk(
                parent_chunk_id="parent-1",
                judgment_header_chunk_id="header-1",
            ),
        ]
        results = await expander.expand(chunks)

        ctx = results[0]
        assert ctx.parent_text is not None
        assert ctx.judgment_header_text is not None

    async def test_chunk_with_no_parent_info(self, expander: ParentDocumentExpander) -> None:
        """Chunk with no parent references gets no expansion."""
        chunks = [_fused_chunk()]  # no parent_chunk_id or header
        results = await expander.expand(chunks)

        ctx = results[0]
        assert ctx.parent_text is None
        assert ctx.judgment_header_text is None
        assert ctx.chunk_text == "Section 420 deals with cheating."

    async def test_redis_key_miss(self, expander: ParentDocumentExpander) -> None:
        """When Redis key not found, parent_text is None (graceful skip)."""
        chunks = [_fused_chunk(parent_chunk_id="nonexistent-parent")]
        results = await expander.expand(chunks)

        ctx = results[0]
        assert ctx.parent_text is None

    async def test_multiple_chunks_expanded(self, expander: ParentDocumentExpander) -> None:
        """Multiple chunks are expanded in order."""
        chunks = [
            _fused_chunk(chunk_id="c1", parent_chunk_id="parent-1"),
            _fused_chunk(chunk_id="c2", parent_chunk_id="parent-2"),
        ]
        results = await expander.expand(chunks)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].parent_text == "Full parent text for context expansion."
        assert results[1].chunk_id == "c2"
        assert results[1].parent_text == "Another parent text."

    async def test_contextualized_text_preferred(self, expander: ParentDocumentExpander) -> None:
        """When contextualized_text is present, it is used as chunk_text."""
        chunks = [
            _fused_chunk(
                text="Original text.",
                contextualized_text="Enriched context text.",
            ),
        ]
        results = await expander.expand(chunks)

        assert results[0].chunk_text == "Enriched context text."

    async def test_plain_text_when_no_contextualized(
        self, expander: ParentDocumentExpander
    ) -> None:
        """When contextualized_text is None, plain text is used."""
        chunks = [_fused_chunk(contextualized_text=None)]
        results = await expander.expand(chunks)

        assert results[0].chunk_text == "Section 420 deals with cheating."

    async def test_empty_chunk_list(self, expander: ParentDocumentExpander) -> None:
        """Expanding an empty list returns an empty list."""
        results = await expander.expand([])
        assert results == []

    async def test_chunk_with_empty_payload(self, expander: ParentDocumentExpander) -> None:
        """Chunk with empty payload (no parent_info) gets no expansion."""
        chunk = FusedChunk(
            chunk_id="bare",
            text="Bare chunk.",
            rrf_score=0.01,
            channels=["dense"],
            payload={},
        )
        results = await expander.expand([chunk])

        ctx = results[0]
        assert ctx.parent_text is None
        assert ctx.judgment_header_text is None


# ===========================================================================
# TestTokenBudget
# ===========================================================================


class TestTokenBudget:
    """Token budget tracking."""

    async def test_respects_max_context_tokens(self, expander: ParentDocumentExpander) -> None:
        """Once budget exceeded, further chunks get no parent/header text."""
        # Use a very small budget
        chunks = [
            _fused_chunk(chunk_id="c1", parent_chunk_id="parent-1"),
            _fused_chunk(chunk_id="c2", parent_chunk_id="parent-2"),
        ]
        # Budget so small that first chunk with parent fills it
        results = await expander.expand(chunks, max_context_tokens=5)

        # First chunk should still have chunk_text (no parent due to budget)
        assert results[0].chunk_text is not None
        # At least one chunk should have parent_text stripped
        no_parent = [r for r in results if r.parent_text is None]
        assert len(no_parent) >= 1

    async def test_budget_includes_all_text(self, expander: ParentDocumentExpander) -> None:
        """Token budget counts chunk_text + parent_text + header_text."""
        chunks = [
            _fused_chunk(
                parent_chunk_id="parent-1",
                judgment_header_chunk_id="header-1",
            ),
        ]
        results = await expander.expand(chunks, max_context_tokens=1_000_000)
        ctx = results[0]

        # total_tokens should be >= tokens of chunk_text alone
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        chunk_tokens = len(enc.encode(ctx.chunk_text))
        assert ctx.total_tokens >= chunk_tokens

    async def test_budget_strips_parent_not_chunk(
        self, settings: RetrievalSettings, mock_redis_client: AsyncMock
    ) -> None:
        """When budget exceeded, chunk_text is always preserved."""
        expander = ParentDocumentExpander(settings)
        expander._client = mock_redis_client

        # Two chunks where second has a huge parent
        huge_text = "word " * 10000
        store_data = {
            "parent:huge-parent": _parent_json("huge-parent", huge_text),
        }

        original_get = mock_redis_client.get.side_effect

        async def extended_get(key: str) -> str | None:
            if key in store_data:
                return store_data[key]
            return await original_get(key)

        mock_redis_client.get = AsyncMock(side_effect=extended_get)

        chunks = [
            _fused_chunk(chunk_id="c1", parent_chunk_id="parent-1"),
            _fused_chunk(chunk_id="c2", parent_chunk_id="huge-parent"),
        ]
        results = await expander.expand(chunks, max_context_tokens=50)

        # Every chunk should still have chunk_text
        for r in results:
            assert r.chunk_text is not None
            assert len(r.chunk_text) > 0


# ===========================================================================
# TestDeduplication
# ===========================================================================


class TestDeduplication:
    """Redis fetch deduplication."""

    async def test_same_parent_fetched_once(
        self, expander: ParentDocumentExpander, mock_redis_client: AsyncMock
    ) -> None:
        """Two chunks sharing the same parent_chunk_id cause only one Redis get."""
        chunks = [
            _fused_chunk(chunk_id="c1", parent_chunk_id="parent-1"),
            _fused_chunk(chunk_id="c2", parent_chunk_id="parent-1"),
        ]
        results = await expander.expand(chunks, max_context_tokens=1_000_000)

        # Both should have same parent text
        assert results[0].parent_text == results[1].parent_text
        assert results[0].parent_text == "Full parent text for context expansion."

        # Count how many times Redis get was called with parent:parent-1
        parent_calls = [
            call
            for call in mock_redis_client.get.call_args_list
            if call.args[0] == "parent:parent-1"
        ]
        assert len(parent_calls) == 1

    async def test_different_parents_fetched_separately(
        self, expander: ParentDocumentExpander, mock_redis_client: AsyncMock
    ) -> None:
        """Two chunks with different parents cause two Redis gets."""
        chunks = [
            _fused_chunk(chunk_id="c1", parent_chunk_id="parent-1"),
            _fused_chunk(chunk_id="c2", parent_chunk_id="parent-2"),
        ]
        await expander.expand(chunks, max_context_tokens=1_000_000)

        get_keys = [call.args[0] for call in mock_redis_client.get.call_args_list]
        assert "parent:parent-1" in get_keys
        assert "parent:parent-2" in get_keys


# ===========================================================================
# TestErrorHandling
# ===========================================================================


class TestErrorHandling:
    """Error paths."""

    async def test_redis_connection_error(self, settings: RetrievalSettings) -> None:
        """Redis connection error raises ContextExpansionError."""
        expander = ParentDocumentExpander(settings)
        client = AsyncMock()
        client.get = AsyncMock(side_effect=ConnectionError("Connection refused"))
        expander._client = client

        chunks = [_fused_chunk(parent_chunk_id="parent-1")]
        with pytest.raises(ContextExpansionError, match="Redis connection error"):
            await expander.expand(chunks)

    async def test_invalid_json_skipped(self, settings: RetrievalSettings) -> None:
        """Invalid JSON in Redis value is skipped gracefully."""
        expander = ParentDocumentExpander(settings)
        client = AsyncMock()
        client.get = AsyncMock(return_value="NOT{VALID}JSON!!!")
        expander._client = client

        chunks = [_fused_chunk(parent_chunk_id="bad-parent")]
        results = await expander.expand(chunks)

        assert len(results) == 1
        assert results[0].parent_text is None  # gracefully skipped

    async def test_redis_not_installed_raises(self, settings: RetrievalSettings) -> None:
        """If redis is not installed, expand raises SearchNotAvailableError."""
        expander = ParentDocumentExpander(settings)
        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            chunks = [_fused_chunk(parent_chunk_id="parent-1")]
            with pytest.raises(SearchNotAvailableError, match="redis is required"):
                await expander.expand(chunks)

    async def test_redis_returns_none_for_key(self, settings: RetrievalSettings) -> None:
        """Redis returning None for a key results in parent_text=None."""
        expander = ParentDocumentExpander(settings)
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        expander._client = client

        chunks = [_fused_chunk(parent_chunk_id="missing-key")]
        results = await expander.expand(chunks)

        assert results[0].parent_text is None


# ===========================================================================
# TestScoring
# ===========================================================================


class TestScoring:
    """Relevance score selection."""

    async def test_rerank_score_preferred(self, expander: ParentDocumentExpander) -> None:
        """When rerank_score is present, it is used as relevance_score."""
        chunks = [_fused_chunk(rrf_score=0.032, rerank_score=0.95)]
        results = await expander.expand(chunks)

        assert results[0].relevance_score == 0.95

    async def test_falls_back_to_rrf_score(self, expander: ParentDocumentExpander) -> None:
        """When rerank_score is None, rrf_score is used."""
        chunks = [_fused_chunk(rrf_score=0.032, rerank_score=None)]
        results = await expander.expand(chunks)

        assert results[0].relevance_score == 0.032

    async def test_zero_rerank_score_used(self, expander: ParentDocumentExpander) -> None:
        """A rerank_score of 0.0 should still be used (not treated as falsy)."""
        chunks = [_fused_chunk(rrf_score=0.5, rerank_score=0.0)]
        results = await expander.expand(chunks)

        assert results[0].relevance_score == 0.0


# ===========================================================================
# TestSettingsFlags
# ===========================================================================


class TestSettingsFlags:
    """Expansion respects settings flags."""

    async def test_include_parent_disabled(self, mock_redis_client: AsyncMock) -> None:
        """When include_parent_chunks=False, parent_text is not fetched."""
        settings = RetrievalSettings(include_parent_chunks=False)
        exp = ParentDocumentExpander(settings)
        exp._client = mock_redis_client

        chunks = [_fused_chunk(parent_chunk_id="parent-1")]
        results = await exp.expand(chunks)

        assert results[0].parent_text is None

    async def test_include_headers_disabled(self, mock_redis_client: AsyncMock) -> None:
        """When include_judgment_headers=False, header_text is not fetched."""
        settings = RetrievalSettings(include_judgment_headers=False)
        exp = ParentDocumentExpander(settings)
        exp._client = mock_redis_client

        chunks = [_fused_chunk(judgment_header_chunk_id="header-1")]
        results = await exp.expand(chunks)

        assert results[0].judgment_header_text is None


# ===========================================================================
# TestTokenCounting
# ===========================================================================


class TestTokenCounting:
    """Verify token counting accuracy."""

    async def test_total_tokens_matches_tiktoken(self, expander: ParentDocumentExpander) -> None:
        """total_tokens should equal the sum of token counts from tiktoken."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        chunks = [
            _fused_chunk(
                text="Section 420 IPC.",
                parent_chunk_id="parent-1",
                judgment_header_chunk_id="header-1",
            ),
        ]
        results = await expander.expand(chunks, max_context_tokens=1_000_000)
        ctx = results[0]

        expected = len(enc.encode(ctx.chunk_text))
        if ctx.parent_text:
            expected += len(enc.encode(ctx.parent_text))
        if ctx.judgment_header_text:
            expected += len(enc.encode(ctx.judgment_header_text))

        assert ctx.total_tokens == expected

    async def test_no_parent_token_count(self, expander: ParentDocumentExpander) -> None:
        """Token count with no parent equals chunk text tokens only."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        chunks = [_fused_chunk(text="Short text.")]
        results = await expander.expand(chunks)

        assert results[0].total_tokens == len(enc.encode("Short text."))


# ===========================================================================
# TestMetadata
# ===========================================================================


class TestMetadata:
    """Metadata propagation."""

    async def test_metadata_from_payload(self, expander: ParentDocumentExpander) -> None:
        """ExpandedContext.metadata should contain the chunk payload."""
        chunks = [_fused_chunk(parent_chunk_id="parent-1")]
        results = await expander.expand(chunks)

        assert "parent_info" in results[0].metadata
        assert results[0].metadata["parent_info"]["parent_chunk_id"] == "parent-1"
