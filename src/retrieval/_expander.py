"""Parent document context expansion via Redis.

Fetches parent text and judgment headers from Redis to provide
broader context around matched chunks during retrieval.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.retrieval._exceptions import ContextExpansionError, SearchNotAvailableError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.retrieval._models import ExpandedContext, FusedChunk, RetrievalSettings

_log = get_logger(__name__)


class ParentDocumentExpander:
    """Expands retrieved chunks with parent/header context from Redis."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._client: object | None = None
        self._encoder: object | None = None

    def _ensure_client(self) -> None:
        """Lazy-initialize the async Redis client.

        Raises:
            SearchNotAvailableError: If the ``redis`` package is not installed.
        """
        if self._client is not None:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            msg = (
                "redis is required for context expansion. Install with: pip install redis[hiredis]"
            )
            raise SearchNotAvailableError(msg) from exc

        self._client = aioredis.from_url(self._settings.redis_url)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        import tiktoken

        if self._encoder is None:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        return len(self._encoder.encode(text))  # type: ignore[union-attr]

    async def expand(
        self,
        chunks: list[FusedChunk],
        max_context_tokens: int = 30_000,
    ) -> list[ExpandedContext]:
        """Expand chunks with parent text and judgment headers from Redis.

        For each chunk, looks up its ``parent_chunk_id`` and
        ``judgment_header_chunk_id`` in Redis, then assembles an
        :class:`ExpandedContext` with the parent/header text appended.

        Token budget tracking stops adding parent/header text once
        *max_context_tokens* is exceeded.

        Args:
            chunks: Fused chunks from RRF / reranking.
            max_context_tokens: Maximum cumulative token budget.

        Returns:
            List of :class:`ExpandedContext` in the same order as *chunks*.

        Raises:
            ContextExpansionError: If the Redis connection fails.
            SearchNotAvailableError: If redis is not installed.
        """
        from src.retrieval._models import ExpandedContext

        self._ensure_client()

        # --- 1. Collect all unique parent/header IDs needed ---
        parent_ids: dict[str, list[int]] = {}  # redis key -> chunk indices
        header_ids: dict[str, list[int]] = {}
        prefix = self._settings.redis_key_prefix

        for idx, chunk in enumerate(chunks):
            parent_info = (chunk.payload or {}).get("parent_info", {})
            pid = parent_info.get("parent_chunk_id")
            hid = parent_info.get("judgment_header_chunk_id")

            if pid is not None and self._settings.include_parent_chunks:
                key = f"{prefix}{pid}"
                parent_ids.setdefault(key, []).append(idx)

            if hid is not None and self._settings.include_judgment_headers:
                key = f"{prefix}{hid}"
                header_ids.setdefault(key, []).append(idx)

        # --- 2. Batch-fetch from Redis (deduplicated) ---
        all_keys = list(set(parent_ids.keys()) | set(header_ids.keys()))
        fetched: dict[str, dict | None] = {}

        for key in all_keys:
            try:
                raw = await self._client.get(key)  # type: ignore[union-attr]
            except Exception as exc:
                msg = f"Redis connection error during context expansion: {exc}"
                raise ContextExpansionError(msg) from exc

            if raw is None:
                fetched[key] = None
                _log.debug("parent_key_miss", key=key)
                continue

            try:
                data = json.loads(raw)
                fetched[key] = data
            except (json.JSONDecodeError, TypeError):
                _log.warning("invalid_parent_json", key=key)
                fetched[key] = None

        # --- 3. Build ExpandedContext for each chunk ---
        results: list[ExpandedContext] = []
        cumulative_tokens = 0
        budget_exceeded = False

        for chunk in chunks:
            parent_info = (chunk.payload or {}).get("parent_info", {})
            pid = parent_info.get("parent_chunk_id")
            hid = parent_info.get("judgment_header_chunk_id")

            # Prefer contextualized_text, fallback to plain text
            chunk_text = chunk.contextualized_text or chunk.text

            # Relevance score: prefer rerank_score, fallback to rrf_score
            relevance_score = (
                chunk.rerank_score if chunk.rerank_score is not None else chunk.rrf_score
            )

            parent_text: str | None = None
            header_text: str | None = None

            if not budget_exceeded:
                # Fetch parent text
                if pid is not None and self._settings.include_parent_chunks:
                    pkey = f"{prefix}{pid}"
                    pdata = fetched.get(pkey)
                    if pdata is not None:
                        parent_text = pdata.get("text")

                # Fetch header text
                if hid is not None and self._settings.include_judgment_headers:
                    hkey = f"{prefix}{hid}"
                    hdata = fetched.get(hkey)
                    if hdata is not None:
                        header_text = hdata.get("text")

            # Count tokens
            total_tokens = self._count_tokens(chunk_text)
            if parent_text is not None:
                total_tokens += self._count_tokens(parent_text)
            if header_text is not None:
                total_tokens += self._count_tokens(header_text)

            # Check if adding this chunk's context exceeds budget
            if (
                cumulative_tokens + total_tokens > max_context_tokens
                and not budget_exceeded
                and (parent_text is not None or header_text is not None)
            ):
                # Still include chunk_text, but strip parent/header
                parent_text = None
                header_text = None
                total_tokens = self._count_tokens(chunk_text)
                budget_exceeded = True

            cumulative_tokens += total_tokens

            results.append(
                ExpandedContext(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk_text,
                    parent_text=parent_text,
                    judgment_header_text=header_text,
                    relevance_score=relevance_score,
                    total_tokens=total_tokens,
                    metadata=chunk.payload or {},
                )
            )

        _log.info(
            "context_expanded",
            chunk_count=len(results),
            total_tokens=cumulative_tokens,
            budget_exceeded=budget_exceeded,
        )
        return results
