"""Contextual Retrieval enricher.

For each chunk, generates a 2-3 sentence context prefix using Claude Haiku
with prompt caching. The full document text is cached in the system message;
only the per-chunk query varies.

Requires ``anthropic`` (optional dependency).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.chunking._token_counter import TokenCounter
from src.enrichment._exceptions import (
    ContextualRetrievalError,
    EnricherNotAvailableError,
    LLMRateLimitError,
)
from src.enrichment.enrichers._base import BaseEnricher

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.enrichment._models import EnrichmentSettings
    from src.parsing._models import ParsedDocument

_log = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = (
    "<document>\n{full_doc_text}\n</document>\n\n"
    "You generate brief context for document chunks from Indian legal texts. "
    "Output only the context — 2-3 sentences. "
    "Do NOT repeat the chunk text."
)

_USER_TEMPLATE = (
    "Here is a chunk from the above document:\n"
    "<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Generate a brief context for this chunk. Include:\n"
    "- The Act/judgment name and section/paragraph number\n"
    "- The legal topic or concept being addressed\n"
    "- How this relates to the surrounding content"
)


class ContextualRetrievalEnricher(BaseEnricher):
    """Enrich chunks with LLM-generated context for BM25 keyword search.

    Uses Anthropic prompt caching: the full document text is placed in the
    system message with ``cache_control``, so it's cached across all chunks
    from the same document. Only the per-chunk user message varies.
    """

    def __init__(self, settings: EnrichmentSettings) -> None:
        super().__init__(settings)
        self._client: Any = None
        self._tc = TokenCounter()

    @property
    def stage_name(self) -> str:
        return "contextual_retrieval"

    async def enrich_document(
        self,
        chunks: list[LegalChunk],
        parsed_doc: ParsedDocument,
    ) -> list[LegalChunk]:
        """Generate contextual text for each chunk.

        Long documents are split into overlapping windows so each fits
        within the model's context window. Chunks are grouped by window
        and processed concurrently within each window group.
        """
        self._ensure_client()

        full_text = parsed_doc.raw_text or ""
        chunks_to_enrich = self._filter_chunks(chunks)

        if not chunks_to_enrich:
            return chunks

        windows = self._build_windows(full_text)
        chunk_window_map = self._assign_chunks_to_windows(chunks_to_enrich, full_text, len(windows))

        semaphore = asyncio.Semaphore(self._settings.concurrency)

        for window_idx, window_text in enumerate(windows):
            window_chunks = chunk_window_map.get(window_idx, [])
            if not window_chunks:
                continue

            system_messages = self._build_system_messages(window_text)

            tasks = [self._enrich_one(chunk, system_messages, semaphore) for chunk in window_chunks]
            await asyncio.gather(*tasks)

        return chunks

    def _filter_chunks(self, chunks: list[LegalChunk]) -> list[LegalChunk]:
        """Filter to chunks eligible for enrichment."""
        result = []
        for chunk in chunks:
            if chunk.ingestion.contextualized:
                continue
            if self._settings.skip_manual_review_chunks and chunk.ingestion.requires_manual_review:
                continue
            result.append(chunk)
        return result

    def _build_windows(self, full_text: str) -> list[str]:
        """Split document text into overlapping windows if needed.

        If the document fits within context_window_tokens, returns a
        single-element list. Otherwise splits into overlapping windows.
        """
        max_tokens = self._settings.context_window_tokens
        overlap_tokens = self._settings.document_window_overlap_tokens

        token_count = self._tc.count(full_text)
        if token_count <= max_tokens:
            return [full_text]

        _log.warning(
            "document_windowed",
            extra={"token_count": token_count, "max_tokens": max_tokens},
        )

        # Split by characters, estimating ~4 chars/token
        chars_per_token = max(1, len(full_text) // max(1, token_count))
        window_chars = max_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token

        windows: list[str] = []
        start = 0
        while start < len(full_text):
            end = start + window_chars
            windows.append(full_text[start:end])
            start = end - overlap_chars
            if start >= len(full_text):
                break

        return windows or [full_text]

    def _assign_chunks_to_windows(
        self,
        chunks: list[LegalChunk],
        full_text: str,
        num_windows: int,
    ) -> dict[int, list[LegalChunk]]:
        """Map each chunk to the window that most likely contains it."""
        if num_windows <= 1:
            return {0: list(chunks)}

        result: dict[int, list[LegalChunk]] = {}
        text_len = max(1, len(full_text))

        for chunk in chunks:
            # Try to find chunk text in document
            pos = full_text.find(chunk.text[:80]) if chunk.text else -1
            if pos >= 0:
                ratio = pos / text_len
            else:
                # Fall back to chunk_index proportion
                max_idx = max(1, max(c.chunk_index for c in chunks))
                ratio = chunk.chunk_index / max_idx

            window_idx = min(int(ratio * num_windows), num_windows - 1)
            result.setdefault(window_idx, []).append(chunk)

        return result

    def _build_system_messages(self, document_text: str) -> list[dict[str, Any]]:
        """Build the system message list with prompt caching."""
        return [
            {
                "type": "text",
                "text": _SYSTEM_TEMPLATE.format(full_doc_text=document_text),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def _enrich_one(
        self,
        chunk: LegalChunk,
        system_messages: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Enrich a single chunk. Errors are isolated — chunk stays unenriched."""
        async with semaphore:
            try:
                context = await self._call_llm(chunk.text, system_messages)
                if context and context.strip():
                    chunk.contextualized_text = f"{context.strip()}\n\n{chunk.text}"
                    chunk.ingestion.contextualized = True
            except LLMRateLimitError:
                raise
            except Exception:
                _log.exception(
                    "contextual_retrieval_failed",
                    extra={"chunk_id": str(chunk.id)},
                )

    async def _call_llm(
        self,
        chunk_text: str,
        system_messages: list[dict[str, Any]],
    ) -> str:
        """Call the LLM to generate context for a chunk."""
        user_message = _USER_TEMPLATE.format(chunk_text=chunk_text)

        try:
            response = await self._client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens_response,
                system=system_messages,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as exc:
            exc_type = type(exc).__name__
            if "RateLimitError" in exc_type:
                msg = f"LLM rate limit exceeded: {exc}"
                raise LLMRateLimitError(msg) from exc
            msg = f"Contextual retrieval LLM call failed: {exc}"
            raise ContextualRetrievalError(msg) from exc

    def _ensure_client(self) -> None:
        """Lazy-load the async Anthropic client."""
        if self._client is not None:
            return
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        except ImportError:
            msg = (
                "ContextualRetrievalEnricher requires anthropic. "
                "Install with: pip install anthropic"
            )
            raise EnricherNotAvailableError(msg) from None
