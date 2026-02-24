"""QuIM-RAG question generation enricher.

For each chunk, generates 3-5 practical questions a lawyer might ask that
the chunk could answer. Questions are stored in a ``QuIMDocument`` sidecar
file and the chunk's ``ingestion.quim_questions`` count is updated.

Requires ``anthropic`` (optional dependency).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.enrichment._exceptions import (
    EnricherNotAvailableError,
    LLMRateLimitError,
)
from src.enrichment._models import QuIMDocument, QuIMEntry
from src.enrichment.enrichers._base import BaseEnricher

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.enrichment._models import EnrichmentSettings
    from src.parsing._models import ParsedDocument

_log = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = (
    "<document>\n{full_doc_text}\n</document>\n\n"
    "You generate practical legal questions that lawyers might ask. "
    "Output one question per line. No numbering. No blank lines."
)

_USER_TEMPLATE = (
    "Given this legal text from {act_or_case}, {section_ref}:\n"
    "<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Generate {n} practical questions a lawyer might ask that this text could answer.\n"
    "Focus on how lawyers phrase real queries, not academic questions."
)


class QuIMRagEnricher(BaseEnricher):
    """Generate retrieval-oriented questions per chunk.

    Uses the same prompt caching pattern as ContextualRetrievalEnricher:
    the full document text is placed in the system message with
    ``cache_control``, and only the per-chunk user message varies.

    After calling :meth:`enrich_document`, call :meth:`get_quim_document`
    to retrieve the generated questions for saving to a sidecar file.
    """

    def __init__(self, settings: EnrichmentSettings) -> None:
        super().__init__(settings)
        self._client: Any = None
        self._last_quim_doc: QuIMDocument | None = None

    @property
    def stage_name(self) -> str:
        return "quim_rag"

    def get_quim_document(self) -> QuIMDocument | None:
        """Return the QuIM document from the last ``enrich_document`` call."""
        return self._last_quim_doc

    async def enrich_document(
        self,
        chunks: list[LegalChunk],
        parsed_doc: ParsedDocument,
    ) -> list[LegalChunk]:
        """Generate questions for each chunk and update quim_questions count."""
        self._ensure_client()
        self._last_quim_doc = None

        if not chunks:
            return chunks

        full_text = parsed_doc.raw_text or ""
        system_messages = self._build_system_messages(full_text)
        semaphore = asyncio.Semaphore(self._settings.concurrency)

        entries: list[QuIMEntry] = []

        async def _process_one(chunk: LegalChunk) -> None:
            async with semaphore:
                entry = await self._generate_for_chunk(chunk, system_messages)
                if entry:
                    entries.append(entry)

        tasks = [_process_one(chunk) for chunk in chunks]
        await asyncio.gather(*tasks)

        # Build QuIM document from all entries
        doc_id = chunks[0].document_id
        self._last_quim_doc = QuIMDocument(
            document_id=doc_id,
            entries=entries,
            model=self._settings.model,
        )

        return chunks

    async def _generate_for_chunk(
        self,
        chunk: LegalChunk,
        system_messages: list[dict[str, Any]],
    ) -> QuIMEntry | None:
        """Generate questions for a single chunk. Errors are isolated."""
        try:
            act_or_case = self._resolve_act_or_case(chunk)
            section_ref = self._resolve_section_ref(chunk)

            user_message = _USER_TEMPLATE.format(
                act_or_case=act_or_case,
                section_ref=section_ref,
                chunk_text=chunk.text,
                n=self._settings.quim_questions_per_chunk,
            )

            response = await self._client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens_response,
                system=system_messages,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text
            questions = _parse_questions(raw_text)

            if questions:
                chunk.ingestion.quim_questions = len(questions)
                return QuIMEntry(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    questions=questions,
                    generated_at=datetime.now(UTC),
                    model=self._settings.model,
                )

        except Exception as exc:
            exc_type = type(exc).__name__
            if "RateLimitError" in exc_type:
                msg = f"LLM rate limit exceeded: {exc}"
                raise LLMRateLimitError(msg) from exc
            _log.exception(
                "quim_generation_failed",
                extra={"chunk_id": str(chunk.id)},
            )

        return None

    def _build_system_messages(self, document_text: str) -> list[dict[str, Any]]:
        """Build the system message list with prompt caching."""
        return [
            {
                "type": "text",
                "text": _SYSTEM_TEMPLATE.format(full_doc_text=document_text),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _resolve_act_or_case(chunk: LegalChunk) -> str:
        """Get the Act name or case citation for the prompt."""
        if chunk.statute:
            return chunk.statute.act_name
        if chunk.judgment:
            return chunk.judgment.case_citation
        return "Unknown Document"

    @staticmethod
    def _resolve_section_ref(chunk: LegalChunk) -> str:
        """Get the section or chunk type reference for the prompt."""
        if chunk.statute and chunk.statute.section_number:
            return f"Section {chunk.statute.section_number}"
        if chunk.judgment:
            return chunk.chunk_type.value
        return "Unknown Section"

    def _ensure_client(self) -> None:
        """Lazy-load the async Anthropic client."""
        if self._client is not None:
            return
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        except ImportError:
            msg = "QuIMRagEnricher requires anthropic. Install with: pip install anthropic"
            raise EnricherNotAvailableError(msg) from None


def _parse_questions(raw_text: str) -> list[str]:
    """Parse LLM output into a clean list of questions.

    Filters out blank lines and very short non-question lines.
    """
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    return [line for line in lines if "?" in line or len(line) > 20]
