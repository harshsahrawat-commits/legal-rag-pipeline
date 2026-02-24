"""Strategy 6: Page-Level Chunker.

Treats each page of the document as a single chunk. The simplest strategy.
Used for degraded scans (OCR < 80%) and schedule documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from src.chunking._metadata_builder import MetadataBuilder
from src.chunking._models import ChunkStrategy, ChunkType, LegalChunk
from src.chunking.chunkers._base import BaseChunker

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings
    from src.chunking._token_counter import TokenCounter
    from src.parsing._models import ParsedDocument


class PageLevelChunker(BaseChunker):
    """Split document by page boundaries (form-feed ``\\f``).

    Each page becomes one chunk. If the document has no page separators,
    the entire raw text becomes a single chunk.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        super().__init__(settings, token_counter)
        self._mb = MetadataBuilder()

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.PAGE_LEVEL

    def can_chunk(self, doc: ParsedDocument) -> bool:
        """PageLevelChunker can handle any document."""
        return True

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        separator = self._settings.page_separator
        pages = doc.raw_text.split(separator) if separator in doc.raw_text else [doc.raw_text]

        is_degraded = (
            doc.ocr_confidence is not None
            and doc.ocr_confidence < self._settings.ocr_confidence_threshold
        )

        source_info = self._mb.build_source_info(doc)
        ingestion = self._mb.build_ingestion_metadata(
            doc,
            ChunkStrategy.PAGE_LEVEL,
            requires_manual_review=is_degraded,
        )

        from src.acquisition._models import DocumentType

        statute_meta = None
        judgment_meta = None
        if doc.document_type == DocumentType.STATUTE:
            statute_meta = self._mb.build_statute_metadata(doc)
        elif doc.document_type == DocumentType.JUDGMENT:
            judgment_meta = self._mb.build_judgment_metadata(doc)

        chunks: list[LegalChunk] = []
        for page_idx, page_text in enumerate(pages):
            page_text = page_text.strip()
            if not page_text:
                continue

            token_count = self._tc.count(page_text)

            # Oversized pages: split at token boundary
            if token_count > self._settings.max_tokens:
                sub_chunks = self._split_oversized_page(
                    page_text,
                    page_idx,
                    doc,
                    source_info,
                    ingestion,
                    statute_meta,
                    judgment_meta,
                )
                chunks.extend(sub_chunks)
                continue

            content_meta = self._mb.build_content_metadata(page_text)

            chunks.append(
                LegalChunk(
                    id=uuid4(),
                    document_id=doc.document_id,
                    text=page_text,
                    document_type=doc.document_type,
                    chunk_type=ChunkType.STATUTORY_TEXT,
                    chunk_index=0,  # assigned by pipeline
                    token_count=token_count,
                    source=source_info,
                    statute=statute_meta,
                    judgment=judgment_meta,
                    content=content_meta,
                    ingestion=ingestion,
                )
            )

        return chunks

    def _split_oversized_page(
        self,
        page_text,
        page_idx,
        doc,
        source_info,
        ingestion,
        statute_meta,
        judgment_meta,
    ) -> list[LegalChunk]:
        """Split an oversized page into sub-chunks at token boundaries."""
        sub_chunks: list[LegalChunk] = []
        remaining = page_text
        while remaining:
            head, tail = self._tc.split_at_token_boundary(remaining, self._settings.max_tokens)
            head = head.strip()
            if not head:
                break
            content_meta = self._mb.build_content_metadata(head)
            sub_chunks.append(
                LegalChunk(
                    id=uuid4(),
                    document_id=doc.document_id,
                    text=head,
                    document_type=doc.document_type,
                    chunk_type=ChunkType.STATUTORY_TEXT,
                    chunk_index=0,
                    token_count=self._tc.count(head),
                    source=source_info,
                    statute=statute_meta,
                    judgment=judgment_meta,
                    content=content_meta,
                    ingestion=ingestion,
                )
            )
            remaining = tail.strip()
        return sub_chunks
