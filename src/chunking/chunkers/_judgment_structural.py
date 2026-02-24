"""Strategy 2: Judgment Structural Chunker.

Chunks judgments by their structural sections: Header, Facts, Issues,
Reasoning, Holding, Order, Dissent, Obiter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.acquisition._models import DocumentType
from src.chunking._metadata_builder import MetadataBuilder
from src.chunking._models import ChunkStrategy, ChunkType, LegalChunk, ParentDocumentInfo
from src.chunking.chunkers._base import BaseChunker
from src.parsing._models import SectionLevel

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings
    from src.chunking._token_counter import TokenCounter
    from src.parsing._models import ParsedDocument, ParsedSection

_JUDGMENT_LEVELS = frozenset(
    {
        SectionLevel.FACTS,
        SectionLevel.ISSUES,
        SectionLevel.REASONING,
        SectionLevel.HOLDING,
        SectionLevel.ORDER,
        SectionLevel.DISSENT,
        SectionLevel.OBITER,
    }
)

# Holding should never be split
_NEVER_SPLIT = frozenset({SectionLevel.HOLDING})


class JudgmentStructuralChunker(BaseChunker):
    """Chunk judgments by structural headings.

    The header becomes a compact chunk. Its ID is set as
    ``judgment_header_chunk_id`` on every other chunk from the same document.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        super().__init__(settings, token_counter)
        self._mb = MetadataBuilder()

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.JUDGMENT_STRUCTURAL

    def can_chunk(self, doc: ParsedDocument) -> bool:
        if doc.document_type != DocumentType.JUDGMENT:
            return False
        judgment_sections = sum(1 for s in doc.sections if s.level in _JUDGMENT_LEVELS)
        return judgment_sections >= self._settings.min_section_count_judgment

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        source = self._mb.build_source_info(doc)
        ingestion = self._mb.build_ingestion_metadata(doc, ChunkStrategy.JUDGMENT_STRUCTURAL)
        judgment_meta = self._mb.build_judgment_metadata(doc)

        chunks: list[LegalChunk] = []
        header_chunk_id: UUID | None = None

        for section in doc.sections:
            if section.level == SectionLevel.HEADER:
                header_chunk = self._make_header_chunk(
                    section,
                    doc,
                    source,
                    ingestion,
                    judgment_meta,
                )
                header_chunk_id = header_chunk.id
                chunks.append(header_chunk)

            elif section.level in _NEVER_SPLIT:
                # Holding: always its own chunk, never split
                text = self._build_section_text(section)
                chunks.append(
                    self._make_chunk(
                        text,
                        section.level,
                        doc,
                        source,
                        ingestion,
                        judgment_meta,
                    )
                )

            elif section.level in _JUDGMENT_LEVELS:
                text = self._build_section_text(section)
                tokens = self._tc.count(text)
                if tokens <= self._settings.max_tokens:
                    chunks.append(
                        self._make_chunk(
                            text,
                            section.level,
                            doc,
                            source,
                            ingestion,
                            judgment_meta,
                        )
                    )
                else:
                    sub_chunks = self._split_section(
                        text,
                        section.level,
                        doc,
                        source,
                        ingestion,
                        judgment_meta,
                    )
                    chunks.extend(sub_chunks)

            elif section.level == SectionLevel.PARAGRAPH:
                # Paragraph fallback — collect and group
                text = self._build_section_text(section)
                if text.strip():
                    chunks.append(
                        self._make_chunk(
                            text,
                            section.level,
                            doc,
                            source,
                            ingestion,
                            judgment_meta,
                        )
                    )
            else:
                # Any other section type
                text = self._build_section_text(section)
                if text.strip():
                    chunks.append(
                        self._make_chunk(
                            text,
                            section.level,
                            doc,
                            source,
                            ingestion,
                            judgment_meta,
                        )
                    )

        # Set judgment_header_chunk_id on all non-header chunks
        if header_chunk_id is not None:
            for c in chunks:
                if c.id != header_chunk_id:
                    c.parent_info = ParentDocumentInfo(
                        parent_chunk_id=c.parent_info.parent_chunk_id,
                        sibling_chunk_ids=c.parent_info.sibling_chunk_ids,
                        judgment_header_chunk_id=header_chunk_id,
                    )

        return chunks

    def _make_header_chunk(
        self,
        section: ParsedSection,
        doc: ParsedDocument,
        source,
        ingestion,
        judgment_meta,
    ) -> LegalChunk:
        """Create the compact header/metadata chunk."""
        text = self._build_section_text(section)
        return LegalChunk(
            id=uuid4(),
            document_id=doc.document_id,
            text=text,
            document_type=doc.document_type,
            chunk_type=ChunkType.STATUTORY_TEXT,
            chunk_index=0,
            token_count=self._tc.count(text),
            source=source,
            judgment=judgment_meta,
            content=self._mb.build_content_metadata(text),
            ingestion=ingestion,
        )

    def _make_chunk(
        self,
        text: str,
        level: SectionLevel,
        doc: ParsedDocument,
        source,
        ingestion,
        judgment_meta,
    ) -> LegalChunk:
        chunk_type = MetadataBuilder.classify_chunk_type(level)
        return LegalChunk(
            id=uuid4(),
            document_id=doc.document_id,
            text=text,
            document_type=doc.document_type,
            chunk_type=chunk_type,
            chunk_index=0,
            token_count=self._tc.count(text),
            source=source,
            judgment=judgment_meta,
            content=self._mb.build_content_metadata(text),
            ingestion=ingestion,
        )

    def _split_section(
        self,
        text: str,
        level: SectionLevel,
        doc: ParsedDocument,
        source,
        ingestion,
        judgment_meta,
    ) -> list[LegalChunk]:
        """Split an oversized section at paragraph boundaries (``\\n\\n``)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        chunks: list[LegalChunk] = []
        parent_id: UUID | None = None
        buffer = ""

        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if self._tc.count(candidate) <= self._settings.max_tokens:
                buffer = candidate
            else:
                # Emit current buffer
                if buffer:
                    chunk = self._make_chunk(buffer, level, doc, source, ingestion, judgment_meta)
                    if parent_id is None:
                        parent_id = chunk.id
                    elif chunk.id != parent_id:
                        chunk.parent_info = ParentDocumentInfo(parent_chunk_id=parent_id)
                    chunks.append(chunk)
                buffer = para

        # Emit remaining buffer
        if buffer:
            chunk = self._make_chunk(buffer, level, doc, source, ingestion, judgment_meta)
            if parent_id is None:
                parent_id = chunk.id
            elif chunk.id != parent_id:
                chunk.parent_info = ParentDocumentInfo(parent_chunk_id=parent_id)
            chunks.append(chunk)

        # If a single paragraph exceeds max_tokens, force-split at token boundary
        final: list[LegalChunk] = []
        for c in chunks:
            if c.token_count > self._settings.max_tokens:
                remaining = c.text
                while remaining.strip():
                    head, tail = self._tc.split_at_token_boundary(
                        remaining,
                        self._settings.max_tokens,
                    )
                    head = head.strip()
                    if not head:
                        break
                    sub = self._make_chunk(head, level, doc, source, ingestion, judgment_meta)
                    if parent_id and sub.id != parent_id:
                        sub.parent_info = ParentDocumentInfo(parent_chunk_id=parent_id)
                    final.append(sub)
                    remaining = tail
            else:
                final.append(c)

        return final

    @staticmethod
    def _build_section_text(section: ParsedSection) -> str:
        """Build text from a section, including title and children."""
        parts: list[str] = []
        if section.title:
            parts.append(section.title)
        if section.text.strip():
            parts.append(section.text.strip())
        for child in section.children:
            child_text = JudgmentStructuralChunker._build_section_text(child)
            if child_text:
                parts.append(child_text)
        return "\n\n".join(parts)
