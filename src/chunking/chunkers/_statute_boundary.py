"""Strategy 1: Statute Boundary Chunker.

Each section becomes one chunk. If a section exceeds ``max_tokens``, it is
split at sub-section boundaries, keeping provisos and explanations with their
parent sub-section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.acquisition._models import DocumentType
from src.chunking._metadata_builder import MetadataBuilder
from src.chunking._models import ChunkStrategy, ChunkType, LegalChunk
from src.chunking.chunkers._base import BaseChunker
from src.parsing._models import SectionLevel

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings
    from src.chunking._token_counter import TokenCounter
    from src.parsing._models import ParsedDocument, ParsedSection

# Levels that must stay attached to their parent
_ATTACHED_LEVELS = frozenset(
    {
        SectionLevel.PROVISO,
        SectionLevel.EXPLANATION,
    }
)

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


class StatuteBoundaryChunker(BaseChunker):
    """Chunk statutes by section boundaries.

    Each SECTION-level node becomes a chunk, with its children (sub-sections,
    provisos, explanations) included in the same chunk text.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        super().__init__(settings, token_counter)
        self._mb = MetadataBuilder()

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.STRUCTURE_BOUNDARY

    def can_chunk(self, doc: ParsedDocument) -> bool:
        if doc.document_type != DocumentType.STATUTE:
            return False
        section_count = _count_sections(doc.sections)
        return section_count >= self._settings.min_section_count_statute

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        source = self._mb.build_source_info(doc)
        ingestion = self._mb.build_ingestion_metadata(doc, ChunkStrategy.STRUCTURE_BOUNDARY)

        # Track current chapter/part for metadata
        current_chapter: str | None = None
        current_part: str | None = None

        chunks: list[LegalChunk] = []
        self._walk_sections(
            doc.sections,
            doc,
            source,
            ingestion,
            current_chapter,
            current_part,
            chunks,
        )
        return chunks

    def _walk_sections(
        self,
        sections: list[ParsedSection],
        doc: ParsedDocument,
        source,
        ingestion,
        current_chapter: str | None,
        current_part: str | None,
        chunks: list[LegalChunk],
    ) -> None:
        for section in sections:
            if section.level == SectionLevel.PART:
                current_part = section.number or section.title
                # Part heading → own chunk if it has text
                if section.text.strip():
                    chunks.append(
                        self._make_chunk(
                            section,
                            doc,
                            source,
                            ingestion,
                            chapter=current_chapter,
                            part=current_part,
                        )
                    )
                self._walk_sections(
                    section.children,
                    doc,
                    source,
                    ingestion,
                    current_chapter,
                    current_part,
                    chunks,
                )

            elif section.level == SectionLevel.CHAPTER:
                current_chapter = section.number or section.title
                if section.text.strip():
                    chunks.append(
                        self._make_chunk(
                            section,
                            doc,
                            source,
                            ingestion,
                            chapter=current_chapter,
                            part=current_part,
                        )
                    )
                self._walk_sections(
                    section.children,
                    doc,
                    source,
                    ingestion,
                    current_chapter,
                    current_part,
                    chunks,
                )

            elif section.level == SectionLevel.PREAMBLE or section.level == SectionLevel.SCHEDULE:
                chunks.append(
                    self._make_chunk(
                        section,
                        doc,
                        source,
                        ingestion,
                        chapter=current_chapter,
                        part=current_part,
                    )
                )

            elif section.level == SectionLevel.SECTION:
                self._chunk_section(
                    section,
                    doc,
                    source,
                    ingestion,
                    current_chapter,
                    current_part,
                    chunks,
                )

            elif section.level == SectionLevel.DEFINITION:
                # Standalone definition outside a section
                chunks.append(
                    self._make_chunk(
                        section,
                        doc,
                        source,
                        ingestion,
                        chapter=current_chapter,
                        part=current_part,
                    )
                )

            else:
                # Paragraph or other — recurse if children, else make chunk
                if section.children:
                    self._walk_sections(
                        section.children,
                        doc,
                        source,
                        ingestion,
                        current_chapter,
                        current_part,
                        chunks,
                    )
                elif section.text.strip():
                    chunks.append(
                        self._make_chunk(
                            section,
                            doc,
                            source,
                            ingestion,
                            chapter=current_chapter,
                            part=current_part,
                        )
                    )

    def _chunk_section(
        self,
        section: ParsedSection,
        doc: ParsedDocument,
        source,
        ingestion,
        chapter: str | None,
        part: str | None,
        chunks: list[LegalChunk],
    ) -> None:
        """Chunk a SECTION-level node.

        If the full section text (including children) fits within max_tokens,
        emit one chunk. Otherwise, split at sub-section boundaries.
        """
        act_name = doc.act_name or doc.title or "Unknown Act"
        header = f"{act_name}, Section {section.number or '?'}"
        if section.title:
            header += f" — {section.title}"

        full_text = _flatten_section_text(section, header)
        total_tokens = self._tc.count(full_text)

        if total_tokens <= self._settings.max_tokens:
            # Fits in one chunk
            chunk_type = self._classify_section(section)
            content = self._mb.build_content_metadata(full_text)
            statute_meta = self._mb.build_statute_metadata(
                doc,
                section_number=section.number,
                chapter=chapter,
                part=part,
            )
            chunks.append(
                LegalChunk(
                    id=uuid4(),
                    document_id=doc.document_id,
                    text=full_text,
                    document_type=doc.document_type,
                    chunk_type=chunk_type,
                    chunk_index=0,
                    token_count=total_tokens,
                    source=source,
                    statute=statute_meta,
                    content=content,
                    ingestion=ingestion,
                )
            )
        else:
            # Split at sub-section boundaries
            self._split_section(
                section,
                header,
                doc,
                source,
                ingestion,
                chapter,
                part,
                chunks,
            )

    def _split_section(
        self,
        section: ParsedSection,
        header: str,
        doc: ParsedDocument,
        source,
        ingestion,
        chapter: str | None,
        part: str | None,
        chunks: list[LegalChunk],
    ) -> None:
        """Split an oversized section at sub-section boundaries.

        Provisos and explanations are kept with their immediate parent.
        The first sub-chunk gets a new UUID; subsequent sub-chunks get
        ``parent_chunk_id`` pointing to the first.
        """
        parent_chunk_id: UUID | None = None

        # Section's own text (without children) as first potential chunk
        section_text = f"{header}\n\n{section.text.strip()}" if section.text.strip() else header

        # Group children: attach provisos/explanations to the preceding non-attached child
        groups = _group_children_with_attachments(section.children)

        # If section text alone is non-trivial, emit it
        if section.text.strip():
            first_id = uuid4()
            parent_chunk_id = first_id
            tokens = self._tc.count(section_text)
            statute_meta = self._mb.build_statute_metadata(
                doc,
                section_number=section.number,
                chapter=chapter,
                part=part,
            )
            chunks.append(
                LegalChunk(
                    id=first_id,
                    document_id=doc.document_id,
                    text=section_text,
                    document_type=doc.document_type,
                    chunk_type=self._classify_section(section),
                    chunk_index=0,
                    token_count=tokens,
                    source=source,
                    statute=statute_meta,
                    content=self._mb.build_content_metadata(section_text),
                    ingestion=ingestion,
                )
            )

        # Emit each group as a sub-chunk
        for group in groups:
            group_text_parts = []
            for child in group:
                child_flat = _flatten_section_text(child, prefix=None)
                group_text_parts.append(child_flat)

            group_text = f"{header}\n\n" + "\n\n".join(group_text_parts)
            tokens = self._tc.count(group_text)

            # If group still oversized, do token-boundary split
            if tokens > self._settings.max_tokens:
                remaining = group_text
                while remaining.strip():
                    head, tail = self._tc.split_at_token_boundary(
                        remaining,
                        self._settings.max_tokens,
                    )
                    head = head.strip()
                    if not head:
                        break
                    cid = uuid4()
                    if parent_chunk_id is None:
                        parent_chunk_id = cid
                    statute_meta = self._mb.build_statute_metadata(
                        doc,
                        section_number=section.number,
                        chapter=chapter,
                        part=part,
                    )
                    from src.chunking._models import ParentDocumentInfo

                    chunks.append(
                        LegalChunk(
                            id=cid,
                            document_id=doc.document_id,
                            text=head,
                            document_type=doc.document_type,
                            chunk_type=self._classify_section(section),
                            chunk_index=0,
                            token_count=self._tc.count(head),
                            source=source,
                            statute=statute_meta,
                            content=self._mb.build_content_metadata(head),
                            ingestion=ingestion,
                            parent_info=ParentDocumentInfo(
                                parent_chunk_id=parent_chunk_id if cid != parent_chunk_id else None,
                            ),
                        )
                    )
                    remaining = tail
            else:
                cid = uuid4()
                if parent_chunk_id is None:
                    parent_chunk_id = cid
                statute_meta = self._mb.build_statute_metadata(
                    doc,
                    section_number=section.number,
                    chapter=chapter,
                    part=part,
                )
                from src.chunking._models import ParentDocumentInfo

                chunks.append(
                    LegalChunk(
                        id=cid,
                        document_id=doc.document_id,
                        text=group_text,
                        document_type=doc.document_type,
                        chunk_type=self._classify_section(section),
                        chunk_index=0,
                        token_count=tokens,
                        source=source,
                        statute=statute_meta,
                        content=self._mb.build_content_metadata(group_text),
                        ingestion=ingestion,
                        parent_info=ParentDocumentInfo(
                            parent_chunk_id=parent_chunk_id if cid != parent_chunk_id else None,
                        ),
                    )
                )

    def _classify_section(self, section: ParsedSection) -> ChunkType:
        """Determine the ChunkType for a section."""
        # If all children are definitions, this is a definitions section
        if section.children and all(c.level == SectionLevel.DEFINITION for c in section.children):
            return ChunkType.DEFINITION
        return MetadataBuilder.classify_chunk_type(section.level)

    def _make_chunk(
        self,
        section: ParsedSection,
        doc: ParsedDocument,
        source,
        ingestion,
        *,
        chapter: str | None = None,
        part: str | None = None,
    ) -> LegalChunk:
        """Create a single LegalChunk from a section node."""
        text = section.text.strip()
        if section.title:
            text = f"{section.title}\n\n{text}" if text else section.title

        # Include children text
        for child in section.children:
            child_text = _flatten_section_text(child, prefix=None)
            if child_text:
                text = f"{text}\n\n{child_text}"

        token_count = self._tc.count(text)
        chunk_type = MetadataBuilder.classify_chunk_type(section.level)
        content = self._mb.build_content_metadata(text)

        schedule_name = None
        if section.level == SectionLevel.SCHEDULE:
            schedule_name = section.title or section.number

        statute_meta = self._mb.build_statute_metadata(
            doc,
            section_number=section.number,
            chapter=chapter,
            part=part,
            schedule=schedule_name,
        )

        return LegalChunk(
            id=uuid4(),
            document_id=doc.document_id,
            text=text,
            document_type=doc.document_type,
            chunk_type=chunk_type,
            chunk_index=0,
            token_count=token_count,
            source=source,
            statute=statute_meta,
            content=content,
            ingestion=ingestion,
        )


def _flatten_section_text(section: ParsedSection, prefix: str | None = None) -> str:
    """Recursively flatten a section tree into a single text string."""
    parts: list[str] = []
    if prefix:
        parts.append(prefix)

    section_text = section.text.strip()
    label = ""
    if section.number:
        label = section.number
    if section.title:
        label = f"{label} {section.title}".strip() if label else section.title

    if label and section_text:
        parts.append(f"{label}: {section_text}")
    elif label:
        parts.append(label)
    elif section_text:
        parts.append(section_text)

    for child in section.children:
        child_flat = _flatten_section_text(child, prefix=None)
        if child_flat:
            parts.append(child_flat)

    return "\n\n".join(parts)


def _count_sections(sections: list[ParsedSection]) -> int:
    """Count SECTION-level nodes in the tree."""
    count = 0
    for s in sections:
        if s.level == SectionLevel.SECTION:
            count += 1
        count += _count_sections(s.children)
    return count


def _group_children_with_attachments(
    children: list[ParsedSection],
) -> list[list[ParsedSection]]:
    """Group children so provisos/explanations attach to their preceding sibling.

    Returns a list of groups, where each group is a list of sections that
    should be kept together in the same chunk.
    """
    if not children:
        return []

    groups: list[list[ParsedSection]] = []
    current_group: list[ParsedSection] = []

    for child in children:
        if child.level in _ATTACHED_LEVELS:
            # Attach to current group
            if current_group:
                current_group.append(child)
            else:
                # Orphaned proviso/explanation — start new group
                current_group = [child]
        else:
            # Non-attached child: finalize previous group, start new
            if current_group:
                groups.append(current_group)
            current_group = [child]

    if current_group:
        groups.append(current_group)

    return groups
