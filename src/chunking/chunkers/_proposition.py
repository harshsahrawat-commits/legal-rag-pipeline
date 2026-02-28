"""Strategy 4: Proposition-Based Chunker.

Decomposes definitions sections into atomic, self-contained propositions
using an LLM. Each definition becomes an independently retrievable chunk
with full context.

Uses the LLM provider abstraction (``get_llm_provider``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from src.chunking._exceptions import ChunkerNotAvailableError
from src.chunking._metadata_builder import MetadataBuilder
from src.chunking._models import ChunkStrategy, ChunkType, LegalChunk
from src.chunking.chunkers._base import BaseChunker
from src.parsing._models import SectionLevel
from src.utils._exceptions import LLMCallError, LLMNotAvailableError
from src.utils._llm_client import LLMMessage, get_llm_provider

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings
    from src.chunking._token_counter import TokenCounter
    from src.parsing._models import ParsedDocument, ParsedSection
    from src.utils._llm_client import BaseLLMProvider

_DEFINITION_PROMPT = """\
Extract each individual definition from this legal text as a self-contained proposition.
Each proposition must:
1. Include the defined term
2. Include the full definition
3. Include the Act name and section for context
4. Be understandable without any surrounding text

Legal text from {act_name}, {section_ref}:
{text}

Return each definition as a separate proposition, one per line.
Do NOT number them. Just output one proposition per line."""


class PropositionChunker(BaseChunker):
    """Decompose definitions sections into atomic propositions via LLM.

    Requires ``anthropic``. Raises ``ChunkerNotAvailableError`` if not installed.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        super().__init__(settings, token_counter)
        self._mb = MetadataBuilder()
        self._provider: BaseLLMProvider | None = None

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.PROPOSITION

    def can_chunk(self, doc: ParsedDocument) -> bool:
        """True if document has definition sections."""
        return _has_definitions(doc.sections)

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        self._ensure_provider()

        source = self._mb.build_source_info(doc)
        ingestion = self._mb.build_ingestion_metadata(doc, ChunkStrategy.PROPOSITION)

        chunks: list[LegalChunk] = []
        def_sections = _find_definition_sections(doc.sections)

        for section in def_sections:
            act_name = doc.act_name or doc.title or "Unknown Act"
            section_ref = f"Section {section.number}" if section.number else "Definitions"

            # Collect text from section + definition children
            text = _collect_definition_text(section)
            if not text.strip():
                continue

            propositions = self._decompose(act_name, section_ref, text)

            for prop in propositions:
                prop = prop.strip()
                if not prop:
                    continue
                content = self._mb.build_content_metadata(prop)
                chunks.append(
                    LegalChunk(
                        id=uuid4(),
                        document_id=doc.document_id,
                        text=prop,
                        document_type=doc.document_type,
                        chunk_type=ChunkType.DEFINITION,
                        chunk_index=0,
                        token_count=self._tc.count(prop),
                        source=source,
                        statute=self._mb.build_statute_metadata(
                            doc,
                            section_number=section.number,
                        ),
                        content=content,
                        ingestion=ingestion,
                    )
                )

        return chunks

    def _ensure_provider(self) -> None:
        """Lazy-load the LLM provider."""
        if self._provider is not None:
            return
        try:
            self._provider = get_llm_provider("proposition")
        except LLMNotAvailableError as exc:
            msg = f"PropositionChunker requires an LLM provider: {exc}"
            raise ChunkerNotAvailableError(msg) from None

    def _decompose(self, act_name: str, section_ref: str, text: str) -> list[str]:
        """Call LLM to decompose definitions into atomic propositions."""
        prompt = _DEFINITION_PROMPT.format(
            act_name=act_name,
            section_ref=section_ref,
            text=text,
        )

        try:
            response = self._provider.complete(
                [LLMMessage(role="user", content=prompt)],
                max_tokens=self._settings.proposition_max_tokens_response,
            )
        except LLMCallError:
            raise

        return [line.strip() for line in response.text.strip().split("\n") if line.strip()]


def _has_definitions(sections: list[ParsedSection]) -> bool:
    """Check if any section in the tree is a definitions section."""
    for s in sections:
        if s.level == SectionLevel.DEFINITION:
            return True
        if s.children and all(c.level == SectionLevel.DEFINITION for c in s.children):
            return True
        if _has_definitions(s.children):
            return True
    return False


def _find_definition_sections(sections: list[ParsedSection]) -> list[ParsedSection]:
    """Find all sections that contain definitions.

    When a parent section has all-definition children, return the parent
    (which includes children text). Don't also return the individual children.
    """
    result: list[ParsedSection] = []
    for s in sections:
        if s.children and all(c.level == SectionLevel.DEFINITION for c in s.children):
            # Parent section contains definitions — include it, skip children
            result.append(s)
        elif s.level == SectionLevel.DEFINITION:
            result.append(s)
        else:
            # Recurse only into non-matched sections
            result.extend(_find_definition_sections(s.children))
    return result


def _collect_definition_text(section: ParsedSection) -> str:
    """Collect all text from a definition section including children."""
    parts: list[str] = []
    if section.text.strip():
        parts.append(section.text.strip())
    for child in section.children:
        if child.text.strip():
            label = child.number or ""
            if label:
                parts.append(f"{label} {child.text.strip()}")
            else:
                parts.append(child.text.strip())
    return "\n".join(parts)
