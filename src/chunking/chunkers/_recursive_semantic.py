"""Strategy 3: Recursive Semantic Chunker (RSC).

Three-phase approach: recursive structural split → semantic merge/split → oversized split.
Requires ``sentence-transformers`` (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from src.chunking._exceptions import ChunkerNotAvailableError
from src.chunking._metadata_builder import MetadataBuilder
from src.chunking._models import ChunkStrategy, ChunkType, LegalChunk
from src.chunking.chunkers._base import BaseChunker

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings
    from src.chunking._token_counter import TokenCounter
    from src.parsing._models import ParsedDocument

_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". "]


class RecursiveSemanticChunker(BaseChunker):
    """RSC: hybrid structural + semantic chunking for partially-structured docs.

    Requires ``sentence-transformers``. Raises ``ChunkerNotAvailableError``
    at chunk time if not installed.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        super().__init__(settings, token_counter)
        self._mb = MetadataBuilder()
        self._model = None

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.RECURSIVE_SEMANTIC

    def can_chunk(self, doc: ParsedDocument) -> bool:
        """True if document has some sections but not enough for structure-aware chunking."""
        from src.acquisition._models import DocumentType

        if doc.document_type == DocumentType.STATUTE:
            from src.chunking.chunkers._statute_boundary import _count_sections

            count = _count_sections(doc.sections)
            return 0 < count < self._settings.min_section_count_statute
        if doc.document_type == DocumentType.JUDGMENT:
            from src.parsing._models import SectionLevel

            judgment_levels = frozenset(
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
            count = sum(1 for s in doc.sections if s.level in judgment_levels)
            return 0 < count < self._settings.min_section_count_judgment
        # Other types with partial structure
        return len(doc.sections) > 0

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        self._ensure_model()

        text = doc.raw_text
        if not text.strip():
            return []

        # Phase 1: Recursive structural splitting
        initial_chunks = _recursive_split(text, _SEPARATORS, self._settings.max_tokens, self._tc)

        # Phase 2: Semantic merge/split
        merged = self._semantic_merge(initial_chunks)

        # Phase 3: Split oversized
        final_texts = self._split_oversized(merged)

        source = self._mb.build_source_info(doc)
        ingestion = self._mb.build_ingestion_metadata(doc, ChunkStrategy.RECURSIVE_SEMANTIC)

        from src.acquisition._models import DocumentType

        statute_meta = None
        judgment_meta = None
        if doc.document_type == DocumentType.STATUTE:
            statute_meta = self._mb.build_statute_metadata(doc)
        elif doc.document_type == DocumentType.JUDGMENT:
            judgment_meta = self._mb.build_judgment_metadata(doc)

        chunks: list[LegalChunk] = []
        for text_part in final_texts:
            text_part = text_part.strip()
            if not text_part:
                continue
            chunks.append(
                LegalChunk(
                    id=uuid4(),
                    document_id=doc.document_id,
                    text=text_part,
                    document_type=doc.document_type,
                    chunk_type=ChunkType.STATUTORY_TEXT,
                    chunk_index=0,
                    token_count=self._tc.count(text_part),
                    source=source,
                    statute=statute_meta,
                    judgment=judgment_meta,
                    content=self._mb.build_content_metadata(text_part),
                    ingestion=ingestion,
                )
            )
        return chunks

    def _ensure_model(self) -> None:
        """Lazy-load sentence-transformers model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._settings.embedding_model)
        except ImportError:
            msg = (
                "RecursiveSemanticChunker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
            raise ChunkerNotAvailableError(msg) from None

    def _encode(self, text: str) -> np.ndarray:
        """Encode text using the embedding model."""
        return self._model.encode(text, normalize_embeddings=True)

    def _semantic_merge(self, chunks: list[str]) -> list[str]:
        """Phase 2: merge semantically similar adjacent chunks."""
        if len(chunks) <= 1:
            return chunks

        result: list[str] = []
        buffer = chunks[0]

        for i in range(1, len(chunks)):
            emb_buf = self._encode(buffer)
            emb_next = self._encode(chunks[i])
            similarity = float(np.dot(emb_buf, emb_next))

            combined = f"{buffer}\n{chunks[i]}"
            combined_tokens = self._tc.count(combined)

            if (
                similarity > self._settings.similarity_threshold
                and combined_tokens <= self._settings.max_tokens
            ):
                buffer = combined
            else:
                result.append(buffer)
                buffer = chunks[i]

        result.append(buffer)
        return result

    def _split_oversized(self, chunks: list[str]) -> list[str]:
        """Phase 3: split chunks that still exceed max_tokens."""
        result: list[str] = []
        for chunk in chunks:
            if self._tc.count(chunk) > self._settings.max_tokens:
                sub = _split_at_lowest_similarity(
                    chunk,
                    self._encode,
                    self._tc,
                    self._settings.max_tokens,
                )
                result.extend(sub)
            else:
                result.append(chunk)
        return result


def _recursive_split(
    text: str,
    separators: list[str],
    max_tokens: int,
    tc,
) -> list[str]:
    """Recursively split text using separators in priority order."""
    if tc.count(text) <= max_tokens:
        return [text]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: list[str] = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if tc.count(part) <= max_tokens:
                    chunks.append(part)
                else:
                    # Recurse with remaining separators
                    remaining_seps = separators[separators.index(sep) + 1 :]
                    if remaining_seps:
                        chunks.extend(_recursive_split(part, remaining_seps, max_tokens, tc))
                    else:
                        # Force split at token boundary
                        remaining = part
                        while remaining.strip():
                            head, tail = tc.split_at_token_boundary(remaining, max_tokens)
                            if head.strip():
                                chunks.append(head.strip())
                            remaining = tail
            return chunks

    # No separator worked — force split
    chunks = []
    remaining = text
    while remaining.strip():
        head, tail = tc.split_at_token_boundary(remaining, max_tokens)
        if head.strip():
            chunks.append(head.strip())
        remaining = tail
    return chunks


def _split_at_lowest_similarity(
    text: str,
    encode_fn,
    tc,
    max_tokens: int,
) -> list[str]:
    """Split text at the point of lowest semantic similarity."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) <= 1:
        # Cannot split by sentences, force token boundary
        result = []
        remaining = text
        while remaining.strip():
            head, tail = tc.split_at_token_boundary(remaining, max_tokens)
            if head.strip():
                result.append(head.strip())
            remaining = tail
        return result

    # Find lowest similarity point
    min_sim = float("inf")
    min_idx = len(sentences) // 2

    for i in range(len(sentences) - 1):
        emb1 = encode_fn(sentences[i])
        emb2 = encode_fn(sentences[i + 1])
        sim = float(np.dot(emb1, emb2))
        if sim < min_sim:
            min_sim = sim
            min_idx = i + 1

    first_half = ". ".join(sentences[:min_idx]) + "."
    second_half = ". ".join(sentences[min_idx:])

    result = []
    for part in [first_half, second_half]:
        if tc.count(part) > max_tokens:
            result.extend(_split_at_lowest_similarity(part, encode_fn, tc, max_tokens))
        else:
            result.append(part)
    return result
