"""Strategy 5: Semantic Max-Min Chunker.

Sentence-level splitting with embedding-based boundary detection.
Requires ``sentence-transformers`` and optionally ``spacy`` (optional dependencies).
"""

from __future__ import annotations

import re
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


class SemanticMaxMinChunker(BaseChunker):
    """Semantic chunking using the Max-Min percentile method.

    Requires ``sentence-transformers``. Optionally uses ``spacy`` for
    sentence boundary detection; falls back to regex if unavailable.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        super().__init__(settings, token_counter)
        self._mb = MetadataBuilder()
        self._model = None
        self._nlp = None

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.SEMANTIC_MAXMIN

    def can_chunk(self, doc: ParsedDocument) -> bool:
        """True for any document without structure (fallback)."""
        return len(doc.sections) == 0

    def chunk(self, doc: ParsedDocument) -> list[LegalChunk]:
        self._ensure_model()

        text = doc.raw_text
        if not text.strip():
            return []

        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Step 2: Embed each sentence
        embeddings = self._model.encode(sentences, normalize_embeddings=True)

        # Step 3: Compute consecutive similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        # Step 4: Calculate split threshold (percentile method)
        if similarities:
            threshold = float(np.percentile(similarities, self._settings.semantic_percentile * 100))
        else:
            threshold = 0.0

        # Step 5: Split at low-similarity boundaries
        groups: list[list[str]] = []
        current_group: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < threshold:
                groups.append(current_group)
                current_group = [sentences[i + 1]]
            else:
                current_group.append(sentences[i + 1])

        if current_group:
            groups.append(current_group)

        # Step 6: Merge tiny groups, split oversized
        chunk_texts = self._merge_and_split(groups)

        source = self._mb.build_source_info(doc)
        ingestion = self._mb.build_ingestion_metadata(doc, ChunkStrategy.SEMANTIC_MAXMIN)

        from src.acquisition._models import DocumentType

        statute_meta = None
        judgment_meta = None
        if doc.document_type == DocumentType.STATUTE:
            statute_meta = self._mb.build_statute_metadata(doc)
        elif doc.document_type == DocumentType.JUDGMENT:
            judgment_meta = self._mb.build_judgment_metadata(doc)

        chunks: list[LegalChunk] = []
        for ct in chunk_texts:
            ct = ct.strip()
            if not ct:
                continue
            chunks.append(
                LegalChunk(
                    id=uuid4(),
                    document_id=doc.document_id,
                    text=ct,
                    document_type=doc.document_type,
                    chunk_type=ChunkType.STATUTORY_TEXT,
                    chunk_index=0,
                    token_count=self._tc.count(ct),
                    source=source,
                    statute=statute_meta,
                    judgment=judgment_meta,
                    content=self._mb.build_content_metadata(ct),
                    ingestion=ingestion,
                )
            )
        return chunks

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._settings.embedding_model)
        except ImportError:
            msg = (
                "SemanticMaxMinChunker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
            raise ChunkerNotAvailableError(msg) from None

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using spacy or regex fallback."""
        try:
            if self._nlp is None:
                import spacy

                self._nlp = spacy.blank("en")
                self._nlp.add_pipe("sentencizer")
            doc = self._nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences
        except ImportError:
            pass

        # Regex fallback
        return _regex_sentence_split(text)

    def _merge_and_split(self, groups: list[list[str]]) -> list[str]:
        """Merge tiny groups with neighbors; split oversized groups."""
        result: list[str] = []

        for group in groups:
            text = " ".join(group)
            tokens = self._tc.count(text)

            if tokens < self._settings.min_chunk_tokens and result:
                # Merge with previous, but check if result becomes oversized
                merged = f"{result[-1]} {text}"
                if self._tc.count(merged) <= self._settings.max_tokens:
                    result[-1] = merged
                else:
                    # Previous chunk is full — start a new one
                    result.append(text)
            elif tokens > self._settings.max_tokens:
                # Split at token boundaries
                remaining = text
                while remaining.strip():
                    head, tail = self._tc.split_at_token_boundary(
                        remaining,
                        self._settings.max_tokens,
                    )
                    if head.strip():
                        result.append(head.strip())
                    remaining = tail
            else:
                result.append(text)

        return result


def _regex_sentence_split(text: str) -> list[str]:
    """Simple regex sentence splitter for legal text."""
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
