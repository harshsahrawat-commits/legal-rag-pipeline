"""BM25 sparse encoder for hybrid search in Qdrant.

Builds a per-document vocabulary and produces BM25-weighted sparse vectors.
Uses simple word-level tokenization suitable for legal text.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embedding._models import SparseVector


class BM25SparseEncoder:
    """Encodes text into BM25-weighted sparse vectors for hybrid search."""

    K1 = 1.2
    B = 0.75

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._built = False

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def build_vocabulary(self, texts: list[str]) -> None:
        """Build vocabulary and compute IDF from a collection of texts.

        Args:
            texts: List of document texts (one per chunk, typically contextualized_text).
        """
        if not texts:
            self._built = True
            return

        n_docs = len(texts)
        doc_freq: dict[str, int] = {}
        total_len = 0

        for text in texts:
            tokens = self._tokenize(text)
            total_len += len(tokens)
            seen = set(tokens)
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._avg_dl = total_len / n_docs if n_docs > 0 else 0.0

        # Build vocab: term -> index
        self._vocab = {term: idx for idx, term in enumerate(sorted(doc_freq))}

        # Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        self._built = True

    def encode(self, text: str) -> SparseVector:
        """Encode a single text into a BM25-weighted sparse vector.

        Args:
            text: Text to encode (typically contextualized_text).

        Returns:
            SparseVector with token indices and BM25 weights.

        Raises:
            RuntimeError: If build_vocabulary() has not been called.
        """
        from src.embedding._models import SparseVector

        if not self._built:
            msg = "build_vocabulary() must be called before encode()"
            raise RuntimeError(msg)

        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Term frequency
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        doc_len = len(tokens)
        indices = []
        values = []

        for term, freq in sorted(tf.items()):
            if term not in self._vocab:
                continue

            idf = self._idf.get(term, 0.0)
            # BM25 TF component
            numerator = freq * (self.K1 + 1)
            denominator = freq + self.K1 * (1 - self.B + self.B * doc_len / max(self._avg_dl, 1.0))
            score = idf * numerator / denominator

            if score > 0:
                indices.append(self._vocab[term])
                values.append(score)

        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word-level tokenizer: lowercase, alphanumeric tokens only."""
        return re.findall(r"[a-z0-9]+", text.lower())
