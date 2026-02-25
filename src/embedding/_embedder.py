"""Late Chunking embedder using transformers directly.

Uses full-document token-level embeddings (last_hidden_state), then slices
per chunk and mean-pools to produce chunk embeddings that are contextually
aware of the entire document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.embedding._exceptions import (
    EmbedderNotAvailableError,
    ModelLoadError,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.embedding._models import EmbeddingSettings

_log = get_logger(__name__)


class LateChunkingEmbedder:
    """Embeds chunks via Late Chunking: full doc -> token embeddings -> slice per chunk."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load the transformer model and tokenizer. Lazy-imports torch/transformers."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            msg = "torch and transformers are required for embedding. Install with: pip install torch transformers"
            raise EmbedderNotAvailableError(msg) from exc

        model_name = self._settings.model_name_or_path
        _log.info("loading_model", model=model_name, device=self._settings.device)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.to(torch.device(self._settings.device))
            self._model.eval()
        except Exception as exc:
            msg = f"Failed to load model {model_name}: {exc}"
            raise ModelLoadError(msg) from exc

        _log.info("model_loaded", model=model_name)

    def embed_document_late_chunking(
        self,
        full_text: str,
        chunks: list[LegalChunk],
    ) -> list[np.ndarray]:
        """Embed chunks using Late Chunking against the full document text.

        Args:
            full_text: The complete document text (from ParsedDocument.raw_text).
            chunks: The chunks whose text appears within full_text.

        Returns:
            List of 768-dim embeddings, one per chunk (in same order as chunks).
        """
        self._ensure_model()

        if not chunks:
            return []

        chunk_texts = [c.text for c in chunks]

        # Tokenize full document with offset mapping
        encoding = self._tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=True,
            padding=False,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"][0].numpy()  # shape: [seq_len, 2]
        seq_len = input_ids.shape[1]

        # Handle windowing if document exceeds model max length
        max_len = self._settings.max_length
        if seq_len <= max_len:
            token_embeddings = self._forward_pass(input_ids, attention_mask)
        else:
            token_embeddings = self._windowed_forward(input_ids, attention_mask, max_len)

        # Map chunks to token spans
        spans = self._find_chunk_token_spans(chunk_texts, full_text, offsets)

        # Slice and pool
        embeddings = []
        for start_tok, end_tok in spans:
            if start_tok < end_tok and end_tok <= token_embeddings.shape[0]:
                chunk_emb = self._mean_pool(token_embeddings[start_tok:end_tok])
            else:
                # Fallback: zero vector if span mapping failed
                _log.warning("chunk_span_fallback", start=start_tok, end=end_tok)
                chunk_emb = np.zeros(self._settings.embedding_dim, dtype=np.float32)
            embeddings.append(chunk_emb)

        return embeddings

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Standard mean-pooled embedding for arbitrary texts (e.g., QuIM questions).

        Args:
            texts: List of texts to embed.

        Returns:
            List of 768-dim embeddings.
        """
        self._ensure_model()
        import torch

        if not texts:
            return []

        embeddings = []
        batch_size = self._settings.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoding = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self._settings.max_length,
                padding=True,
            )
            input_ids = encoding["input_ids"].to(self._settings.device)
            attention_mask = encoding["attention_mask"].to(self._settings.device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state.cpu().numpy()  # [batch, seq, dim]
                mask = attention_mask.cpu().numpy()  # [batch, seq]

            for j in range(hidden.shape[0]):
                # Mask-aware mean pool
                m = mask[j][:, np.newaxis]  # [seq, 1]
                pooled = (hidden[j] * m).sum(axis=0) / m.sum()
                embeddings.append(pooled.astype(np.float32))

        return embeddings

    def matryoshka_slice(self, embedding: np.ndarray) -> np.ndarray:
        """Slice a full embedding to the Matryoshka fast dimension.

        Args:
            embedding: Full embedding (768-dim by default).

        Returns:
            Sliced embedding (64-dim by default), L2-normalized.
        """
        dim = self._settings.matryoshka_dim
        sliced = embedding[:dim].copy()
        norm = np.linalg.norm(sliced)
        if norm > 0:
            sliced /= norm
        return sliced

    def _ensure_model(self) -> None:
        """Ensure model is loaded, raise if not."""
        if self._model is None or self._tokenizer is None:
            msg = "Model not loaded. Call load_model() first."
            raise ModelLoadError(msg)

    def _forward_pass(self, input_ids: object, attention_mask: object) -> np.ndarray:
        """Run a single forward pass and return token embeddings.

        Returns:
            numpy array of shape [seq_len, embedding_dim].
        """
        import torch

        device = self._settings.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[0].cpu().numpy()

    def _windowed_forward(
        self,
        input_ids: object,
        attention_mask: object,
        max_len: int,
    ) -> np.ndarray:
        """Handle documents longer than model max_length via overlapping windows.

        Returns:
            Stitched token embeddings of shape [total_tokens, embedding_dim].
        """
        import torch

        seq_len = input_ids.shape[1]
        overlap = self._settings.window_overlap_tokens
        stride = max_len - overlap

        all_ids = input_ids[0]  # [seq_len]
        all_mask = attention_mask[0]  # [seq_len]

        result = np.zeros((seq_len, self._settings.embedding_dim), dtype=np.float32)
        counts = np.zeros(seq_len, dtype=np.float32)

        for start in range(0, seq_len, stride):
            end = min(start + max_len, seq_len)
            window_ids = all_ids[start:end].unsqueeze(0)
            window_mask = all_mask[start:end].unsqueeze(0)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=window_ids.to(self._settings.device),
                    attention_mask=window_mask.to(self._settings.device),
                )
                window_embs = outputs.last_hidden_state[0].cpu().numpy()

            result[start:end] += window_embs
            counts[start:end] += 1.0

            if end >= seq_len:
                break

        # Average overlapping regions
        counts = np.maximum(counts, 1.0)
        result /= counts[:, np.newaxis]
        return result

    def _find_chunk_token_spans(
        self,
        chunk_texts: list[str],
        full_text: str,
        offsets: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Map chunk texts to token index spans using character offset mapping.

        Args:
            chunk_texts: The text of each chunk.
            full_text: The full document text.
            offsets: Token offset mapping from tokenizer, shape [seq_len, 2].

        Returns:
            List of (start_token_idx, end_token_idx) per chunk.
        """
        spans = []
        search_start = 0

        for text in chunk_texts:
            # Find chunk in full text by matching a prefix
            prefix = text[:80] if len(text) > 80 else text
            char_start = full_text.find(prefix, search_start)

            if char_start == -1:
                # Retry from beginning if not found after search_start
                char_start = full_text.find(prefix)

            if char_start == -1:
                # Complete fallback: proportional estimation
                _log.warning("chunk_text_not_found", prefix=prefix[:40])
                spans.append((0, 0))
                continue

            char_end = char_start + len(text)
            search_start = char_start + len(prefix)

            # Map char offsets to token indices
            start_tok = self._char_to_token(offsets, char_start, start=True)
            end_tok = self._char_to_token(offsets, char_end, start=False)

            spans.append((start_tok, end_tok))

        return spans

    @staticmethod
    def _char_to_token(
        offsets: np.ndarray,
        char_pos: int,
        *,
        start: bool,
    ) -> int:
        """Convert a character position to a token index.

        For start=True, find first token whose start offset <= char_pos.
        For start=False, find first token whose end offset >= char_pos.
        """
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue  # Skip special tokens
            if start and tok_start >= char_pos:
                return i
            if not start and tok_end >= char_pos:
                return i + 1
        return len(offsets)

    @staticmethod
    def _mean_pool(token_embeddings: np.ndarray) -> np.ndarray:
        """Mean-pool token embeddings to produce a single vector."""
        return token_embeddings.mean(axis=0).astype(np.float32)
