"""Cross-encoder reranker using BGE-reranker-v2-m3.

Lazy-loads a cross-encoder model (``transformers.AutoModelForSequenceClassification``)
and scores each ``(query, chunk_text)`` pair.  Scores are written into
:pyattr:`FusedChunk.rerank_score` and results are returned sorted descending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.retrieval._exceptions import RerankerError, RerankerNotAvailableError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from typing import Any

    from src.retrieval._models import FusedChunk, RetrievalSettings

_log = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks fused chunks using a cross-encoder model.

    Uses ``BAAI/bge-reranker-v2-m3`` (or the model specified in
    :pyattr:`RetrievalSettings.reranker_model`) via the ``transformers``
    library.  The model is loaded lazily on first call to :pymeth:`rerank`.
    """

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._torch: Any = None  # cached torch module reference

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the model and tokenizer have been loaded."""
        return self._model is not None and self._tokenizer is not None

    def load_model(self) -> None:
        """Load the cross-encoder model and tokenizer.

        Raises:
            RerankerNotAvailableError: If ``torch`` / ``transformers`` are not
                installed.
            RerankerError: If the model fails to load for any other reason.
        """
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            msg = (
                "torch and transformers are required for reranking. "
                "Install with: pip install torch transformers"
            )
            raise RerankerNotAvailableError(msg) from exc

        model_name = self._settings.reranker_model
        _log.info("loading_reranker", model=model_name)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
            )
            self._torch = torch
            self._device = torch.device(self._settings.device)
            self._model.to(self._device)
            self._model.eval()
        except (RerankerNotAvailableError, RerankerError):
            raise
        except Exception as exc:
            msg = f"Failed to load reranker model {model_name}: {exc}"
            raise RerankerError(msg) from exc

        _log.info("reranker_loaded", model=model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[FusedChunk],
        top_k: int = 20,
    ) -> list[FusedChunk]:
        """Rerank *chunks* by ``(query, chunk_text)`` relevance.

        Args:
            query: The user's natural-language query.
            chunks: Fused chunks from RRF (typically pre-sorted by RRF
                score, but ordering is not required).
            top_k: Maximum number of results to return.

        Returns:
            Up to *top_k* :class:`FusedChunk` instances sorted by
            ``rerank_score`` descending.  Each chunk's
            :pyattr:`rerank_score` field is populated.

        Raises:
            RerankerError: If the model has not been loaded or inference
                fails.
        """
        if not self.is_loaded:
            msg = "Reranker model is not loaded. Call load_model() before rerank()."
            raise RerankerError(msg)

        if not chunks:
            return []

        # Build (query, text) pairs — prefer contextualized_text if present.
        pairs = [[query, c.contextualized_text or c.text] for c in chunks]

        scores = self._score_pairs(pairs)

        # Write scores back onto the chunks.
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = score

        # Sort descending by rerank_score and truncate.
        ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """Batch-tokenize and score ``(query, text)`` pairs.

        Returns a flat list of sigmoid-normalised relevance scores aligned
        with *pairs*.
        """
        torch = self._torch
        batch_size = self._settings.rerank_batch_size
        all_scores: list[float] = []

        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]

            try:
                encoding = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoding = {k: v.to(self._device) for k, v in encoding.items()}

                with torch.no_grad():
                    logits = self._model(**encoding).logits.squeeze(-1)
                    scores = torch.sigmoid(logits)

                # Handle single-element batch (0-d tensor after squeeze).
                if scores.dim() == 0:
                    all_scores.append(float(scores.item()))
                else:
                    all_scores.extend(scores.cpu().tolist())

            except (RerankerError, RerankerNotAvailableError):
                raise
            except Exception as exc:
                msg = f"Reranker inference failed on batch starting at index {batch_start}: {exc}"
                raise RerankerError(msg) from exc

        return all_scores
