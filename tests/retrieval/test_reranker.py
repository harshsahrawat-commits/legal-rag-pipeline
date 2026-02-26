"""Tests for the cross-encoder reranker (``src.retrieval._reranker``)."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval._exceptions import RerankerError, RerankerNotAvailableError
from src.retrieval._models import FusedChunk, RetrievalSettings

# ---------------------------------------------------------------------------
# Helpers — mock torch + transformers
# ---------------------------------------------------------------------------


def _make_fake_logits(scores: list[float]) -> MagicMock:
    """Return a mock ``logits`` tensor that ``squeeze(-1)`` yields *scores*.

    The scores are *pre-sigmoid* (we set ``torch.sigmoid`` to identity in
    our mock), so these values will be returned as-is.
    """
    inner = MagicMock()
    inner.dim.return_value = 1
    inner.cpu.return_value = inner
    inner.tolist.return_value = scores
    inner.item.return_value = scores[0] if len(scores) == 1 else scores

    squeezed = MagicMock()
    squeezed.dim.return_value = 1 if len(scores) > 1 else 0
    squeezed.cpu.return_value = squeezed
    squeezed.tolist.return_value = scores
    squeezed.item.return_value = scores[0] if len(scores) == 1 else scores

    inner.squeeze.return_value = squeezed
    return inner


def _make_model_output(logits_mock: MagicMock) -> MagicMock:
    """Wrap logits in a model output object."""
    output = MagicMock()
    output.logits = logits_mock
    return output


def _build_mock_modules(
    logits_per_batch: list[list[float]] | None = None,
) -> tuple[ModuleType, ModuleType]:
    """Build fake ``torch`` and ``transformers`` modules.

    Args:
        logits_per_batch: Per-batch list of score lists.  If ``None`` a
            single batch returning ``[0.9, 0.7, 0.5]`` is used.
    """
    if logits_per_batch is None:
        logits_per_batch = [[0.9, 0.7, 0.5]]

    # --- torch ---
    torch_mod = ModuleType("torch")
    torch_mod.device = MagicMock(side_effect=lambda d: d)  # type: ignore[attr-defined]

    # no_grad as context manager
    no_grad_ctx = MagicMock()
    no_grad_ctx.__enter__ = MagicMock(return_value=None)
    no_grad_ctx.__exit__ = MagicMock(return_value=False)
    torch_mod.no_grad = MagicMock(return_value=no_grad_ctx)  # type: ignore[attr-defined]

    # sigmoid is identity — tests supply final scores directly.
    torch_mod.sigmoid = MagicMock(side_effect=lambda x: x)  # type: ignore[attr-defined]

    # --- transformers ---
    transformers_mod = ModuleType("transformers")

    # Tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance
    transformers_mod.AutoTokenizer = mock_tokenizer_cls  # type: ignore[attr-defined]

    # Model — returns successive batches of logits
    batch_iter = iter(logits_per_batch)

    def _model_call(**_kwargs: Any) -> MagicMock:
        try:
            scores = next(batch_iter)
        except StopIteration:
            scores = logits_per_batch[-1]  # repeat last batch
        return _make_model_output(_make_fake_logits(scores))

    mock_model_instance = MagicMock()
    mock_model_instance.side_effect = _model_call
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = None

    mock_model_cls = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model_instance
    transformers_mod.AutoModelForSequenceClassification = mock_model_cls  # type: ignore[attr-defined]

    return torch_mod, transformers_mod


def _make_fused_chunks(
    n: int,
    *,
    contextualized: bool = False,
) -> list[FusedChunk]:
    """Create *n* FusedChunks with distinct text."""
    chunks = []
    for i in range(n):
        chunks.append(
            FusedChunk(
                chunk_id=f"chunk-{i}",
                text=f"Text for chunk {i}.",
                contextualized_text=f"Context for chunk {i}." if contextualized else None,
                rrf_score=1.0 / (60 + i),
                channels=["dense"],
                payload={"id": f"chunk-{i}"},
            )
        )
    return chunks


def _load_reranker_with_mocks(
    settings: RetrievalSettings | None = None,
    logits_per_batch: list[list[float]] | None = None,
) -> Any:
    """Instantiate a ``CrossEncoderReranker`` with mocked deps, already loaded."""
    torch_mod, transformers_mod = _build_mock_modules(logits_per_batch)
    settings = settings or RetrievalSettings()
    with patch.dict(
        sys.modules,
        {"torch": torch_mod, "transformers": transformers_mod},
    ):
        from src.retrieval._reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings)
        reranker.load_model()
    return reranker


# ===================================================================
# TestModelLoading
# ===================================================================


class TestModelLoading:
    """Tests for ``load_model``."""

    def test_load_model_success(self) -> None:
        """Model + tokenizer are loaded and reranker reports is_loaded."""
        reranker = _load_reranker_with_mocks()
        assert reranker.is_loaded is True

    def test_missing_deps_raises_not_available(self) -> None:
        """When torch/transformers are not importable, RerankerNotAvailableError."""
        settings = RetrievalSettings()

        # Remove torch and transformers from sys.modules so the lazy import fails
        fake_modules = {
            "torch": None,
            "transformers": None,
        }
        with patch.dict(sys.modules, fake_modules):
            from src.retrieval._reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings)
            with pytest.raises(RerankerNotAvailableError, match="torch and transformers"):
                reranker.load_model()

    def test_model_pretrained_failure_raises_reranker_error(self) -> None:
        """If from_pretrained explodes, we get a RerankerError."""
        torch_mod, transformers_mod = _build_mock_modules()
        transformers_mod.AutoModelForSequenceClassification.from_pretrained.side_effect = (  # type: ignore[attr-defined]
            RuntimeError("download failed")
        )

        settings = RetrievalSettings()
        with patch.dict(
            sys.modules,
            {"torch": torch_mod, "transformers": transformers_mod},
        ):
            from src.retrieval._reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings)
            with pytest.raises(RerankerError, match="Failed to load reranker model"):
                reranker.load_model()

    def test_tokenizer_failure_raises_reranker_error(self) -> None:
        """If tokenizer load fails, we get a RerankerError."""
        torch_mod, transformers_mod = _build_mock_modules()
        transformers_mod.AutoTokenizer.from_pretrained.side_effect = (  # type: ignore[attr-defined]
            OSError("tokenizer not found")
        )

        settings = RetrievalSettings()
        with patch.dict(
            sys.modules,
            {"torch": torch_mod, "transformers": transformers_mod},
        ):
            from src.retrieval._reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings)
            with pytest.raises(RerankerError, match="Failed to load reranker model"):
                reranker.load_model()

    def test_is_loaded_false_initially(self) -> None:
        """Before load_model, is_loaded is False."""
        settings = RetrievalSettings()
        from src.retrieval._reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings)
        assert reranker.is_loaded is False

    def test_custom_model_name_used(self) -> None:
        """Settings.reranker_model is passed to from_pretrained."""
        torch_mod, transformers_mod = _build_mock_modules()
        settings = RetrievalSettings(reranker_model="my-org/custom-reranker")
        with patch.dict(
            sys.modules,
            {"torch": torch_mod, "transformers": transformers_mod},
        ):
            from src.retrieval._reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings)
            reranker.load_model()

        transformers_mod.AutoTokenizer.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
            "my-org/custom-reranker",
        )
        transformers_mod.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
            "my-org/custom-reranker",
        )

    def test_device_applied(self) -> None:
        """The model is moved to the configured device."""
        torch_mod, transformers_mod = _build_mock_modules()
        settings = RetrievalSettings(device="cuda")
        with patch.dict(
            sys.modules,
            {"torch": torch_mod, "transformers": transformers_mod},
        ):
            from src.retrieval._reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings)
            reranker.load_model()

        model_instance = (
            transformers_mod.AutoModelForSequenceClassification.from_pretrained.return_value
        )  # type: ignore[attr-defined]
        model_instance.to.assert_called_once()
        model_instance.eval.assert_called_once()


# ===================================================================
# TestRerank
# ===================================================================


class TestRerank:
    """Tests for the ``rerank`` method."""

    def test_scores_set_correctly(self) -> None:
        """Each chunk gets its rerank_score populated."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9, 0.3, 0.6]])
        chunks = _make_fused_chunks(3)

        result = reranker.rerank("test query", chunks, top_k=10)

        assert len(result) == 3
        scores = [c.rerank_score for c in result]
        # All scores should be set (not None)
        assert all(s is not None for s in scores)

    def test_ordering_descending(self) -> None:
        """Results are sorted by rerank_score descending."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.3, 0.9, 0.6]])
        chunks = _make_fused_chunks(3)

        result = reranker.rerank("test query", chunks, top_k=10)

        scores = [c.rerank_score for c in result]
        assert scores == sorted(scores, reverse=True)
        # The highest score chunk should be first
        assert result[0].chunk_id == "chunk-1"  # 0.9
        assert result[1].chunk_id == "chunk-2"  # 0.6
        assert result[2].chunk_id == "chunk-0"  # 0.3

    def test_top_k_truncation(self) -> None:
        """Only top_k results are returned."""
        reranker = _load_reranker_with_mocks(
            logits_per_batch=[[0.9, 0.8, 0.7, 0.6, 0.5]],
        )
        chunks = _make_fused_chunks(5)

        result = reranker.rerank("test query", chunks, top_k=3)

        assert len(result) == 3

    def test_top_k_larger_than_input(self) -> None:
        """If top_k > len(chunks), all chunks returned."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9, 0.5]])
        chunks = _make_fused_chunks(2)

        result = reranker.rerank("test query", chunks, top_k=100)

        assert len(result) == 2

    def test_empty_input_returns_empty(self) -> None:
        """Empty chunk list returns empty list without calling model."""
        reranker = _load_reranker_with_mocks()

        result = reranker.rerank("test query", [], top_k=10)

        assert result == []

    def test_model_not_loaded_raises_error(self) -> None:
        """Calling rerank before load_model raises RerankerError."""
        settings = RetrievalSettings()
        from src.retrieval._reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings)

        with pytest.raises(RerankerError, match="not loaded"):
            reranker.rerank("test query", _make_fused_chunks(3))

    def test_single_chunk(self) -> None:
        """Reranking a single chunk works (0-d tensor edge case)."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.85]])
        chunks = _make_fused_chunks(1)

        result = reranker.rerank("test query", chunks, top_k=5)

        assert len(result) == 1
        assert result[0].rerank_score == pytest.approx(0.85)

    def test_rerank_preserves_chunk_fields(self) -> None:
        """Non-score fields on FusedChunk are preserved after reranking."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9, 0.5]])
        chunks = _make_fused_chunks(2)
        chunks[0].channels = ["dense", "bm25"]
        chunks[0].payload = {"id": "chunk-0", "custom": "data"}

        result = reranker.rerank("test query", chunks, top_k=10)

        top = next(c for c in result if c.chunk_id == "chunk-0")
        assert top.channels == ["dense", "bm25"]
        assert top.payload["custom"] == "data"
        assert top.rrf_score == pytest.approx(1.0 / 60)

    def test_inference_failure_raises_reranker_error(self) -> None:
        """If the model call throws, we wrap in RerankerError."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.5]])
        chunks = _make_fused_chunks(2)

        # Sabotage the model to raise on call
        reranker._model.side_effect = RuntimeError("CUDA OOM")

        with pytest.raises(RerankerError, match="inference failed"):
            reranker.rerank("test query", chunks)


# ===================================================================
# TestBatchProcessing
# ===================================================================


class TestBatchProcessing:
    """Tests for batch processing behaviour."""

    def test_large_input_batched(self) -> None:
        """When chunks exceed batch_size, multiple batches are processed."""
        settings = RetrievalSettings(rerank_batch_size=2)
        # 5 chunks → 3 batches (2, 2, 1)
        reranker = _load_reranker_with_mocks(
            settings=settings,
            logits_per_batch=[
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5],
            ],
        )
        chunks = _make_fused_chunks(5)

        result = reranker.rerank("test query", chunks, top_k=10)

        assert len(result) == 5
        # Verify descending order
        scores = [c.rerank_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_batch_size_equal_to_input(self) -> None:
        """When batch_size == len(chunks), processed in one batch."""
        settings = RetrievalSettings(rerank_batch_size=3)
        reranker = _load_reranker_with_mocks(
            settings=settings,
            logits_per_batch=[[0.9, 0.8, 0.7]],
        )
        chunks = _make_fused_chunks(3)

        result = reranker.rerank("test query", chunks, top_k=10)

        assert len(result) == 3

    def test_batch_size_larger_than_input(self) -> None:
        """When batch_size > len(chunks), processed in one batch."""
        settings = RetrievalSettings(rerank_batch_size=100)
        reranker = _load_reranker_with_mocks(
            settings=settings,
            logits_per_batch=[[0.9, 0.5]],
        )
        chunks = _make_fused_chunks(2)

        result = reranker.rerank("test query", chunks, top_k=10)

        assert len(result) == 2

    def test_batch_boundary_scores_correct(self) -> None:
        """Scores across batch boundaries are assigned to the right chunks."""
        settings = RetrievalSettings(rerank_batch_size=2)
        reranker = _load_reranker_with_mocks(
            settings=settings,
            logits_per_batch=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        )
        chunks = _make_fused_chunks(4)

        result = reranker.rerank("test query", chunks, top_k=10)

        # Original order: chunk-0=0.1, chunk-1=0.2, chunk-2=0.3, chunk-3=0.4
        # Sorted descending: chunk-3, chunk-2, chunk-1, chunk-0
        assert result[0].chunk_id == "chunk-3"
        assert result[0].rerank_score == pytest.approx(0.4)
        assert result[3].chunk_id == "chunk-0"
        assert result[3].rerank_score == pytest.approx(0.1)


# ===================================================================
# TestContextualizedText
# ===================================================================


class TestContextualizedText:
    """Tests that contextualized_text is preferred over plain text."""

    def test_uses_contextualized_text_when_available(self) -> None:
        """Pairs are built with contextualized_text, not text."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9]])
        chunks = [
            FusedChunk(
                chunk_id="c1",
                text="plain text",
                contextualized_text="richer contextualized text",
                rrf_score=0.01,
                channels=["dense"],
            ),
        ]

        result = reranker.rerank("query", chunks, top_k=5)

        assert len(result) == 1
        # Verify the tokenizer was called with the contextualized_text
        tokenizer = reranker._tokenizer
        call_args = tokenizer.call_args
        batch = call_args[0][0]
        assert batch == [["query", "richer contextualized text"]]

    def test_falls_back_to_text_when_no_contextualized(self) -> None:
        """When contextualized_text is None, falls back to text."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.8]])
        chunks = [
            FusedChunk(
                chunk_id="c1",
                text="only plain text",
                contextualized_text=None,
                rrf_score=0.01,
                channels=["dense"],
            ),
        ]

        result = reranker.rerank("query", chunks, top_k=5)

        assert len(result) == 1
        tokenizer = reranker._tokenizer
        call_args = tokenizer.call_args
        batch = call_args[0][0]
        assert batch == [["query", "only plain text"]]

    def test_mixed_contextualized_and_plain(self) -> None:
        """Mix of chunks with and without contextualized_text."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9, 0.7]])
        chunks = [
            FusedChunk(
                chunk_id="c1",
                text="plain",
                contextualized_text="contextualized",
                rrf_score=0.01,
                channels=["dense"],
            ),
            FusedChunk(
                chunk_id="c2",
                text="only plain",
                contextualized_text=None,
                rrf_score=0.01,
                channels=["bm25"],
            ),
        ]

        result = reranker.rerank("query", chunks, top_k=5)

        assert len(result) == 2
        tokenizer = reranker._tokenizer
        call_args = tokenizer.call_args
        batch = call_args[0][0]
        assert batch == [["query", "contextualized"], ["query", "only plain"]]


# ===================================================================
# TestTokenizerParams
# ===================================================================


class TestTokenizerParams:
    """Tests for tokenizer call parameters."""

    def test_tokenizer_called_with_correct_params(self) -> None:
        """Tokenizer receives padding, truncation, max_length, return_tensors."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9]])
        chunks = _make_fused_chunks(1)

        reranker.rerank("q", chunks, top_k=5)

        tokenizer = reranker._tokenizer
        call_args = tokenizer.call_args
        assert call_args[1]["padding"] is True
        assert call_args[1]["truncation"] is True
        assert call_args[1]["max_length"] == 512
        assert call_args[1]["return_tensors"] == "pt"


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_all_same_scores(self) -> None:
        """When all chunks have the same score, all are returned (stable)."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.5, 0.5, 0.5]])
        chunks = _make_fused_chunks(3)

        result = reranker.rerank("query", chunks, top_k=10)

        assert len(result) == 3
        assert all(c.rerank_score == pytest.approx(0.5) for c in result)

    def test_top_k_zero(self) -> None:
        """top_k=0 returns an empty list."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.9]])
        chunks = _make_fused_chunks(1)

        result = reranker.rerank("query", chunks, top_k=0)

        assert result == []

    def test_rerank_score_populated_on_original_chunks(self) -> None:
        """rerank_score is written to the original chunk objects (mutation)."""
        reranker = _load_reranker_with_mocks(logits_per_batch=[[0.75, 0.25]])
        chunks = _make_fused_chunks(2)

        reranker.rerank("query", chunks, top_k=10)

        # The original objects should be mutated
        assert chunks[0].rerank_score == pytest.approx(0.75)
        assert chunks[1].rerank_score == pytest.approx(0.25)

    def test_reranker_with_conftest_fused_chunks(
        self,
        sample_fused_chunks: list[FusedChunk],
    ) -> None:
        """Works with the shared conftest fixture."""
        reranker = _load_reranker_with_mocks(
            logits_per_batch=[[0.88, 0.72]],
        )

        result = reranker.rerank(
            "What is Section 420 of IPC?",
            sample_fused_chunks,
            top_k=5,
        )

        assert len(result) == 2
        assert result[0].rerank_score == pytest.approx(0.88)
        assert result[1].rerank_score == pytest.approx(0.72)
