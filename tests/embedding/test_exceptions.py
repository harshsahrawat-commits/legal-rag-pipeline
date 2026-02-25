from __future__ import annotations

import pytest

from src.embedding._exceptions import (
    CollectionCreationError,
    EmbedderNotAvailableError,
    EmbeddingError,
    EmbeddingInferenceError,
    IndexingError,
    ModelLoadError,
    RedisStoreError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    """All embedding exceptions inherit from EmbeddingError -> LegalRAGError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            EmbeddingError,
            ModelLoadError,
            EmbeddingInferenceError,
            IndexingError,
            RedisStoreError,
            EmbedderNotAvailableError,
            CollectionCreationError,
        ],
    )
    def test_inherits_legal_rag_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, LegalRAGError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            ModelLoadError,
            EmbeddingInferenceError,
            IndexingError,
            RedisStoreError,
            EmbedderNotAvailableError,
            CollectionCreationError,
        ],
    )
    def test_inherits_embedding_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, EmbeddingError)

    def test_embedding_error_is_base(self) -> None:
        assert issubclass(EmbeddingError, LegalRAGError)
        assert not issubclass(LegalRAGError, EmbeddingError)


class TestExceptionMessages:
    def test_model_load_error_message(self) -> None:
        exc = ModelLoadError("Failed to load BAAI/bge-m3")
        assert "BAAI/bge-m3" in str(exc)

    def test_indexing_error_message(self) -> None:
        exc = IndexingError("Qdrant upsert failed")
        assert "upsert" in str(exc)

    def test_redis_store_error_message(self) -> None:
        exc = RedisStoreError("Connection refused")
        assert "Connection refused" in str(exc)

    def test_embedder_not_available_message(self) -> None:
        exc = EmbedderNotAvailableError("torch not installed")
        assert "torch" in str(exc)

    def test_collection_creation_error_message(self) -> None:
        exc = CollectionCreationError("legal_chunks already exists with different schema")
        assert "legal_chunks" in str(exc)

    def test_embedding_inference_error_message(self) -> None:
        exc = EmbeddingInferenceError("CUDA out of memory")
        assert "CUDA" in str(exc)


class TestExceptionRaising:
    def test_catch_as_embedding_error(self) -> None:
        with pytest.raises(EmbeddingError):
            raise ModelLoadError("test")

    def test_catch_as_legal_rag_error(self) -> None:
        with pytest.raises(LegalRAGError):
            raise IndexingError("test")

    def test_exception_chaining(self) -> None:
        original = RuntimeError("torch error")
        try:
            raise ModelLoadError("model load failed") from original
        except ModelLoadError as exc:
            assert exc.__cause__ is original
