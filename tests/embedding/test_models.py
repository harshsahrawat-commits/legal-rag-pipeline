from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.embedding._models import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingSettings,
    SparseVector,
)


class TestSparseVector:
    def test_basic_creation(self) -> None:
        sv = SparseVector(indices=[0, 3, 7], values=[1.5, 0.8, 2.1])
        assert sv.indices == [0, 3, 7]
        assert sv.values == [1.5, 0.8, 2.1]

    def test_empty_sparse_vector(self) -> None:
        sv = SparseVector(indices=[], values=[])
        assert sv.indices == []
        assert sv.values == []

    def test_round_trip_json(self) -> None:
        sv = SparseVector(indices=[1, 2], values=[0.5, 0.7])
        data = sv.model_dump(mode="json")
        restored = SparseVector.model_validate(data)
        assert restored == sv


class TestEmbeddingSettings:
    def test_defaults(self) -> None:
        s = EmbeddingSettings()
        assert s.input_dir == Path("data/enriched")
        assert s.parsed_dir == Path("data/parsed")
        assert s.model_name_or_path == "BAAI/bge-m3"
        assert s.embedding_dim == 768
        assert s.matryoshka_dim == 64
        assert s.device == "cpu"
        assert s.batch_size == 16
        assert s.max_length == 8192
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.chunks_collection == "legal_chunks"
        assert s.quim_collection == "quim_questions"
        assert s.redis_url == "redis://localhost:6379/0"
        assert s.redis_key_prefix == "parent:"
        assert s.window_overlap_tokens == 128
        assert s.skip_existing is True

    def test_custom_values(self) -> None:
        s = EmbeddingSettings(
            model_name_or_path="custom/model",
            embedding_dim=1024,
            matryoshka_dim=128,
            device="cuda",
            batch_size=32,
        )
        assert s.model_name_or_path == "custom/model"
        assert s.embedding_dim == 1024
        assert s.matryoshka_dim == 128
        assert s.device == "cuda"
        assert s.batch_size == 32

    def test_path_coercion(self) -> None:
        s = EmbeddingSettings(input_dir="my/input", parsed_dir="my/parsed")
        assert isinstance(s.input_dir, Path)
        assert isinstance(s.parsed_dir, Path)


class TestEmbeddingConfig:
    def test_default_settings(self) -> None:
        config = EmbeddingConfig()
        assert isinstance(config.settings, EmbeddingSettings)
        assert config.settings.embedding_dim == 768

    def test_from_dict(self) -> None:
        data = {"settings": {"model_name_or_path": "test/model", "device": "cuda"}}
        config = EmbeddingConfig.model_validate(data)
        assert config.settings.model_name_or_path == "test/model"
        assert config.settings.device == "cuda"

    def test_round_trip(self) -> None:
        config = EmbeddingConfig(settings=EmbeddingSettings(batch_size=64))
        data = config.model_dump(mode="json")
        restored = EmbeddingConfig.model_validate(data)
        assert restored.settings.batch_size == 64


class TestEmbeddingResult:
    def test_defaults(self) -> None:
        r = EmbeddingResult()
        assert r.documents_found == 0
        assert r.documents_indexed == 0
        assert r.documents_skipped == 0
        assert r.documents_failed == 0
        assert r.chunks_embedded == 0
        assert r.quim_questions_embedded == 0
        assert r.parent_entries_stored == 0
        assert r.errors == []
        assert r.source_type is None
        assert r.finished_at is None
        assert isinstance(r.started_at, datetime)

    def test_counts_are_mutable(self) -> None:
        r = EmbeddingResult()
        r.documents_indexed += 5
        r.chunks_embedded += 100
        r.quim_questions_embedded += 50
        r.parent_entries_stored += 10
        assert r.documents_indexed == 5
        assert r.chunks_embedded == 100

    def test_errors_accumulate(self) -> None:
        r = EmbeddingResult()
        r.errors.append("doc1 failed")
        r.errors.append("doc2 failed")
        assert len(r.errors) == 2

    def test_finished_at_settable(self) -> None:
        r = EmbeddingResult()
        now = datetime.now(UTC)
        r.finished_at = now
        assert r.finished_at == now
