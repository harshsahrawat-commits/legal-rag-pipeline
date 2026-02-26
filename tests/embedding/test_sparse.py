"""Tests for BM25SparseEncoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.embedding._models import SparseVector
from src.embedding._sparse import BM25SparseEncoder


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = BM25SparseEncoder._tokenize("Hello World!")
        assert tokens == ["hello", "world"]

    def test_numbers_included(self) -> None:
        tokens = BM25SparseEncoder._tokenize("Section 302 IPC")
        assert "302" in tokens

    def test_punctuation_removed(self) -> None:
        tokens = BM25SparseEncoder._tokenize("sub-section (1)(a)")
        assert tokens == ["sub", "section", "1", "a"]

    def test_empty_string(self) -> None:
        assert BM25SparseEncoder._tokenize("") == []

    def test_only_punctuation(self) -> None:
        assert BM25SparseEncoder._tokenize("---...!!!") == []

    def test_mixed_case(self) -> None:
        tokens = BM25SparseEncoder._tokenize("Supreme COURT of India")
        assert tokens == ["supreme", "court", "of", "india"]


class TestBuildVocabulary:
    def test_builds_vocab(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["hello world", "world foo"])
        assert enc.vocab_size == 3  # foo, hello, world

    def test_empty_texts(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary([])
        assert enc.vocab_size == 0
        assert enc._built

    def test_single_doc(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["section 302 ipc"])
        assert enc.vocab_size == 3

    def test_idf_computed(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["hello world", "hello foo", "bar baz"])
        # "hello" appears in 2/3 docs — lower IDF than "world" (1/3)
        assert enc._idf["hello"] < enc._idf["world"]

    def test_avg_doc_length(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["a b c", "d e f g h"])
        # total tokens: 3 + 5 = 8, avg = 4.0
        assert enc._avg_dl == 4.0

    def test_vocab_sorted(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["zebra apple mango"])
        keys = list(enc._vocab.keys())
        assert keys == sorted(keys)


class TestEncode:
    def test_encode_produces_sparse_vector(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["hello world", "foo bar"])
        result = enc.encode("hello world")
        assert isinstance(result, SparseVector)
        assert len(result.indices) > 0
        assert len(result.indices) == len(result.values)

    def test_encode_empty_text(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["hello"])
        result = enc.encode("")
        assert result.indices == []
        assert result.values == []

    def test_encode_unknown_terms_ignored(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["hello world"])
        result = enc.encode("unknown terms only")
        assert result.indices == []

    def test_raises_if_not_built(self) -> None:
        enc = BM25SparseEncoder()
        with pytest.raises(RuntimeError, match="build_vocabulary"):
            enc.encode("hello")

    def test_higher_tf_higher_score(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["contract contract breach", "breach penalty"])
        result1 = enc.encode("contract contract contract")
        result2 = enc.encode("contract")
        # "contract" score should be higher when tf is higher (BM25 saturates)
        if result1.indices and result2.indices:
            idx = enc._vocab["contract"]
            s1 = dict(zip(result1.indices, result1.values, strict=True)).get(idx, 0)
            s2 = dict(zip(result2.indices, result2.values, strict=True)).get(idx, 0)
            assert s1 > s2

    def test_all_values_positive(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["section 10 contract", "section 11 competent"])
        result = enc.encode("section 10 contract")
        assert all(v > 0 for v in result.values)

    def test_indices_are_valid(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["alpha beta gamma"])
        result = enc.encode("alpha beta gamma")
        assert all(0 <= i < enc.vocab_size for i in result.indices)

    def test_indices_sorted(self) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["zebra apple mango banana"])
        result = enc.encode("zebra apple mango banana")
        assert result.indices == sorted(result.indices)


class TestSaveLoadVocabulary:
    """Tests for BM25 vocabulary persistence (save/load roundtrip)."""

    def test_roundtrip_produces_same_encode(self, tmp_path: Path) -> None:
        corpus = [
            "section 420 indian penal code cheating",
            "section 302 murder punishment",
            "contract agreement breach remedy",
        ]
        enc = BM25SparseEncoder()
        enc.build_vocabulary(corpus)
        original = enc.encode("section 420 cheating")

        vocab_path = tmp_path / "bm25_vocab.json"
        enc.save_vocabulary(vocab_path)

        loaded = BM25SparseEncoder.load_vocabulary(vocab_path)
        restored = loaded.encode("section 420 cheating")

        assert original.indices == restored.indices
        assert original.values == pytest.approx(restored.values)

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            BM25SparseEncoder.load_vocabulary(missing)

    def test_save_before_build_raises(self, tmp_path: Path) -> None:
        enc = BM25SparseEncoder()
        with pytest.raises(RuntimeError, match="build_vocabulary"):
            enc.save_vocabulary(tmp_path / "vocab.json")

    def test_loaded_encoder_can_encode_new_text(self, tmp_path: Path) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["hello world foo bar"])
        vocab_path = tmp_path / "vocab.json"
        enc.save_vocabulary(vocab_path)

        loaded = BM25SparseEncoder.load_vocabulary(vocab_path)
        result = loaded.encode("hello foo baz")
        # "baz" is OOV, but "hello" and "foo" should produce indices
        assert len(result.indices) == 2
        assert all(v > 0 for v in result.values)

    def test_loaded_vocab_size_matches(self, tmp_path: Path) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["alpha beta gamma delta"])
        vocab_path = tmp_path / "vocab.json"
        enc.save_vocabulary(vocab_path)

        loaded = BM25SparseEncoder.load_vocabulary(vocab_path)
        assert loaded.vocab_size == enc.vocab_size

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        enc = BM25SparseEncoder()
        enc.build_vocabulary(["a b c"])
        nested = tmp_path / "a" / "b" / "vocab.json"
        enc.save_vocabulary(nested)
        assert nested.exists()
