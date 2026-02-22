from __future__ import annotations

from src.utils._hashing import content_hash


class TestContentHash:
    def test_string_input(self):
        result = content_hash("hello")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_bytes_input(self):
        result = content_hash(b"hello")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_string_and_bytes_match(self):
        assert content_hash("hello") == content_hash(b"hello")

    def test_deterministic(self):
        assert content_hash("test data") == content_hash("test data")

    def test_different_inputs_differ(self):
        assert content_hash("a") != content_hash("b")

    def test_empty_string(self):
        result = content_hash("")
        assert len(result) == 64

    def test_known_sha256(self):
        # SHA-256 of "hello" is well-known
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert content_hash("hello") == expected
