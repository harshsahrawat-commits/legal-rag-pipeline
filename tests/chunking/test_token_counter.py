from __future__ import annotations

import pytest

from src.chunking._token_counter import TokenCounter


@pytest.fixture()
def tc() -> TokenCounter:
    return TokenCounter()


class TestTokenCounter:
    def test_count_empty(self, tc: TokenCounter):
        assert tc.count("") == 0

    def test_count_simple(self, tc: TokenCounter):
        count = tc.count("hello world")
        assert count == 2

    def test_count_longer_text(self, tc: TokenCounter):
        text = (
            "Section 302 of the Indian Penal Code, 1860 provides that "
            "whoever commits murder shall be punished with death."
        )
        count = tc.count(text)
        assert 15 < count < 30  # reasonable token range

    def test_truncate_short_text(self, tc: TokenCounter):
        text = "hello world"
        result = tc.truncate(text, max_tokens=100)
        assert result == text

    def test_truncate_long_text(self, tc: TokenCounter):
        text = " ".join(f"word{i}" for i in range(200))
        result = tc.truncate(text, max_tokens=10)
        assert tc.count(result) <= 10

    def test_truncate_preserves_exact_tokens(self, tc: TokenCounter):
        text = "one two three four five six seven eight nine ten"
        truncated = tc.truncate(text, max_tokens=5)
        assert tc.count(truncated) == 5

    def test_split_at_token_boundary_short(self, tc: TokenCounter):
        text = "hello"
        head, tail = tc.split_at_token_boundary(text, target_tokens=10)
        assert head == text
        assert tail == ""

    def test_split_at_token_boundary_long(self, tc: TokenCounter):
        text = " ".join(f"word{i}" for i in range(100))
        head, tail = tc.split_at_token_boundary(text, target_tokens=20)
        assert tc.count(head) == 20
        assert tc.count(tail) > 0
        # Recombined should have the same tokens
        assert tc.count(head) + tc.count(tail) == tc.count(text)

    def test_split_preserves_content(self, tc: TokenCounter):
        text = "The quick brown fox jumps over the lazy dog"
        head, tail = tc.split_at_token_boundary(text, target_tokens=4)
        # head + tail should reconstruct approximately the original
        assert head != ""
        assert tail != ""

    def test_custom_encoding(self):
        tc = TokenCounter(encoding_name="cl100k_base")
        assert tc.count("test") > 0

    def test_hindi_text(self, tc: TokenCounter):
        text = "भारतीय दंड संहिता की धारा 302"
        count = tc.count(text)
        assert count > 0

    def test_count_legal_section_references(self, tc: TokenCounter):
        text = "Section 302 IPC r/w Section 34 IPC"
        count = tc.count(text)
        assert count > 5
