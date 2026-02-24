from __future__ import annotations

import tiktoken


class TokenCounter:
    """Thin wrapper around tiktoken for consistent token counting.

    Uses ``cl100k_base`` encoding (GPT-4 / text-embedding-ada-002).
    One instance is shared across all chunkers.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        if not text:
            return 0
        return len(self._enc.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate *text* to at most *max_tokens* tokens."""
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens])

    def split_at_token_boundary(self, text: str, target_tokens: int) -> tuple[str, str]:
        """Split *text* near *target_tokens* into (head, tail).

        The split happens at the exact token boundary closest to *target_tokens*.
        Returns ``(head, tail)`` where ``head`` has at most *target_tokens* tokens.
        If the text is shorter than *target_tokens*, ``tail`` is empty.
        """
        tokens = self._enc.encode(text)
        if len(tokens) <= target_tokens:
            return text, ""
        head = self._enc.decode(tokens[:target_tokens])
        tail = self._enc.decode(tokens[target_tokens:])
        return head, tail
