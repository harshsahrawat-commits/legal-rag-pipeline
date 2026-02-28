"""Tests for Selective HyDE component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.query._hyde import _HYDE_PROMPT, SelectiveHyDE
from src.query._models import QuerySettings
from src.retrieval._models import QueryRoute
from src.utils._llm_client import LLMResponse

# --- Helpers ---


def _make_mock_provider(response_text: str) -> MagicMock:
    """Create a mock LLM provider that returns predefined text."""
    provider = MagicMock()
    response = LLMResponse(text=response_text, model="mock", provider="mock")
    provider.complete.return_value = response
    provider.is_available = True
    provider.provider_name = "mock"
    return provider


# --- Fixtures ---


@pytest.fixture
def settings() -> QuerySettings:
    return QuerySettings()


@pytest.fixture
def disabled_settings() -> QuerySettings:
    return QuerySettings(hyde_enabled=False)


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder that returns deterministic embeddings."""
    embedder = MagicMock()
    embedder.embed_texts.return_value = [np.ones(768, dtype=np.float32)]
    return embedder


# --- is_available ---


class TestIsAvailable:
    def test_available_when_enabled_and_provider_available(
        self, settings: QuerySettings
    ) -> None:
        mock_provider = _make_mock_provider("text")
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            assert hyde.is_available is True

    def test_not_available_when_disabled(self, disabled_settings: QuerySettings) -> None:
        hyde = SelectiveHyDE(disabled_settings)
        assert hyde.is_available is False

    def test_not_available_when_provider_unavailable(self, settings: QuerySettings) -> None:
        mock_provider = MagicMock()
        mock_provider.is_available = False
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            assert hyde.is_available is False


# --- maybe_generate ---


class TestMaybeGenerate:
    def test_skipped_for_simple_route(self, settings: QuerySettings) -> None:
        hyde = SelectiveHyDE(settings)
        result = hyde.maybe_generate("What is Section 302?", QueryRoute.SIMPLE)
        assert result.generated is False
        assert result.hypothetical_text is None

    def test_skipped_for_standard_route(self, settings: QuerySettings) -> None:
        hyde = SelectiveHyDE(settings)
        result = hyde.maybe_generate("What is the punishment for cheating?", QueryRoute.STANDARD)
        assert result.generated is False

    def test_generated_for_complex_route(
        self,
        settings: QuerySettings,
        mock_embedder: MagicMock,
    ) -> None:
        mock_provider = _make_mock_provider(
            "Under Section 498A of the Indian Penal Code..."
        )
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "What is the interplay between Section 498A IPC and the DV Act?",
                QueryRoute.COMPLEX,
                embedder=mock_embedder,
            )

        assert result.generated is True
        assert result.hypothetical_text is not None
        assert "498A" in result.hypothetical_text
        assert result.hyde_embedding is not None
        assert len(result.hyde_embedding) == 768

    def test_generated_for_analytical_route(
        self,
        settings: QuerySettings,
    ) -> None:
        mock_provider = _make_mock_provider("The evolution of Section 377...")
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Trace the evolution of Section 377 jurisprudence",
                QueryRoute.ANALYTICAL,
            )

        assert result.generated is True
        assert result.hypothetical_text is not None
        # No embedder provided → no embedding
        assert result.hyde_embedding is None

    def test_skipped_when_disabled(self, disabled_settings: QuerySettings) -> None:
        hyde = SelectiveHyDE(disabled_settings)
        result = hyde.maybe_generate(
            "Compare eviction grounds",
            QueryRoute.COMPLEX,
        )
        assert result.generated is False

    def test_skipped_when_provider_unavailable(self, settings: QuerySettings) -> None:
        mock_provider = MagicMock()
        mock_provider.is_available = False
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Compare eviction grounds",
                QueryRoute.COMPLEX,
            )
        assert result.generated is False

    def test_graceful_on_llm_error(
        self,
        settings: QuerySettings,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.provider_name = "mock"
        mock_provider.complete.side_effect = RuntimeError("API timeout")
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Complex legal question",
                QueryRoute.COMPLEX,
            )

        assert result.generated is False
        assert result.hypothetical_text is None

    def test_graceful_on_empty_response(
        self,
        settings: QuerySettings,
    ) -> None:
        mock_provider = _make_mock_provider("   ")
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Complex legal question",
                QueryRoute.COMPLEX,
            )

        assert result.generated is False

    def test_embedder_failure_graceful(
        self,
        settings: QuerySettings,
    ) -> None:
        mock_provider = _make_mock_provider("A hypothetical answer about law.")
        bad_embedder = MagicMock()
        bad_embedder.embed_texts.side_effect = RuntimeError("CUDA OOM")

        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Complex legal question",
                QueryRoute.COMPLEX,
                embedder=bad_embedder,
            )

        # Generated text is set, but embedding failed gracefully
        assert result.generated is True
        assert result.hypothetical_text is not None
        assert result.hyde_embedding is None

    def test_custom_hyde_routes(self) -> None:
        """Only ANALYTICAL in hyde_routes — COMPLEX should be skipped."""
        settings = QuerySettings(hyde_routes=["analytical"])
        hyde = SelectiveHyDE(settings)
        result = hyde.maybe_generate("Compare laws", QueryRoute.COMPLEX)
        assert result.generated is False


# --- _embed_hypothetical ---


class TestEmbedHypothetical:
    def test_returns_embedding(self, mock_embedder: MagicMock) -> None:
        result = SelectiveHyDE._embed_hypothetical("hypothetical text", mock_embedder)
        assert result is not None
        assert len(result) == 768

    def test_returns_none_without_embedder(self) -> None:
        result = SelectiveHyDE._embed_hypothetical("hypothetical text", None)
        assert result is None

    def test_returns_none_on_error(self) -> None:
        bad_embedder = MagicMock()
        bad_embedder.embed_texts.side_effect = RuntimeError("model error")
        result = SelectiveHyDE._embed_hypothetical("text", bad_embedder)
        assert result is None

    def test_returns_none_on_empty_result(self) -> None:
        empty_embedder = MagicMock()
        empty_embedder.embed_texts.return_value = []
        result = SelectiveHyDE._embed_hypothetical("text", empty_embedder)
        assert result is None


# --- _ensure_provider ---


class TestEnsureProvider:
    def test_initializes_provider(
        self,
        settings: QuerySettings,
    ) -> None:
        mock_provider = _make_mock_provider("text")
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            assert hyde._provider is None
            hyde._ensure_provider()
            assert hyde._provider is not None

    def test_provider_init_error_graceful(
        self,
        settings: QuerySettings,
    ) -> None:
        from src.utils._exceptions import LLMNotAvailableError

        with patch(
            "src.query._hyde.get_llm_provider",
            side_effect=LLMNotAvailableError("no provider"),
        ):
            hyde = SelectiveHyDE(settings)
            hyde._ensure_provider()
            assert hyde._provider is None

    def test_no_double_init(
        self,
        settings: QuerySettings,
    ) -> None:
        mock_provider = _make_mock_provider("text")
        with patch("src.query._hyde.get_llm_provider", return_value=mock_provider):
            hyde = SelectiveHyDE(settings)
            hyde._ensure_provider()
            first = hyde._provider
            hyde._ensure_provider()
            assert hyde._provider is first


# --- Prompt content ---


class TestPromptContent:
    def test_prompt_contains_indian_lawyer_instruction(self) -> None:
        assert "Indian lawyer" in _HYDE_PROMPT

    def test_prompt_contains_query_placeholder(self) -> None:
        assert "{query}" in _HYDE_PROMPT

    def test_prompt_requests_legal_terminology(self) -> None:
        assert "legal terminology" in _HYDE_PROMPT

    def test_prompt_format_works(self) -> None:
        formatted = _HYDE_PROMPT.format(query="What is Section 302?")
        assert "What is Section 302?" in formatted
