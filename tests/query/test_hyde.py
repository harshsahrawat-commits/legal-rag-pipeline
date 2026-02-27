"""Tests for Selective HyDE component."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.query._hyde import _HYDE_PROMPT, SelectiveHyDE
from src.query._models import QuerySettings
from src.retrieval._models import QueryRoute

# --- Fixtures ---


@pytest.fixture
def settings() -> QuerySettings:
    return QuerySettings()


@pytest.fixture
def disabled_settings() -> QuerySettings:
    return QuerySettings(hyde_enabled=False)


@pytest.fixture
def mock_anthropic_module() -> types.ModuleType:
    """Build a fake anthropic module for testing."""
    mod = types.ModuleType("anthropic")
    mock_client_cls = MagicMock()
    mod.Anthropic = mock_client_cls  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder that returns deterministic embeddings."""
    embedder = MagicMock()
    embedder.embed_texts.return_value = [np.ones(768, dtype=np.float32)]
    return embedder


# --- is_available ---


class TestIsAvailable:
    def test_available_when_enabled_and_anthropic_present(
        self, settings: QuerySettings, mock_anthropic_module: types.ModuleType
    ) -> None:
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            hyde = SelectiveHyDE(settings)
            assert hyde.is_available is True

    def test_not_available_when_disabled(self, disabled_settings: QuerySettings) -> None:
        hyde = SelectiveHyDE(disabled_settings)
        assert hyde.is_available is False

    def test_not_available_when_anthropic_missing(self, settings: QuerySettings) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
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
        mock_anthropic_module: types.ModuleType,
        mock_embedder: MagicMock,
    ) -> None:
        # Set up mock client response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Under Section 498A of the Indian Penal Code...")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
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
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The evolution of Section 377...")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
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

    def test_skipped_when_anthropic_missing(self, settings: QuerySettings) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Compare eviction grounds",
                QueryRoute.COMPLEX,
            )
        assert result.generated is False

    def test_graceful_on_llm_error(
        self,
        settings: QuerySettings,
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API timeout")
        mock_anthropic_module.Anthropic.return_value = mock_client  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
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
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="   ")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            hyde = SelectiveHyDE(settings)
            result = hyde.maybe_generate(
                "Complex legal question",
                QueryRoute.COMPLEX,
            )

        assert result.generated is False

    def test_embedder_failure_graceful(
        self,
        settings: QuerySettings,
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A hypothetical answer about law.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client  # type: ignore[attr-defined]

        bad_embedder = MagicMock()
        bad_embedder.embed_texts.side_effect = RuntimeError("CUDA OOM")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
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


# --- _resolve_model_id ---


class TestResolveModelId:
    def test_claude_haiku_alias(self) -> None:
        assert SelectiveHyDE._resolve_model_id("claude-haiku") == "claude-haiku-4-5-20251001"

    def test_claude_sonnet_alias(self) -> None:
        assert SelectiveHyDE._resolve_model_id("claude-sonnet") == "claude-sonnet-4-6-20250514"

    def test_claude_opus_alias(self) -> None:
        assert SelectiveHyDE._resolve_model_id("claude-opus") == "claude-opus-4-6-20250514"

    def test_passthrough_for_unknown(self) -> None:
        assert SelectiveHyDE._resolve_model_id("my-custom-model") == "my-custom-model"


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


# --- _ensure_client ---


class TestEnsureClient:
    def test_initializes_client(
        self,
        settings: QuerySettings,
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            hyde = SelectiveHyDE(settings)
            assert hyde._client is None
            hyde._ensure_client()
            assert hyde._client is not None

    def test_client_init_error_graceful(
        self,
        settings: QuerySettings,
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        mock_anthropic_module.Anthropic.side_effect = RuntimeError("no API key")  # type: ignore[attr-defined]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            hyde = SelectiveHyDE(settings)
            hyde._ensure_client()
            assert hyde._client is None

    def test_no_double_init(
        self,
        settings: QuerySettings,
        mock_anthropic_module: types.ModuleType,
    ) -> None:
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            hyde = SelectiveHyDE(settings)
            hyde._ensure_client()
            first_client = hyde._client
            hyde._ensure_client()
            assert hyde._client is first_client


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
