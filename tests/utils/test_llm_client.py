"""Tests for the LLM provider abstraction layer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils._exceptions import LLMCallError, LLMNotAvailableError
from src.utils._llm_client import (
    AnthropicProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    NvidiaProvider,
    OllamaProvider,
    ProviderConfig,
    _OpenAICompatibleProvider,
    clear_provider_cache,
    get_langchain_llm,
    get_llm_provider,
    load_llm_config,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset module-level caches between tests."""
    clear_provider_cache()
    yield
    clear_provider_cache()


@pytest.fixture()
def sample_config() -> LLMConfig:
    return LLMConfig(
        providers={
            "anthropic": ProviderConfig(
                default_model="claude-haiku-4-5-20251001", timeout_seconds=60
            ),
            "ollama": ProviderConfig(
                base_url="http://localhost:11434",
                default_model="qwen3:14b",
                timeout_seconds=120,
            ),
            "nvidia": ProviderConfig(
                base_url="https://integrate.api.nvidia.com/v1",
                default_model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                timeout_seconds=60,
            ),
        },
        routing={
            "hyde": "ollama",
            "contextual_retrieval": "ollama",
            "genground": "ollama",
            "ragas": "nvidia",
            "flare": "anthropic",
        },
        model_overrides={
            "hyde": "qwen3:14b",
            "ragas": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        },
        ollama_options={
            "contextual_retrieval": {"temperature": 0.3, "num_ctx": 8192},
        },
    )


@pytest.fixture()
def llm_yaml_path(tmp_path: Path) -> Path:
    content = {
        "providers": {
            "ollama": {
                "base_url": "http://localhost:11434",
                "default_model": "qwen3:14b",
                "timeout_seconds": 120,
            },
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "default_model": "nemotron-super-49b",
                "timeout_seconds": 60,
            },
        },
        "routing": {"hyde": "ollama", "ragas": "nvidia"},
        "model_overrides": {"hyde": "qwen3:14b"},
        "ollama_options": {"hyde": {"temperature": 0.7, "num_ctx": 4096}},
    }
    import yaml

    p = tmp_path / "llm.yaml"
    p.write_text(yaml.dump(content), encoding="utf-8")
    return p


def _user_msg(text: str = "Hello") -> list[LLMMessage]:
    return [LLMMessage(role="user", content=text)]


def _system_msg(text: str = "You are helpful.", cached: bool = False) -> list[LLMMessage]:
    cc = {"type": "ephemeral"} if cached else None
    return [LLMMessage(role="system", content=text, cache_control=cc)]


# ── Model tests ─────────────────────────────────────────────────────────


class TestPydanticModels:
    def test_provider_config_defaults(self):
        cfg = ProviderConfig()
        assert cfg.base_url == ""
        assert cfg.default_model == ""
        assert cfg.timeout_seconds == 120

    def test_llm_config_defaults(self):
        cfg = LLMConfig()
        assert cfg.providers == {}
        assert cfg.routing == {}
        assert cfg.model_overrides == {}
        assert cfg.ollama_options == {}

    def test_llm_message_basic(self):
        msg = LLMMessage(role="user", content="hi")
        assert msg.role == "user"
        assert msg.content == "hi"
        assert msg.cache_control is None

    def test_llm_message_with_cache_control(self):
        msg = LLMMessage(role="system", content="ctx", cache_control={"type": "ephemeral"})
        assert msg.cache_control == {"type": "ephemeral"}

    def test_llm_response_defaults(self):
        resp = LLMResponse(text="answer")
        assert resp.text == "answer"
        assert resp.model == ""
        assert resp.provider == ""
        assert resp.usage == {}

    def test_llm_response_full(self):
        resp = LLMResponse(
            text="ok",
            model="test-model",
            provider="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.usage["prompt_tokens"] == 10


# ── Config loading tests ────────────────────────────────────────────────


class TestLoadLLMConfig:
    def test_loads_from_yaml(self, llm_yaml_path: Path):
        cfg = load_llm_config(llm_yaml_path)
        assert "ollama" in cfg.providers
        assert cfg.routing["hyde"] == "ollama"
        assert cfg.model_overrides["hyde"] == "qwen3:14b"
        assert cfg.ollama_options["hyde"]["num_ctx"] == 4096

    def test_returns_defaults_when_file_missing(self, tmp_path: Path):
        cfg = load_llm_config(tmp_path / "nonexistent.yaml")
        assert cfg.providers == {}
        assert cfg.routing == {}

    def test_returns_defaults_on_invalid_yaml(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : invalid", encoding="utf-8")
        cfg = load_llm_config(bad)
        assert isinstance(cfg, LLMConfig)

    def test_caches_default_path(self, llm_yaml_path: Path):
        """When using default path, config should be cached."""
        with patch("src.utils._llm_client._DEFAULT_CONFIG_PATH", llm_yaml_path):
            cfg1 = load_llm_config()
            cfg2 = load_llm_config()
            assert cfg1 is cfg2

    def test_explicit_path_not_cached(self, llm_yaml_path: Path):
        cfg1 = load_llm_config(llm_yaml_path)
        cfg2 = load_llm_config(llm_yaml_path)
        # Both valid, but not the same object (no caching for explicit paths)
        assert cfg1 is not cfg2


# ── AnthropicProvider tests ─────────────────────────────────────────────


class TestAnthropicProvider:
    def test_provider_name(self):
        p = AnthropicProvider(model="test")
        assert p.provider_name == "anthropic"

    def test_model_property(self):
        p = AnthropicProvider(model="claude-haiku-4-5-20251001")
        assert p.model == "claude-haiku-4-5-20251001"

    def test_is_available_no_anthropic(self):
        with patch("importlib.util.find_spec", return_value=None):
            p = AnthropicProvider(model="test")
            assert p.is_available is False

    def test_is_available_no_api_key(self):
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict("os.environ", {}, clear=True),
        ):
            p = AnthropicProvider(model="test")
            assert p.is_available is False

    def test_is_available_with_key(self):
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}),
        ):
            p = AnthropicProvider(model="test")
            assert p.is_available is True

    def test_complete_sync(self):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        content_block = SimpleNamespace(text="hello world")
        mock_response = SimpleNamespace(content=[content_block], model="test-m", usage=usage)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        p = AnthropicProvider(model="test-m")
        p._sync_client = mock_client

        resp = p.complete(_user_msg("hi"), max_tokens=100)

        assert resp.text == "hello world"
        assert resp.model == "test-m"
        assert resp.provider == "anthropic"
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5
        mock_client.messages.create.assert_called_once()

    async def test_acomplete_async(self):
        usage = SimpleNamespace(input_tokens=8, output_tokens=3)
        content_block = SimpleNamespace(text="async result")
        mock_response = SimpleNamespace(content=[content_block], model="a-m", usage=usage)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        p = AnthropicProvider(model="a-m")
        p._async_client = mock_client

        resp = await p.acomplete(_user_msg("test"), max_tokens=200)
        assert resp.text == "async result"
        assert resp.provider == "anthropic"

    def test_cache_control_preserved_in_system(self):
        mock_client = MagicMock()
        content_block = SimpleNamespace(text="ok")
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[content_block], model="m", usage=None
        )

        p = AnthropicProvider(model="m")
        p._sync_client = mock_client

        p.complete(
            _user_msg("chunk text"),
            system=_system_msg("full doc", cached=True),
            max_tokens=512,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        sys_block = call_kwargs["system"][0]
        assert sys_block["cache_control"] == {"type": "ephemeral"}
        assert sys_block["type"] == "text"

    def test_temperature_passed(self):
        mock_client = MagicMock()
        content_block = SimpleNamespace(text="ok")
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[content_block], model="m", usage=None
        )

        p = AnthropicProvider(model="m")
        p._sync_client = mock_client

        p.complete(_user_msg(), temperature=0.5, max_tokens=100)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_temperature_omitted_when_none(self):
        mock_client = MagicMock()
        content_block = SimpleNamespace(text="ok")
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[content_block], model="m", usage=None
        )

        p = AnthropicProvider(model="m")
        p._sync_client = mock_client

        p.complete(_user_msg(), max_tokens=100)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    def test_complete_raises_llm_call_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")

        p = AnthropicProvider(model="m")
        p._sync_client = mock_client

        with pytest.raises(LLMCallError, match="Anthropic call failed"):
            p.complete(_user_msg())

    def test_ensure_sync_raises_without_import(self):
        p = AnthropicProvider(model="m")
        with (
            patch("builtins.__import__", side_effect=ImportError("no anthropic")),
            pytest.raises(LLMNotAvailableError, match="anthropic package required"),
        ):
            p._ensure_sync()

    def test_ensure_async_raises_without_import(self):
        p = AnthropicProvider(model="m")
        with (
            patch("builtins.__import__", side_effect=ImportError("no anthropic")),
            pytest.raises(LLMNotAvailableError, match="anthropic package required"),
        ):
            p._ensure_async()

    def test_empty_content_returns_empty_text(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[], model="m", usage=None
        )
        p = AnthropicProvider(model="m")
        p._sync_client = mock_client

        resp = p.complete(_user_msg())
        assert resp.text == ""


# ── OllamaProvider tests ───────────────────────────────────────────────


class TestOllamaProvider:
    def _openai_response(
        self, text: str = "ollama says", model: str = "qwen3:14b"
    ) -> dict[str, Any]:
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": model,
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }

    def test_provider_name(self):
        p = OllamaProvider(model="qwen3:14b")
        assert p.provider_name == "ollama"

    def test_is_available_with_httpx(self):
        p = OllamaProvider(model="qwen3:14b")
        assert p.is_available is True  # httpx is installed (transitive dep)

    def test_complete_sync(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="qwen3:14b")
        p._sync_client = mock_client

        resp = p.complete(_user_msg("test"), max_tokens=256)

        assert resp.text == "ollama says"
        assert resp.provider == "ollama"
        assert resp.model == "qwen3:14b"
        assert resp.usage["prompt_tokens"] == 20

    async def test_acomplete_async(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response("async ollama")
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="qwen3:14b")
        p._async_client = mock_client

        resp = await p.acomplete(_user_msg("test"), max_tokens=256)
        assert resp.text == "async ollama"

    def test_system_message_prepended(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="qwen3:14b")
        p._sync_client = mock_client

        p.complete(
            _user_msg("chunk"),
            system=_system_msg("full document"),
            max_tokens=256,
        )

        payload = mock_client.post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "full document"
        assert payload["messages"][1]["role"] == "user"

    def test_cache_control_stripped(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="qwen3:14b")
        p._sync_client = mock_client

        p.complete(
            _user_msg("chunk"),
            system=_system_msg("doc", cached=True),
            max_tokens=256,
        )

        payload = mock_client.post.call_args[1]["json"]
        sys_msg = payload["messages"][0]
        assert "cache_control" not in sys_msg

    def test_ollama_options_applied(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="qwen3:14b", options={"temperature": 0.3, "num_ctx": 8192})
        p._sync_client = mock_client

        p.complete(_user_msg(), max_tokens=100)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["temperature"] == 0.3
        assert payload["options"]["num_ctx"] == 8192

    def test_explicit_temperature_overrides_options(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="qwen3:14b", options={"temperature": 0.3})
        p._sync_client = mock_client

        p.complete(_user_msg(), max_tokens=100, temperature=0.9)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["temperature"] == 0.9

    def test_chat_url_appends_v1(self):
        p = OllamaProvider(model="m", base_url="http://localhost:11434")
        assert p._chat_url() == "http://localhost:11434/v1/chat/completions"

    def test_chat_url_preserves_existing_v1(self):
        p = OllamaProvider(model="m", base_url="http://localhost:11434/v1")
        assert p._chat_url() == "http://localhost:11434/v1/chat/completions"

    def test_complete_raises_on_error(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = RuntimeError("connection refused")

        p = OllamaProvider(model="m")
        p._sync_client = mock_client

        with pytest.raises(LLMCallError, match="ollama call failed"):
            p.complete(_user_msg())

    def test_stream_false_in_payload(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        p = OllamaProvider(model="m")
        p._sync_client = mock_client
        p.complete(_user_msg(), max_tokens=100)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["stream"] is False


# ── NvidiaProvider tests ────────────────────────────────────────────────


class TestNvidiaProvider:
    def _openai_response(
        self,
        text: str = "nvidia says",
        model: str = "nemotron",
        use_reasoning: bool = False,
    ) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant"}
        if use_reasoning:
            msg["content"] = None
            msg["reasoning_content"] = text
        else:
            msg["content"] = text
        return {
            "choices": [{"message": msg}],
            "model": model,
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }

    def test_provider_name(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
            p = NvidiaProvider(model="nemotron")
        assert p.provider_name == "nvidia"

    def test_is_available_with_key(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            p = NvidiaProvider(model="nemotron")
            assert p.is_available is True

    def test_is_available_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            p = NvidiaProvider(model="nemotron")
            assert p.is_available is False

    def test_api_key_in_headers(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-123"}):
            p = NvidiaProvider(model="nemotron")
        headers = p._headers()
        assert headers["Authorization"] == "Bearer nvapi-123"

    def test_complete_normal_content(self):
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response("nvidia result")
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
            p = NvidiaProvider(model="nemotron")
        p._sync_client = mock_client

        resp = p.complete(_user_msg(), max_tokens=200)
        assert resp.text == "nvidia result"
        assert resp.provider == "nvidia"

    def test_reasoning_content_fallback(self):
        """NVIDIA Ultra model returns response in reasoning_content."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._openai_response(
            "ultra reasoning", use_reasoning=True
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
            p = NvidiaProvider(model="ultra")
        p._sync_client = mock_client

        resp = p.complete(_user_msg(), max_tokens=200)
        assert resp.text == "ultra reasoning"

    def test_chat_url_no_double_v1(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
            p = NvidiaProvider(model="m", base_url="https://api.nvidia.com/v1")
        assert p._chat_url() == "https://api.nvidia.com/v1/chat/completions"

    def test_empty_choices_returns_empty(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [], "model": "m", "usage": {}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
            p = NvidiaProvider(model="m")
        p._sync_client = mock_client

        resp = p.complete(_user_msg())
        assert resp.text == ""


# ── OpenAI-compatible base tests ────────────────────────────────────────


class TestOpenAICompatibleBase:
    def test_flatten_system_single(self):
        msgs = [LLMMessage(role="system", content="You are helpful.")]
        assert _OpenAICompatibleProvider._flatten_system(msgs) == "You are helpful."

    def test_flatten_system_multiple(self):
        msgs = [
            LLMMessage(role="system", content="Part 1"),
            LLMMessage(role="system", content="Part 2"),
        ]
        assert _OpenAICompatibleProvider._flatten_system(msgs) == "Part 1\n\nPart 2"

    def test_flatten_system_strips_cache_control(self):
        msgs = [
            LLMMessage(role="system", content="cached doc", cache_control={"type": "ephemeral"}),
        ]
        result = _OpenAICompatibleProvider._flatten_system(msgs)
        assert result == "cached doc"
        assert "ephemeral" not in result


# ── Factory tests ───────────────────────────────────────────────────────


class TestGetLLMProvider:
    def test_returns_anthropic_for_anthropic_route(self, sample_config: LLMConfig):
        p = get_llm_provider("flare", config=sample_config)
        assert isinstance(p, AnthropicProvider)
        assert p.model == "claude-haiku-4-5-20251001"

    def test_returns_ollama_for_ollama_route(self, sample_config: LLMConfig):
        p = get_llm_provider("hyde", config=sample_config)
        assert isinstance(p, OllamaProvider)
        assert p.model == "qwen3:14b"

    def test_returns_nvidia_for_nvidia_route(self, sample_config: LLMConfig):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
            p = get_llm_provider("ragas", config=sample_config)
        assert isinstance(p, NvidiaProvider)
        assert p.model == "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    def test_model_override_applied(self, sample_config: LLMConfig):
        p = get_llm_provider("hyde", config=sample_config)
        assert p.model == "qwen3:14b"

    def test_ollama_options_applied(self, sample_config: LLMConfig):
        p = get_llm_provider("contextual_retrieval", config=sample_config)
        assert isinstance(p, OllamaProvider)
        assert p._options["temperature"] == 0.3
        assert p._options["num_ctx"] == 8192

    def test_fallback_to_anthropic_when_no_routing(self, sample_config: LLMConfig):
        p = get_llm_provider("unknown_component", config=sample_config)
        assert isinstance(p, AnthropicProvider)

    def test_fallback_model_when_no_override(self):
        cfg = LLMConfig()
        p = get_llm_provider("anything", config=cfg)
        assert isinstance(p, AnthropicProvider)
        assert p.model == "claude-haiku-4-5-20251001"

    def test_cache_returns_same_instance(self, sample_config: LLMConfig):
        # First call with explicit config won't cache
        # Default (no config) will cache
        with patch("src.utils._llm_client.load_llm_config", return_value=sample_config):
            p1 = get_llm_provider("hyde")
            p2 = get_llm_provider("hyde")
            assert p1 is p2

    def test_explicit_config_bypasses_cache(self, sample_config: LLMConfig):
        p1 = get_llm_provider("hyde", config=sample_config)
        p2 = get_llm_provider("hyde", config=sample_config)
        assert p1 is not p2

    def test_clear_cache_works(self, sample_config: LLMConfig):
        with patch("src.utils._llm_client.load_llm_config", return_value=sample_config):
            p1 = get_llm_provider("hyde")
            clear_provider_cache()
            p2 = get_llm_provider("hyde")
            assert p1 is not p2

    def test_unknown_provider_raises(self):
        cfg = LLMConfig(routing={"x": "gpt4all"})
        with pytest.raises(LLMNotAvailableError, match="Unknown LLM provider"):
            get_llm_provider("x", config=cfg)


# ── LangChain helper tests ─────────────────────────────────────────────


class TestGetLangchainLLM:
    def test_anthropic_route(self, sample_config: LLMConfig):
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_cls)}):
            get_langchain_llm("flare", config=sample_config)
        mock_cls.assert_called_once_with(model="claude-haiku-4-5-20251001")

    def test_ollama_route(self, sample_config: LLMConfig):
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_cls)}):
            get_langchain_llm("hyde", config=sample_config)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "qwen3:14b"
        assert "11434/v1" in call_kwargs["base_url"]
        assert call_kwargs["api_key"] == "ollama"

    def test_nvidia_route(self, sample_config: LLMConfig):
        mock_cls = MagicMock()
        with (
            patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_cls)}),
            patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}),
        ):
            get_langchain_llm("ragas", config=sample_config)
        call_kwargs = mock_cls.call_args[1]
        assert "nvidia" in call_kwargs["base_url"]
        assert call_kwargs["api_key"] == "nvapi-test"

    def test_nvidia_route_no_key_raises(self, sample_config: LLMConfig):
        mock_cls = MagicMock()
        with (
            patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_cls)}),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(LLMNotAvailableError, match="NVIDIA_API_KEY"),
        ):
            get_langchain_llm("ragas", config=sample_config)

    def test_anthropic_import_error(self, sample_config: LLMConfig):
        with (
            patch.dict("sys.modules", {"langchain_anthropic": None}),
            pytest.raises(LLMNotAvailableError, match="langchain-anthropic"),
        ):
            get_langchain_llm("flare", config=sample_config)

    def test_openai_import_error(self, sample_config: LLMConfig):
        with (
            patch.dict("sys.modules", {"langchain_openai": None}),
            pytest.raises(LLMNotAvailableError, match="langchain-openai"),
        ):
            get_langchain_llm("hyde", config=sample_config)

    def test_unknown_provider_raises(self):
        cfg = LLMConfig(routing={"x": "gpt4all"})
        with pytest.raises(LLMNotAvailableError, match="No LangChain wrapper"):
            get_langchain_llm("x", config=cfg)


# ── Integration-style: factory → provider → call ───────────────────────


class TestFactoryToProviderFlow:
    def test_factory_ollama_complete_flow(self, sample_config: LLMConfig):
        """Factory creates OllamaProvider, which can complete a call."""
        provider = get_llm_provider("hyde", config=sample_config)
        assert isinstance(provider, OllamaProvider)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hypothetical answer"}}],
            "model": "qwen3:14b",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        provider._sync_client = mock_client

        resp = provider.complete(_user_msg("What is Section 302?"), max_tokens=200)
        assert resp.text == "hypothetical answer"
        assert resp.provider == "ollama"

    def test_factory_anthropic_with_cache_control(self, sample_config: LLMConfig):
        """Factory creates AnthropicProvider, cache_control preserved."""
        provider = get_llm_provider("flare", config=sample_config)
        assert isinstance(provider, AnthropicProvider)

        content_block = SimpleNamespace(text="flare result")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[content_block], model="claude", usage=None
        )
        provider._sync_client = mock_client

        resp = provider.complete(
            _user_msg("assess this"),
            system=_system_msg("context doc", cached=True),
            max_tokens=256,
        )
        assert resp.text == "flare result"

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
