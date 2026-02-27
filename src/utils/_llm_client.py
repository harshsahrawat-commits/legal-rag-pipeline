"""LLM provider abstraction — Anthropic, Ollama, NVIDIA NIM.

Provides a unified interface for calling LLMs across different backends.
Provider routing is configured in ``configs/llm.yaml``.

Usage::

    from src.utils._llm_client import LLMMessage, get_llm_provider

    provider = get_llm_provider("contextual_retrieval")
    response = await provider.acomplete(
        [LLMMessage(role="user", content="Summarise this section.")],
        max_tokens=512,
    )
    print(response.text)
"""

from __future__ import annotations

import importlib.util
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.utils._exceptions import LLMCallError, LLMNotAvailableError
from src.utils._logging import get_logger

_log = get_logger(__name__)

# ── Default paths ───────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path("configs/llm.yaml")

# ── Pydantic models ────────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    base_url: str = ""
    default_model: str = ""
    timeout_seconds: int = 120


class LLMConfig(BaseModel):
    """Root model that mirrors ``configs/llm.yaml``."""

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: dict[str, str] = Field(default_factory=dict)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    ollama_options: dict[str, dict[str, Any]] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """A single message in the conversation."""

    role: str
    content: str
    cache_control: dict[str, str] | None = None


class LLMResponse(BaseModel):
    """Normalised response from any provider."""

    text: str
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


# ── Abstract base ──────────────────────────────────────────────────────


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(
        self,
        model: str,
        base_url: str = "",
        timeout_seconds: int = 120,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._options = options or {}

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        system: list[LLMMessage] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        system: list[LLMMessage] | None = None,
    ) -> LLMResponse: ...


# ── Anthropic provider ─────────────────────────────────────────────────


class AnthropicProvider(BaseLLMProvider):
    """Uses the native ``anthropic`` Python SDK.

    Preserves ``cache_control`` on system messages for prompt caching.
    """

    def __init__(self, model: str, timeout_seconds: int = 60, **kwargs: Any) -> None:
        super().__init__(model=model, timeout_seconds=timeout_seconds, **kwargs)
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("anthropic") is None:
                return False
        except (ValueError, ModuleNotFoundError):
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    # -- client lifecycle ---------------------------------------------------

    def _ensure_sync(self) -> None:
        if self._sync_client is not None:
            return
        try:
            import anthropic
        except ImportError:
            raise LLMNotAvailableError(
                "anthropic package required. Install with: pip install anthropic"
            ) from None
        self._sync_client = anthropic.Anthropic(
            timeout=float(self._timeout),
        )
        _log.info("anthropic_sync_client_initialized")

    def _ensure_async(self) -> None:
        if self._async_client is not None:
            return
        try:
            import anthropic
        except ImportError:
            raise LLMNotAvailableError(
                "anthropic package required. Install with: pip install anthropic"
            ) from None
        self._async_client = anthropic.AsyncAnthropic(
            timeout=float(self._timeout),
        )
        _log.info("anthropic_async_client_initialized")

    # -- kwargs builder -----------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float | None,
        system: list[LLMMessage] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            sys_blocks: list[dict[str, Any]] = []
            for s in system:
                block: dict[str, Any] = {"type": "text", "text": s.content}
                if s.cache_control:
                    block["cache_control"] = s.cache_control
                sys_blocks.append(block)
            kwargs["system"] = sys_blocks
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        text = response.content[0].text if response.content else ""
        usage: dict[str, int] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
            }
        return LLMResponse(text=text, model=response.model, provider="anthropic", usage=usage)

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        system: list[LLMMessage] | None = None,
    ) -> LLMResponse:
        self._ensure_sync()
        kwargs = self._build_kwargs(messages, max_tokens, temperature, system)
        try:
            response = self._sync_client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMCallError(f"Anthropic call failed: {exc}") from exc
        return self._parse_response(response)

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        system: list[LLMMessage] | None = None,
    ) -> LLMResponse:
        self._ensure_async()
        kwargs = self._build_kwargs(messages, max_tokens, temperature, system)
        try:
            response = await self._async_client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMCallError(f"Anthropic call failed: {exc}") from exc
        return self._parse_response(response)


# ── OpenAI-compatible base (shared by Ollama & NVIDIA) ─────────────────


class _OpenAICompatibleProvider(BaseLLMProvider):
    """Base for providers using the OpenAI chat-completions format."""

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        timeout_seconds: int = 120,
        api_key: str = "",
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            options=options,
            **kwargs,
        )
        self._name = name
        self._api_key = api_key
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def provider_name(self) -> str:
        return self._name

    # -- client lifecycle ---------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _chat_url(self) -> str:
        base = self._base_url
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _ensure_sync(self) -> None:
        if self._sync_client is not None:
            return
        try:
            import httpx
        except ImportError:
            raise LLMNotAvailableError(
                "httpx package required for Ollama/NVIDIA providers"
            ) from None
        self._sync_client = httpx.Client(timeout=float(self._timeout), headers=self._headers())
        _log.info("openai_compat_sync_client_initialized", provider=self._name)

    def _ensure_async(self) -> None:
        if self._async_client is not None:
            return
        try:
            import httpx
        except ImportError:
            raise LLMNotAvailableError(
                "httpx package required for Ollama/NVIDIA providers"
            ) from None
        self._async_client = httpx.AsyncClient(
            timeout=float(self._timeout), headers=self._headers()
        )
        _log.info("openai_compat_async_client_initialized", provider=self._name)

    # -- payload building ---------------------------------------------------

    @staticmethod
    def _flatten_system(system: list[LLMMessage]) -> str:
        """Convert system messages to plain text, stripping cache_control."""
        return "\n\n".join(s.content for s in system)

    def _build_payload(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float | None,
        system: list[LLMMessage] | None,
    ) -> dict[str, Any]:
        chat_messages: list[dict[str, str]] = []
        if system:
            chat_messages.append({"role": "system", "content": self._flatten_system(system)})
        for m in messages:
            chat_messages.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        elif "temperature" in self._options:
            payload["temperature"] = self._options["temperature"]
        return payload

    # -- response parsing ---------------------------------------------------

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        text = ""
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content") or msg.get("reasoning_content") or ""
        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
        }
        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            provider=self._name,
            usage=usage,
        )

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        system: list[LLMMessage] | None = None,
    ) -> LLMResponse:
        self._ensure_sync()
        payload = self._build_payload(messages, max_tokens, temperature, system)
        url = self._chat_url()
        try:
            resp = self._sync_client.post(url, json=payload)
            resp.raise_for_status()
        except Exception as exc:
            raise LLMCallError(f"{self._name} call failed: {exc}") from exc
        return self._parse_response(resp.json())

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        system: list[LLMMessage] | None = None,
    ) -> LLMResponse:
        self._ensure_async()
        payload = self._build_payload(messages, max_tokens, temperature, system)
        url = self._chat_url()
        try:
            resp = await self._async_client.post(url, json=payload)
            resp.raise_for_status()
        except Exception as exc:
            raise LLMCallError(f"{self._name} call failed: {exc}") from exc
        return self._parse_response(resp.json())


# ── Ollama provider ────────────────────────────────────────────────────


class OllamaProvider(_OpenAICompatibleProvider):
    """Ollama via the OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="ollama",
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            options=options,
            **kwargs,
        )

    @property
    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("httpx") is None:
                return False
        except (ValueError, ModuleNotFoundError):
            return False
        return bool(self._base_url)

    def _build_payload(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float | None,
        system: list[LLMMessage] | None,
    ) -> dict[str, Any]:
        payload = super()._build_payload(messages, max_tokens, temperature, system)
        # Pass Ollama-specific options (num_ctx, etc.)
        if "num_ctx" in self._options:
            payload.setdefault("options", {})["num_ctx"] = self._options["num_ctx"]
        return payload


# ── NVIDIA NIM provider ────────────────────────────────────────────────


class NvidiaProvider(_OpenAICompatibleProvider):
    """NVIDIA NIM via OpenAI-compatible API.

    Handles the Ultra model quirk where responses may appear
    in ``reasoning_content`` instead of ``content``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout_seconds: int = 60,
        **kwargs: Any,
    ) -> None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        super().__init__(
            name="nvidia",
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            **kwargs,
        )

    @property
    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("httpx") is None:
                return False
        except (ValueError, ModuleNotFoundError):
            return False
        return bool(os.environ.get("NVIDIA_API_KEY"))


# ── Config loading & factory ───────────────────────────────────────────

_config_cache: LLMConfig | None = None
_provider_cache: dict[str, BaseLLMProvider] = {}


def load_llm_config(config_path: Path | None = None) -> LLMConfig:
    """Load LLM config from ``configs/llm.yaml``.

    Returns a default (empty) config if the file is missing,
    which causes :func:`get_llm_provider` to fall back to Anthropic.
    """
    global _config_cache
    path = config_path or _DEFAULT_CONFIG_PATH

    # Only use cache when using the default path
    if config_path is None and _config_cache is not None:
        return _config_cache

    if not path.exists():
        _log.info("llm_config_not_found", path=str(path))
        cfg = LLMConfig()
        if config_path is None:
            _config_cache = cfg
        return cfg

    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = LLMConfig.model_validate(raw)
    except Exception as exc:
        _log.warning("llm_config_load_failed", error=str(exc))
        cfg = LLMConfig()

    if config_path is None:
        _config_cache = cfg
    return cfg


def get_llm_provider(
    component: str,
    config: LLMConfig | None = None,
) -> BaseLLMProvider:
    """Get a configured LLM provider for a pipeline component.

    Args:
        component: Key matching ``routing`` in ``configs/llm.yaml``
            (e.g. ``"contextual_retrieval"``, ``"hyde"``, ``"flare"``).
        config: Optional pre-loaded config.  Loads from disk if *None*.

    Returns:
        A :class:`BaseLLMProvider` instance (cached per component).
    """
    if config is None and component in _provider_cache:
        return _provider_cache[component]

    cfg = config or load_llm_config()

    provider_name = cfg.routing.get(component, "anthropic")
    provider_cfg = cfg.providers.get(provider_name, ProviderConfig())
    model = cfg.model_overrides.get(component) or provider_cfg.default_model
    if not model:
        model = "claude-haiku-4-5-20251001"

    provider: BaseLLMProvider
    if provider_name == "anthropic":
        provider = AnthropicProvider(
            model=model,
            timeout_seconds=provider_cfg.timeout_seconds,
        )
    elif provider_name == "ollama":
        options = cfg.ollama_options.get(component, {})
        provider = OllamaProvider(
            model=model,
            base_url=provider_cfg.base_url or "http://localhost:11434",
            timeout_seconds=provider_cfg.timeout_seconds,
            options=options,
        )
    elif provider_name == "nvidia":
        provider = NvidiaProvider(
            model=model,
            base_url=provider_cfg.base_url or "https://integrate.api.nvidia.com/v1",
            timeout_seconds=provider_cfg.timeout_seconds,
        )
    else:
        msg = f"Unknown LLM provider: {provider_name!r}"
        raise LLMNotAvailableError(msg)

    if config is None:
        _provider_cache[component] = provider

    _log.info(
        "llm_provider_created",
        component=component,
        provider=provider_name,
        model=model,
    )
    return provider


def clear_provider_cache() -> None:
    """Reset cached config and providers.  Useful for testing."""
    global _config_cache
    _provider_cache.clear()
    _config_cache = None


def get_langchain_llm(component: str = "ragas", config: LLMConfig | None = None) -> Any:
    """Get a LangChain-compatible chat model for a pipeline component.

    Returns :class:`ChatAnthropic` for the Anthropic route or
    :class:`ChatOpenAI` for Ollama / NVIDIA routes.
    """
    cfg = config or load_llm_config()
    provider_name = cfg.routing.get(component, "anthropic")
    provider_cfg = cfg.providers.get(provider_name, ProviderConfig())
    model = cfg.model_overrides.get(component) or provider_cfg.default_model
    if not model:
        model = "claude-haiku-4-5-20251001"

    if provider_name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise LLMNotAvailableError(
                "langchain-anthropic required for RAGAS with Anthropic provider. "
                "Install with: pip install langchain-anthropic"
            ) from None
        return ChatAnthropic(model=model)

    # Ollama and NVIDIA both use OpenAI-compatible API
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise LLMNotAvailableError(
            "langchain-openai required for RAGAS with Ollama/NVIDIA providers. "
            "Install with: pip install langchain-openai"
        ) from None

    base_url = provider_cfg.base_url or ""
    if provider_name == "ollama":
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        return ChatOpenAI(model=model, base_url=base_url, api_key="ollama")

    if provider_name == "nvidia":
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise LLMNotAvailableError("NVIDIA_API_KEY environment variable not set")
        return ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

    msg = f"No LangChain wrapper for provider: {provider_name!r}"
    raise LLMNotAvailableError(msg)
