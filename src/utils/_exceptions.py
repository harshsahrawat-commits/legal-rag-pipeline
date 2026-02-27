from __future__ import annotations


class LegalRAGError(Exception):
    """Root exception for the entire Legal RAG pipeline."""


class ConfigurationError(LegalRAGError):
    """Invalid or missing configuration."""


class ValidationError(LegalRAGError):
    """Data validation failed."""


class LLMError(LegalRAGError):
    """Base exception for LLM provider errors."""


class LLMNotAvailableError(LLMError):
    """Required LLM dependency not installed or provider unreachable."""


class LLMCallError(LLMError):
    """An LLM API call failed."""
