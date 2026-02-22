from __future__ import annotations


class LegalRAGError(Exception):
    """Root exception for the entire Legal RAG pipeline."""


class ConfigurationError(LegalRAGError):
    """Invalid or missing configuration."""


class ValidationError(LegalRAGError):
    """Data validation failed."""
