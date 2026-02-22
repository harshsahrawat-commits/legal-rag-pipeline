from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class AcquisitionError(LegalRAGError):
    """Base exception for the acquisition module."""


class ScrapingError(AcquisitionError):
    """Error during document scraping."""


class RateLimitError(AcquisitionError):
    """Rate limit was exceeded for a source."""


class SourceUnreachableError(AcquisitionError):
    """A configured source is not reachable."""


class ContentValidationError(AcquisitionError):
    """Downloaded content failed validation checks."""
