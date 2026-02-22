from __future__ import annotations

from src.utils._exceptions import LegalRAGError


class ParsingError(LegalRAGError):
    """Base exception for the parsing module."""


class PDFDownloadError(ParsingError):
    """Failed to download a PDF from a remote URL."""


class DocumentStructureError(ParsingError):
    """Could not detect structural elements in the document."""


class QualityValidationError(ParsingError):
    """Document failed quality validation checks."""


class UnsupportedFormatError(ParsingError):
    """Document format is not supported by any available parser."""


class ParserNotAvailableError(ParsingError):
    """A required parser library is not installed."""
