from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.acquisition._models import RawDocument
    from src.parsing._models import ParsedDocument, ParserType


class BaseParser(abc.ABC):
    """Abstract base for all document parsers.

    Each parser transforms raw content (HTML string or PDF file path)
    into a ParsedDocument. Parsers are stateless — one instance can
    parse multiple documents.
    """

    @abc.abstractmethod
    def parse(
        self,
        content_path: Path,
        raw_doc: RawDocument,
    ) -> ParsedDocument:
        """Parse a single document.

        Args:
            content_path: Path to the raw content file (HTML or PDF).
            raw_doc: The RawDocument metadata from Phase 1.

        Returns:
            ParsedDocument with structural extraction and metadata.
        """

    @abc.abstractmethod
    def can_parse(self, raw_doc: RawDocument) -> bool:
        """Check if this parser can handle the given document."""

    @property
    @abc.abstractmethod
    def parser_type(self) -> ParserType:
        """Return the ParserType enum value for this parser."""
