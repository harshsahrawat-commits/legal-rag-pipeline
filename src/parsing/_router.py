from __future__ import annotations

from typing import TYPE_CHECKING

from src.parsing._exceptions import UnsupportedFormatError
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.acquisition._models import RawDocument
    from src.parsing._models import ParsingSettings
    from src.parsing.parsers._base import BaseParser

_log = get_logger(__name__)


class ParserRouter:
    """Selects the appropriate parser based on document source and format.

    Parsers are registered in priority order. The first parser that
    reports ``can_parse(raw_doc) == True`` is selected.
    """

    def __init__(self, settings: ParsingSettings) -> None:
        self._settings = settings
        self._parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        """Register a parser. Earlier registrations have higher priority."""
        self._parsers.append(parser)
        _log.info("parser_registered", parser_type=parser.parser_type)

    @property
    def available_parsers(self) -> list[BaseParser]:
        """Return the list of registered parsers."""
        return list(self._parsers)

    def select_parser(self, raw_doc: RawDocument) -> BaseParser:
        """Select the best available parser for a document.

        Raises:
            UnsupportedFormatError: If no parser can handle the document.
        """
        for parser in self._parsers:
            if parser.can_parse(raw_doc):
                _log.debug(
                    "parser_selected",
                    parser_type=parser.parser_type,
                    source_type=raw_doc.source_type,
                    content_format=raw_doc.content_format,
                )
                return parser

        raise UnsupportedFormatError(
            f"No parser available for source_type={raw_doc.source_type}, "
            f"content_format={raw_doc.content_format}"
        )
