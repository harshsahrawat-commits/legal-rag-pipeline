"""Tiered chunker router.

Routes each ``ParsedDocument`` to the best available chunker based on
document type, structure quality, and OCR confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import ChunkingSettings
    from src.chunking._token_counter import TokenCounter
    from src.chunking.chunkers._base import BaseChunker
    from src.parsing._models import ParsedDocument

_log = get_logger(__name__)


class ChunkerRouter:
    """Select the best chunker for a given document.

    Chunkers are registered in priority order. The router evaluates each
    chunker's ``can_chunk()`` predicate and returns the first match.
    """

    def __init__(self, settings: ChunkingSettings, token_counter: TokenCounter) -> None:
        self._settings = settings
        self._tc = token_counter
        self._chunkers: list[BaseChunker] = []

    def register(self, chunker: BaseChunker) -> None:
        """Register a chunker. Call in priority order (highest first)."""
        self._chunkers.append(chunker)

    def select(self, doc: ParsedDocument) -> BaseChunker:
        """Select the best chunker for *doc*.

        Evaluates registered chunkers in order. Returns the first whose
        ``can_chunk(doc)`` returns True. Falls through to the last registered
        chunker (expected to be PageLevelChunker, which always returns True).

        Raises:
            ValueError: If no chunker can handle the document.
        """
        for chunker in self._chunkers:
            try:
                if chunker.can_chunk(doc):
                    _log.debug(
                        "chunker_selected",
                        strategy=chunker.strategy.value,
                        doc_type=doc.document_type.value,
                    )
                    return chunker
            except Exception:
                _log.warning(
                    "chunker_check_failed",
                    strategy=chunker.strategy.value,
                    exc_info=True,
                )
                continue

        msg = f"No chunker available for document type={doc.document_type}"
        raise ValueError(msg)

    @property
    def chunkers(self) -> list[BaseChunker]:
        """Read-only view of registered chunkers."""
        return list(self._chunkers)
