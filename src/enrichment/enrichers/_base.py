from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.enrichment._models import EnrichmentSettings
    from src.parsing._models import ParsedDocument


class BaseEnricher(abc.ABC):
    """Abstract base for all enrichment stages.

    Enrichers are pure transforms: they accept chunks and a parsed document
    in memory and return enriched chunks. I/O is handled by the pipeline.
    """

    def __init__(self, settings: EnrichmentSettings) -> None:
        self._settings = settings

    @abc.abstractmethod
    async def enrich_document(
        self,
        chunks: list[LegalChunk],
        parsed_doc: ParsedDocument,
    ) -> list[LegalChunk]:
        """Enrich all chunks for a single document.

        Args:
            chunks: Chunks produced by Phase 3 for this document.
            parsed_doc: The full parsed document (for full-text context).

        Returns:
            The same chunks with enrichment fields populated.
        """

    @property
    @abc.abstractmethod
    def stage_name(self) -> str:
        """Identifier for this enrichment stage (e.g. 'contextual_retrieval')."""
