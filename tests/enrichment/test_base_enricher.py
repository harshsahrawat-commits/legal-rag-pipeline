from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.enrichment._models import EnrichmentSettings
from src.enrichment.enrichers._base import BaseEnricher

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.parsing._models import ParsedDocument


class ConcreteEnricher(BaseEnricher):
    """Minimal concrete subclass for testing."""

    @property
    def stage_name(self) -> str:
        return "test_stage"

    async def enrich_document(
        self,
        chunks: list[LegalChunk],
        parsed_doc: ParsedDocument,
    ) -> list[LegalChunk]:
        return chunks


class TestBaseEnricher:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseEnricher(EnrichmentSettings())

    def test_concrete_subclass_instantiates(self):
        enricher = ConcreteEnricher(EnrichmentSettings())
        assert enricher is not None

    def test_stage_name_property(self):
        enricher = ConcreteEnricher(EnrichmentSettings())
        assert enricher.stage_name == "test_stage"

    def test_settings_stored(self):
        settings = EnrichmentSettings(concurrency=20)
        enricher = ConcreteEnricher(settings)
        assert enricher._settings.concurrency == 20

    async def test_enrich_document_returns_chunks(
        self,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ConcreteEnricher(EnrichmentSettings())
        result = await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        assert result is sample_statute_chunks

    def test_missing_stage_name_raises(self):
        class NoStageName(BaseEnricher):
            async def enrich_document(self, chunks, parsed_doc):
                return chunks

        with pytest.raises(TypeError, match="abstract"):
            NoStageName(EnrichmentSettings())

    def test_missing_enrich_document_raises(self):
        class NoEnrichMethod(BaseEnricher):
            @property
            def stage_name(self) -> str:
                return "test"

        with pytest.raises(TypeError, match="abstract"):
            NoEnrichMethod(EnrichmentSettings())

    async def test_enrich_document_with_empty_chunks(self, sample_parsed_doc: ParsedDocument):
        enricher = ConcreteEnricher(EnrichmentSettings())
        result = await enricher.enrich_document([], sample_parsed_doc)
        assert result == []
