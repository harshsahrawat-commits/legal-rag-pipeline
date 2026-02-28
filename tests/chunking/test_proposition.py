from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._exceptions import ChunkerNotAvailableError
from src.chunking._models import ChunkingSettings, ChunkStrategy, ChunkType
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._proposition import (
    PropositionChunker,
    _collect_definition_text,
    _find_definition_sections,
    _has_definitions,
)
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)
from src.utils._llm_client import LLMResponse


def _make_statute_with_definitions() -> ParsedDocument:
    sections = [
        ParsedSection(
            id="s2",
            level=SectionLevel.SECTION,
            number="2",
            title="Interpretation clause",
            text="In this Act the following words are used:",
            children=[
                ParsedSection(
                    id="s2_a",
                    level=SectionLevel.DEFINITION,
                    number="(a)",
                    text=(
                        "When one person signifies to another his willingness "
                        'to do anything, he is said to make a "proposal";'
                    ),
                    parent_id="s2",
                ),
                ParsedSection(
                    id="s2_b",
                    level=SectionLevel.DEFINITION,
                    number="(b)",
                    text=(
                        "When the person to whom the proposal is made signifies "
                        'his assent thereto, the proposal is said to be "accepted";'
                    ),
                    parent_id="s2",
                ),
            ],
        ),
        ParsedSection(id="s10", level=SectionLevel.SECTION, number="10", text="Section 10."),
        ParsedSection(id="s11", level=SectionLevel.SECTION, number="11", text="Section 11."),
    ]
    return ParsedDocument(
        document_id=uuid4(),
        source_type=SourceType.INDIAN_KANOON,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.HTML,
        raw_text="dummy",
        sections=sections,
        act_name="Indian Contract Act",
        act_number="9 of 1872",
        parser_used=ParserType.HTML_INDIAN_KANOON,
        quality=QualityReport(overall_score=0.9, passed=True),
        raw_content_path="data/raw/test.html",
    )


def _make_mock_provider(response_text: str) -> MagicMock:
    """Create a mock LLM provider that returns predefined text."""
    provider = MagicMock()
    response = LLMResponse(text=response_text, model="mock", provider="mock")
    provider.complete.return_value = response
    provider.is_available = True
    provider.provider_name = "mock"
    return provider


class TestStrategy:
    def test_strategy_value(self):
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.PROPOSITION


class TestCanChunk:
    def test_with_definitions(self):
        doc = _make_statute_with_definitions()
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(doc) is True

    def test_without_definitions(self, sample_judgment_doc: ParsedDocument):
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_judgment_doc) is False

    def test_no_sections(self, sample_unstructured_doc: ParsedDocument):
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_unstructured_doc) is False


class TestChunkWithMockedProvider:
    def test_decomposes_definitions(self):
        doc = _make_statute_with_definitions()
        response = (
            'Under Section 2(a) of the Indian Contract Act, 1872, a "proposal" '
            "means when one person signifies to another his willingness to do anything.\n"
            'Under Section 2(b) of the Indian Contract Act, 1872, a proposal is "accepted" '
            "when the person to whom the proposal is made signifies his assent thereto."
        )
        provider = _make_mock_provider(response)
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        chunker._provider = provider
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert all(c.chunk_type == ChunkType.DEFINITION for c in chunks)
        assert "proposal" in chunks[0].text
        assert "accepted" in chunks[1].text

    def test_metadata_populated(self):
        doc = _make_statute_with_definitions()
        provider = _make_mock_provider("Definition one.\nDefinition two.")
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        chunker._provider = provider
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.statute is not None
            assert c.statute.act_name == "Indian Contract Act"
            assert c.ingestion.chunk_strategy == ChunkStrategy.PROPOSITION
            assert c.token_count > 0
            assert c.document_id == doc.document_id

    def test_empty_response(self):
        doc = _make_statute_with_definitions()
        provider = _make_mock_provider("")
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        chunker._provider = provider
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_provider_called_with_max_tokens(self):
        doc = _make_statute_with_definitions()
        provider = _make_mock_provider("Def one.\nDef two.")
        settings = ChunkingSettings(proposition_model="claude-haiku-4-5-20251001")
        chunker = PropositionChunker(settings, TokenCounter())
        chunker._provider = provider
        chunker.chunk(doc)
        call_kwargs = provider.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == settings.proposition_max_tokens_response

    def test_unique_ids(self):
        doc = _make_statute_with_definitions()
        provider = _make_mock_provider("A.\nB.\nC.")
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        chunker._provider = provider
        chunks = chunker.chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestMissingDependency:
    def test_raises_when_provider_unavailable(self):
        from src.utils._exceptions import LLMNotAvailableError

        doc = _make_statute_with_definitions()
        chunker = PropositionChunker(ChunkingSettings(), TokenCounter())
        with (
            patch(
                "src.chunking.chunkers._proposition.get_llm_provider",
                side_effect=LLMNotAvailableError("no provider"),
            ),
            pytest.raises(ChunkerNotAvailableError, match="LLM provider"),
        ):
            chunker._provider = None
            chunker.chunk(doc)


class TestHelpers:
    def test_has_definitions_true(self):
        sections = [
            ParsedSection(
                id="s2",
                level=SectionLevel.SECTION,
                text="defs",
                children=[
                    ParsedSection(id="d1", level=SectionLevel.DEFINITION, text="def 1"),
                ],
            ),
        ]
        assert _has_definitions(sections) is True

    def test_has_definitions_false(self):
        sections = [
            ParsedSection(id="s1", level=SectionLevel.SECTION, text="no defs"),
        ]
        assert _has_definitions(sections) is False

    def test_find_definition_sections(self):
        sections = [
            ParsedSection(
                id="s2",
                level=SectionLevel.SECTION,
                number="2",
                text="defs",
                children=[
                    ParsedSection(id="d1", level=SectionLevel.DEFINITION, text="def 1"),
                    ParsedSection(id="d2", level=SectionLevel.DEFINITION, text="def 2"),
                ],
            ),
            ParsedSection(id="s3", level=SectionLevel.SECTION, text="no defs"),
        ]
        result = _find_definition_sections(sections)
        assert len(result) == 1
        assert result[0].id == "s2"

    def test_collect_definition_text(self):
        section = ParsedSection(
            id="s2",
            level=SectionLevel.SECTION,
            text="Interpretation:",
            children=[
                ParsedSection(id="d1", level=SectionLevel.DEFINITION, number="(a)", text="Def A."),
                ParsedSection(id="d2", level=SectionLevel.DEFINITION, number="(b)", text="Def B."),
            ],
        )
        text = _collect_definition_text(section)
        assert "Interpretation:" in text
        assert "(a) Def A." in text
        assert "(b) Def B." in text
