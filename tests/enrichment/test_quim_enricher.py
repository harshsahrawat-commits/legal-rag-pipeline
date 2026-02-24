from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment._exceptions import EnricherNotAvailableError, LLMRateLimitError
from src.enrichment._models import EnrichmentSettings
from src.enrichment.enrichers._quim import QuIMRagEnricher, _parse_questions
from tests.enrichment.conftest import make_mock_async_anthropic

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.parsing._models import ParsedDocument

_SAMPLE_QUESTIONS = (
    "What constitutes a valid contract under Section 10?\n"
    "Who is competent to enter into a contract?\n"
    "What does free consent mean in the context of contracts?\n"
    "When can a contract be declared void?\n"
    "What are the requirements for enforceability of an agreement?"
)


class TestStageNameAndInit:
    def test_stage_name(self, enrichment_settings: EnrichmentSettings):
        enricher = QuIMRagEnricher(enrichment_settings)
        assert enricher.stage_name == "quim_rag"

    def test_client_starts_none(self, enrichment_settings: EnrichmentSettings):
        enricher = QuIMRagEnricher(enrichment_settings)
        assert enricher._client is None

    def test_last_quim_doc_starts_none(self, enrichment_settings: EnrichmentSettings):
        enricher = QuIMRagEnricher(enrichment_settings)
        assert enricher.get_quim_document() is None


class TestGenerateQuestions:
    async def test_generates_questions_for_each_chunk(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        quim_doc = enricher.get_quim_document()
        assert quim_doc is not None
        assert len(quim_doc.entries) == 3  # One per chunk

    async def test_quim_questions_count_set(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        for chunk in sample_statute_chunks:
            assert chunk.ingestion.quim_questions == 5

    async def test_questions_are_strings(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        quim_doc = enricher.get_quim_document()
        for entry in quim_doc.entries:
            for q in entry.questions:
                assert isinstance(q, str)
                assert len(q) > 0

    async def test_questions_contain_question_marks(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        quim_doc = enricher.get_quim_document()
        for entry in quim_doc.entries:
            assert any("?" in q for q in entry.questions)

    async def test_quim_document_has_correct_chunk_ids(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        quim_doc = enricher.get_quim_document()
        entry_chunk_ids = {e.chunk_id for e in quim_doc.entries}
        expected_ids = {c.id for c in sample_statute_chunks}
        assert entry_chunk_ids == expected_ids

    async def test_quim_document_has_model(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        quim_doc = enricher.get_quim_document()
        assert quim_doc.model == enrichment_settings.model

    async def test_quim_document_json_round_trip(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        quim_doc = enricher.get_quim_document()
        from src.enrichment._models import QuIMDocument

        json_str = quim_doc.model_dump_json()
        restored = QuIMDocument.model_validate_json(json_str)
        assert len(restored.entries) == len(quim_doc.entries)
        assert restored.document_id == quim_doc.document_id

    async def test_empty_chunks_returns_unchanged(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        result = await enricher.enrich_document([], sample_parsed_doc)
        assert result == []
        assert enricher.get_quim_document() is None


class TestPromptContent:
    async def test_cache_control_in_system_message(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    async def test_act_name_in_user_message_for_statute(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks[:1], sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "Indian Contract Act" in user_content

    async def test_case_citation_in_user_message_for_judgment(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_judgment_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        enricher._client = client
        await enricher.enrich_document(sample_judgment_chunks[:1], sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "AIR 2024 SC 1500" in user_content

    async def test_calls_correct_model(
        self,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        settings = EnrichmentSettings(model="custom-model", concurrency=2)
        enricher = QuIMRagEnricher(settings)
        client = make_mock_async_anthropic(_SAMPLE_QUESTIONS)
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks[:1], sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"


class TestErrorIsolation:
    async def test_llm_failure_is_isolated(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """If one chunk fails, others still get questions."""
        enricher = QuIMRagEnricher(enrichment_settings)
        client = MagicMock()

        content_ok = MagicMock()
        content_ok.text = _SAMPLE_QUESTIONS
        response_ok = MagicMock()
        response_ok.content = [content_ok]

        client.messages.create = AsyncMock(
            side_effect=[
                RuntimeError("LLM error"),
                response_ok,
                response_ok,
            ]
        )
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        # First chunk should have 0 questions
        assert sample_statute_chunks[0].ingestion.quim_questions == 0
        # Others should succeed
        assert sample_statute_chunks[1].ingestion.quim_questions == 5
        assert sample_statute_chunks[2].ingestion.quim_questions == 5

        quim_doc = enricher.get_quim_document()
        assert len(quim_doc.entries) == 2  # Two succeeded

    async def test_empty_response_yields_no_questions(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        for chunk in sample_statute_chunks:
            assert chunk.ingestion.quim_questions == 0

    async def test_rate_limit_raises(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = QuIMRagEnricher(enrichment_settings)
        client = MagicMock()
        rate_err = type("RateLimitError", (Exception,), {})("rate limited")
        client.messages.create = AsyncMock(side_effect=rate_err)
        enricher._client = client

        with pytest.raises(LLMRateLimitError, match="rate limit"):
            await enricher.enrich_document(sample_statute_chunks[:1], sample_parsed_doc)


class TestMissingDependency:
    def test_raises_when_no_anthropic(self, enrichment_settings: EnrichmentSettings):
        enricher = QuIMRagEnricher(enrichment_settings)
        with (
            patch.dict("sys.modules", {"anthropic": None}),
            pytest.raises(EnricherNotAvailableError, match="anthropic"),
        ):
            enricher._client = None
            enricher._ensure_client()


class TestParseQuestions:
    def test_basic_parsing(self):
        raw = "What is Section 10?\nWho can make a contract?\nWhen is consent free?"
        result = _parse_questions(raw)
        assert len(result) == 3

    def test_filters_blank_lines(self):
        raw = "Q1?\n\n\nQ2?\n\nQ3?"
        result = _parse_questions(raw)
        assert len(result) == 3

    def test_filters_short_non_questions(self):
        raw = "Q1?\nOK\nQ2?"
        result = _parse_questions(raw)
        assert len(result) == 2  # "OK" is too short and has no "?"

    def test_keeps_long_lines_without_question_mark(self):
        raw = "This is a statement about legal provisions that is quite long"
        result = _parse_questions(raw)
        assert len(result) == 1

    def test_empty_input(self):
        assert _parse_questions("") == []
        assert _parse_questions("   ") == []


class TestResolveHelpers:
    def test_resolve_act_for_statute(self, sample_statute_chunks: list[LegalChunk]):
        assert (
            QuIMRagEnricher._resolve_act_or_case(sample_statute_chunks[0]) == "Indian Contract Act"
        )

    def test_resolve_act_for_judgment(self, sample_judgment_chunks: list[LegalChunk]):
        assert QuIMRagEnricher._resolve_act_or_case(sample_judgment_chunks[0]) == "AIR 2024 SC 1500"

    def test_resolve_section_for_statute(self, sample_statute_chunks: list[LegalChunk]):
        assert QuIMRagEnricher._resolve_section_ref(sample_statute_chunks[0]) == "Section 10"

    def test_resolve_section_for_judgment(self, sample_judgment_chunks: list[LegalChunk]):
        assert QuIMRagEnricher._resolve_section_ref(sample_judgment_chunks[0]) == "facts"
