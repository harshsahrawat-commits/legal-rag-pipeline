from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment._exceptions import (
    EnricherNotAvailableError,
    LLMRateLimitError,
)
from src.enrichment._models import EnrichmentSettings
from src.enrichment.enrichers._contextual import ContextualRetrievalEnricher
from tests.enrichment.conftest import make_mock_async_anthropic

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.parsing._models import ParsedDocument


class TestStageNameAndInit:
    def test_stage_name(self, enrichment_settings: EnrichmentSettings):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        assert enricher.stage_name == "contextual_retrieval"

    def test_client_starts_none(self, enrichment_settings: EnrichmentSettings):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        assert enricher._client is None


class TestEnrichChunks:
    async def test_enriches_all_chunks(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(
            "This chunk is from Section 10 of the Indian Contract Act."
        )
        result = await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        assert len(result) == 3
        for chunk in result:
            assert chunk.ingestion.contextualized is True
            assert chunk.contextualized_text is not None

    async def test_contextualized_text_contains_original(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context prefix here.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        for chunk in sample_statute_chunks:
            assert chunk.text in chunk.contextualized_text

    async def test_contextualized_text_format(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Contextualized text should be: context + \\n\\n + original text."""
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context prefix.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        chunk = sample_statute_chunks[0]
        assert chunk.contextualized_text.startswith("Context prefix.")
        assert "\n\n" in chunk.contextualized_text
        assert chunk.contextualized_text.endswith(chunk.text)

    async def test_contextualized_flag_set(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        for chunk in sample_statute_chunks:
            assert chunk.ingestion.contextualized is True

    async def test_original_text_unchanged(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        original_texts = [c.text for c in sample_statute_chunks]
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        for chunk, orig in zip(sample_statute_chunks, original_texts, strict=True):
            assert chunk.text == orig


class TestPromptCaching:
    async def test_cache_control_in_system_message(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        client = make_mock_async_anthropic("Context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    async def test_system_message_contains_document_text(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        client = make_mock_async_anthropic("Context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        system_text = call_kwargs["system"][0]["text"]
        assert "<document>" in system_text
        assert "Indian Contract Act" in system_text

    async def test_user_message_contains_chunk_text(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        client = make_mock_async_anthropic("Context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "<chunk>" in user_content

    async def test_calls_correct_model(
        self,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        settings = EnrichmentSettings(model="custom-model-v1", concurrency=2)
        enricher = ContextualRetrievalEnricher(settings)
        client = make_mock_async_anthropic("Context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "custom-model-v1"

    async def test_max_tokens_response_respected(
        self,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        settings = EnrichmentSettings(max_tokens_response=256, concurrency=2)
        enricher = ContextualRetrievalEnricher(settings)
        client = make_mock_async_anthropic("Context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256


class TestErrorIsolation:
    async def test_llm_failure_is_isolated(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """If one chunk's LLM call fails, others still succeed."""
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        client = MagicMock()

        content_ok = MagicMock()
        content_ok.text = "Context for this chunk."
        response_ok = MagicMock()
        response_ok.content = [content_ok]

        # First call fails, second and third succeed
        client.messages.create = AsyncMock(
            side_effect=[
                RuntimeError("LLM unavailable"),
                response_ok,
                response_ok,
            ]
        )
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        # First chunk should NOT be contextualized
        assert sample_statute_chunks[0].ingestion.contextualized is False
        assert sample_statute_chunks[0].contextualized_text is None
        # Other chunks should succeed
        assert sample_statute_chunks[1].ingestion.contextualized is True
        assert sample_statute_chunks[2].ingestion.contextualized is True

    async def test_rate_limit_raises(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Rate limit errors should propagate as LLMRateLimitError."""
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        client = MagicMock()

        # Simulate a rate limit error with the right class name
        rate_err = type("RateLimitError", (Exception,), {})("rate limited")
        client.messages.create = AsyncMock(side_effect=rate_err)
        enricher._client = client

        with pytest.raises(LLMRateLimitError, match="rate limit"):
            await enricher.enrich_document(sample_statute_chunks[:1], sample_parsed_doc)

    async def test_empty_llm_response_handled(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Empty or whitespace-only LLM response leaves chunk unenriched."""
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("   ")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        for chunk in sample_statute_chunks:
            assert chunk.ingestion.contextualized is False
            assert chunk.contextualized_text is None


class TestMissingDependency:
    def test_raises_when_no_anthropic(self, enrichment_settings: EnrichmentSettings):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        with (
            patch.dict("sys.modules", {"anthropic": None}),
            pytest.raises(EnricherNotAvailableError, match="anthropic"),
        ):
            enricher._client = None
            enricher._ensure_client()


class TestDocumentWindowing:
    async def test_short_doc_single_window(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Documents under context_window_tokens use a single window."""
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context.")
        windows = enricher._build_windows(sample_parsed_doc.raw_text)
        assert len(windows) == 1

    async def test_long_doc_multiple_windows(
        self,
        enrichment_settings: EnrichmentSettings,
    ):
        """Documents exceeding context_window_tokens get split."""
        settings = EnrichmentSettings(
            context_window_tokens=10,  # Very low to trigger windowing
            document_window_overlap_tokens=2,
            concurrency=2,
        )
        enricher = ContextualRetrievalEnricher(settings)
        # ~100 tokens of text
        long_text = " ".join(["word"] * 100)
        windows = enricher._build_windows(long_text)
        assert len(windows) > 1

    async def test_windowed_chunks_all_assigned(
        self,
        sample_statute_chunks: list[LegalChunk],
    ):
        """Every chunk gets assigned to exactly one window."""
        settings = EnrichmentSettings(
            context_window_tokens=10,
            document_window_overlap_tokens=2,
            concurrency=2,
        )
        enricher = ContextualRetrievalEnricher(settings)
        text = " ".join(["word"] * 100)
        windows = enricher._build_windows(text)
        mapping = enricher._assign_chunks_to_windows(sample_statute_chunks, text, len(windows))
        assigned_chunks = []
        for chunk_list in mapping.values():
            assigned_chunks.extend(chunk_list)
        assert len(assigned_chunks) == len(sample_statute_chunks)


class TestSkipBehavior:
    async def test_skips_already_contextualized(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Chunks already contextualized should be skipped."""
        for chunk in sample_statute_chunks:
            chunk.ingestion.contextualized = True
            chunk.contextualized_text = "Already done.\n\n" + chunk.text

        enricher = ContextualRetrievalEnricher(enrichment_settings)
        client = make_mock_async_anthropic("New context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        # LLM should NOT have been called
        client.messages.create.assert_not_called()

    async def test_skips_manual_review_when_configured(
        self,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """When skip_manual_review_chunks=True, manual review chunks are skipped."""
        settings = EnrichmentSettings(skip_manual_review_chunks=True, concurrency=2)
        sample_statute_chunks[0].ingestion.requires_manual_review = True

        enricher = ContextualRetrievalEnricher(settings)
        client = make_mock_async_anthropic("Context.")
        enricher._client = client
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        # First chunk skipped, other two enriched
        assert sample_statute_chunks[0].ingestion.contextualized is False
        assert sample_statute_chunks[1].ingestion.contextualized is True
        assert sample_statute_chunks[2].ingestion.contextualized is True

    async def test_does_not_skip_manual_review_by_default(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """By default, manual review chunks ARE enriched."""
        sample_statute_chunks[0].ingestion.requires_manual_review = True

        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)

        assert sample_statute_chunks[0].ingestion.contextualized is True

    async def test_empty_chunks_returns_unchanged(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context.")
        result = await enricher.enrich_document([], sample_parsed_doc)
        assert result == []


class TestJudgmentChunks:
    async def test_enriches_judgment_chunks(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_judgment_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic(
            "This chunk is from the facts section of AIR 2024 SC 1500."
        )
        await enricher.enrich_document(sample_judgment_chunks, sample_parsed_doc)
        for chunk in sample_judgment_chunks:
            assert chunk.ingestion.contextualized is True
            assert "AIR 2024 SC 1500" in chunk.contextualized_text


class TestConcurrency:
    async def test_respects_semaphore(
        self,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        """Verify concurrency limit is applied (calls still complete)."""
        settings = EnrichmentSettings(concurrency=1)
        enricher = ContextualRetrievalEnricher(settings)
        enricher._client = make_mock_async_anthropic("Context.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        # All chunks should still be enriched despite concurrency=1
        assert all(c.ingestion.contextualized for c in sample_statute_chunks)


class TestChunkIds:
    async def test_chunk_ids_preserved(
        self,
        enrichment_settings: EnrichmentSettings,
        sample_statute_chunks: list[LegalChunk],
        sample_parsed_doc: ParsedDocument,
    ):
        original_ids = [c.id for c in sample_statute_chunks]
        enricher = ContextualRetrievalEnricher(enrichment_settings)
        enricher._client = make_mock_async_anthropic("Context.")
        await enricher.enrich_document(sample_statute_chunks, sample_parsed_doc)
        assert [c.id for c in sample_statute_chunks] == original_ids
