from __future__ import annotations

from src.enrichment._exceptions import (
    ContextualRetrievalError,
    DocumentTextTooLargeError,
    EnricherNotAvailableError,
    EnrichmentError,
    LLMRateLimitError,
    QuIMGenerationError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    def test_enrichment_error_is_legal_rag_error(self):
        assert issubclass(EnrichmentError, LegalRAGError)

    def test_contextual_retrieval_error_is_enrichment_error(self):
        assert issubclass(ContextualRetrievalError, EnrichmentError)

    def test_quim_generation_error_is_enrichment_error(self):
        assert issubclass(QuIMGenerationError, EnrichmentError)

    def test_enricher_not_available_error_is_enrichment_error(self):
        assert issubclass(EnricherNotAvailableError, EnrichmentError)

    def test_llm_rate_limit_error_is_enrichment_error(self):
        assert issubclass(LLMRateLimitError, EnrichmentError)

    def test_document_text_too_large_error_is_enrichment_error(self):
        assert issubclass(DocumentTextTooLargeError, EnrichmentError)

    def test_all_exceptions_carry_message(self):
        exc = EnrichmentError("test message")
        assert str(exc) == "test message"

    def test_all_exceptions_are_catchable_as_base(self):
        for exc_cls in (
            ContextualRetrievalError,
            QuIMGenerationError,
            EnricherNotAvailableError,
            LLMRateLimitError,
            DocumentTextTooLargeError,
        ):
            try:
                raise exc_cls("test")
            except LegalRAGError:
                pass  # Expected
