from __future__ import annotations

from src.chunking._exceptions import (
    ChunkerNotAvailableError,
    ChunkingError,
    DocumentStructureError,
    MetadataBuildError,
    TokenLimitExceededError,
)
from src.utils._exceptions import LegalRAGError


class TestExceptionHierarchy:
    def test_chunking_error_is_legal_rag_error(self):
        assert issubclass(ChunkingError, LegalRAGError)

    def test_token_limit_exceeded_is_chunking_error(self):
        assert issubclass(TokenLimitExceededError, ChunkingError)

    def test_chunker_not_available_is_chunking_error(self):
        assert issubclass(ChunkerNotAvailableError, ChunkingError)

    def test_document_structure_error_is_chunking_error(self):
        assert issubclass(DocumentStructureError, ChunkingError)

    def test_metadata_build_error_is_chunking_error(self):
        assert issubclass(MetadataBuildError, ChunkingError)

    def test_exception_message(self):
        exc = TokenLimitExceededError("chunk exceeds 1500 tokens")
        assert str(exc) == "chunk exceeds 1500 tokens"

    def test_catch_as_legal_rag_error(self):
        with __import__("pytest").raises(LegalRAGError):
            raise ChunkingError("test")

    def test_each_exception_is_distinct(self):
        classes = {
            ChunkingError,
            TokenLimitExceededError,
            ChunkerNotAvailableError,
            DocumentStructureError,
            MetadataBuildError,
        }
        assert len(classes) == 5
