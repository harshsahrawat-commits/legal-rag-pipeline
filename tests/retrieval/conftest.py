"""Shared fixtures for retrieval tests."""

from __future__ import annotations

from typing import Any

import pytest

from src.retrieval._models import (
    ExpandedContext,
    FusedChunk,
    QueryRoute,
    RetrievalQuery,
    RetrievalSettings,
    ScoredChunk,
)


@pytest.fixture()
def retrieval_settings() -> RetrievalSettings:
    """Default retrieval settings for tests."""
    return RetrievalSettings()


@pytest.fixture()
def sample_query() -> RetrievalQuery:
    """A basic retrieval query for testing."""
    return RetrievalQuery(text="What is Section 420 of IPC?")


@pytest.fixture()
def sample_query_with_embedding() -> RetrievalQuery:
    """A query with pre-computed embeddings."""
    return RetrievalQuery(
        text="What is Section 420 of IPC?",
        query_embedding=[0.1] * 768,
        query_embedding_fast=[0.2] * 64,
        route=QueryRoute.STANDARD,
    )


@pytest.fixture()
def sample_scored_chunks() -> list[ScoredChunk]:
    """Sample scored chunks from different channels."""
    return [
        ScoredChunk(
            chunk_id="chunk-1",
            text="Section 420 deals with cheating.",
            score=0.95,
            channel="dense",
            document_type="statute",
            chunk_type="statutory_text",
            payload=_sample_payload("chunk-1"),
        ),
        ScoredChunk(
            chunk_id="chunk-2",
            text="Cheating is defined under Section 415.",
            score=0.88,
            channel="dense",
            document_type="statute",
            chunk_type="definition",
            payload=_sample_payload("chunk-2"),
        ),
        ScoredChunk(
            chunk_id="chunk-3",
            text="The court held that mens rea is essential for S.420.",
            score=0.82,
            channel="bm25",
            document_type="judgment",
            chunk_type="reasoning",
            payload=_sample_payload("chunk-3"),
        ),
    ]


@pytest.fixture()
def sample_fused_chunks() -> list[FusedChunk]:
    """Sample fused chunks after RRF."""
    return [
        FusedChunk(
            chunk_id="chunk-1",
            text="Section 420 deals with cheating.",
            rrf_score=0.032,
            channels=["dense", "bm25"],
            payload=_sample_payload("chunk-1"),
        ),
        FusedChunk(
            chunk_id="chunk-2",
            text="Cheating is defined under Section 415.",
            rrf_score=0.028,
            channels=["dense"],
            payload=_sample_payload("chunk-2"),
        ),
    ]


@pytest.fixture()
def sample_expanded_contexts() -> list[ExpandedContext]:
    """Sample expanded contexts."""
    return [
        ExpandedContext(
            chunk_id="chunk-1",
            chunk_text="Section 420 deals with cheating.",
            parent_text="Chapter XVII — Of Offences Against Property...",
            relevance_score=0.95,
            total_tokens=150,
        ),
    ]


def _sample_payload(chunk_id: str) -> dict[str, Any]:
    """Build a minimal chunk payload for test fixtures."""
    return {
        "id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "text": f"Text for {chunk_id}",
        "document_type": "statute",
        "chunk_type": "statutory_text",
        "parent_info": {
            "parent_chunk_id": None,
            "judgment_header_chunk_id": None,
            "sibling_chunk_ids": [],
        },
    }
