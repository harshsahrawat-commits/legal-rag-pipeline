"""Tests for retrieval searcher classes: Dense, Sparse, QuIM, Graph."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval._exceptions import SearchError, SearchNotAvailableError
from src.retrieval._models import (
    RetrievalSettings,
    ScoredChunk,
)

# ---------------------------------------------------------------------------
# Mock Qdrant module — shared across Dense, Sparse, QuIM tests
# ---------------------------------------------------------------------------


def _build_mock_qdrant_module():
    """Build a fake qdrant_client module with required model classes."""
    mod = ModuleType("qdrant_client")
    models_mod = ModuleType("qdrant_client.models")

    models_mod.FieldCondition = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.MatchAny = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.Filter = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.NamedSparseVector = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.SparseVector = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models_mod.SearchRequest = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    mod.models = models_mod
    mod.QdrantClient = MagicMock
    return mod, models_mod


@pytest.fixture(autouse=True)
def _mock_qdrant_module():
    """Inject a fake qdrant_client module for all tests in this file."""
    mod, models_mod = _build_mock_qdrant_module()
    with patch.dict(sys.modules, {"qdrant_client": mod, "qdrant_client.models": models_mod}):
        yield


# ---------------------------------------------------------------------------
# Helper — build mock ScoredPoint
# ---------------------------------------------------------------------------


def _mock_point(
    point_id: str = "chunk-1",
    score: float = 0.95,
    text: str = "Section 420 deals with cheating.",
    document_type: str = "statute",
    chunk_type: str = "statutory_text",
    extra_payload: dict | None = None,
) -> MagicMock:
    """Create a mock Qdrant ScoredPoint."""
    pt = MagicMock()
    pt.id = point_id
    pt.score = score
    payload = {
        "id": point_id,
        "text": text,
        "document_type": document_type,
        "chunk_type": chunk_type,
        "parent_info": {
            "parent_chunk_id": None,
            "judgment_header_chunk_id": None,
            "sibling_chunk_ids": [],
        },
    }
    if extra_payload:
        payload.update(extra_payload)
    pt.payload = payload
    return pt


# ===================================================================
# DenseSearcher
# ===================================================================


class TestDenseSearcher:
    """Tests for the 2-stage Matryoshka funnel search."""

    def _make_searcher(self, mock_client: MagicMock) -> object:
        from src.retrieval._searchers import DenseSearcher

        s = DenseSearcher(RetrievalSettings())
        s._client = mock_client
        return s

    def test_successful_two_stage_search(self) -> None:
        client = MagicMock()
        # Stage 1 returns 2 candidates
        stage1 = [_mock_point("c-1", 0.9), _mock_point("c-2", 0.85)]
        # Stage 2 returns 1 refined result
        stage2 = [_mock_point("c-1", 0.95)]
        client.search.side_effect = [stage1, stage2]

        searcher = self._make_searcher(client)
        results = searcher.search(
            embedding_full=[0.1] * 768,
            embedding_fast=[0.2] * 64,
            top_k_fast=100,
            top_k_full=10,
        )
        assert len(results) == 1
        assert results[0].chunk_id == "c-1"
        assert results[0].channel == "dense"
        assert results[0].score == 0.95

    def test_empty_stage1_returns_empty(self) -> None:
        client = MagicMock()
        client.search.return_value = []

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768, [0.2] * 64)
        assert results == []
        # Only one search call (stage 1), no stage 2
        assert client.search.call_count == 1

    def test_stage1_error_raises_search_error(self) -> None:
        client = MagicMock()
        client.search.side_effect = RuntimeError("connection refused")

        searcher = self._make_searcher(client)
        with pytest.raises(SearchError, match="stage-1 failed"):
            searcher.search([0.1] * 768, [0.2] * 64)

    def test_stage2_error_raises_search_error(self) -> None:
        client = MagicMock()
        client.search.side_effect = [
            [_mock_point("c-1")],
            RuntimeError("timeout"),
        ]

        searcher = self._make_searcher(client)
        with pytest.raises(SearchError, match="stage-2 failed"):
            searcher.search([0.1] * 768, [0.2] * 64)

    def test_multiple_results_from_stage2(self) -> None:
        client = MagicMock()
        pts = [_mock_point(f"c-{i}", 0.9 - i * 0.1) for i in range(5)]
        client.search.side_effect = [pts, pts[:3]]

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768, [0.2] * 64)
        assert len(results) == 3
        assert all(r.channel == "dense" for r in results)

    def test_missing_dep_raises_not_available(self) -> None:
        from src.retrieval._searchers import DenseSearcher

        searcher = DenseSearcher(RetrievalSettings())
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(SearchNotAvailableError, match="qdrant-client"),
        ):
            searcher._ensure_client()

    def test_result_carries_document_type(self) -> None:
        client = MagicMock()
        pt = _mock_point("j-1", 0.92, document_type="judgment", chunk_type="reasoning")
        client.search.side_effect = [[pt], [pt]]

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768, [0.2] * 64)
        assert results[0].document_type == "judgment"
        assert results[0].chunk_type == "reasoning"

    def test_result_is_scored_chunk(self) -> None:
        client = MagicMock()
        client.search.side_effect = [[_mock_point()], [_mock_point()]]

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768, [0.2] * 64)
        assert isinstance(results[0], ScoredChunk)

    def test_stage2_none_returns_empty(self) -> None:
        """If Qdrant returns None for stage 2, treat as empty."""
        client = MagicMock()
        client.search.side_effect = [[_mock_point()], None]

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768, [0.2] * 64)
        assert results == []


# ===================================================================
# SparseSearcher
# ===================================================================


class TestSparseSearcher:
    """Tests for BM25 sparse vector search."""

    def _make_searcher(self, mock_client: MagicMock) -> object:
        from src.retrieval._searchers import SparseSearcher

        s = SparseSearcher(RetrievalSettings())
        s._client = mock_client
        return s

    def test_successful_search(self) -> None:
        client = MagicMock()
        client.search.return_value = [
            _mock_point("c-1", 0.88),
            _mock_point("c-2", 0.72),
        ]

        searcher = self._make_searcher(client)
        results = searcher.search(
            sparse_indices=[10, 42, 99],
            sparse_values=[1.5, 0.8, 2.1],
        )
        assert len(results) == 2
        assert results[0].channel == "bm25"
        assert results[0].chunk_id == "c-1"

    def test_empty_indices_returns_empty(self) -> None:
        client = MagicMock()

        searcher = self._make_searcher(client)
        results = searcher.search(sparse_indices=[], sparse_values=[])
        assert results == []
        # Client should not be called
        client.search.assert_not_called()

    def test_qdrant_error_raises_search_error(self) -> None:
        client = MagicMock()
        client.search.side_effect = RuntimeError("Qdrant down")

        searcher = self._make_searcher(client)
        with pytest.raises(SearchError, match="BM25 sparse search failed"):
            searcher.search([10], [1.5])

    def test_missing_dep_raises_not_available(self) -> None:
        from src.retrieval._searchers import SparseSearcher

        searcher = SparseSearcher(RetrievalSettings())
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(SearchNotAvailableError, match="qdrant-client"),
        ):
            searcher._ensure_client()

    def test_result_is_scored_chunk(self) -> None:
        client = MagicMock()
        client.search.return_value = [_mock_point()]

        searcher = self._make_searcher(client)
        results = searcher.search([0], [1.0])
        assert isinstance(results[0], ScoredChunk)

    def test_top_k_passed_to_qdrant(self) -> None:
        client = MagicMock()
        client.search.return_value = []

        searcher = self._make_searcher(client)
        searcher.search([0], [1.0], top_k=42)
        _, kwargs = client.search.call_args
        assert kwargs["limit"] == 42

    def test_none_result_treated_as_empty(self) -> None:
        client = MagicMock()
        client.search.return_value = None

        searcher = self._make_searcher(client)
        results = searcher.search([10], [1.5])
        assert results == []


# ===================================================================
# QuIMSearcher
# ===================================================================


class TestQuIMSearcher:
    """Tests for QuIM question-based search."""

    def _make_searcher(self, mock_client: MagicMock) -> object:
        from src.retrieval._searchers import QuIMSearcher

        s = QuIMSearcher(RetrievalSettings())
        s._client = mock_client
        return s

    def test_successful_search(self) -> None:
        client = MagicMock()
        pt = MagicMock()
        pt.id = "chunk-1-q0"
        pt.score = 0.91
        pt.payload = {
            "source_chunk_id": "chunk-1",
            "document_id": "doc-1",
            "question": "What is Section 420?",
        }
        client.search.return_value = [pt]

        searcher = self._make_searcher(client)
        results = searcher.search(embedding=[0.1] * 768)
        assert len(results) == 1
        assert results[0].channel == "quim"
        assert results[0].chunk_id == "chunk-1"
        assert results[0].text == "What is Section 420?"
        assert results[0].score == 0.91

    def test_maps_source_chunk_id_correctly(self) -> None:
        client = MagicMock()
        pt = MagicMock()
        pt.id = "c5-q2"
        pt.score = 0.85
        pt.payload = {
            "source_chunk_id": "original-chunk-5",
            "question": "Q?",
        }
        client.search.return_value = [pt]

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768)
        assert results[0].chunk_id == "original-chunk-5"

    def test_empty_results(self) -> None:
        client = MagicMock()
        client.search.return_value = []

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768)
        assert results == []

    def test_qdrant_error_raises_search_error(self) -> None:
        client = MagicMock()
        client.search.side_effect = RuntimeError("connection refused")

        searcher = self._make_searcher(client)
        with pytest.raises(SearchError, match="QuIM search failed"):
            searcher.search([0.1] * 768)

    def test_missing_dep_raises_not_available(self) -> None:
        from src.retrieval._searchers import QuIMSearcher

        searcher = QuIMSearcher(RetrievalSettings())
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(SearchNotAvailableError, match="qdrant-client"),
        ):
            searcher._ensure_client()

    def test_multiple_results(self) -> None:
        client = MagicMock()
        pts = []
        for i in range(3):
            pt = MagicMock()
            pt.id = f"q-{i}"
            pt.score = 0.9 - i * 0.1
            pt.payload = {
                "source_chunk_id": f"chunk-{i}",
                "question": f"Question {i}?",
            }
            pts.append(pt)
        client.search.return_value = pts

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768, top_k=3)
        assert len(results) == 3
        assert [r.chunk_id for r in results] == ["chunk-0", "chunk-1", "chunk-2"]

    def test_searches_quim_collection(self) -> None:
        client = MagicMock()
        client.search.return_value = []
        settings = RetrievalSettings(quim_collection="my_quim")

        from src.retrieval._searchers import QuIMSearcher

        searcher = QuIMSearcher(settings)
        searcher._client = client
        searcher.search([0.1] * 768)
        _, kwargs = client.search.call_args
        assert kwargs["collection_name"] == "my_quim"

    def test_fallback_chunk_id_from_point_id(self) -> None:
        """When source_chunk_id is missing, fall back to point.id."""
        client = MagicMock()
        pt = MagicMock()
        pt.id = "fallback-id"
        pt.score = 0.5
        pt.payload = {"question": "Q?"}
        client.search.return_value = [pt]

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768)
        assert results[0].chunk_id == "fallback-id"

    def test_none_result_treated_as_empty(self) -> None:
        client = MagicMock()
        client.search.return_value = None

        searcher = self._make_searcher(client)
        results = searcher.search([0.1] * 768)
        assert results == []


# ===================================================================
# GraphSearcher
# ===================================================================


class TestGraphSearcher:
    """Tests for knowledge-graph-backed search."""

    @pytest.fixture()
    def _mock_kg_modules(self):
        """Mock the knowledge_graph imports for GraphSearcher."""
        with (
            patch("src.retrieval._searchers.GraphSearcher._ensure_client_async") as mock_ensure,
        ):
            mock_ensure.return_value = None
            yield

    @pytest.mark.asyncio
    async def test_extracts_section_ref_and_queries_kg(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.return_value = [
            {"citation": "AIR 2020 SC 123", "court": "Supreme Court", "date_decided": "2020-01-15"},
        ]
        mock_qb.hierarchy_navigation.return_value = [
            {"number": "420", "chapter": "XVII", "is_in_force": True},
        ]

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        results = await searcher.search("Section 420 of Indian Penal Code")
        assert len(results) >= 1
        channels = {r.channel for r in results}
        assert channels == {"graph"}

    @pytest.mark.asyncio
    async def test_no_references_returns_empty(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = AsyncMock()
        searcher._client = MagicMock()

        results = await searcher.search("What is the meaning of life?")
        assert results == []

    @pytest.mark.asyncio
    async def test_abbreviation_expanded(self) -> None:
        """S. 302 IPC should expand IPC -> Indian Penal Code."""
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.return_value = []
        mock_qb.hierarchy_navigation.return_value = []

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        await searcher.search("S. 302 IPC")
        mock_qb.citation_traversal.assert_called_once_with(section="302", act="Indian Penal Code")

    @pytest.mark.asyncio
    async def test_kg_error_isolated(self) -> None:
        """KG errors should not crash — results from other refs still returned."""
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.side_effect = RuntimeError("KG down")
        mock_qb.hierarchy_navigation.side_effect = RuntimeError("KG down")

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        # Should not raise
        results = await searcher.search("Section 420 of Indian Penal Code")
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_references(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.return_value = [
            {"citation": "AIR 2020 SC 123"},
        ]
        mock_qb.hierarchy_navigation.return_value = []

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        await searcher.search("Section 10 of Indian Contract Act, Section 302 of Indian Penal Code")
        # Both refs queried
        assert mock_qb.citation_traversal.call_count == 2

    @pytest.mark.asyncio
    async def test_graph_results_score_is_one(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.return_value = [
            {"citation": "2023 SCC 456"},
        ]
        mock_qb.hierarchy_navigation.return_value = []

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        results = await searcher.search("Section 5 of Limitation Act")
        for r in results:
            assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_deduplicates_results(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        # Same citation returned for both calls
        mock_qb.citation_traversal.return_value = [
            {"citation": "AIR 2020 SC 123"},
            {"citation": "AIR 2020 SC 123"},
        ]
        mock_qb.hierarchy_navigation.return_value = []

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        results = await searcher.search("Section 420 of Indian Penal Code")
        chunk_ids = [r.chunk_id for r in results]
        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.asyncio
    async def test_max_results_cap(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.return_value = [{"citation": f"CASE-{i}"} for i in range(100)]
        mock_qb.hierarchy_navigation.return_value = []

        searcher = GraphSearcher(RetrievalSettings(graph_max_results=5))
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        results = await searcher.search("Section 420 of Indian Penal Code")
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        searcher = GraphSearcher(RetrievalSettings())
        mock_client = AsyncMock()
        searcher._client = mock_client
        searcher._query_builder = AsyncMock()

        await searcher.close()
        mock_client.close.assert_called_once()
        assert searcher._client is None
        assert searcher._query_builder is None

    @pytest.mark.asyncio
    async def test_close_noop_when_not_initialised(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        searcher = GraphSearcher(RetrievalSettings())
        await searcher.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_hierarchy_results_have_statute_type(self) -> None:
        from src.retrieval._searchers import GraphSearcher

        mock_qb = AsyncMock()
        mock_qb.citation_traversal.return_value = []
        mock_qb.hierarchy_navigation.return_value = [
            {"number": "10", "chapter": "I", "is_in_force": True},
        ]

        searcher = GraphSearcher(RetrievalSettings())
        searcher._query_builder = mock_qb
        searcher._client = MagicMock()

        results = await searcher.search("Section 10 of Indian Contract Act")
        statute_results = [r for r in results if r.document_type == "statute"]
        assert len(statute_results) >= 1


# ===================================================================
# extract_section_references helper
# ===================================================================


class TestExtractSectionReferences:
    """Tests for the regex-based section reference extractor."""

    def test_section_x_of_y(self) -> None:
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("Section 420 of Indian Penal Code")
        assert len(refs) >= 1
        assert refs[0] == ("420", "Indian Penal Code")

    def test_s_dot_pattern(self) -> None:
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("S. 302 IPC")
        assert len(refs) >= 1
        assert refs[0][0] == "302"
        assert refs[0][1] == "Indian Penal Code"

    def test_sec_pattern(self) -> None:
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("Sec 10 Contract Act")
        assert len(refs) >= 1
        assert refs[0][0] == "10"

    def test_no_references(self) -> None:
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("What is the meaning of justice?")
        assert refs == []

    def test_alphanumeric_section(self) -> None:
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("Section 498A of Indian Penal Code")
        assert any(r[0] == "498A" for r in refs)

    def test_crpc_alias(self) -> None:
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("S. 144 CrPC")
        assert any(r[1] == "Code of Criminal Procedure" for r in refs)

    def test_section_of_the_act(self) -> None:
        """'Section X of the Y Act' pattern."""
        from src.retrieval._searchers import extract_section_references

        refs = extract_section_references("Section 10 of the Contract Act")
        assert len(refs) >= 1
        assert refs[0][0] == "10"


# ===================================================================
# _point_to_scored_chunk helper
# ===================================================================


class TestPointToScoredChunk:
    def test_converts_to_scored_chunk(self) -> None:
        from src.retrieval._searchers import _point_to_scored_chunk

        pt = _mock_point("c-1", 0.95, "some text", "statute", "statutory_text")
        result = _point_to_scored_chunk(pt, "dense")
        assert isinstance(result, ScoredChunk)
        assert result.chunk_id == "c-1"
        assert result.score == 0.95
        assert result.channel == "dense"
        assert result.text == "some text"

    def test_missing_payload_fields(self) -> None:
        from src.retrieval._searchers import _point_to_scored_chunk

        pt = MagicMock()
        pt.id = "x"
        pt.score = 0.5
        pt.payload = {}
        result = _point_to_scored_chunk(pt, "bm25")
        assert result.chunk_id == "x"
        assert result.text == ""
        assert result.document_type is None

    def test_none_payload(self) -> None:
        from src.retrieval._searchers import _point_to_scored_chunk

        pt = MagicMock()
        pt.id = "y"
        pt.score = 0.3
        pt.payload = None
        result = _point_to_scored_chunk(pt, "quim")
        assert result.chunk_id == "y"
