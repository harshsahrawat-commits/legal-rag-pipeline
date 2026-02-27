"""Shared fixtures for evaluation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.evaluation._models import (
    EvaluationInput,
    EvaluationSettings,
    TestQuery,
    TestQueryDataset,
)


@pytest.fixture()
def evaluation_settings() -> EvaluationSettings:
    """Default evaluation settings for tests."""
    return EvaluationSettings()


@pytest.fixture()
def sample_test_query() -> TestQuery:
    """A single test query fixture."""
    return TestQuery(
        query_id="test_001",
        query_text="What is the punishment for cheating under BNS?",
        practice_area="criminal",
        query_type="factual",
        expected_route="standard",
        expected_citations=["Section 318 BNS"],
        expected_answer_contains=["imprisonment", "fine"],
        reference_answer="Section 318 BNS prescribes imprisonment and fine for cheating.",
        expected_sections=["318"],
    )


@pytest.fixture()
def sample_test_dataset(sample_test_query: TestQuery) -> TestQueryDataset:
    """A minimal test query dataset."""
    return TestQueryDataset(
        version="1.0",
        description="Test fixture dataset",
        queries=[
            sample_test_query,
            TestQuery(
                query_id="test_002",
                query_text="What was the bail provision under CrPC before BNSS?",
                practice_area="criminal",
                query_type="temporal",
                expected_route="complex",
                expected_citations=["Section 437 CrPC", "Section 478 BNSS"],
                expected_answer_contains=["bail", "non-bailable"],
                reference_answer="Section 437 CrPC provided bail in non-bailable cases.",
                expected_sections=["437", "478"],
                temporal_test=True,
            ),
            TestQuery(
                query_id="test_003",
                query_text="How do Sections 73 and 74 Contract Act interact?",
                practice_area="civil_contract",
                query_type="cross_reference",
                expected_route="complex",
                expected_citations=["Section 73 Contract Act", "Section 74 Contract Act"],
                expected_answer_contains=["damages", "compensation"],
                reference_answer="Section 73 provides general damages, Section 74 liquidated damages.",
                expected_sections=["73", "74"],
                cross_reference_test=True,
            ),
        ],
    )


@pytest.fixture()
def sample_evaluation_input() -> EvaluationInput:
    """An EvaluationInput with upstream results populated."""
    return EvaluationInput(
        query_id="test_001",
        query_text="What is the punishment for cheating under BNS?",
        practice_area="criminal",
        query_type="factual",
        expected_route="standard",
        expected_citations=["Section 318 BNS"],
        expected_answer_contains=["imprisonment", "fine"],
        expected_sections=["318"],
        reference_answer="Section 318 BNS prescribes imprisonment and fine for cheating.",
        response_text="Under Section 318 of the Bharatiya Nyaya Sanhita, cheating is punishable with imprisonment up to 3 years and fine.",
        retrieved_contexts=[
            "Section 318. Cheating.—Whoever cheats shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
            "Section 319. Cheating by personation.—Whoever cheats by personation shall be punished with imprisonment for a term which may extend to five years.",
        ],
        qi_result={"cache_hit": False, "route": "standard"},
        retrieval_result={
            "flare_retrievals": 0,
            "chunks": [{"parent_text": "Full act text...", "chunk_text": "Section 318..."}],
        },
        verification_result={
            "summary": {"supported_claims": 2, "total_claims": 2},
        },
        total_elapsed_ms=450.0,
    )


@pytest.fixture()
def sample_evaluation_inputs() -> list[EvaluationInput]:
    """Multiple EvaluationInputs for batch testing."""
    return [
        EvaluationInput(
            query_id="batch_001",
            query_text="What is Section 302 IPC?",
            practice_area="criminal",
            query_type="factual",
            expected_route="simple",
            expected_citations=["Section 302 IPC"],
            expected_sections=["302"],
            response_text="Section 302 of the Indian Penal Code prescribes punishment for murder.",
            retrieved_contexts=[
                "Section 302. Punishment for murder.—Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            ],
            qi_result={"cache_hit": True, "route": "simple"},
            retrieval_result={"flare_retrievals": 0, "chunks": [{"parent_text": None}]},
            verification_result={"summary": {"supported_claims": 1, "total_claims": 1}},
            total_elapsed_ms=150.0,
            temporal_test=True,
        ),
        EvaluationInput(
            query_id="batch_002",
            query_text="How do Sections 7 and 9 IBC differ?",
            practice_area="corporate_commercial",
            query_type="cross_reference",
            expected_route="complex",
            expected_citations=["Section 7 IBC", "Section 9 IBC"],
            expected_sections=["7", "9"],
            cross_reference_test=True,
            response_text="Section 7 IBC allows financial creditors to initiate insolvency. Section 9 IBC allows operational creditors.",
            retrieved_contexts=[
                "Section 7. Initiation of corporate insolvency resolution process by financial creditor.",
                "Section 9. Application for initiation of corporate insolvency resolution process by operational creditor.",
            ],
            qi_result={"cache_hit": False, "route": "complex"},
            retrieval_result={
                "flare_retrievals": 0,
                "chunks": [
                    {"parent_text": "Full IBC text..."},
                    {"parent_text": "Full IBC text..."},
                ],
            },
            verification_result={"summary": {"supported_claims": 3, "total_claims": 3}},
            total_elapsed_ms=1200.0,
        ),
        EvaluationInput(
            query_id="batch_003",
            query_text="What deductions under Section 80C?",
            practice_area="tax",
            query_type="factual",
            expected_route="standard",
            expected_citations=["Section 80C IT Act"],
            expected_sections=["80C"],
            response_text="Section 80C allows deduction up to Rs. 1.5 lakh for PPF, ELSS, and other investments.",
            retrieved_contexts=[
                "Section 80C. Deduction in respect of life insurance premia, deferred annuity, contributions to provident fund, subscription to certain equity shares or debentures, etc.",
            ],
            qi_result={"cache_hit": False, "route": "standard"},
            retrieval_result={
                "flare_retrievals": 0,
                "chunks": [{"parent_text": "Full IT Act text..."}],
            },
            verification_result={"summary": {"supported_claims": 2, "total_claims": 2}},
            total_elapsed_ms=600.0,
        ),
    ]


@pytest.fixture()
def test_queries_path(tmp_path: Path, sample_test_dataset: TestQueryDataset) -> Path:
    """Write sample dataset to a temp file and return the path."""
    path = tmp_path / "test_queries.json"
    path.write_text(sample_test_dataset.model_dump_json(indent=2), encoding="utf-8")
    return path
