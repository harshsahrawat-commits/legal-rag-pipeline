"""Tests for Layer 4: GenGround Refiner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.hallucination._exceptions import GenGroundError, GenGroundNotAvailableError
from src.hallucination._genground_refiner import GenGroundRefiner, _parse_json
from src.hallucination._models import (
    ClaimVerdictType,
    HallucinationSettings,
)
from src.retrieval._models import ExpandedContext, ScoredChunk


@pytest.fixture()
def settings() -> HallucinationSettings:
    return HallucinationSettings()


@pytest.fixture()
def disabled_settings() -> HallucinationSettings:
    return HallucinationSettings(genground_enabled=False)


@pytest.fixture()
def sample_chunks() -> list[ExpandedContext]:
    return [
        ExpandedContext(
            chunk_id="c1",
            chunk_text="Section 420 deals with cheating.",
            relevance_score=0.9,
        ),
        ExpandedContext(
            chunk_id="c2",
            chunk_text="Mens rea is essential for S.420.",
            relevance_score=0.85,
        ),
    ]


def _mock_anthropic_response(text: str) -> MagicMock:
    """Build a mock Anthropic message response."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


class TestParseJson:
    def test_valid_json(self) -> None:
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_array(self) -> None:
        result = _parse_json('[{"text": "claim"}]')
        assert isinstance(result, list)

    def test_markdown_code_block(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        assert _parse_json(raw) == {"key": "value"}

    def test_invalid_json(self) -> None:
        assert _parse_json("not json at all") == {}

    def test_empty_string(self) -> None:
        assert _parse_json("") == {}


class TestGenGroundDisabled:
    async def test_returns_unmodified(
        self,
        disabled_settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(disabled_settings)
        modified, verdicts = await refiner.verify("test response", sample_chunks)
        assert modified == "test response"
        assert verdicts == []
        assert refiner.llm_calls == 0


class TestSimpleAudit:
    async def test_supported_response(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(
                json.dumps({"verdict": "supported", "issues": []})
            )
        )
        refiner._client = mock_client

        modified, verdicts = await refiner.verify(
            "Section 420 punishes cheating.", sample_chunks, is_simple=True
        )
        assert modified == "Section 420 punishes cheating."
        assert len(verdicts) == 1
        assert verdicts[0].verdict == ClaimVerdictType.SUPPORTED
        assert refiner.llm_calls == 1

    async def test_partially_supported(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(
                json.dumps(
                    {
                        "verdict": "partially_supported",
                        "issues": ["Maximum sentence claim unverified"],
                    }
                )
            )
        )
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=True)
        assert verdicts[0].verdict == ClaimVerdictType.PARTIALLY_SUPPORTED
        assert len(verdicts[0].issues) == 1

    async def test_unsupported(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(
                json.dumps({"verdict": "unsupported", "issues": ["No evidence"]})
            )
        )
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=True)
        assert verdicts[0].verdict == ClaimVerdictType.UNSUPPORTED


class TestFullVerify:
    async def test_claim_extraction_and_alignment(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()

        # Call 1: claim extraction
        # Call 2: claim alignment
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_anthropic_response(
                    json.dumps(
                        [
                            {"claim_id": 1, "text": "Section 420 punishes cheating"},
                        ]
                    )
                ),
                _mock_anthropic_response(
                    json.dumps(
                        {
                            "verdict": "supported",
                            "confidence": 0.95,
                            "reasoning": "matches source",
                            "issues": [],
                        }
                    )
                ),
            ]
        )
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify(
            "Section 420 punishes cheating.", sample_chunks, is_simple=False
        )
        assert len(verdicts) == 1
        assert verdicts[0].verdict == ClaimVerdictType.SUPPORTED
        assert verdicts[0].confidence == 0.95
        assert refiner.llm_calls == 2

    async def test_multiple_claims(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_anthropic_response(
                    json.dumps(
                        [
                            {"claim_id": 1, "text": "claim 1"},
                            {"claim_id": 2, "text": "claim 2"},
                        ]
                    )
                ),
                _mock_anthropic_response(
                    json.dumps(
                        {"verdict": "supported", "confidence": 0.9, "reasoning": "ok", "issues": []}
                    )
                ),
                _mock_anthropic_response(
                    json.dumps(
                        {
                            "verdict": "unsupported",
                            "confidence": 0.2,
                            "reasoning": "no evidence",
                            "issues": ["no source"],
                        }
                    )
                ),
            ]
        )
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response text", sample_chunks, is_simple=False)
        assert len(verdicts) == 2
        assert verdicts[0].verdict == ClaimVerdictType.SUPPORTED
        assert verdicts[1].verdict == ClaimVerdictType.UNSUPPORTED
        assert refiner.llm_calls == 3

    async def test_unsupported_claims_add_caveat(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_anthropic_response(json.dumps([{"claim_id": 1, "text": "false claim"}])),
                _mock_anthropic_response(
                    json.dumps(
                        {
                            "verdict": "unsupported",
                            "confidence": 0.1,
                            "reasoning": "no match",
                            "issues": ["fabricated"],
                        }
                    )
                ),
            ]
        )
        refiner._client = mock_client

        modified, _verdicts = await refiner.verify("Some response.", sample_chunks, is_simple=False)
        assert "could not be verified" in modified

    async def test_partial_claims_add_caveat(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_anthropic_response(json.dumps([{"claim_id": 1, "text": "partial claim"}])),
                _mock_anthropic_response(
                    json.dumps(
                        {
                            "verdict": "partially_supported",
                            "confidence": 0.5,
                            "reasoning": "half match",
                            "issues": ["incomplete"],
                        }
                    )
                ),
            ]
        )
        refiner._client = mock_client

        modified, _verdicts = await refiner.verify("Some response.", sample_chunks, is_simple=False)
        assert "partially" in modified.lower()

    async def test_no_claims_extracted(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()

        mock_client.messages.create = AsyncMock(return_value=_mock_anthropic_response("[]"))
        refiner._client = mock_client

        modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=False)
        assert modified == "response"
        assert verdicts == []


class TestReRetrieval:
    async def test_with_retrieval_engine(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        mock_engine = AsyncMock()
        mock_engine.hybrid_search = AsyncMock(
            return_value=[
                ScoredChunk(
                    chunk_id="re-1",
                    text="Re-retrieved evidence",
                    score=0.9,
                    channel="dense",
                ),
            ]
        )
        refiner = GenGroundRefiner(settings, retrieval_engine=mock_engine)
        mock_client = AsyncMock()

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_anthropic_response(json.dumps([{"claim_id": 1, "text": "test claim"}])),
                _mock_anthropic_response(
                    json.dumps(
                        {
                            "verdict": "supported",
                            "confidence": 0.95,
                            "reasoning": "ok",
                            "issues": [],
                        }
                    )
                ),
            ]
        )
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=False)
        mock_engine.hybrid_search.assert_called_once()
        assert verdicts[0].evidence_chunk_ids == ["re-1"]

    async def test_re_retrieval_failure_fallback(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        mock_engine = AsyncMock()
        mock_engine.hybrid_search = AsyncMock(side_effect=Exception("search failed"))
        refiner = GenGroundRefiner(settings, retrieval_engine=mock_engine)
        mock_client = AsyncMock()

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_anthropic_response(json.dumps([{"claim_id": 1, "text": "test claim"}])),
                _mock_anthropic_response(
                    json.dumps(
                        {"verdict": "supported", "confidence": 0.8, "reasoning": "ok", "issues": []}
                    )
                ),
            ]
        )
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=False)
        # Falls back to initial chunks, doesn't crash
        assert len(verdicts) == 1


class TestLLMErrors:
    async def test_llm_call_failure(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))
        refiner._client = mock_client

        with pytest.raises(GenGroundError, match="LLM call failed"):
            await refiner.verify("response", sample_chunks, is_simple=True)

    async def test_anthropic_not_installed(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        # Don't set _client — force _ensure_client to run
        with (
            patch.dict("sys.modules", {"anthropic": None}),
            pytest.raises(GenGroundNotAvailableError),
        ):
            await refiner.verify("response", sample_chunks, is_simple=True)

    async def test_invalid_json_from_llm(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response("not valid json")
        )
        refiner._client = mock_client

        # Simple audit with invalid JSON — should handle gracefully
        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=True)
        assert len(verdicts) == 1
        # Invalid JSON → empty issues, default verdict

    async def test_empty_llm_response(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        refiner = GenGroundRefiner(settings)
        mock_client = AsyncMock()
        response = MagicMock()
        response.content = []
        mock_client.messages.create = AsyncMock(return_value=response)
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=True)
        # Empty response → parsed as {} → default handling
        assert len(verdicts) == 1


class TestMaxClaims:
    async def test_claims_capped(
        self,
        settings: HallucinationSettings,
        sample_chunks: list[ExpandedContext],
    ) -> None:
        settings_capped = HallucinationSettings(genground_max_claims=2)
        refiner = GenGroundRefiner(settings_capped)
        mock_client = AsyncMock()

        # Extract 5 claims but only 2 should be verified
        claims = [{"claim_id": i, "text": f"claim {i}"} for i in range(5)]
        responses = [
            _mock_anthropic_response(json.dumps(claims)),
            _mock_anthropic_response(
                json.dumps(
                    {"verdict": "supported", "confidence": 0.9, "reasoning": "ok", "issues": []}
                )
            ),
            _mock_anthropic_response(
                json.dumps(
                    {"verdict": "supported", "confidence": 0.9, "reasoning": "ok", "issues": []}
                )
            ),
        ]
        mock_client.messages.create = AsyncMock(side_effect=responses)
        refiner._client = mock_client

        _modified, verdicts = await refiner.verify("response", sample_chunks, is_simple=False)
        assert len(verdicts) == 2  # Capped at max_claims
        assert refiner.llm_calls == 3  # 1 extraction + 2 alignments
