"""Live smoke test: verify LLM provider abstraction works against real Ollama.

Exercises 4 components end-to-end:
1. Provider factory + basic completion
2. SelectiveHyDE (sync, COMPLEX route)
3. ContextualRetrievalEnricher (async, prompt caching)
4. GenGroundRefiner (async, JSON parsing)

Requires: Ollama running at localhost:11434 with qwen3:14b loaded.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_1_provider_factory() -> None:
    """Test: get_llm_provider returns a working OllamaProvider."""
    _banner("Test 1: Provider Factory + Basic Completion")

    from src.utils._llm_client import LLMMessage, get_llm_provider

    t0 = time.perf_counter()
    provider = get_llm_provider("hyde")
    print(f"  Provider: {provider.provider_name}")
    print(f"  Available: {provider.is_available}")

    response = provider.complete(
        [LLMMessage(role="user", content="What is Section 302 of the Indian Penal Code? Answer in one sentence. /no_think")],
        max_tokens=256,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Model: {response.model}")
    print(f"  Response ({len(response.text)} chars, {elapsed:.1f}s):")
    print(f"    {response.text[:200]}")
    assert len(response.text) > 10, "Response too short"
    assert response.provider == "ollama"
    print("  PASSED")


def test_2_hyde() -> None:
    """Test: SelectiveHyDE generates a hypothetical answer for COMPLEX queries."""
    _banner("Test 2: SelectiveHyDE (sync, COMPLEX route)")

    from src.query._hyde import SelectiveHyDE
    from src.query._models import QuerySettings
    from src.retrieval._models import QueryRoute

    settings = QuerySettings()
    hyde = SelectiveHyDE(settings)

    print(f"  Available: {hyde.is_available}")

    t0 = time.perf_counter()
    result = hyde.maybe_generate(
        "What is the interplay between Section 498A IPC and the Domestic Violence Act?",
        QueryRoute.COMPLEX,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Generated: {result.generated}")
    if result.hypothetical_text:
        print(f"  Hypothetical ({len(result.hypothetical_text)} chars, {elapsed:.1f}s):")
        print(f"    {result.hypothetical_text[:200]}...")
    assert result.generated, "HyDE should generate for COMPLEX route"
    assert result.hypothetical_text and len(result.hypothetical_text) > 20
    print("  PASSED")


async def test_3_contextual_enrichment() -> None:
    """Test: ContextualRetrievalEnricher contextualizes a chunk."""
    _banner("Test 3: ContextualRetrievalEnricher (async)")

    from uuid import uuid4

    from src.acquisition._models import ContentFormat, DocumentType, SourceType
    from src.chunking._models import (
        ChunkStrategy,
        ChunkType,
        ContentMetadata,
        IngestionMetadata,
        LegalChunk,
        ParentDocumentInfo,
        SourceInfo,
        StatuteMetadata,
    )
    from src.enrichment._models import EnrichmentSettings
    from src.enrichment.enrichers._contextual import ContextualRetrievalEnricher
    from src.parsing._models import ParsedDocument, ParserType, QualityReport

    doc = ParsedDocument(
        document_id=uuid4(),
        source_type=SourceType.INDIA_CODE,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.HTML,
        raw_text="Section 10. All agreements are contracts which are made by free consent of parties.",
        sections=[],
        act_name="Indian Contract Act",
        act_number="9 of 1872",
        parser_used=ParserType.HTML_INDIAN_KANOON,
        quality=QualityReport(overall_score=0.9, passed=True),
        raw_content_path="data/raw/test.html",
    )

    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    chunk = LegalChunk(
        id=uuid4(),
        document_id=doc.document_id,
        text="Section 10. All agreements are contracts which are made by free consent of parties.",
        document_type=DocumentType.STATUTE,
        chunk_type=ChunkType.STATUTORY_TEXT,
        chunk_index=0,
        token_count=15,
        source=SourceInfo(
            url="https://example.com",
            source_name="India Code",
            scraped_at=now,
            last_verified=now,
        ),
        statute=StatuteMetadata(act_name="Indian Contract Act", section_number="10"),
        content=ContentMetadata(),
        ingestion=IngestionMetadata(
            ingested_at=now,
            parser="test",
            chunk_strategy=ChunkStrategy.STRUCTURE_BOUNDARY,
        ),
        parent_info=ParentDocumentInfo(),
    )

    settings = EnrichmentSettings(concurrency=1)
    enricher = ContextualRetrievalEnricher(settings)

    t0 = time.perf_counter()
    result = await enricher.enrich_document([chunk], doc)
    elapsed = time.perf_counter() - t0

    enriched = result[0]
    print(f"  Contextualized: {enriched.ingestion.contextualized}")
    if enriched.contextualized_text:
        print(f"  Text ({len(enriched.contextualized_text)} chars, {elapsed:.1f}s):")
        # Show the context prefix (before \n\n)
        parts = enriched.contextualized_text.split("\n\n", 1)
        print(f"    Context: {parts[0][:200]}")
    assert enriched.ingestion.contextualized, "Chunk should be contextualized"
    assert enriched.contextualized_text and enriched.text in enriched.contextualized_text
    print("  PASSED")


async def test_4_genground() -> None:
    """Test: GenGroundRefiner does a simple audit."""
    _banner("Test 4: GenGroundRefiner (async, JSON parsing)")

    from src.hallucination._genground_refiner import GenGroundRefiner
    from src.hallucination._models import HallucinationSettings
    from src.retrieval._models import ExpandedContext

    settings = HallucinationSettings(genground_enabled=True)
    refiner = GenGroundRefiner(settings)

    chunks = [
        ExpandedContext(
            chunk_id="c1",
            chunk_text="Section 302 of the Indian Penal Code deals with punishment for murder. The punishment is death or imprisonment for life, and also fine.",
            relevance_score=0.95,
        ),
    ]

    t0 = time.perf_counter()
    modified, verdicts = await refiner.verify(
        "Section 302 IPC prescribes death or life imprisonment for murder.",
        chunks,
        is_simple=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"  LLM calls: {refiner.llm_calls}")
    print(f"  Verdicts: {len(verdicts)}")
    if verdicts:
        v = verdicts[0]
        print(f"  Verdict: {v.verdict.value} (confidence: {v.confidence})")
        print(f"  Issues: {v.issues}")
    print(f"  Elapsed: {elapsed:.1f}s")
    assert len(verdicts) == 1, "Should produce exactly 1 verdict"
    assert refiner.llm_calls == 1
    print("  PASSED")


async def main() -> None:
    print("Legal RAG Pipeline — LLM Provider Smoke Test")
    print(f"Target: Ollama @ localhost:11434 (qwen3:14b)")

    passed = 0
    failed = 0

    # Test 1: sync
    try:
        test_1_provider_factory()
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 2: sync
    try:
        test_2_hyde()
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 3: async
    try:
        await test_3_contextual_enrichment()
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 4: async
    try:
        await test_4_genground()
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    _banner(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
